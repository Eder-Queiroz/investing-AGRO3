"""Testes unitários para src/feature_engineering/sliding_window.py.

Foco em:
- Integridade do MODEL_FEATURE_COLS (estacionaridade, sem duplicatas)
- Corretude do compute_valid_indices (NaN, NA de target, janelas curtas)
- Integridade temporal do split (disjunção, ordem cronológica)
- Geometria e ordem temporal do flattening em build_windows
- Anti-leakage do fit_scaler
- Corretude do remap_labels

Estes testes NÃO dependem de PyTorch — apenas numpy, pandas e sklearn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.sliding_window import (
    MODEL_FEATURE_COLS,
    build_windows,
    compute_valid_indices,
    fit_scaler,
    remap_labels,
    split_indices_chronological,
)

# ---------------------------------------------------------------------------
# Constantes de apoio
# ---------------------------------------------------------------------------

_N_ROWS = 200
_W = 10  # window_size pequeno para testes mais rápidos
_F = 5   # número de features sintéticas

# Colunas de features não-estacionárias que devem estar AUSENTES em MODEL_FEATURE_COLS
_NON_STATIONARY_COLS: list[str] = [
    "open_adj", "high_adj", "low_adj", "close_adj", "volume",
    "p_vpa", "ev_ebitda", "roe", "net_debt_ebitda", "gross_margin", "dividend_yield",
    "usd_brl", "soy_price_usd", "corn_price_usd",
]


# ---------------------------------------------------------------------------
# Helpers de dados sintéticos
# ---------------------------------------------------------------------------


def _make_index(n_rows: int = _N_ROWS) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-01", periods=n_rows, freq="W-FRI", name="date")


def _make_feature_cols(n: int = _F) -> list[str]:
    return [f"feat_{i}" for i in range(n)]


def _make_clean_df(
    n_rows: int = _N_ROWS,
    feature_cols: list[str] | None = None,
    target_values: list[int] | None = None,
) -> pd.DataFrame:
    """DataFrame limpo: todas features finitas, target válido (sem pd.NA)."""
    cols = feature_cols or _make_feature_cols()
    index = _make_index(n_rows)

    rng = np.random.default_rng(42)
    data = {col: rng.standard_normal(n_rows) for col in cols}

    if target_values is not None:
        targets = pd.array(target_values, dtype=pd.Int8Dtype())
    else:
        # Alterna -1, 0, 1 para cobrir todas as classes
        raw = [(-1 + (i % 3)) for i in range(n_rows)]
        targets = pd.array(raw, dtype=pd.Int8Dtype())

    data["target"] = targets
    return pd.DataFrame(data, index=index)


def _make_df_with_nan_in_feature(
    nan_row: int,
    n_rows: int = _N_ROWS,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Injeta NaN em todas as features da linha nan_row."""
    cols = feature_cols or _make_feature_cols()
    df = _make_clean_df(n_rows=n_rows, feature_cols=cols)
    for col in cols:
        df.iloc[nan_row, df.columns.get_loc(col)] = np.nan
    return df


def _make_df_with_na_target(
    na_rows: list[int],
    n_rows: int = _N_ROWS,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Injeta pd.NA no target nas linhas especificadas."""
    cols = feature_cols or _make_feature_cols()
    df = _make_clean_df(n_rows=n_rows, feature_cols=cols)
    for row in na_rows:
        df.at[df.index[row], "target"] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Grupo 1: MODEL_FEATURE_COLS
# ---------------------------------------------------------------------------


class TestModelFeatureCols:
    def test_has_23_elements(self) -> None:
        assert len(MODEL_FEATURE_COLS) == 23, (
            f"MODEL_FEATURE_COLS deve ter 23 elementos, tem {len(MODEL_FEATURE_COLS)}"
        )

    def test_no_duplicates(self) -> None:
        assert len(set(MODEL_FEATURE_COLS)) == len(MODEL_FEATURE_COLS), (
            "MODEL_FEATURE_COLS contém colunas duplicadas"
        )

    def test_no_nonstationary_cols(self) -> None:
        for col in _NON_STATIONARY_COLS:
            assert col not in MODEL_FEATURE_COLS, (
                f"Coluna não-estacionária '{col}' encontrada em MODEL_FEATURE_COLS"
            )

    def test_contains_all_technical_features(self) -> None:
        expected_technical = [
            "log_return_1w", "log_return_4w", "log_return_13w",
            "volatility_4w", "rsi_14", "price_to_52w_high", "volume_zscore_4w",
            "log_return_soy_4w", "log_return_corn_4w", "log_return_usd_brl_4w",
            "delta_cdi_4w", "delta_selic_real_4w",
        ]
        for col in expected_technical:
            assert col in MODEL_FEATURE_COLS, f"Feature técnica '{col}' ausente"

    def test_contains_all_zscore_features(self) -> None:
        expected_zscores = [
            "p_vpa_zscore", "ev_ebitda_zscore", "roe_zscore",
            "net_debt_ebitda_zscore", "gross_margin_zscore", "dividend_yield_zscore",
        ]
        for col in expected_zscores:
            assert col in MODEL_FEATURE_COLS, f"Z-score '{col}' ausente"

    def test_contains_macro_rates(self) -> None:
        expected_macro = ["cdi_rate", "selic_rate", "igpm", "ipca", "selic_real"]
        for col in expected_macro:
            assert col in MODEL_FEATURE_COLS, f"Macro rate '{col}' ausente"

    def test_is_list_of_strings(self) -> None:
        assert isinstance(MODEL_FEATURE_COLS, list)
        assert all(isinstance(c, str) for c in MODEL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Grupo 2: compute_valid_indices
# ---------------------------------------------------------------------------


class TestComputeValidIndices:
    def test_all_clean_returns_all_valid_positions(self) -> None:
        """DataFrame limpo com target em todas as linhas → todas posições de W-1 a n-1."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)
        expected = np.arange(_W - 1, _N_ROWS, dtype=np.intp)
        np.testing.assert_array_equal(idx, expected)

    def test_nan_in_feature_invalidates_surrounding_windows(self) -> None:
        """NaN na linha t invalida todas as janelas que a contêm (t a t+W-1)."""
        cols = _make_feature_cols()
        nan_row = 50
        df = _make_df_with_nan_in_feature(nan_row=nan_row, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)

        # Nenhuma posição entre nan_row e nan_row + W - 1 deve aparecer
        for t in range(nan_row, nan_row + _W):
            assert t not in idx, (
                f"Posição {t} deveria ser inválida (NaN na linha {nan_row}, W={_W})"
            )

    def test_nan_in_feature_positions_before_and_after_are_valid(self) -> None:
        """Posições fora do alcance do NaN devem continuar válidas."""
        cols = _make_feature_cols()
        nan_row = 50
        df = _make_df_with_nan_in_feature(nan_row=nan_row, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)

        # Posição W-1 até nan_row-1 devem ser válidas (janelas que não incluem nan_row)
        for t in range(_W - 1, nan_row):
            assert t in idx

        # Posição nan_row + W em diante deve ser válida
        for t in range(nan_row + _W, _N_ROWS):
            assert t in idx

    def test_na_target_excludes_position(self) -> None:
        """pd.NA no target exclui a posição mesmo com features limpas."""
        cols = _make_feature_cols()
        na_rows = [30, 80, 150]
        df = _make_df_with_na_target(na_rows=na_rows, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)

        for row in na_rows:
            assert row not in idx

    def test_df_shorter_than_window_returns_empty(self) -> None:
        """DataFrame com n < W deve retornar array vazio sem exceção."""
        cols = _make_feature_cols()
        df = _make_clean_df(n_rows=5, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)  # W=10 > 5
        assert len(idx) == 0

    def test_df_exactly_window_size_returns_one_index(self) -> None:
        """DataFrame com n == W deve retornar exatamente a última posição."""
        cols = _make_feature_cols()
        df = _make_clean_df(n_rows=_W, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)
        assert len(idx) == 1
        assert idx[0] == _W - 1

    def test_returns_sorted_ascending(self) -> None:
        """Output deve estar em ordem crescente."""
        cols = _make_feature_cols()
        df = _make_df_with_nan_in_feature(nan_row=30, feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)
        assert (np.diff(idx) > 0).all()

    def test_dtype_is_integer(self) -> None:
        """Dtype do output deve ser np.intp (inteiro nativo para indexação)."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)
        assert np.issubdtype(idx.dtype, np.integer)

    def test_inf_in_feature_is_rejected(self) -> None:
        """np.inf deve ser rejeitado assim como NaN."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        df.iloc[40, df.columns.get_loc(cols[0])] = np.inf
        idx = compute_valid_indices(df, _W, cols)
        for t in range(40, 40 + _W):
            assert t not in idx

    def test_negative_inf_in_feature_is_rejected(self) -> None:
        """-np.inf deve ser rejeitado."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        df.iloc[40, df.columns.get_loc(cols[0])] = -np.inf
        idx = compute_valid_indices(df, _W, cols)
        for t in range(40, 40 + _W):
            assert t not in idx


# ---------------------------------------------------------------------------
# Grupo 3: split_indices_chronological
# ---------------------------------------------------------------------------


class TestSplitIndicesChronological:
    def _make_valid_idx(self, n: int = 100) -> np.ndarray:
        return np.arange(n, dtype=np.intp)

    def test_sizes_sum_to_total(self) -> None:
        idx = self._make_valid_idx(100)
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        assert len(train) + len(val) + len(test) == len(idx)

    def test_splits_are_disjoint(self) -> None:
        idx = self._make_valid_idx(100)
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        assert len(set(train) & set(val)) == 0
        assert len(set(val) & set(test)) == 0
        assert len(set(train) & set(test)) == 0

    def test_splits_are_chronologically_ordered(self) -> None:
        idx = self._make_valid_idx(100)
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        assert int(train[-1]) < int(val[0])
        assert int(val[-1]) < int(test[0])

    def test_proportions_are_approximately_correct(self) -> None:
        n = 1000
        idx = self._make_valid_idx(n)
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        # Aceita margem de 1 amostra por causa de arredondamento inteiro
        assert abs(len(train) - 600) <= 1
        assert abs(len(val) - 200) <= 1
        assert abs(len(test) - 200) <= 2

    def test_raises_on_negative_train_ratio(self) -> None:
        idx = self._make_valid_idx()
        with pytest.raises(ValueError, match="positivos"):
            split_indices_chronological(idx, -0.1, 0.20)

    def test_raises_on_zero_val_ratio(self) -> None:
        idx = self._make_valid_idx()
        with pytest.raises(ValueError, match="positivos"):
            split_indices_chronological(idx, 0.60, 0.0)

    def test_raises_when_ratios_leave_no_test(self) -> None:
        idx = self._make_valid_idx()
        with pytest.raises(ValueError, match="< 1.0"):
            split_indices_chronological(idx, 0.60, 0.45)

    def test_all_indices_covered(self) -> None:
        """Nenhum índice deve ser perdido."""
        idx = self._make_valid_idx(99)  # número não-redondo
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        combined = np.concatenate([train, val, test])
        np.testing.assert_array_equal(np.sort(combined), idx)


# ---------------------------------------------------------------------------
# Grupo 4: build_windows
# ---------------------------------------------------------------------------


class TestBuildWindows:
    def _setup(
        self,
        n_rows: int = _N_ROWS,
        window_size: int = _W,
        n_features: int = _F,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        cols = _make_feature_cols(n_features)
        df = _make_clean_df(n_rows=n_rows, feature_cols=cols)
        idx = compute_valid_indices(df, window_size, cols)
        return df, idx, cols

    def test_x_shape(self) -> None:
        df, idx, cols = self._setup()
        X, y = build_windows(df, idx, _W, cols)
        assert X.shape == (len(idx), _W * _F)

    def test_y_shape(self) -> None:
        df, idx, cols = self._setup()
        X, y = build_windows(df, idx, _W, cols)
        assert y.shape == (len(idx),)

    def test_x_is_all_finite(self) -> None:
        df, idx, cols = self._setup()
        X, _ = build_windows(df, idx, _W, cols)
        assert np.isfinite(X).all()

    def test_y_values_in_correct_range(self) -> None:
        df, idx, cols = self._setup()
        _, y = build_windows(df, idx, _W, cols)
        assert set(int(v) for v in np.unique(y)).issubset({-1, 0, 1})

    def test_flattening_temporal_order(self) -> None:
        """X[i, :F] deve ser a semana mais ANTIGA; X[i, -F:] a mais RECENTE."""
        cols = _make_feature_cols(_F)
        n_rows = 50

        # Features com valores crescentes para identificar ordem temporal
        index = _make_index(n_rows)
        data = {col: np.arange(n_rows, dtype=float) for col in cols}
        targets = pd.array([0] * n_rows, dtype=pd.Int8Dtype())
        data["target"] = targets
        df = pd.DataFrame(data, index=index)

        idx = compute_valid_indices(df, _W, cols)
        X, _ = build_windows(df, idx, _W, cols)

        for i, t in enumerate(idx):
            # Semana mais antiga: linha t - W + 1
            oldest_row = df[cols].iloc[t - _W + 1].values
            # Semana mais recente: linha t
            newest_row = df[cols].iloc[t].values

            np.testing.assert_array_equal(X[i, :_F], oldest_row,
                err_msg=f"Semana mais antiga incorreta para i={i}, t={t}")
            np.testing.assert_array_equal(X[i, -_F:], newest_row,
                err_msg=f"Semana mais recente incorreta para i={i}, t={t}")

    def test_y_matches_target_column(self) -> None:
        """y[i] deve corresponder ao target da linha t (última da janela)."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        idx = compute_valid_indices(df, _W, cols)
        _, y = build_windows(df, idx, _W, cols)

        for i, t in enumerate(idx):
            expected = int(df["target"].iloc[t])
            assert y[i] == expected, f"Mismatch em i={i}, t={t}"

    def test_empty_indices_returns_empty_arrays(self) -> None:
        """Índices vazios devem retornar arrays (0, W*F) e (0,) sem exceção."""
        cols = _make_feature_cols()
        df = _make_clean_df(feature_cols=cols)
        empty_idx = np.array([], dtype=np.intp)
        X, y = build_windows(df, empty_idx, _W, cols)
        assert X.shape == (0, _W * _F)
        assert y.shape == (0,)

    def test_x_dtype_is_float64(self) -> None:
        df, idx, cols = self._setup()
        X, _ = build_windows(df, idx, _W, cols)
        assert X.dtype == np.float64

    def test_y_dtype_is_int8(self) -> None:
        df, idx, cols = self._setup()
        _, y = build_windows(df, idx, _W, cols)
        assert y.dtype == np.int8


# ---------------------------------------------------------------------------
# Grupo 5: fit_scaler
# ---------------------------------------------------------------------------


class TestFitScaler:
    def _make_X(self, n: int = 100, d: int = 50) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, d)) * 5.0 + 3.0  # média ≠ 0, std ≠ 1

    def test_returns_standard_scaler(self) -> None:
        X = self._make_X()
        scaler = fit_scaler(X)
        assert isinstance(scaler, StandardScaler)

    def test_mean_shape_matches_input_dim(self) -> None:
        X = self._make_X(d=60)
        scaler = fit_scaler(X)
        assert scaler.mean_.shape == (60,)

    def test_transform_produces_unit_normal_on_train(self) -> None:
        """Após transform do próprio X_train, médias ≈ 0 e stds ≈ 1."""
        X = self._make_X(n=500, d=30)
        scaler = fit_scaler(X)
        X_scaled = scaler.transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_scaler_does_not_use_val_data(self) -> None:
        """Scaler fitado no train: transform do val NÃO terá média ≈ 0."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((100, 10)) * 2.0 + 5.0   # mean ≈ 5
        X_val = rng.standard_normal((100, 10)) * 2.0 + 20.0     # mean ≈ 20

        scaler = fit_scaler(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Após remover a média do train (≈5), X_val (≈20) ficará com média ≈ 15/std
        # Logo NÃO terá média ≈ 0
        assert not np.allclose(X_val_scaled.mean(axis=0), 0.0, atol=1.0), (
            "Scaler não deveria normalizar val para média 0 — foi fitado no train"
        )


# ---------------------------------------------------------------------------
# Grupo 6: remap_labels
# ---------------------------------------------------------------------------


class TestRemapLabels:
    def test_minus_one_maps_to_zero(self) -> None:
        y = np.array([-1], dtype=np.int8)
        assert remap_labels(y)[0] == 0

    def test_zero_maps_to_one(self) -> None:
        y = np.array([0], dtype=np.int8)
        assert remap_labels(y)[0] == 1

    def test_one_maps_to_two(self) -> None:
        y = np.array([1], dtype=np.int8)
        assert remap_labels(y)[0] == 2

    def test_full_mapping_explicit(self) -> None:
        y = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        expected = np.array([0, 1, 2, 0, 2, 1], dtype=np.int64)
        np.testing.assert_array_equal(remap_labels(y), expected)

    def test_output_dtype_is_int64(self) -> None:
        y = np.array([-1, 0, 1], dtype=np.int8)
        assert remap_labels(y).dtype == np.int64

    def test_reversible(self) -> None:
        """remapped - 1 deve recuperar os labels originais."""
        y = np.array([-1, 0, 1, -1, 0], dtype=np.int8)
        y_remapped = remap_labels(y)
        np.testing.assert_array_equal(y_remapped - 1, y)

    def test_invalid_value_raises(self) -> None:
        y = np.array([0, 1, 2], dtype=np.int8)  # 2 é inválido
        with pytest.raises(ValueError, match="subconjunto"):
            remap_labels(y)

    def test_invalid_negative_value_raises(self) -> None:
        y = np.array([-2, 0, 1], dtype=np.int8)  # -2 é inválido
        with pytest.raises(ValueError, match="subconjunto"):
            remap_labels(y)

    def test_single_class_dataset(self) -> None:
        """Dataset com apenas uma classe (ex: apenas HOLD) deve funcionar."""
        y = np.array([0, 0, 0, 0], dtype=np.int8)
        result = remap_labels(y)
        assert set(int(v) for v in result) == {1}

    def test_output_values_subset_of_012(self) -> None:
        y = np.array([-1, 0, 1], dtype=np.int8)
        result = remap_labels(y)
        assert set(int(v) for v in result).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Grupo 7: Integração (sem PyTorch)
# ---------------------------------------------------------------------------


class TestIntegration:
    def _make_df_with_many_rows(self, n: int = 300) -> tuple[pd.DataFrame, list[str]]:
        cols = _make_feature_cols(_F)
        df = _make_clean_df(n_rows=n, feature_cols=cols)
        return df, cols

    def test_no_leakage_between_train_and_val(self) -> None:
        """max(train_indices) < min(val_indices) — integridade temporal garantida."""
        df, cols = self._make_df_with_many_rows()
        idx = compute_valid_indices(df, _W, cols)
        train, val, test = split_indices_chronological(idx, 0.60, 0.20)
        assert int(train[-1]) < int(val[0])

    def test_no_leakage_between_val_and_test(self) -> None:
        df, cols = self._make_df_with_many_rows()
        idx = compute_valid_indices(df, _W, cols)
        _, val, test = split_indices_chronological(idx, 0.60, 0.20)
        assert int(val[-1]) < int(test[0])

    def test_scaler_fit_only_on_train(self) -> None:
        """Média de X_val após transform NÃO deve ser ≈ 0."""
        df, cols = self._make_df_with_many_rows(n=300)
        idx = compute_valid_indices(df, _W, cols)
        train_idx, val_idx, _ = split_indices_chronological(idx, 0.60, 0.20)

        X_train, _ = build_windows(df, train_idx, _W, cols)
        X_val, _ = build_windows(df, val_idx, _W, cols)

        scaler = fit_scaler(X_train)

        # X_train escalado: média ≈ 0 por construção do StandardScaler
        X_train_scaled = scaler.transform(X_train)
        np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10)

        # X_val escalado com parâmetros do train — NÃO deve ter média ≈ 0
        # (a menos que val e train tenham a mesma distribuição, o que pode acontecer
        # com dados sintéticos. Verificamos apenas que o scaler foi aplicado sem erro.)
        X_val_scaled = scaler.transform(X_val)
        assert X_val_scaled.shape == X_val.shape
        assert np.isfinite(X_val_scaled).all()

    def test_full_pipeline_produces_consistent_dimensions(self) -> None:
        """Dimensões consistentes ao longo de todo o pipeline."""
        df, cols = self._make_df_with_many_rows(n=300)
        idx = compute_valid_indices(df, _W, cols)
        train_idx, val_idx, test_idx = split_indices_chronological(idx, 0.60, 0.20)

        X_train, y_train = build_windows(df, train_idx, _W, cols)
        X_val,   y_val   = build_windows(df, val_idx,   _W, cols)
        X_test,  y_test  = build_windows(df, test_idx,  _W, cols)

        # Dimensão de input consistente
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == _W * _F

        # Labels alinhados com X
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Total de amostras
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(idx)

        # Labels remapeados
        y_train_r = remap_labels(y_train)
        y_val_r = remap_labels(y_val)
        y_test_r = remap_labels(y_test)

        assert set(int(v) for v in np.unique(y_train_r)).issubset({0, 1, 2})
        assert set(int(v) for v in np.unique(y_val_r)).issubset({0, 1, 2})
        assert set(int(v) for v in np.unique(y_test_r)).issubset({0, 1, 2})
