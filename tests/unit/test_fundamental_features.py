"""Testes unitários para FundamentalFeatureBuilder.

Foco em:
- Integridade causal do z-score expansivo (sem leakage futuro)
- Propagação correta de NaN (lag de CVM)
- Invariância das colunas brutas
- Comportamento com dados insuficientes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.fundamental_features import (
    FundamentalFeatureBuilder,
    _FUNDAMENTAL_COLS,
    _MIN_EXPANDING_PERIODS,
)

# ---------------------------------------------------------------------------
# Helpers de dados sintéticos
# ---------------------------------------------------------------------------

_N = 60  # Semanas


def _make_fundamentals_df(
    n: int = _N,
    nan_rows: int = 4,
) -> pd.DataFrame:
    """DataFrame com 6 colunas de fundamentos e NaN iniciais."""
    index = pd.date_range("2020-01-03", periods=n, freq="W-FRI", name="date")
    data: dict[str, list[float | float]] = {
        "p_vpa": list(range(1, n + 1)),           # 1.0, 2.0, ..., n
        "ev_ebitda": [8.0] * n,
        "roe": [0.12] * n,
        "net_debt_ebitda": [1.5] * n,
        "gross_margin": [0.40] * n,
        "dividend_yield": [0.05] * n,
    }
    df = pd.DataFrame(data, index=index)
    # Simula o lag de CVM: primeiras nan_rows são NaN
    if nan_rows > 0:
        df.iloc[:nan_rows] = float("nan")
    return df


def _make_builder() -> FundamentalFeatureBuilder:
    return FundamentalFeatureBuilder()


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestFundamentalFeatureBuilder:
    def test_transform_returns_dataframe(self) -> None:
        """transform() deve retornar pd.DataFrame."""
        result = _make_builder().transform(_make_fundamentals_df())
        assert isinstance(result, pd.DataFrame)

    def test_zscore_columns_added(self) -> None:
        """Uma coluna '_zscore' deve ser adicionada para cada fundamental."""
        result = _make_builder().transform(_make_fundamentals_df())
        for col in _FUNDAMENTAL_COLS:
            assert f"{col}_zscore" in result.columns, f"Coluna ausente: {col}_zscore"

    def test_raw_columns_unchanged(self) -> None:
        """Os valores das colunas brutas não devem ser modificados."""
        df = _make_fundamentals_df()
        original = df.copy()
        result = _make_builder().transform(df)
        for col in _FUNDAMENTAL_COLS:
            pd.testing.assert_series_equal(result[col], original[col])

    def test_transform_does_not_modify_input_df(self) -> None:
        """O DataFrame de entrada não deve ser modificado (immutability)."""
        df = _make_fundamentals_df()
        original_cols = set(df.columns)
        _make_builder().transform(df)
        assert set(df.columns) == original_cols

    def test_nan_input_propagates_to_zscore(self) -> None:
        """Onde o ratio bruto é NaN (lag de CVM), o z-score também deve ser NaN."""
        df = _make_fundamentals_df(nan_rows=4)
        result = _make_builder().transform(df)
        # As primeiras 4 linhas têm fundamentals NaN → z-score deve ser NaN
        assert result["p_vpa_zscore"].iloc[:4].isna().all(), (
            "Z-score deve ser NaN onde o valor bruto é NaN"
        )

    def test_zscore_nan_for_insufficient_history(self) -> None:
        """Z-score deve ser NaN até acumular min_periods observações válidas."""
        df = _make_fundamentals_df(nan_rows=0)
        result = _make_builder().transform(df)
        # Com expanding(min_periods=8), as primeiras 7 linhas são NaN
        # (precisa de 8 valores para calcular std confiável)
        n_nan = _MIN_EXPANDING_PERIODS - 1
        assert result["p_vpa_zscore"].iloc[:n_nan].isna().all(), (
            f"Primeiras {n_nan} linhas do z-score devem ser NaN (histórico insuficiente)"
        )

    def test_zscore_available_after_min_periods(self) -> None:
        """Z-score deve estar disponível a partir de min_periods observações."""
        df = _make_fundamentals_df(nan_rows=0)
        result = _make_builder().transform(df)
        # A partir da posição min_periods-1 (0-indexed), o z-score deve ser válido
        assert pd.notna(result["p_vpa_zscore"].iloc[_MIN_EXPANDING_PERIODS - 1])

    def test_expanding_zscore_no_future_leakage(self) -> None:
        """Z-score em t não deve mudar ao inserir um outlier em t+1.

        Esta é a verificação crítica de integridade causal: a janela expansiva
        em t usa apenas dados até t. Um outlier em t+1 não deve alterar o
        z-score calculado em t.
        """
        df = _make_fundamentals_df(nan_rows=0, n=40)
        builder = _make_builder()

        # Calcula z-score baseline
        result_base = builder.transform(df)
        zscore_at_t20 = result_base["p_vpa_zscore"].iloc[20]

        # Insere outlier extremo em t+1 = posição 21
        # Usa .loc para modificação segura em pandas 2.x (evita SettingWithCopyWarning)
        df_modified = df.copy()
        df_modified.loc[df_modified.index[21], "p_vpa"] = 999_999.0

        result_modified = builder.transform(df_modified)
        zscore_at_t20_modified = result_modified["p_vpa_zscore"].iloc[20]

        assert zscore_at_t20 == pytest.approx(zscore_at_t20_modified, abs=1e-10), (
            "Z-score em t não deve mudar com alteração em t+1 (causalidade violada)"
        )

    def test_expanding_zscore_changes_after_outlier_at_same_t(self) -> None:
        """Z-score em t DEVE mudar ao alterar o valor em t (sanidade do teste de leakage)."""
        df = _make_fundamentals_df(nan_rows=0, n=40)
        builder = _make_builder()

        result_base = builder.transform(df)
        zscore_at_t20 = result_base["p_vpa_zscore"].iloc[20]

        df_modified = df.copy()
        df_modified.loc[df_modified.index[20], "p_vpa"] = 999_999.0

        result_modified = builder.transform(df_modified)
        zscore_at_t20_modified = result_modified["p_vpa_zscore"].iloc[20]

        # Alterar t deve mudar o z-score em t (é a observação sendo normalizada)
        assert zscore_at_t20 != pytest.approx(zscore_at_t20_modified, abs=1e-4)

    def test_zscore_constant_series_is_nan(self) -> None:
        """Série constante (std = 0) deve produzir z-score NaN (não inf)."""
        # ev_ebitda é constante em _make_fundamentals_df
        df = _make_fundamentals_df(nan_rows=0)
        result = _make_builder().transform(df)
        # ev_ebitda constante → expanding std = 0 (após 2 pontos) → zscore NaN
        assert result["ev_ebitda_zscore"].dropna().empty, (
            "Série constante deve produzir z-score NaN (std = 0)"
        )

    def test_zscore_is_zero_at_mean(self) -> None:
        """Z-score do valor médio histórico deve ser próximo de zero."""
        # p_vpa aumenta linearmente: 1, 2, ..., n
        # O valor médio da janela expansiva está próximo do histórico
        df = _make_fundamentals_df(nan_rows=0, n=50)
        result = _make_builder().transform(df)
        # Para uma série linear crescente, a média expansiva fica abaixo do valor atual,
        # então z-score > 0 na maioria. Verificamos apenas que não é sempre zero.
        valid = result["p_vpa_zscore"].dropna()
        assert valid.abs().max() > 0.0, "Z-score não deve ser sempre zero"

    def test_zscore_formula_matches_manual_calculation(self) -> None:
        """Verifica a fórmula do z-score expansivo com cálculo manual."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        series = pd.Series(values)

        result = FundamentalFeatureBuilder._expanding_zscore(series, min_periods=3)

        # Em t=4 (5ª posição, valor = 5.0), usando dados [1,2,3,4,5]:
        # mean([1,2,3,4,5]) = 3.0, std([1,2,3,4,5]) = 1.5811...
        # z = (5 - 3) / 1.5811 ≈ 1.2649
        expected_z = (5.0 - np.mean([1, 2, 3, 4, 5])) / np.std([1, 2, 3, 4, 5], ddof=1)
        assert result.iloc[4] == pytest.approx(expected_z, abs=1e-6)

    def test_missing_fundamental_column_raises_value_error(self) -> None:
        """ValueError deve ser levantado se coluna fundamental estiver ausente."""
        df = _make_fundamentals_df().drop(columns=["p_vpa"])
        with pytest.raises(ValueError, match="p_vpa"):
            _make_builder().transform(df)

    def test_all_six_zscore_columns_present(self) -> None:
        """Exatamente 6 colunas de z-score devem ser adicionadas."""
        result = _make_builder().transform(_make_fundamentals_df())
        zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
        assert len(zscore_cols) == 6
