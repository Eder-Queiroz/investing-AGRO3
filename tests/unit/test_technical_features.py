"""Testes unitários para TechnicalFeatureBuilder.

Verificações matemáticas das transformações de estacionariedade,
invariantes estruturais (RSI ∈ [0,100], price_to_52w_high ≤ 1)
e proteção contra divisão por zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.technical_features import TechnicalFeatureBuilder

# ---------------------------------------------------------------------------
# Helpers de dados sintéticos
# ---------------------------------------------------------------------------

_N = 80  # Semanas — suficiente para janelas de 52 semanas


def _make_price_df(
    n: int = _N,
    weekly_growth: float = 0.005,
    volume: float = 500_000.0,
    cdi: float = 13.75,
    soy: float = 14.0,
    corn: float = 5.5,
    usd_brl: float = 5.0,
) -> pd.DataFrame:
    """DataFrame consolidado sintético com todas as colunas obrigatórias."""
    index = pd.date_range("2020-01-03", periods=n, freq="W-FRI", name="date")
    close = pd.Series(
        [10.0 * ((1.0 + weekly_growth) ** i) for i in range(n)], index=index
    )
    return pd.DataFrame(
        {
            "open_adj": close * 0.99,
            "high_adj": close * 1.01,
            "low_adj": close * 0.98,
            "close_adj": close,
            "volume": [volume] * n,
            "cdi_rate": [cdi] * n,
            "usd_brl": [usd_brl] * n,
            "selic_rate": [cdi + 0.25] * n,
            "igpm": [0.5] * n,
            "ipca": [0.4] * n,
            "soy_price_usd": [soy] * n,
            "corn_price_usd": [corn] * n,
            "selic_real": [cdi - 5.0] * n,
        },
        index=index,
    )


def _make_builder(cfg: dict | None = None) -> TechnicalFeatureBuilder:
    config = cfg or {
        "feature_engineering": {
            "rsi_period": 14,
            "volatility_window_weeks": 4,
            "momentum_windows_weeks": [1, 4, 13],
        }
    }
    return TechnicalFeatureBuilder(config=config)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestTechnicalFeatureBuilder:
    def test_transform_returns_dataframe(self) -> None:
        """transform() deve retornar pd.DataFrame."""
        builder = _make_builder()
        result = builder.transform(_make_price_df())
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self) -> None:
        """Colunas de entrada devem estar presentes no resultado."""
        df = _make_price_df()
        builder = _make_builder()
        result = builder.transform(df)
        for col in df.columns:
            assert col in result.columns

    def test_transform_does_not_modify_input(self) -> None:
        """O DataFrame de entrada não deve ser modificado (immutability)."""
        df = _make_price_df()
        original_cols = set(df.columns)
        original_shape = df.shape
        _make_builder().transform(df)
        assert set(df.columns) == original_cols
        assert df.shape == original_shape

    # --- Log-retornos ---

    def test_log_return_1w_has_one_nan_at_start(self) -> None:
        """log_return_1w deve ter exatamente 1 NaN no início."""
        result = _make_builder().transform(_make_price_df())
        assert pd.isna(result["log_return_1w"].iloc[0])
        assert result["log_return_1w"].iloc[1:].notna().all()

    def test_log_return_4w_has_four_nans_at_start(self) -> None:
        """log_return_4w deve ter exatamente 4 NaN no início."""
        result = _make_builder().transform(_make_price_df())
        assert result["log_return_4w"].iloc[:4].isna().all()
        assert result["log_return_4w"].iloc[4:].notna().all()

    def test_log_return_13w_has_thirteen_nans_at_start(self) -> None:
        """log_return_13w deve ter exatamente 13 NaN no início."""
        result = _make_builder().transform(_make_price_df())
        assert result["log_return_13w"].iloc[:13].isna().all()
        assert result["log_return_13w"].iloc[13:].notna().all()

    def test_log_return_static_method_formula(self) -> None:
        """_log_return deve seguir log(price[t] / price[t-lag])."""
        close = pd.Series([100.0, 110.0, 121.0])
        result = TechnicalFeatureBuilder._log_return(close, lag=1)
        expected_1 = np.log(110.0 / 100.0)
        expected_2 = np.log(121.0 / 110.0)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(expected_1, abs=1e-10)
        assert result.iloc[2] == pytest.approx(expected_2, abs=1e-10)

    def test_constant_price_log_return_is_zero(self) -> None:
        """Preço constante → log-retorno = 0 (exceto NaN inicial)."""
        close = pd.Series([10.0] * 20)
        result = TechnicalFeatureBuilder._log_return(close, lag=1)
        assert result.iloc[1:].eq(0.0).all()

    # --- Volatilidade ---

    def test_volatility_4w_nan_for_first_window_minus_one(self) -> None:
        """volatility_4w deve ser NaN para as primeiras 3 linhas (window-1=3)."""
        result = _make_builder().transform(_make_price_df())
        # log_return_1w tem 1 NaN + rolling(4) precisa de 4 valores → total 4 NaN
        assert result["volatility_4w"].iloc[:4].isna().all()

    def test_volatility_constant_price_is_zero(self) -> None:
        """Preço constante → log-retorno = 0 → volatilidade = 0."""
        df = _make_price_df(weekly_growth=0.0)
        result = _make_builder().transform(df)
        non_nan = result["volatility_4w"].dropna()
        assert (non_nan == 0.0).all()

    def test_volatility_annualized_via_sqrt52(self) -> None:
        """Verifica a fórmula de anualização: vol_anual = vol_semanal × sqrt(52)."""
        # Gera log-retornos conhecidos e verifica a correspondência
        close = pd.Series([10.0, 10.1, 9.9, 10.05, 10.2, 10.15])
        log_ret = TechnicalFeatureBuilder._log_return(close, lag=1)
        vol = TechnicalFeatureBuilder._annualized_volatility(log_ret, window=4)
        # Janela de 4 sobre 5 valores disponíveis: posição 4 (0-indexed) deve ter valor
        expected = log_ret.iloc[1:5].std() * np.sqrt(52)
        assert vol.iloc[4] == pytest.approx(expected, abs=1e-10)

    # --- RSI ---

    def test_rsi_bounded_0_100(self) -> None:
        """RSI deve sempre estar no intervalo [0, 100]."""
        df = _make_price_df()
        result = _make_builder().transform(df)
        valid_rsi = result["rsi_14"].dropna()
        assert (valid_rsi >= 0.0).all()
        assert (valid_rsi <= 100.0).all()

    def test_rsi_wilder_differs_from_rolling_mean(self) -> None:
        """O RSI de Wilder (EWM) deve diferir de uma implementação por rolling mean.

        Usa série com alternância de ganhos e perdas para garantir avg_loss > 0
        em ambas as implementações. Para série pura de alta/baixa, avg_loss = 0
        e ambas convergiriam para RSI = 100/0, tornando a comparação inútil.
        """
        # Série alternando subidas e descidas para forçar ganhos E perdas
        rng = list(range(30))
        close = pd.Series([10.0 + (i % 3) * 2.0 - (i % 5) * 1.5 for i in rng])
        rsi_wilder = TechnicalFeatureBuilder._rsi(close, period=14)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain_rolling = gain.rolling(14).mean()
        avg_loss_rolling = loss.rolling(14).mean()
        rs_rolling = avg_gain_rolling / avg_loss_rolling.replace(0, float("nan"))
        rsi_rolling = 100.0 - (100.0 / (1.0 + rs_rolling))

        # Ambas devem ter valores válidos com avg_loss > 0
        valid_wilder = rsi_wilder.dropna()
        valid_rolling = rsi_rolling.dropna()
        assert len(valid_wilder) > 0 and len(valid_rolling) > 0

        common_idx = valid_wilder.index.intersection(valid_rolling.index)
        assert len(common_idx) > 0, "Ambas devem ter posições válidas em comum"

        # Com avg_loss > 0 em ambas, os resultados NÃO devem ser iguais
        # pois o EWM (Wilder) pesa os dados de forma diferente do rolling mean
        are_all_equal = (
            (rsi_wilder[common_idx] - rsi_rolling[common_idx]).abs() < 1e-10
        ).all()
        assert not are_all_equal, (
            "RSI Wilder (EWM) e RSI rolling mean não devem ser idênticos para série mista"
        )

    def test_rsi_constant_uptrend_approaches_100(self) -> None:
        """Preço subindo monotonicamente → RSI deve se aproximar de 100."""
        close = pd.Series([float(i) for i in range(1, 60)])
        rsi = TechnicalFeatureBuilder._rsi(close, period=14)
        # Após convergência, RSI deve ser > 95 (avg_loss → 0)
        assert rsi.dropna().iloc[-1] > 95.0

    def test_rsi_constant_downtrend_approaches_0(self) -> None:
        """Preço caindo monotonicamente → RSI deve se aproximar de 0."""
        close = pd.Series([float(60 - i) for i in range(60)])
        rsi = TechnicalFeatureBuilder._rsi(close, period=14)
        assert rsi.dropna().iloc[-1] < 5.0

    def test_rsi_all_loss_no_gain_gives_zero(self) -> None:
        """Quando avg_gain = 0 e avg_loss > 0, RSI deve ser 0."""
        # Série monotonicamente decrescente por muitos períodos
        close = pd.Series([100.0 - i * 0.5 for i in range(50)])
        rsi = TechnicalFeatureBuilder._rsi(close, period=14)
        # O RSI de uma série decrescente constante deve convergir para 0
        assert rsi.dropna().iloc[-1] < 5.0

    def test_rsi_all_gain_no_loss_gives_100(self) -> None:
        """Quando avg_loss = 0, RSI deve ser 100 (não NaN)."""
        close = pd.Series([10.0 + i * 0.5 for i in range(50)])
        rsi = TechnicalFeatureBuilder._rsi(close, period=14)
        last_valid = rsi.dropna().iloc[-1]
        assert last_valid == pytest.approx(100.0, abs=1.0)

    # --- Price to 52-week high ---

    def test_price_to_52w_high_leq_one(self) -> None:
        """price_to_52w_high nunca deve superar 1.0."""
        df = _make_price_df()
        result = _make_builder().transform(df)
        valid = result["price_to_52w_high"].dropna()
        assert (valid <= 1.0).all()

    def test_price_to_52w_high_gt_zero(self) -> None:
        """price_to_52w_high deve ser positivo."""
        df = _make_price_df()
        result = _make_builder().transform(df)
        valid = result["price_to_52w_high"].dropna()
        assert (valid > 0.0).all()

    def test_price_to_52w_high_nan_for_first_51_rows(self) -> None:
        """Primeiras 51 linhas devem ser NaN (janela incompleta de 52 semanas)."""
        df = _make_price_df()
        result = _make_builder().transform(df)
        assert result["price_to_52w_high"].iloc[:51].isna().all()

    def test_price_to_52w_high_equals_one_at_new_high(self) -> None:
        """No máximo absoluto da série, price_to_52w_high deve ser 1.0."""
        # Série sempre crescente: o último valor é sempre o máximo → ratio = 1
        df = _make_price_df(weekly_growth=0.01, n=_N)
        result = _make_builder().transform(df)
        valid = result["price_to_52w_high"].dropna()
        # Em uma série monotonicamente crescente, close = rolling_max → ratio = 1
        # Usa comparação numérica direta (pytest.approx não funciona com Series)
        assert (valid - 1.0).abs().max() < 1e-10

    # --- Volume z-score ---

    def test_volume_zscore_nan_for_first_window_rows(self) -> None:
        """volume_zscore_4w deve ser NaN nas primeiras window-1 linhas."""
        df = _make_price_df()
        result = _make_builder().transform(df)
        assert result["volume_zscore_4w"].iloc[:3].isna().all()

    def test_volume_zscore_zero_std_produces_nan(self) -> None:
        """Se volume for constante na janela, std=0 → zscore deve ser NaN."""
        # Volume constante → std=0 → NaN (não inf/div_by_zero)
        df = _make_price_df(volume=500_000.0)
        result = _make_builder().transform(df)
        # Com volume constante, std = 0 → zscore = NaN
        assert result["volume_zscore_4w"].dropna().empty

    def test_volume_zscore_large_spike_gives_positive_z(self) -> None:
        """Volume muito acima da média deve produzir z-score positivo."""
        df = _make_price_df(volume=100_000.0).copy()
        # Usa .loc para modificação segura em pandas 2.x (evita SettingWithCopyWarning)
        df.loc[df.index[20], "volume"] = 10_000_000.0
        result = _make_builder().transform(df)
        # O z-score na semana 20 ou nas próximas 4 deve ser positivo
        assert result["volume_zscore_4w"].iloc[20:25].max() > 0.0

    # --- Features de commodities e câmbio ---

    def test_log_return_soy_4w_has_four_nans(self) -> None:
        """log_return_soy_4w deve ter 4 NaN no início."""
        result = _make_builder().transform(_make_price_df())
        assert result["log_return_soy_4w"].iloc[:4].isna().all()
        assert result["log_return_soy_4w"].iloc[4:].notna().all()

    def test_log_return_corn_4w_has_four_nans(self) -> None:
        result = _make_builder().transform(_make_price_df())
        assert result["log_return_corn_4w"].iloc[:4].isna().all()

    def test_log_return_usd_brl_4w_has_four_nans(self) -> None:
        result = _make_builder().transform(_make_price_df())
        assert result["log_return_usd_brl_4w"].iloc[:4].isna().all()

    def test_constant_commodity_log_return_is_zero(self) -> None:
        """Commodity com preço constante → log-retorno = 0."""
        df = _make_price_df(soy=14.0)
        result = _make_builder().transform(df)
        non_nan = result["log_return_soy_4w"].dropna()
        assert (non_nan == 0.0).all()

    # --- Delta de taxas macro ---

    def test_delta_cdi_4w_has_four_nans(self) -> None:
        result = _make_builder().transform(_make_price_df())
        assert result["delta_cdi_4w"].iloc[:4].isna().all()

    def test_delta_cdi_constant_rate_is_zero(self) -> None:
        """CDI constante → delta = 0."""
        df = _make_price_df(cdi=13.75)
        result = _make_builder().transform(df)
        non_nan = result["delta_cdi_4w"].dropna()
        # Usa comparação numérica direta (pytest.approx não funciona com Series)
        assert non_nan.abs().max() < 1e-10

    def test_delta_selic_real_4w_present(self) -> None:
        result = _make_builder().transform(_make_price_df())
        assert "delta_selic_real_4w" in result.columns

    # --- Validação de schema ---

    def test_missing_close_adj_raises_value_error(self) -> None:
        df = _make_price_df().drop(columns=["close_adj"])
        with pytest.raises(ValueError, match="close_adj"):
            _make_builder().transform(df)

    def test_missing_volume_raises_value_error(self) -> None:
        df = _make_price_df().drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            _make_builder().transform(df)

    def test_missing_cdi_rate_raises_value_error(self) -> None:
        df = _make_price_df().drop(columns=["cdi_rate"])
        with pytest.raises(ValueError, match="cdi_rate"):
            _make_builder().transform(df)
