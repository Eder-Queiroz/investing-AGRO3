"""Testes unitários para TargetBuilder.

Foco em verificação matemática do target (CDI composto, retorno forward)
e na integridade do dtype nullable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.target_builder import TargetBuilder

# ---------------------------------------------------------------------------
# Helpers de dados sintéticos
# ---------------------------------------------------------------------------

_N_WEEKS = 120  # Suficiente para ter 120 - 52 = 68 amostras válidas


def _make_df(
    weekly_price_growth: float = 0.01,
    cdi_annual_pct: float = 13.75,
    n_weeks: int = _N_WEEKS,
) -> pd.DataFrame:
    """Cria DataFrame sintético com preço crescendo a taxa constante."""
    index = pd.date_range("2018-01-05", periods=n_weeks, freq="W-FRI", name="date")
    close = pd.Series(
        [10.0 * ((1.0 + weekly_price_growth) ** i) for i in range(n_weeks)],
        index=index,
        name="close_adj",
    )
    cdi = pd.Series([cdi_annual_pct] * n_weeks, index=index, name="cdi_rate")
    return pd.DataFrame({"close_adj": close, "cdi_rate": cdi})


def _make_builder(buy_pp: float = 5.0, sell_pp: float = -5.0, horizon: int = 52) -> TargetBuilder:
    config = {
        "target": {
            "buy_threshold_pp": buy_pp,
            "sell_threshold_pp": sell_pp,
            "horizon_weeks": horizon,
        }
    }
    return TargetBuilder(config=config)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestTargetBuilder:
    def test_build_returns_series_named_target(self) -> None:
        """build() deve retornar pd.Series com name='target'."""
        builder = _make_builder()
        result = builder.build(_make_df())
        assert isinstance(result, pd.Series)
        assert result.name == "target"

    def test_dtype_is_int8_nullable(self) -> None:
        """dtype deve ser pd.Int8Dtype() para suportar pd.NA."""
        builder = _make_builder()
        result = builder.build(_make_df())
        assert result.dtype == pd.Int8Dtype()

    def test_index_preserved(self) -> None:
        """O índice do DataFrame de entrada deve ser preservado no output."""
        df = _make_df()
        builder = _make_builder()
        result = builder.build(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_last_52_rows_are_na(self) -> None:
        """Últimas horizon_weeks linhas devem ser pd.NA (sem janela futura)."""
        builder = _make_builder(horizon=52)
        result = builder.build(_make_df(n_weeks=_N_WEEKS))
        assert result.iloc[-52:].isna().all(), (
            "Últimas 52 linhas devem ser pd.NA — sem janela futura completa"
        )

    def test_first_rows_have_valid_labels(self) -> None:
        """Linhas com janela futura completa devem ter label em {-1, 0, 1}."""
        builder = _make_builder()
        result = builder.build(_make_df(n_weeks=_N_WEEKS))
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.isin([-1, 0, 1]).all()

    def test_strong_uptrend_generates_buy_signals(self) -> None:
        """Retorno ~68% em 52s (1%/s) vs CDI ~13.75% → margem ~54pp >> 5pp → COMPRAR."""
        # 1% semanal: (1.01)^52 - 1 ≈ 0.678 = 67.8%
        # CDI 13.75% a.a. composto semanalmente por 52s ≈ 13.75%
        # margem ≈ 67.8% - 13.75% ≈ 54pp >> 5pp → tudo COMPRAR
        builder = _make_builder(buy_pp=5.0)
        df = _make_df(weekly_price_growth=0.01, cdi_annual_pct=13.75)
        result = builder.build(df)
        valid = result.dropna()
        assert (valid == 1).all(), (
            "Tendência de alta forte (+1%/sem) vs CDI 13.75% deve gerar apenas COMPRAR"
        )

    def test_flat_price_equal_to_cdi_generates_hold(self) -> None:
        """Preço crescendo à mesma taxa do CDI → margem ≈ 0 → AGUARDAR."""
        # CDI = 5.5% a.a. → weekly factor = (1.055)^(1/52) ≈ 1.001036
        cdi_annual = 5.5
        weekly_factor = (1.0 + cdi_annual / 100.0) ** (1.0 / 52.0)
        weekly_growth = weekly_factor - 1.0  # Preço cresce exatamente ao CDI semanal

        builder = _make_builder(buy_pp=5.0, sell_pp=-5.0)
        df = _make_df(weekly_price_growth=weekly_growth, cdi_annual_pct=cdi_annual)
        result = builder.build(df)
        valid = result.dropna()
        assert (valid == 0).all(), (
            "Retorno exatamente igual ao CDI deve gerar AGUARDAR"
        )

    def test_declining_price_generates_sell_signals(self) -> None:
        """Preço caindo 1%/sem → retorno ~-40% vs CDI 13.75% → margem ~-54pp << -5pp → VENDER."""
        builder = _make_builder(sell_pp=-5.0)
        df = _make_df(weekly_price_growth=-0.01, cdi_annual_pct=13.75)
        result = builder.build(df)
        valid = result.dropna()
        assert (valid == -1).all(), (
            "Tendência de queda (-1%/sem) vs CDI 13.75% deve gerar apenas VENDER"
        )

    def test_forward_cdi_compound_math(self) -> None:
        """CDI de 13.75% a.a. composto semanalmente por 52s deve ≈ 13.75%."""
        # (1 + 0.1375)^(1/52) ^ 52 = 1.1375 → retorno = 13.75%
        builder = _make_builder()
        cdi_rate = pd.Series([13.75] * 110)
        forward_cdi = builder._compute_forward_cdi(cdi_rate)
        # As primeiras linhas têm janela futura completa de 52s
        expected = (1.0 + 13.75 / 100.0) - 1.0  # = 0.1375
        first_valid = forward_cdi.dropna().iloc[0]
        assert first_valid == pytest.approx(expected, abs=1e-4), (
            f"CDI composto esperado ≈ {expected:.4f}, obtido {first_valid:.4f}"
        )

    def test_custom_buy_threshold_respected(self) -> None:
        """Threshold de compra customizado deve ser usado na classificação."""
        # Com buy_threshold=50pp, o uptrend de 68% ainda deve gerar COMPRAR
        builder = _make_builder(buy_pp=50.0, sell_pp=-50.0)
        df = _make_df(weekly_price_growth=0.01, cdi_annual_pct=13.75)
        result = builder.build(df)
        valid = result.dropna()
        assert (valid == 1).all()

    def test_extreme_threshold_generates_hold_for_uptrend(self) -> None:
        """Com threshold de compra de 90pp, o retorno de 68% cai em AGUARDAR."""
        # margem ≈ 54pp < 90pp → AGUARDAR (não atinge buy threshold)
        # margem ≈ 54pp > -90pp → não VENDER
        builder = _make_builder(buy_pp=90.0, sell_pp=-90.0)
        df = _make_df(weekly_price_growth=0.01, cdi_annual_pct=13.75)
        result = builder.build(df)
        valid = result.dropna()
        assert (valid == 0).all()

    def test_missing_close_adj_raises_value_error(self) -> None:
        """ValueError deve ser levantado se 'close_adj' estiver ausente."""
        df = pd.DataFrame({"cdi_rate": [13.75] * 60})
        with pytest.raises(ValueError, match="close_adj"):
            _make_builder().build(df)

    def test_missing_cdi_rate_raises_value_error(self) -> None:
        """ValueError deve ser levantado se 'cdi_rate' estiver ausente."""
        df = pd.DataFrame({"close_adj": [10.0] * 60})
        with pytest.raises(ValueError, match="cdi_rate"):
            _make_builder().build(df)

    def test_forward_return_last_horizon_are_nan(self) -> None:
        """_compute_forward_return deve retornar NaN nas últimas horizon posições."""
        builder = _make_builder(horizon=10)
        close = pd.Series([10.0 + i for i in range(30)])
        result = builder._compute_forward_return(close)
        assert result.iloc[-10:].isna().all()
        assert result.iloc[:-10].notna().all()

    def test_label_uses_pandas_native_assignment_not_numpy(self) -> None:
        """pd.NA (não np.nan) deve ser usado nas posições sem janela futura."""
        builder = _make_builder()
        result = builder.build(_make_df())
        # pd.NA é distinto de np.nan — verifica via isna() do pandas Int8
        na_mask = result.isna()
        assert na_mask.iloc[-52:].all()
        # Garante que são pd.NA, não np.nan: o dtype mantém pd.Int8Dtype()
        assert result.dtype == pd.Int8Dtype()

    def test_exactly_52_nan_at_end_with_default_horizon(self) -> None:
        """Com horizon=52, exatamente 52 linhas finais devem ser pd.NA."""
        builder = _make_builder(horizon=52)
        df = _make_df(n_weeks=100)
        result = builder.build(df)
        assert int(result.isna().sum()) == 52
