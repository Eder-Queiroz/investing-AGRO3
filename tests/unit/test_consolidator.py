"""Testes unitários para DataConsolidator.

Todas as chamadas a fetcher.load() são mockadas — zero acesso a disco real.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_engineering.consolidator import DataConsolidator

# ---------------------------------------------------------------------------
# Helpers de dados sintéticos
# ---------------------------------------------------------------------------

_WEEKLY_INDEX = pd.date_range("2020-01-03", periods=20, freq="W-FRI", name="date")


def _make_market_df(index: pd.DatetimeIndex = _WEEKLY_INDEX) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_adj": [10.0] * len(index),
            "high_adj": [11.0] * len(index),
            "low_adj": [9.0] * len(index),
            "close_adj": [10.5] * len(index),
            "volume": [500_000.0] * len(index),
        },
        index=index,
    )


def _make_fundamentals_df(index: pd.DatetimeIndex = _WEEKLY_INDEX) -> pd.DataFrame:
    data = {col: [1.0] * len(index) for col in
            ["p_vpa", "ev_ebitda", "roe", "net_debt_ebitda", "gross_margin", "dividend_yield"]}
    df = pd.DataFrame(data, index=index)
    # Simula NaN nas primeiras 4 semanas (lag de CVM)
    df.iloc[:4] = float("nan")
    return df


def _make_macro_df(index: pd.DatetimeIndex = _WEEKLY_INDEX) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cdi_rate": [13.75] * len(index),
            "usd_brl": [5.0] * len(index),
            "selic_rate": [14.0] * len(index),
            "igpm": [0.5] * len(index),
            "ipca": [0.4] * len(index),
            "soy_price_usd": [14.0] * len(index),
            "corn_price_usd": [5.5] * len(index),
            "selic_real": [13.6] * len(index),
        },
        index=index,
    )


def _make_consolidator(
    market: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
) -> DataConsolidator:
    """Cria DataConsolidator com fetchers mockados."""
    market_mock = MagicMock()
    market_mock.load.return_value = market if market is not None else _make_market_df()

    fundamentals_mock = MagicMock()
    fundamentals_mock.load.return_value = (
        fundamentals if fundamentals is not None else _make_fundamentals_df()
    )

    macro_mock = MagicMock()
    macro_mock.load.return_value = macro if macro is not None else _make_macro_df()

    return DataConsolidator(
        market_fetcher=market_mock,
        fundamentals_fetcher=fundamentals_mock,
        macro_fetcher=macro_mock,
    )


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestDataConsolidator:
    def test_consolidate_returns_dataframe(self) -> None:
        """consolidate() deve retornar um pd.DataFrame."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        assert isinstance(df, pd.DataFrame)

    def test_all_19_columns_present(self) -> None:
        """Resultado deve conter todas as 19 colunas (5 market + 6 fund + 8 macro)."""
        expected = [
            "open_adj", "high_adj", "low_adj", "close_adj", "volume",
            "p_vpa", "ev_ebitda", "roe", "net_debt_ebitda", "gross_margin", "dividend_yield",
            "cdi_rate", "usd_brl", "selic_rate", "igpm", "ipca",
            "soy_price_usd", "corn_price_usd", "selic_real",
        ]
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        for col in expected:
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_fundamentals_nan_preserved_in_early_rows(self) -> None:
        """NaN nas primeiras linhas de fundamentals deve ser preservado no resultado."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        # As primeiras 4 linhas de fundamentals são NaN por design (_make_fundamentals_df)
        assert df["p_vpa"].iloc[:4].isna().all(), (
            "NaN iniciais de fundamentals devem ser preservados no join"
        )

    def test_index_equals_market_spine(self) -> None:
        """O índice do resultado deve ser idêntico ao índice do market DataFrame."""
        market_df = _make_market_df()
        consolidator = _make_consolidator(market=market_df)
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        pd.testing.assert_index_equal(df.index, market_df.index)

    def test_date_filtering_applied(self) -> None:
        """Apenas as semanas dentro do intervalo solicitado devem estar no resultado."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-03", "2020-02-28")
        assert df.index.min() >= pd.Timestamp("2020-01-03")
        assert df.index.max() <= pd.Timestamp("2020-02-28")

    def test_index_is_monotonic_increasing(self) -> None:
        """O índice do resultado deve estar ordenado cronologicamente."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        assert df.index.is_monotonic_increasing

    def test_no_duplicate_timestamps(self) -> None:
        """Não deve haver timestamps duplicados no resultado."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        assert not df.index.duplicated().any()

    def test_raises_if_market_load_raises_file_not_found(self) -> None:
        """FileNotFoundError do fetcher deve ser propagado."""
        market_mock = MagicMock()
        market_mock.load.side_effect = FileNotFoundError("parquet não encontrado")
        fund_mock = MagicMock()
        fund_mock.load.return_value = _make_fundamentals_df()
        macro_mock = MagicMock()
        macro_mock.load.return_value = _make_macro_df()

        consolidator = DataConsolidator(
            market_fetcher=market_mock,
            fundamentals_fetcher=fund_mock,
            macro_fetcher=macro_mock,
        )
        with pytest.raises(FileNotFoundError, match="parquet não encontrado"):
            consolidator.consolidate("2020-01-01", "2021-12-31")

    def test_raises_if_market_missing_required_column(self) -> None:
        """ValueError deve ser levantado se colunas obrigatórias de market estiverem ausentes."""
        bad_market = _make_market_df().drop(columns=["close_adj"])
        consolidator = _make_consolidator(market=bad_market)
        with pytest.raises(ValueError, match="close_adj"):
            consolidator.consolidate("2020-01-01", "2021-12-31")

    def test_raises_if_fundamentals_missing_required_column(self) -> None:
        """ValueError deve ser levantado se colunas obrigatórias de fundamentals estiverem ausentes."""
        bad_fund = _make_fundamentals_df().drop(columns=["p_vpa"])
        consolidator = _make_consolidator(fundamentals=bad_fund)
        with pytest.raises(ValueError, match="p_vpa"):
            consolidator.consolidate("2020-01-01", "2021-12-31")

    def test_raises_if_macro_missing_required_column(self) -> None:
        """ValueError deve ser levantado se colunas obrigatórias de macro estiverem ausentes."""
        bad_macro = _make_macro_df().drop(columns=["cdi_rate"])
        consolidator = _make_consolidator(macro=bad_macro)
        with pytest.raises(ValueError, match="cdi_rate"):
            consolidator.consolidate("2020-01-01", "2021-12-31")

    def test_raises_on_empty_date_range(self) -> None:
        """ValueError deve ser levantado se nenhuma semana estiver dentro do intervalo."""
        consolidator = _make_consolidator()
        # O índice sintético começa em 2020-01-03; filtrar em 2019 deve retornar vazio
        with pytest.raises(ValueError, match="Nenhum dado encontrado"):
            consolidator.consolidate("2019-01-01", "2019-12-31")

    def test_macro_values_preserved_correctly(self) -> None:
        """Valores de macro devem ser transferidos corretamente após o join."""
        consolidator = _make_consolidator()
        df = consolidator.consolidate("2020-01-01", "2021-12-31")
        assert (df["cdi_rate"] == 13.75).all()
        assert (df["usd_brl"] == 5.0).all()
