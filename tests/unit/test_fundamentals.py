"""Testes unitários para FundamentalsFetcher.

Foco especial em:
- Lag de 45 dias (anti-leakage crítico)
- merge_asof sem future data
- EBITDA negativo não quebra o pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data_ingestion.fundamentals import FundamentalsFetcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_Q_DATES = pd.to_datetime(["2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"])

_BALANCE = pd.DataFrame(
    {
        "Stockholders Equity": [500_000_000.0] * 4,
        "Total Debt": [200_000_000.0] * 4,
        "Cash And Cash Equivalents": [50_000_000.0] * 4,
    },
    index=_Q_DATES,
)

_INCOME = pd.DataFrame(
    {
        "Total Revenue": [800_000_000.0] * 4,
        "Gross Profit": [300_000_000.0] * 4,
        "EBIT": [150_000_000.0] * 4,
        "Net Income": [100_000_000.0] * 4,
    },
    index=_Q_DATES,
)

_CASHFLOW = pd.DataFrame(
    {
        "Depreciation And Amortization": [20_000_000.0] * 4,
    },
    index=_Q_DATES,
)

_PRICE_SERIES = pd.Series(
    [30.0] * 300,
    index=pd.date_range("2023-01-01", periods=300, freq="B"),
)

_DIVIDENDS = pd.Series(
    [0.50] * 300,
    index=pd.date_range("2023-01-01", periods=300, freq="B"),
)


def _make_fetcher(tmp_path: Path) -> FundamentalsFetcher:
    fetcher = FundamentalsFetcher(
        config={"data_ingestion": {"output": {"fundamentals": str(tmp_path) + "/"}}}
    )
    fetcher._output_path = tmp_path / "fundamentals_quarterly.parquet"
    return fetcher


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestFundamentalsFetcher:
    def test_reporting_lag_applied_45_days(self) -> None:
        """Q1 (31/mar) não deve aparecer antes de 15/mai no spine semanal."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        q1_date = pd.Timestamp("2024-03-31")
        quarterly = pd.DataFrame({"p_vpa": [1.5]}, index=[q1_date])
        quarterly.index = pd.DatetimeIndex(quarterly.index)

        lagged = fetcher._apply_reporting_lag(quarterly)

        expected_earliest = pd.Timestamp("2024-05-15")
        assert lagged.index[0] >= expected_earliest, (
            f"Lag incorreto: lag-date {lagged.index[0]} < {expected_earliest}"
        )

    def test_merge_asof_no_future_leakage(self) -> None:
        """Semanas antes do lag-date devem ter NaN nos fundamentos."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        spine = pd.date_range("2024-01-05", periods=20, freq="W-FRI", name="date")

        # Q4/2023 termina 31/dez → lag-date = 14/fev/2024
        # Índice deve ter nome "date" para o merge_asof funcionar corretamente
        quarterly = pd.DataFrame(
            {"p_vpa": [2.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-02-14")], name="date"),
        )
        merged = fetcher._merge_to_weekly_spine(quarterly, spine)

        # Semanas de jan a 09/fev devem ser NaN
        before_lag = merged.loc[:"2024-02-09", "p_vpa"]
        assert before_lag.isna().all(), "Leakage detectado: valores antes do lag-date"

        # 16/fev em diante deve ter 2.0
        after_lag = merged.loc["2024-02-16":, "p_vpa"]
        assert (after_lag - 2.0).abs().lt(1e-6).all()

    def test_negative_ebitda_yields_nan_not_error(self) -> None:
        """EBITDA negativo não deve lançar exceção — deve gerar NaN."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)

        neg_income = _INCOME.copy()
        neg_income["EBIT"] = -50_000_000.0  # EBIT negativo → EBITDA negativo

        ratios = fetcher._compute_ratios(
            _BALANCE, neg_income, _CASHFLOW, _PRICE_SERIES, 100_000_000, _DIVIDENDS
        )
        # ev_ebitda e net_debt_ebitda devem ser NaN quando EBITDA < 0
        assert ratios["ev_ebitda"].isna().all(), "ev_ebitda deveria ser NaN com EBITDA negativo"
        assert ratios["net_debt_ebitda"].isna().all()

    def test_p_vpa_computed_from_price_and_equity(self) -> None:
        """P/VP = preço / (equity / shares)."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        # equity = 500M, shares = 100M → book/share = 5.0
        # price = 30.0 → P/VP = 6.0
        ratios = fetcher._compute_ratios(
            _BALANCE, _INCOME, _CASHFLOW, _PRICE_SERIES, 100_000_000, _DIVIDENDS
        )
        assert ratios["p_vpa"].dropna().iloc[0] == pytest.approx(6.0, rel=0.01)

    def test_gross_margin_computed_correctly(self) -> None:
        """gross_margin = gross_profit / total_revenue = 300M / 800M ≈ 0.375."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        ratios = fetcher._compute_ratios(
            _BALANCE, _INCOME, _CASHFLOW, _PRICE_SERIES, 100_000_000, _DIVIDENDS
        )
        assert ratios["gross_margin"].dropna().iloc[0] == pytest.approx(300 / 800, rel=0.01)

    def test_safe_get_column_finds_first_candidate(self) -> None:
        df = pd.DataFrame({"Total Revenue": [100.0], "Operating Revenue": [200.0]})
        result = FundamentalsFetcher._safe_get_column(df, ["Total Revenue", "Operating Revenue"])
        assert result.iloc[0] == pytest.approx(100.0)

    def test_safe_get_column_falls_back_to_second_candidate(self) -> None:
        df = pd.DataFrame({"Operating Revenue": [200.0]})
        result = FundamentalsFetcher._safe_get_column(df, ["Total Revenue", "Operating Revenue"])
        assert result.iloc[0] == pytest.approx(200.0)

    def test_safe_get_column_raises_if_no_candidate_found(self) -> None:
        df = pd.DataFrame({"Other Column": [100.0]})
        with pytest.raises(KeyError, match="Nenhum dos candidatos"):
            FundamentalsFetcher._safe_get_column(df, ["Total Revenue", "Operating Revenue"])

    def test_leading_nan_rows_acceptable(self, tmp_path: Path) -> None:
        """Primeiras semanas sem dados são NaN — comportamento esperado."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        # Dados só a partir de 2024-05-15 (após lag do Q1)
        quarterly = pd.DataFrame(
            {"p_vpa": [2.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-05-15")], name="date"),
        )
        spine = pd.date_range("2024-01-05", periods=30, freq="W-FRI", name="date")
        merged = fetcher._merge_to_weekly_spine(quarterly, spine)
        # Antes de mai/2024 → NaN
        nan_count = merged.loc[:"2024-05-09", "p_vpa"].isna().sum()
        assert nan_count > 0, "Deveria haver NaN antes do lag-date"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        quarterly = pd.DataFrame(
            {
                col: [1.0] * 4
                for col in [
                    "p_vpa",
                    "ev_ebitda",
                    "roe",
                    "net_debt_ebitda",
                    "gross_margin",
                    "dividend_yield",
                ]
            },
            index=pd.date_range("2024-01-01", periods=4, freq="W-FRI", name="date"),
        )
        fetcher.save(quarterly)
        loaded = fetcher.load()
        # check_freq=False: Parquet não preserva freq do DatetimeIndex
        pd.testing.assert_frame_equal(quarterly, loaded, check_freq=False)

    def test_load_raises_if_not_found(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with pytest.raises(FileNotFoundError):
            fetcher.load()
