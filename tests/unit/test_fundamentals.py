"""Testes unitários para FundamentalsFetcher e StatusInvestClient.

FundamentalsFetcher:
- Lag de 45 dias (anti-leakage crítico)
- merge_asof sem future data
- EBITDA negativo não quebra o pipeline
- Ratios computados corretamente com fixtures do Status Invest

StatusInvestClient:
- Parsing de rótulos de trimestre ('3T2024' → 2024-09-30)
- Parsing do gridLineModel (None → NaN, shape, ordenação)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data_ingestion.fundamentals import FundamentalsFetcher
from src.data_ingestion.status_invest import StatusInvestClient

# ---------------------------------------------------------------------------
# Fixtures — estrutura Status Invest (não yfinance)
# ---------------------------------------------------------------------------

_Q_DATES = pd.to_datetime(["2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"])

# DRE: campos conforme retornados pelo Status Invest
# MargemBruta e Roe são pré-calculados em % (ex: 37.5 = 37.5%)
_DRE = pd.DataFrame(
    {
        "ReceitaLiquida": [800_000_000.0] * 4,
        "Ebitda": [170_000_000.0] * 4,  # EBIT 150M + D&A 20M
        "Ebit": [150_000_000.0] * 4,
        "LucroLiquido": [100_000_000.0] * 4,
        "MargemBruta": [37.5] * 4,  # % → /100 → 0.375
        "ROE": [20.0] * 4,  # % → /100 → 0.20  (chave real da API: maiúsculas)
        "DividaBruta": [200_000_000.0] * 4,  # disponível no DRE
    },
    index=_Q_DATES,
)

# Balanço Patrimonial: campos conforme retornados pelo Status Invest (/acao/getativos)
# DividaBruta vem do DRE; aqui apenas equity e cash.
_BP = pd.DataFrame(
    {
        "PatrimonioLiquidoConsolidado": [500_000_000.0] * 4,
        "CaixaeEquivalentesdeCaixa": [50_000_000.0] * 4,
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
    fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
    fetcher._output_path = tmp_path / "fundamentals_quarterly.parquet"
    return fetcher


# ---------------------------------------------------------------------------
# FundamentalsFetcher — lag e alinhamento temporal
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

        neg_dre = _DRE.copy()
        neg_dre["Ebitda"] = -50_000_000.0

        ratios = fetcher._compute_ratios(neg_dre, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)
        assert ratios["ev_ebitda"].isna().all(), "ev_ebitda deveria ser NaN com EBITDA negativo"
        assert ratios["net_debt_ebitda"].isna().all()

    def test_p_vpa_computed_from_price_and_equity(self) -> None:
        """P/VP = preço / (equity / shares).

        equity=500M, shares=100M → book/share=5.0; price=30.0 → P/VP=6.0.
        """
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        ratios = fetcher._compute_ratios(_DRE, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)
        assert ratios["p_vpa"].dropna().iloc[0] == pytest.approx(6.0, rel=0.01)

    def test_gross_margin_converted_from_percentage(self) -> None:
        """gross_margin = MargemBruta% / 100 = 37.5 / 100 = 0.375."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        ratios = fetcher._compute_ratios(_DRE, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)
        assert ratios["gross_margin"].dropna().iloc[0] == pytest.approx(0.375, rel=0.01)

    def test_roe_converted_from_percentage(self) -> None:
        """roe = Roe% / 100 = 20.0 / 100 = 0.20."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        ratios = fetcher._compute_ratios(_DRE, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)
        assert ratios["roe"].dropna().iloc[0] == pytest.approx(0.20, rel=0.01)

    def test_net_debt_ebitda_computed_correctly(self) -> None:
        """net_debt_ebitda = (DividaBruta - Caixa) / Ebitda = 150M / 170M."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        ratios = fetcher._compute_ratios(_DRE, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)
        expected = (200_000_000 - 50_000_000) / 170_000_000
        assert ratios["net_debt_ebitda"].dropna().iloc[0] == pytest.approx(expected, rel=0.01)

    def test_missing_dre_field_raises_key_error(self) -> None:
        """KeyError claro quando campo do DRE está ausente."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        bad_dre = _DRE.drop(columns=["Ebitda"])
        with pytest.raises(KeyError, match="Campo do DRE"):
            fetcher._compute_ratios(bad_dre, _BP, _PRICE_SERIES, 100_000_000, _DIVIDENDS)

    def test_missing_bp_field_raises_key_error(self) -> None:
        """KeyError claro quando campo do Balanço está ausente."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        bad_bp = _BP.drop(columns=["PatrimonioLiquidoConsolidado"])
        with pytest.raises(KeyError, match="Campo do Balanço"):
            fetcher._compute_ratios(_DRE, bad_bp, _PRICE_SERIES, 100_000_000, _DIVIDENDS)

    def test_leading_nan_rows_acceptable(self) -> None:
        """Primeiras semanas sem dados são NaN — comportamento esperado."""
        fetcher = FundamentalsFetcher.__new__(FundamentalsFetcher)
        quarterly = pd.DataFrame(
            {"p_vpa": [2.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-05-15")], name="date"),
        )
        spine = pd.date_range("2024-01-05", periods=30, freq="W-FRI", name="date")
        merged = fetcher._merge_to_weekly_spine(quarterly, spine)
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


# ---------------------------------------------------------------------------
# StatusInvestClient — parsing de datas e grid
# ---------------------------------------------------------------------------


class TestStatusInvestClient:
    def test_parse_quarter_dates_q1(self) -> None:
        """'1T2024' → último dia de março de 2024."""
        client = StatusInvestClient()
        result = client._parse_quarter_dates(["1T2024"])
        assert result[0] == pd.Timestamp("2024-03-31")

    def test_parse_quarter_dates_q2(self) -> None:
        """'2T2024' → último dia de junho de 2024."""
        client = StatusInvestClient()
        result = client._parse_quarter_dates(["2T2024"])
        assert result[0] == pd.Timestamp("2024-06-30")

    def test_parse_quarter_dates_q3(self) -> None:
        """'3T2024' → último dia de setembro de 2024."""
        client = StatusInvestClient()
        result = client._parse_quarter_dates(["3T2024"])
        assert result[0] == pd.Timestamp("2024-09-30")

    def test_parse_quarter_dates_q4(self) -> None:
        """'4T2023' → último dia de dezembro de 2023."""
        client = StatusInvestClient()
        result = client._parse_quarter_dates(["4T2023"])
        assert result[0] == pd.Timestamp("2023-12-31")

    def test_parse_quarter_dates_all_quarters(self) -> None:
        """Q1-Q4 do mesmo ano mapeiam para datas de fim de trimestre corretas."""
        client = StatusInvestClient()
        labels = ["4T2023", "3T2023", "2T2023", "1T2023"]
        result = client._parse_quarter_dates(labels)
        expected = pd.DatetimeIndex([
            pd.Timestamp("2023-12-31"),
            pd.Timestamp("2023-09-30"),
            pd.Timestamp("2023-06-30"),
            pd.Timestamp("2023-03-31"),
        ])
        pd.testing.assert_index_equal(result, expected)

    def test_parse_quarter_dates_invalid_format_raises(self) -> None:
        """Rótulo em formato inválido levanta ValueError."""
        client = StatusInvestClient()
        with pytest.raises(ValueError, match="Rótulo de trimestre inesperado"):
            client._parse_quarter_dates(["MARÇO/2024"])

    # ------------------------------------------------------------------
    # Helpers para montar fixtures com a estrutura real da API
    # ------------------------------------------------------------------

    @staticmethod
    def _make_grid(labels: list[str], rows: list[dict]) -> dict:
        """Monta um dict com a estrutura real de data['data']['grid']."""
        header_cols = [{"value": "#"}]
        for lbl in labels:
            header_cols += [
                {"name": "DATA", "value": lbl},
                {"name": "AH", "value": "AH"},
                {"name": "AV", "value": "AV"},
            ]
        grid = [{"isHeader": True, "row": 0, "columns": header_cols}]
        for i, row in enumerate(rows, start=1):
            grid.append({
                "isHeader": False,
                "row": i,
                "columns": [],
                "gridLineModel": row,
            })
        return {"grid": grid}

    # ------------------------------------------------------------------
    # _parse_grid tests
    # ------------------------------------------------------------------

    def test_parse_grid_shape(self) -> None:
        """DataFrame resultante tem shape (n_quarters, n_metrics)."""
        client = StatusInvestClient()
        data = self._make_grid(
            labels=["2T2024", "1T2024"],
            rows=[
                {"key": "Ebitda", "values": [170.0, 160.0]},
                {"key": "Roe", "values": [20.0, 18.0]},
            ],
        )
        df = client._parse_grid(data)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Ebitda", "Roe"]

    def test_parse_grid_none_becomes_nan(self) -> None:
        """Valores None na API são convertidos para np.nan."""
        client = StatusInvestClient()
        data = self._make_grid(
            labels=["2T2024", "1T2024"],
            rows=[{"key": "Ebitda", "values": [170.0, None]}],
        )
        df = client._parse_grid(data)
        assert np.isnan(df.loc[pd.Timestamp("2024-03-31"), "Ebitda"])
        assert df.loc[pd.Timestamp("2024-06-30"), "Ebitda"] == pytest.approx(170.0)

    def test_parse_grid_sorts_ascending(self) -> None:
        """Status Invest retorna mais recente primeiro; resultado deve ser ascendente."""
        client = StatusInvestClient()
        data = self._make_grid(
            labels=["3T2024", "2T2024", "1T2024"],
            rows=[{"key": "Ebitda", "values": [300.0, 200.0, 100.0]}],
        )
        df = client._parse_grid(data)
        assert df.index.is_monotonic_increasing
        # Q1 (mais antigo) deve ser a primeira linha com valor 100.0
        assert df.index[0] == pd.Timestamp("2024-03-31")
        assert df["Ebitda"].iloc[0] == pytest.approx(100.0)

    def test_parse_grid_index_name_is_date(self) -> None:
        """index.name deve ser 'date' para compatibilidade com merge_asof."""
        client = StatusInvestClient()
        data = self._make_grid(
            labels=["1T2024"],
            rows=[{"key": "Ebitda", "values": [100.0]}],
        )
        df = client._parse_grid(data)
        assert df.index.name == "date"
