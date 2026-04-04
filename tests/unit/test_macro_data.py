"""Testes unitários para MacroDataFetcher.

Todas as chamadas de rede são mockadas — zero tráfego real durante os testes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from src.data_ingestion.macro_data import MacroDataFetcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DAILY_INDEX = pd.date_range("2024-01-02", periods=30, freq="B")

_SAMPLE_BCB_DAILY = pd.DataFrame(
    {
        "cdi_rate": [12.65] * 30,
        "usd_brl": [5.0] * 30,
        "selic_rate": [13.75] * 30,
    },
    index=_DAILY_INDEX,
)

_MONTHLY_INDEX = pd.date_range("2024-01-31", periods=3, freq="ME")

_SAMPLE_BCB_MONTHLY = pd.DataFrame(
    {
        "igpm": [0.5] * 3,
        "ipca": [0.4] * 3,
    },
    index=_MONTHLY_INDEX,
)

_SAMPLE_FUTURES = pd.DataFrame(
    {
        "soy_price_usd": [1300.0] * 30,
        "corn_price_usd": [500.0] * 30,
    },
    index=_DAILY_INDEX,
)


def _make_fetcher(tmp_path: Path) -> MacroDataFetcher:
    fetcher = MacroDataFetcher(
        config={"data_ingestion": {"output": {"macro": str(tmp_path) + "/"}}}
    )
    fetcher._output_path = tmp_path / "macro_weekly.parquet"
    return fetcher


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestMacroDataFetcher:
    def test_fetch_returns_dataframe_with_required_columns(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)

        with (
            patch.object(
                fetcher, "_fetch_bcb_series", side_effect=[_SAMPLE_BCB_DAILY, _SAMPLE_BCB_MONTHLY]
            ),
            patch.object(fetcher, "_fetch_futures", return_value=_SAMPLE_FUTURES),
        ):
            df = fetcher.fetch("2024-01-01", "2024-03-31")

        required = [
            "cdi_rate",
            "usd_brl",
            "selic_rate",
            "igpm",
            "ipca",
            "soy_price_usd",
            "corn_price_usd",
            "selic_real",
        ]
        for col in required:
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_weekly_index_anchored_to_friday(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with (
            patch.object(
                fetcher, "_fetch_bcb_series", side_effect=[_SAMPLE_BCB_DAILY, _SAMPLE_BCB_MONTHLY]
            ),
            patch.object(fetcher, "_fetch_futures", return_value=_SAMPLE_FUTURES),
        ):
            df = fetcher.fetch("2024-01-01", "2024-03-31")
        assert all(ts.weekday() == 4 for ts in df.index)

    def test_selic_real_computed_correctly(self, tmp_path: Path) -> None:
        """selic_real deve ser selic_rate - ipca."""
        fetcher = _make_fetcher(tmp_path)
        with (
            patch.object(
                fetcher, "_fetch_bcb_series", side_effect=[_SAMPLE_BCB_DAILY, _SAMPLE_BCB_MONTHLY]
            ),
            patch.object(fetcher, "_fetch_futures", return_value=_SAMPLE_FUTURES),
        ):
            df = fetcher.fetch("2024-01-01", "2024-03-31")
        # selic = 13.75, ipca = 0.4 (por mês — ffill para semanas)
        # selic_real = 13.75 - 0.4 = 13.35
        expected = df["selic_rate"] - df["ipca"]
        pd.testing.assert_series_equal(df["selic_real"], expected, check_names=False)

    def test_ffill_applied_to_monthly_bcb_series(self) -> None:
        """Séries mensais (IGPM, IPCA) devem ser forward-filled ao spine semanal."""
        fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
        spine = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        monthly = pd.DataFrame(
            {"igpm": [0.5, 0.6]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        aligned = fetcher._align_to_weekly(monthly, spine)
        # Semanas de jan (antes de 31/jan) podem ser NaN → bfill(4) resolve o início
        # Semanas de fev devem ter 0.5 (ffill do jan)
        feb_rows = aligned.loc["2024-02-01":"2024-02-28"]
        assert (feb_rows["igpm"] - 0.5).abs().lt(1e-6).all()

    def test_bcb_empty_response_raises_value_error(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        empty_df = pd.DataFrame()
        with patch("src.data_ingestion.macro_data.sgs.get", return_value=empty_df):  # noqa: SIM117
            with pytest.raises(ValueError, match="vazio"):
                fetcher._fetch_bcb_series({"cdi_rate": 4389}, "2024-01-01", "2024-12-31")

    def test_futures_failure_logs_warning_not_raises(self, tmp_path: Path) -> None:
        """Se futuros falharem, o fetch deve continuar sem lançar exceção."""
        fetcher = _make_fetcher(tmp_path)
        with (
            patch.object(
                fetcher, "_fetch_bcb_series", side_effect=[_SAMPLE_BCB_DAILY, _SAMPLE_BCB_MONTHLY]
            ),
            patch.object(fetcher, "_download_futures_single", side_effect=RuntimeError("timeout")),
        ):
            # Não deve lançar — deve retornar DF com colunas de commodities ausentes/NaN
            df = fetcher.fetch("2024-01-01", "2024-03-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_align_to_weekly_removes_timezone(self) -> None:
        fetcher = MacroDataFetcher.__new__(MacroDataFetcher)
        spine = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        raw_with_tz = pd.DataFrame(
            {"cdi_rate": [12.0] * 5},
            index=pd.date_range("2024-01-02", periods=5, freq="B", tz="America/Sao_Paulo"),
        )
        aligned = fetcher._align_to_weekly(raw_with_tz, spine)
        assert aligned.index.tz is None

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with (
            patch.object(
                fetcher, "_fetch_bcb_series", side_effect=[_SAMPLE_BCB_DAILY, _SAMPLE_BCB_MONTHLY]
            ),
            patch.object(fetcher, "_fetch_futures", return_value=_SAMPLE_FUTURES),
        ):
            df_original = fetcher.fetch("2024-01-01", "2024-03-31")
        fetcher.save(df_original)
        df_loaded = fetcher.load()
        # check_freq=False: Parquet não preserva freq do DatetimeIndex
        pd.testing.assert_frame_equal(df_original, df_loaded, check_freq=False)

    def test_load_raises_if_not_found(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with pytest.raises(FileNotFoundError):
            fetcher.load()
