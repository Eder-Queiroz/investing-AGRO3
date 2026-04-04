"""Testes unitários para MarketDataFetcher.

Todas as chamadas de rede são mockadas — zero tráfego real durante os testes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from src.data_ingestion.market_data import MarketDataFetcher

# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

_DAILY_INDEX = pd.date_range("2024-01-02", periods=10, freq="B")  # seg–sex

_SAMPLE_DAILY = pd.DataFrame(
    {
        "Open": [10.0] * 10,
        "High": [11.0] * 10,
        "Low": [9.0] * 10,
        "Close": [10.5] * 10,
        "Volume": [100_000] * 10,
    },
    index=_DAILY_INDEX,
)

_SAMPLE_DAILY_TZ = _SAMPLE_DAILY.copy()
_SAMPLE_DAILY_TZ.index = _SAMPLE_DAILY_TZ.index.tz_localize("America/Sao_Paulo")

_MINIMAL_CONFIG = {
    "data_ingestion": {
        "output": {"market": "data/raw/market/"},
    }
}


def _make_fetcher(tmp_path: Path) -> MarketDataFetcher:
    cfg = {
        "data_ingestion": {
            "output": {"market": str(tmp_path) + "/"},
        }
    }
    fetcher = MarketDataFetcher(config=cfg)
    fetcher._output_path = tmp_path / "agro3_weekly.parquet"
    return fetcher


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestMarketDataFetcher:
    def test_fetch_happy_path_returns_dataframe(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df = fetcher.fetch("2024-01-01", "2024-01-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df = fetcher.fetch("2024-01-01", "2024-01-31")
        for col in ["close_adj", "open_adj", "high_adj", "low_adj", "volume"]:
            assert col in df.columns, f"Coluna ausente: {col}"

    def test_weekly_index_anchored_to_friday(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df = fetcher.fetch("2024-01-01", "2024-01-31")
        # Todas as datas do índice devem ser sextas (weekday == 4)
        assert all(ts.weekday() == 4 for ts in df.index), "Índice não está ancorado em sextas"

    def test_empty_yfinance_response_raises_value_error(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        empty_df = pd.DataFrame()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_df
        with patch("src.data_ingestion.market_data.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(ValueError, match="vazio"):
                fetcher._download_daily("2024-01-01", "2024-01-31")

    def test_timezone_stripped_from_index(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        weekly = fetcher._resample_to_weekly(_SAMPLE_DAILY_TZ)
        assert weekly.index.tz is None, "Timezone não foi removida do índice"

    def test_volume_is_sum_not_last(self, tmp_path: Path) -> None:
        """Volume semanal deve ser a soma dos volumes diários da semana."""
        fetcher = _make_fetcher(tmp_path)
        weekly = fetcher._resample_to_weekly(_SAMPLE_DAILY)
        # 5 dias de vol 100_000 → 500_000 por semana completa
        assert weekly["volume"].max() == pytest.approx(500_000, rel=0.01)

    def test_close_is_last_trading_day(self, tmp_path: Path) -> None:
        """Close semanal deve ser o fechamento do último pregão da semana."""
        daily = _SAMPLE_DAILY.copy()
        # _DAILY_INDEX começa em 2024-01-02 (Terça).
        # Sexta da 1ª semana = 2024-01-05 → índice 3 (Ter, Qua, Qui, Sex)
        friday_idx = 3
        daily.iloc[friday_idx, daily.columns.get_loc("Close")] = 99.0
        fetcher = _make_fetcher(tmp_path)
        weekly = fetcher._resample_to_weekly(daily)
        assert weekly["close_adj"].iloc[0] == pytest.approx(99.0)

    def test_save_creates_parquet_file(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df = fetcher.fetch("2024-01-01", "2024-01-31")
        fetcher.save(df)
        assert fetcher._output_path.exists()
        assert fetcher._output_path.suffix == ".parquet"

    def test_load_roundtrip_preserves_dtypes(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df_original = fetcher.fetch("2024-01-01", "2024-01-31")
        fetcher.save(df_original)
        df_loaded = fetcher.load()
        # check_freq=False: Parquet não preserva freq do DatetimeIndex
        pd.testing.assert_frame_equal(df_original, df_loaded, check_freq=False)

    def test_load_raises_if_file_not_found(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with pytest.raises(FileNotFoundError, match="fetch()"):
            fetcher.load()

    def test_index_name_is_date(self, tmp_path: Path) -> None:
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, "_download_daily", return_value=_SAMPLE_DAILY):
            df = fetcher.fetch("2024-01-01", "2024-01-31")
        assert df.index.name == "date"
