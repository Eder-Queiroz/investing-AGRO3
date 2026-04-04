"""
Ingestão de dados de mercado — AGRO3.SA via yfinance.

Busca preços históricos diários e reamostrado para frequência semanal (W-FRI).
Usa auto_adjust=True para evitar a armadilha dos dividendos: os preços
retornados já são retroativamente ajustados, logo um ex-date não gera queda
espúria de preço para a MLP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns, validate_no_future_leakage

logger: logging.Logger = get_logger(__name__)

_REQUIRED_COLS = ["close_adj", "open_adj", "high_adj", "low_adj", "volume"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MarketDataFetcher:
    """Fetcher de dados de mercado para AGRO3.SA.

    Responsável por:
    - Baixar OHLCV diário via yfinance (auto-ajustado para proventos/splits)
    - Reamostrar para frequência semanal (último pregão da semana = sexta)
    - Salvar/carregar como Parquet em data/raw/market/

    Exemplo de uso:
        fetcher = MarketDataFetcher()
        df = fetcher.fetch("2006-01-01", "2025-12-31")
        fetcher.save(df)
    """

    TICKER: str = "AGRO3.SA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_pipeline_config()
        raw_output: str = cfg["data_ingestion"]["output"]["market"]
        self._output_path: Path = _PROJECT_ROOT / raw_output / "agro3_weekly.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Baixa e reamostrou dados de mercado da AGRO3 para frequência semanal.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'.
            end_date: Data final no formato 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex semanal (W-FRI), colunas:
            close_adj, open_adj, high_adj, low_adj, volume.

        Raises:
            ValueError: Se o yfinance retornar DataFrame vazio.
        """
        logger.info(f"Buscando dados de mercado: {self.TICKER} de {start_date} a {end_date}")
        daily_df = self._download_daily(start_date, end_date)
        weekly_df = self._resample_to_weekly(daily_df)

        validate_columns(weekly_df, _REQUIRED_COLS, context="MarketDataFetcher.fetch")
        validate_no_future_leakage(
            weekly_df.reset_index(), "date", context="MarketDataFetcher.fetch"
        )

        logger.info(
            f"Dados de mercado prontos: {len(weekly_df)} semanas ({start_date} → {end_date})"
        )
        return weekly_df

    def save(self, df: pd.DataFrame) -> None:
        """Persiste o DataFrame como Parquet comprimido (snappy).

        Args:
            df: DataFrame retornado por fetch().
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._output_path, engine="pyarrow", compression="snappy")
        logger.info(f"Salvo {len(df)} linhas em {self._output_path}")

    def load(self) -> pd.DataFrame:
        """Carrega o Parquet previamente salvo.

        Returns:
            DataFrame com as mesmas colunas e índice de fetch().

        Raises:
            FileNotFoundError: Se o arquivo ainda não existir.
        """
        if not self._output_path.exists():
            raise FileNotFoundError(
                f"Dados de mercado não encontrados em {self._output_path}. "
                "Execute fetch() e save() primeiro."
            )
        return pd.read_parquet(self._output_path, engine="pyarrow")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _download_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Baixa OHLCV diário via yfinance com retry automático.

        auto_adjust=True: todos os preços são retroativamente ajustados para
        dividendos e splits. Isso previne a "armadilha dos dividendos": o
        modelo não aprende que um pagamento de proventos = queda de valor.

        Args:
            start_date: Data inicial 'YYYY-MM-DD'.
            end_date: Data final 'YYYY-MM-DD'.

        Returns:
            DataFrame diário com colunas Open, High, Low, Close, Volume.

        Raises:
            ValueError: Se yfinance retornar DataFrame vazio.
        """
        ticker = yf.Ticker(self.TICKER)
        df = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,  # ajuste retroativo de dividendos/splits
            actions=False,  # sem colunas Dividends/Stock Splits na saída
        )
        if df.empty:
            raise ValueError(
                f"yfinance retornou DataFrame vazio para {self.TICKER} "
                f"no período {start_date} → {end_date}"
            )
        logger.debug(f"Download diário: {len(df)} linhas para {self.TICKER}")
        return df

    def _resample_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Reamostro de diário para semanal (âncora sexta-feira).

        Estratégia de agregação:
        - Open: primeiro pregão da semana
        - High: máximo da semana
        - Low: mínimo da semana
        - Close: último pregão da semana (sexta ou último dia útil anterior)
        - Volume: soma dos volumes diários

        Args:
            daily_df: DataFrame diário retornado por _download_daily().

        Returns:
            DataFrame semanal com colunas renomeadas para snake_case.
        """
        # Remove timezone se presente (yfinance pode retornar America/Sao_Paulo)
        if daily_df.index.tz is not None:
            daily_df = daily_df.copy()
            daily_df.index = daily_df.index.tz_convert(None)

        weekly = daily_df.resample("W-FRI").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

        # Remove semanas sem pregão (ex.: feriados que cobrem a semana inteira)
        weekly = weekly.dropna(subset=["Close"])

        weekly = weekly.rename(
            columns={
                "Open": "open_adj",
                "High": "high_adj",
                "Low": "low_adj",
                "Close": "close_adj",
                "Volume": "volume",
            }
        )

        weekly.index.name = "date"
        return weekly


# ---------------------------------------------------------------------------
# Execução standalone: uv run python -m src.data_ingestion.market_data
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import pandas as pd

    cfg = load_pipeline_config()
    di_cfg = cfg["data_ingestion"]
    end = di_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    fetcher = MarketDataFetcher(config=cfg)
    df = fetcher.fetch(start_date=di_cfg["start_date"], end_date=end)
    fetcher.save(df)

    print(f"\nAmostras:\n{df.tail()}")
    print(f"\nInfo:\n{df.dtypes}")
