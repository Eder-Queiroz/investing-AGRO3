"""
Ingestão de dados macroeconômicos.

Fontes:
- Banco Central do Brasil (BCB/SGS via python-bcb):
    CDI, Selic, USD/BRL (PTAX), IGPM, IPCA
- CBOT via yfinance (futuros contínuos):
    Soja (ZS=F), Milho (ZC=F)

Coluna derivada:
    selic_real = selic_rate - ipca (ambos anualizados em %)

Todas as séries são alinhadas ao spine semanal (W-FRI) com forward-fill.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from bcb import sgs
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns

logger: logging.Logger = get_logger(__name__)

_REQUIRED_COLS = [
    "cdi_rate",
    "usd_brl",
    "selic_rate",
    "igpm",
    "ipca",
    "soy_price_usd",
    "corn_price_usd",
    "selic_real",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MacroDataFetcher:
    """Fetcher de dados macroeconômicos para o modelo AGRO3.

    Responsável por:
    - Buscar séries do BCB/SGS (CDI, Selic, USD/BRL, IGPM, IPCA)
    - Buscar futuros de commodities CBOT (Soja, Milho) via yfinance
    - Alinhar tudo ao spine semanal sexta-feira com forward-fill
    - Computar Selic real = Selic - IPCA
    - Salvar/carregar como Parquet em data/raw/macro/

    Nota sobre pipeline_config.yaml: o bloco fred_series contém apenas
    placeholders — este módulo usa yfinance para commodities, não FRED.
    O bloco FRED é preservado no YAML para uso futuro.

    Exemplo de uso:
        fetcher = MacroDataFetcher()
        df = fetcher.fetch("2006-01-01", "2025-12-31")
        fetcher.save(df)
    """

    # BCB série → nome da coluna de saída
    BCB_DAILY_SERIES: dict[str, int] = {
        "cdi_rate": 4389,  # CDI Over (% a.a.)
        "usd_brl": 1,  # PTAX USD/BRL
        "selic_rate": 4390,  # Selic (% a.a.)
    }
    BCB_MONTHLY_SERIES: dict[str, int] = {
        "igpm": 189,  # IGPM mensal (% a.m.)
        "ipca": 433,  # IPCA mensal (% a.m.)
    }

    # Tickers yfinance → nome da coluna de saída
    FUTURES_TICKERS: dict[str, str] = {
        "soy_price_usd": "ZS=F",  # CBOT Soybeans (USD/bushel)
        "corn_price_usd": "ZC=F",  # CBOT Corn (USD/bushel)
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_pipeline_config()
        raw_output: str = cfg["data_ingestion"]["output"]["macro"]
        self._output_path: Path = _PROJECT_ROOT / raw_output / "macro_weekly.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Busca e alinha todos os dados macro à frequência semanal.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'.
            end_date: Data final no formato 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex semanal (W-FRI), colunas:
            cdi_rate, usd_brl, selic_rate, igpm, ipca,
            soy_price_usd, corn_price_usd, selic_real.

        Raises:
            ValueError: Se o BCB retornar dados vazios para as séries diárias.
        """
        logger.info(f"Buscando dados macro de {start_date} a {end_date}")

        spine = self._build_weekly_spine(start_date, end_date)

        # --- BCB ---
        daily_bcb = self._fetch_bcb_series(self.BCB_DAILY_SERIES, start_date, end_date)
        monthly_bcb = self._fetch_bcb_series(self.BCB_MONTHLY_SERIES, start_date, end_date)
        raw_bcb = pd.concat([daily_bcb, monthly_bcb], axis=1)

        # --- Futuros CBOT ---
        futures_df = self._fetch_futures(start_date, end_date)

        # --- Combinar e alinhar ---
        raw_all = pd.concat([raw_bcb, futures_df], axis=1) if not futures_df.empty else raw_bcb
        aligned = self._align_to_weekly(raw_all, spine)

        # Garante colunas de commodities mesmo se futuros falharem completamente
        import numpy as np

        for commodity_col in ["soy_price_usd", "corn_price_usd"]:
            if commodity_col not in aligned.columns:
                aligned[commodity_col] = np.nan
                logger.warning(
                    f"Coluna '{commodity_col}' ausente"
                    f"(fetch de futuros falhou) — preenchida com NaN"
                )

        # --- Selic real (coluna derivada) ---
        aligned["selic_real"] = aligned["selic_rate"] - aligned["ipca"]

        validate_columns(aligned, _REQUIRED_COLS, context="MacroDataFetcher.fetch")

        logger.info(f"Dados macro prontos: {len(aligned)} semanas, {aligned.shape[1]} colunas")
        return aligned

    def save(self, df: pd.DataFrame) -> None:
        """Persiste o DataFrame como Parquet comprimido (snappy)."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._output_path, engine="pyarrow", compression="snappy")
        logger.info(f"Salvo {len(df)} linhas em {self._output_path}")

    def load(self) -> pd.DataFrame:
        """Carrega o Parquet previamente salvo.

        Raises:
            FileNotFoundError: Se o arquivo ainda não existir.
        """
        if not self._output_path.exists():
            raise FileNotFoundError(
                f"Dados macro não encontrados em {self._output_path}. "
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
    def _fetch_bcb_series(
        self,
        series: dict[str, int],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Busca séries do BCB/SGS com retry automático.

        Args:
            series: Mapeamento {nome_coluna: código_bcb}.
            start: Data inicial 'YYYY-MM-DD'.
            end: Data final 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex diário/mensal e colunas nomeadas.

        Raises:
            ValueError: Se a resposta do BCB for vazia.
        """
        # sgs.get aceita {nome: código} e retorna DataFrame com essas colunas
        # A inversão é necessária: sgs.get espera {código: nome} ou
        # aceita dict diretamente dependendo da versão do python-bcb.
        # A API atual de python-bcb >= 0.3.0 aceita dict {nome: código}.
        raw = sgs.get(series, start=start, end=end)
        if raw.empty:
            raise ValueError(f"BCB SGS retornou DataFrame vazio para séries: {list(series.keys())}")
        # Garante float64 — BCB pode retornar object dtype
        return raw.astype(float)

    def _fetch_futures(self, start: str, end: str) -> pd.DataFrame:
        """Busca preços de futuros CBOT via yfinance.

        Usa front-month continuous contracts (ZS=F, ZC=F).
        Gaps de roll (1-2 dias) são resolvidos pelo ffill no alinhamento.
        Se um ticker falhar completamente, registra warning e continua
        sem aquela coluna — não lança exceção.

        Args:
            start: Data inicial 'YYYY-MM-DD'.
            end: Data final 'YYYY-MM-DD'.

        Returns:
            DataFrame com colunas soy_price_usd e/ou corn_price_usd.
        """
        frames: list[pd.Series] = []
        for col_name, ticker_sym in self.FUTURES_TICKERS.items():
            try:
                df = self._download_futures_single(ticker_sym, start, end)
                if df.empty:
                    logger.warning(
                        f"yfinance retornou dados vazios para {ticker_sym} — coluna será NaN"
                    )
                    continue
                series = df["Close"].rename(col_name)
                if series.index.tz is not None:
                    series.index = series.index.tz_convert(None)
                frames.append(series)
            except Exception as exc:
                logger.warning(f"Falha ao buscar {ticker_sym}: {exc} — coluna será NaN")

        if not frames:
            logger.warning(
                "Nenhum ticker de futuros retornou dados — colunas de commodities serão NaN"
            )
            return pd.DataFrame()

        return pd.concat(frames, axis=1)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _download_futures_single(self, ticker_sym: str, start: str, end: str) -> pd.DataFrame:
        """Download de um único ticker de futuros com retry."""
        ticker = yf.Ticker(ticker_sym)
        return ticker.history(start=start, end=end, auto_adjust=True, actions=False)

    @staticmethod
    def _build_weekly_spine(start: str, end: str) -> pd.DatetimeIndex:
        """Constrói o spine semanal (sextas-feiras) para o período.

        Args:
            start: Data inicial 'YYYY-MM-DD'.
            end: Data final 'YYYY-MM-DD'.

        Returns:
            DatetimeIndex com freq='W-FRI', timezone-naive.
        """
        return pd.date_range(start=start, end=end, freq="W-FRI", name="date")

    @staticmethod
    def _align_to_weekly(raw: pd.DataFrame, spine: pd.DatetimeIndex) -> pd.DataFrame:
        """Alinha um DataFrame de frequência mista ao spine semanal.

        Estratégia:
        1. Remove timezone se presente
        2. Converte tudo para float64
        3. Reindex ao spine com forward-fill (carrega o último valor conhecido)
        4. Back-fill limitado (máx. 4 semanas) apenas para NaN no início

        Args:
            raw: DataFrame com qualquer DatetimeIndex (diário, mensal, etc.).
            spine: DatetimeIndex semanal de destino.

        Returns:
            DataFrame alinhado ao spine, sem timezone.
        """
        if raw.empty:
            return pd.DataFrame(index=spine)

        if raw.index.tz is not None:
            raw = raw.copy()
            raw.index = raw.index.tz_convert(None)

        raw = raw.astype(float)

        aligned = raw.reindex(spine, method="ffill")

        # Back-fill limitado para NaN iniciais (série começou depois do spine)
        aligned = aligned.bfill(limit=4)

        return aligned


# ---------------------------------------------------------------------------
# Execução standalone: uv run python -m src.data_ingestion.macro_data
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import pandas as pd

    cfg = load_pipeline_config()
    di_cfg = cfg["data_ingestion"]
    end = di_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    fetcher = MacroDataFetcher(config=cfg)
    df = fetcher.fetch(start_date=di_cfg["start_date"], end_date=end)
    fetcher.save(df)

    print(f"\nAmostras:\n{df.tail()}")
    print(f"\nInfo:\n{df.dtypes}")
