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
import requests
import yfinance as yf
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
        "cdi_rate": 4389,  # CDI Anualizado (% a.a.)
        "usd_brl": 1,  # PTAX USD/BRL
        "selic_rate": 1178,  # Selic Anualizada base 252 (% a.a.) - CORRIGIDO!
    }
    BCB_MONTHLY_SERIES: dict[str, int] = {
        "igpm": 189,  # IGPM mensal (% a.m.) - O BCB não tem IGPM 12m limpo, mantemos mensal
        "ipca": 13522,  # IPCA Acumulado 12 Meses (% a.a.) - CORRIGIDO!
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
        raw_bcb = pd.concat([daily_bcb, monthly_bcb], axis=1, sort=False)

        # --- Futuros CBOT ---
        futures_df = self._fetch_futures(start_date, end_date)

        # --- Combinar e alinhar ---
        raw_all = (
            pd.concat([raw_bcb, futures_df], axis=1, sort=False)
            if not futures_df.empty
            else raw_bcb
        )
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

    def _fetch_bcb_series(
        self,
        series: dict[str, int],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Busca séries do BCB/SGS, uma por vez, combinando ao final.

        Fetching individualmente isola falhas: se uma série falhar, as demais
        continuam. Cada chamada individual tem retry próprio via
        _fetch_single_bcb_series.

        Args:
            series: Mapeamento {nome_coluna: código_bcb}.
            start: Data inicial 'YYYY-MM-DD'.
            end: Data final 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex e colunas nomeadas.

        Raises:
            ValueError: Se TODAS as séries falharem.
        """
        frames: dict[str, pd.Series] = {}
        for col_name, code in series.items():
            try:
                s = self._fetch_single_bcb_series(col_name, code, start, end)
                frames[col_name] = s
            except Exception as exc:
                logger.warning(f"BCB série {col_name} (código {code}) falhou: {exc}")

        if not frames:
            raise ValueError(f"Todas as séries BCB falharam: {list(series.keys())}")

        return pd.DataFrame(frames).astype(float)

    def _fetch_single_bcb_series(
        self,
        col_name: str,
        code: int,
        start: str,
        end: str,
        chunk_years: int = 5,  # 5 anos garante segurança contra o limite de 10 anos do BCB
    ) -> pd.Series:
        """Busca uma série BCB em chunks anuais.

        A API do BCB exige dataInicial para séries diárias e limita a consulta
        a um máximo de 10 anos por requisição. Divide o período em blocos de
        `chunk_years` para contornar essa limitação.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        chunks: list[pd.Series] = []

        chunk_start = start_ts
        while chunk_start <= end_ts:
            chunk_end = min(
                chunk_start + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1),
                end_ts,
            )
            try:
                chunk = self._fetch_bcb_chunk(code, chunk_start, chunk_end)
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception as exc:
                logger.warning(
                    f"BCB chunk {chunk_start.date()}–{chunk_end.date()} código={code} falhou: {exc}"
                )
            chunk_start = chunk_end + pd.Timedelta(days=1)

        if not chunks:
            raise ValueError(f"Todos os chunks BCB falharam para código {code}")

        s = pd.concat(chunks).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = col_name
        return s

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_bcb_chunk(
        self,
        code: int,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """Busca um único chunk da API REST do BCB camuflado como navegador.

        Inclui headers para evitar o bloqueio WAF (406 Not Acceptable).
        """
        start_br = start.strftime("%d/%m/%Y")
        end_br = end.strftime("%d/%m/%Y")

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
            f"?formato=json&dataInicial={start_br}&dataFinal={end_br}"
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        logger.debug(f"BCB GET código={code} {start_br}–{end_br}")
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        if not data:
            return pd.Series(dtype=float)

        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        s = df.set_index("data")["valor"]
        s.index.name = "date"

        return s

    def _fetch_futures(self, start: str, end: str) -> pd.DataFrame:
        """Busca preços de futuros CBOT via yfinance.

        Garante que o índice seja do tipo Date (sem horas e timezone) para
        alinhar perfeitamente com os dados do Banco Central no merge_asof.
        """
        frames: list[pd.Series] = []
        for col_name, ticker_sym in self.FUTURES_TICKERS.items():
            try:
                df = self._download_futures_single(ticker_sym, start, end)
                if df.empty:
                    logger.warning(f"yfinance retornou vazio para {ticker_sym}")
                    continue

                series = df["Close"].rename(col_name)

                # A CORREÇÃO: Remove a hora e o timezone, forçando a ser apenas 'Data'
                if series.index.tz is not None:
                    series.index = series.index.tz_convert(None)
                series.index = series.index.normalize()  # Força a hora para 00:00:00

                frames.append(series)
            except Exception as exc:
                logger.warning(f"Falha ao buscar {ticker_sym}: {exc}")

        if not frames:
            logger.warning("Nenhum ticker retornou dados.")
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
        if raw.empty:
            return pd.DataFrame(index=spine)

        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(None)

        raw = raw.astype(float)

        # 1. Limpa duplicatas e garante a ordem do tempo
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()

        # 2. A MÁGICA: Propaga a Selic e o IPCA para todos os dias vazios seguintes!
        raw = raw.ffill()

        # 3. Cria o alvo e faz o merge seguro (puxando as sextas-feiras)
        target_df = pd.DataFrame(index=spine)
        aligned = pd.merge_asof(
            target_df, raw, left_index=True, right_index=True, direction="backward"
        )

        # 4. Limpa possíveis NaNs residuais bem no começo do DataFrame
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
