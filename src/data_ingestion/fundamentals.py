"""
Ingestão de dados fundamentalistas — AGRO3.SA via Status Invest API.

Fonte primária:
    Status Invest (/acao/getdre, /acao/getativos) — DRE e Balanço Patrimonial
    trimestrais de 2011 a 2025 (~59 trimestres).

Fonte secundária (via yfinance):
    Preços ajustados, ações em circulação e dividendos — necessários para
    computar P/VP, EV/EBITDA e Dividend Yield.

Ratios computados:
    p_vpa           — Preço / Valor Patrimonial por Ação
    ev_ebitda       — Enterprise Value / EBITDA
    roe             — Return on Equity (direto do Status Invest, em %)
    net_debt_ebitda — Dívida Líquida / EBITDA
    gross_margin    — Margem Bruta (direta do Status Invest, em %)
    dividend_yield  — DY trailing 12 meses

Cobertura:
    2006-2011: NaN (anterior à cobertura do Status Invest)
    2011-2025: dados completos (~59 trimestres)

Proteção contra data leakage:
    Lag de 45 dias aplicado ao índice trimestral antes do merge_asof.
    Q1 (31/mar) → disponível a partir de 15/mai.
    Q2 (30/jun) → disponível a partir de 14/ago. Etc.

Alinhamento temporal:
    merge_asof(direction='backward') para o spine semanal W-FRI.
    Cada semana recebe o valor do trimestre mais recente cujo
    lag-date é <= data da semana.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_ingestion.status_invest import StatusInvestClient
from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns

logger: logging.Logger = get_logger(__name__)

_REQUIRED_COLS = [
    "p_vpa",
    "ev_ebitda",
    "roe",
    "net_debt_ebitda",
    "gross_margin",
    "dividend_yield",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Nomes dos campos no DRE do Status Invest (chaves exatas da API)
_DRE_FIELDS: dict[str, str] = {
    "ebitda": "Ebitda",
    "gross_margin": "MargemBruta",  # pré-calculado em % → dividir por 100
    "roe": "ROE",  # pré-calculado em % → dividir por 100
    "total_debt": "DividaBruta",  # disponível no DRE — evita chamada extra ao getativos
}

# Nomes dos campos no Balanço Patrimonial do Status Invest (/acao/getativos)
# DividaBruta vem do DRE (mais completo); aqui só equity e cash.
_BP_FIELDS: dict[str, str] = {
    "equity": "PatrimonioLiquidoConsolidado",
    "cash": "CaixaeEquivalentesdeCaixa",
}


class FundamentalsFetcher:
    """Fetcher de dados fundamentalistas para AGRO3.SA.

    Fonte primária: Status Invest API (/acao/getdre, /acao/getativos).
    Fonte secundária: yfinance (preços, ações, dividendos).

    Responsável por:
    - Baixar demonstrativos trimestrais via StatusInvestClient
    - Computar ratios (P/VP, EV/EBITDA, ROE, etc.)
    - Aplicar lag de 45 dias para evitar data leakage
    - Alinhar ao spine semanal via merge_asof(direction='backward')
    - Salvar/carregar como Parquet em data/raw/fundamentals/

    Exemplo de uso:
        fetcher = FundamentalsFetcher()
        df = fetcher.fetch("2006-01-01", "2025-12-31")
        fetcher.save(df)
    """

    TICKER: str = "AGRO3.SA"
    CODE: str = "AGRO3"
    REPORTING_LAG_DAYS: int = 45

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_pipeline_config()
        raw_output: str = cfg["data_ingestion"]["output"]["fundamentals"]
        self._output_path: Path = _PROJECT_ROOT / raw_output / "fundamentals_quarterly.parquet"
        self._client: StatusInvestClient = StatusInvestClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Baixa fundamentos e alinha ao spine semanal com lag de reporting.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'.
            end_date: Data final no formato 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex semanal (W-FRI), colunas:
            p_vpa, ev_ebitda, roe, net_debt_ebitda, gross_margin, dividend_yield.
            Semanas de 2006-2011 (anterior à cobertura do Status Invest) terão NaN.

        Raises:
            requests.HTTPError: Se o Status Invest não responder após 3 tentativas.
            KeyError: Se um campo esperado não existir na resposta do Status Invest.
        """
        logger.info(f"Buscando fundamentos: {self.TICKER} de {start_date} a {end_date}")

        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year

        dre, bp = self._fetch_quarterly_statements(start_year, end_year)
        price_series, shares_outstanding, dividends = self._fetch_price_and_dividends(
            start_date, end_date
        )

        quarterly_df = self._compute_ratios(dre, bp, price_series, shares_outstanding, dividends)
        lagged_df = self._apply_reporting_lag(quarterly_df)
        spine = pd.date_range(start=start_date, end=end_date, freq="W-FRI", name="date")
        weekly_df = self._merge_to_weekly_spine(lagged_df, spine)

        validate_columns(weekly_df, _REQUIRED_COLS, context="FundamentalsFetcher.fetch")

        nan_count = weekly_df[_REQUIRED_COLS].isna().any(axis=1).sum()
        if nan_count > 0:
            logger.debug(
                f"{nan_count} semanas com NaN em fundamentos "
                "(esperado para período 2006-2011 anterior à cobertura do Status Invest)"
            )

        logger.info(f"Fundamentos prontos: {len(weekly_df)} semanas")
        return weekly_df

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
                f"Fundamentos não encontrados em {self._output_path}. "
                "Execute fetch() e save() primeiro."
            )
        return pd.read_parquet(self._output_path, engine="pyarrow")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_quarterly_statements(
        self, start_year: int, end_year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Busca DRE e Balanço Patrimonial trimestrais via Status Invest.

        Args:
            start_year: Ano inicial.
            end_year: Ano final.

        Returns:
            Tupla (dre_df, bp_df) com DatetimeIndex de datas de fim de trimestre.
        """
        dre = self._client.fetch_dre(self.CODE, start_year, end_year)
        bp = self._client.fetch_balance(self.CODE, start_year, end_year)

        logger.debug(
            f"Demonstrativos obtidos: DRE={len(dre)} trimestres "
            f"({dre.index.min()} → {dre.index.max()}), "
            f"BP={len(bp)} trimestres"
        )
        return dre, bp

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_price_and_dividends(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.Series, float, pd.Series]:
        """Baixa preços ajustados e dividendos para cálculo de ratios.

        Ainda usa yfinance pois preços/ações/dividendos históricos estão
        disponíveis (ao contrário dos demonstrativos financeiros trimestrais).

        Args:
            start_date: Data inicial.
            end_date: Data final.

        Returns:
            Tupla (price_series, shares_outstanding, dividends_series).

        Raises:
            ValueError: Se o yfinance retornar preços vazios.
        """
        ticker = yf.Ticker(self.TICKER)
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=True, actions=True)
        if hist.empty:
            raise ValueError(f"Preços vazios para {self.TICKER}")

        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)

        price_series: pd.Series = hist["Close"]
        dividends: pd.Series = (
            hist["Dividends"] if "Dividends" in hist.columns else pd.Series(dtype=float)
        )

        info = ticker.info
        shares_outstanding: float = float(
            info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0
        )
        if shares_outstanding == 0:
            logger.warning(
                "Número de ações não disponível no yfinance info — "
                "usando 1 como fallback (P/VP será incorreto)"
            )
            shares_outstanding = 1.0

        return price_series, shares_outstanding, dividends

    def _compute_ratios(
        self,
        dre: pd.DataFrame,
        bp: pd.DataFrame,
        price_series: pd.Series,
        shares_outstanding: float,
        dividends: pd.Series,
    ) -> pd.DataFrame:
        """Computa os ratios fundamentalistas por trimestre.

        ROE e Margem Bruta são fornecidos pelo Status Invest em % (ex: 15.3 = 15.3%).
        P/VP, EV/EBITDA e Dívida Líq./EBITDA são calculados a partir dos dados
        do Balanço + DRE + preços yfinance.

        Args:
            dre: DRE trimestral do StatusInvestClient (index=quarter end dates).
            bp: Balanço Patrimonial trimestral do StatusInvestClient.
            price_series: Série diária de preços ajustados (yfinance).
            shares_outstanding: Número de ações em circulação (yfinance).
            dividends: Série diária de dividendos por ação (yfinance).

        Returns:
            DataFrame trimestral com colunas de ratios.

        Raises:
            KeyError: Se um campo esperado não existir no DRE ou BP.
        """
        ratios: dict[str, pd.Series] = {}

        # quarter_dates é o índice canônico — DRE define a granularidade.
        # BP pode ter datas ligeiramente diferentes; reindexar para dre.index
        # antes de qualquer aritmética evita union-index silencioso.
        quarter_dates = dre.index

        # --- Extrai itens do DRE ---
        try:
            ebitda = dre[_DRE_FIELDS["ebitda"]].reindex(quarter_dates).astype(float)
            gross_margin_pct = (
                dre[_DRE_FIELDS["gross_margin"]].reindex(quarter_dates).astype(float)
            )
            roe_pct = dre[_DRE_FIELDS["roe"]].reindex(quarter_dates).astype(float)
            total_debt = dre[_DRE_FIELDS["total_debt"]].reindex(quarter_dates).astype(float)
        except KeyError as exc:
            raise KeyError(
                f"Campo do DRE não encontrado: {exc}. "
                f"Colunas disponíveis: {dre.columns.tolist()}"
            ) from exc

        # --- Extrai itens do Balanço (equity e cash apenas) ---
        try:
            equity = bp[_BP_FIELDS["equity"]].reindex(quarter_dates).astype(float)
            cash = bp[_BP_FIELDS["cash"]].reindex(quarter_dates).astype(float)
        except KeyError as exc:
            raise KeyError(
                f"Campo do Balanço Patrimonial não encontrado: {exc}. "
                f"Colunas disponíveis: {bp.columns.tolist()}"
            ) from exc

        # --- Preço no final de cada trimestre ---
        price_at_quarter_end = price_series.reindex(quarter_dates, method="nearest")

        # --- Dívida líquida e Market Cap ---
        net_debt = (total_debt - cash).fillna(0.0)
        market_cap = price_at_quarter_end * shares_outstanding

        # --- ROE (Status Invest fornece em %) ---
        ratios["roe"] = pd.Series(
            (roe_pct / 100.0).values,
            index=quarter_dates,
        )

        # --- Margem Bruta (Status Invest fornece em %) ---
        ratios["gross_margin"] = pd.Series(
            (gross_margin_pct / 100.0).values,
            index=quarter_dates,
        )

        # --- Dívida Líquida / EBITDA ---
        ratios["net_debt_ebitda"] = pd.Series(
            np.where(ebitda > 0, net_debt / ebitda, np.nan),
            index=quarter_dates,
        )

        # --- P/VPA ---
        book_value_per_share = equity / shares_outstanding
        ratios["p_vpa"] = pd.Series(
            np.where(book_value_per_share > 0, price_at_quarter_end / book_value_per_share, np.nan),
            index=quarter_dates,
        )

        # --- EV/EBITDA ---
        ev = market_cap + net_debt
        ratios["ev_ebitda"] = pd.Series(
            np.where(ebitda > 0, ev / ebitda, np.nan),
            index=quarter_dates,
        )

        # --- Dividend Yield (trailing 12 meses) ---
        dy_values: list[float] = []
        for q_date in quarter_dates:
            trailing_12m_start = q_date - pd.DateOffset(months=12)
            mask = (dividends.index >= trailing_12m_start) & (dividends.index <= q_date)
            divs_sum = dividends.loc[mask].sum() if not dividends.empty else 0.0
            price = price_at_quarter_end.get(q_date, np.nan)
            dy = divs_sum / price if (price and price > 0) else np.nan
            dy_values.append(dy)

        ratios["dividend_yield"] = pd.Series(dy_values, index=quarter_dates)

        result = pd.DataFrame(ratios)
        result.index.name = "date"
        result.index = pd.DatetimeIndex(result.index)
        return result.sort_index()

    def _apply_reporting_lag(self, quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica lag de 45 dias às datas de trimestre.

        CVM permite até 45 dias para publicação de ITRs. Este lag é a
        proteção central contra data leakage: informação do Q1 (31/mar)
        só entra no modelo a partir de 15/mai.

        Args:
            quarterly_df: DataFrame com DatetimeIndex de datas de final de trimestre.

        Returns:
            DataFrame com datas deslocadas em REPORTING_LAG_DAYS.
        """
        df = quarterly_df.copy()
        df.index = df.index + pd.Timedelta(days=self.REPORTING_LAG_DAYS)
        return df

    @staticmethod
    def _merge_to_weekly_spine(
        quarterly_df: pd.DataFrame,
        spine: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Alinha fundamentos trimestrais (com lag) ao spine semanal.

        Para cada sexta-feira do spine, usa o trimestre mais recente
        cujo lag-date é <= aquela sexta. Implementa causalidade estrita.

        Args:
            quarterly_df: DataFrame trimestral com lag aplicado.
            spine: DatetimeIndex semanal (W-FRI).

        Returns:
            DataFrame alinhado ao spine com forward-fill implícito do merge_asof.
        """
        spine_df = pd.DataFrame(index=spine).reset_index()

        # reset_index() produz "date" se o índice tiver esse nome,
        # ou "index" caso contrário. Normaliza para "date".
        quarterly_reset = quarterly_df.sort_index().reset_index()
        if "index" in quarterly_reset.columns:
            quarterly_reset = quarterly_reset.rename(columns={"index": "date"})

        merged = pd.merge_asof(
            left=spine_df,
            right=quarterly_reset,
            on="date",
            direction="backward",  # usa o trimestre cujo lag-date <= sexta-feira
        )

        merged = merged.set_index("date")
        return merged


# ---------------------------------------------------------------------------
# Execução standalone: uv run python -m src.data_ingestion.fundamentals
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    cfg = load_pipeline_config()
    di_cfg = cfg["data_ingestion"]
    end = di_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    fetcher = FundamentalsFetcher(config=cfg)
    df = fetcher.fetch(start_date=di_cfg["start_date"], end_date=end)
    fetcher.save(df)

    print(f"\nAmostras:\n{df.tail()}")
    print(f"\nInfo:\n{df.dtypes}")
    print(f"\nNaN por coluna:\n{df.isna().sum()}")
