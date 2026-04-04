"""
Ingestão de dados fundamentalistas — AGRO3.SA via yfinance.

Fonte: demonstrativos financeiros trimestrais (quarterly_balance_sheet,
quarterly_income_stmt, quarterly_cashflow) do yfinance.

Ratios computados:
    p_vpa           — Preço / Valor Patrimonial por Ação
    ev_ebitda       — Enterprise Value / EBITDA
    roe             — Return on Equity
    net_debt_ebitda — Dívida Líquida / EBITDA
    gross_margin    — Margem Bruta
    dividend_yield  — DY trailing 12 meses

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

# Nomes alternativos por item do demonstrativo (yfinance muda entre versões)
_BALANCE_CANDIDATES: dict[str, list[str]] = {
    "stockholders_equity": [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
    ],
    "total_debt": [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
    ],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ],
}

_INCOME_CANDIDATES: dict[str, list[str]] = {
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebit": ["EBIT", "Operating Income"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}

_CASHFLOW_CANDIDATES: dict[str, list[str]] = {
    "depreciation": [
        "Depreciation And Amortization",
        "Reconciled Depreciation",
        "Depreciation Depletion And Amortization",
    ],
}


class FundamentalsFetcher:
    """Fetcher de dados fundamentalistas para AGRO3.SA.

    Responsável por:
    - Baixar demonstrativos trimestrais via yfinance
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
    REPORTING_LAG_DAYS: int = 45

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_pipeline_config()
        raw_output: str = cfg["data_ingestion"]["output"]["fundamentals"]
        self._output_path: Path = _PROJECT_ROOT / raw_output / "fundamentals_quarterly.parquet"

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
            As primeiras semanas pós-IPO (antes do primeiro lag-date) terão NaN.

        Raises:
            ValueError: Se o yfinance não retornar demonstrativos trimestrais.
        """
        logger.info(f"Buscando fundamentos: {self.TICKER} de {start_date} a {end_date}")

        balance, income, cashflow = self._fetch_raw_statements()
        price_series, shares_outstanding, dividends = self._fetch_price_and_dividends(
            start_date, end_date
        )

        quarterly_df = self._compute_ratios(
            balance, income, cashflow, price_series, shares_outstanding, dividends
        )

        lagged_df = self._apply_reporting_lag(quarterly_df)
        spine = pd.date_range(start=start_date, end=end_date, freq="W-FRI", name="date")
        weekly_df = self._merge_to_weekly_spine(lagged_df, spine)

        validate_columns(weekly_df, _REQUIRED_COLS, context="FundamentalsFetcher.fetch")

        nan_count = weekly_df[_REQUIRED_COLS].isna().any(axis=1).sum()
        if nan_count > 0:
            logger.debug(
                f"{nan_count} semanas com NaN em fundamentos "
                "(esperado para período pré-lag do primeiro trimestre)"
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_raw_statements(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Baixa os três demonstrativos trimestrais do yfinance.

        yfinance retorna DataFrames onde:
        - Colunas = datas de final de trimestre (pd.Timestamp)
        - Índice = nome do item financeiro
        Transpomos (.T) para obter row=trimestre, col=item.

        Returns:
            Tupla (balance_sheet, income_stmt, cashflow) transpostos.

        Raises:
            ValueError: Se os demonstrativos estiverem vazios.
        """
        ticker = yf.Ticker(self.TICKER)

        balance = ticker.quarterly_balance_sheet
        income = ticker.quarterly_income_stmt
        cashflow = ticker.quarterly_cashflow

        if balance.empty or income.empty:
            raise ValueError(f"yfinance não retornou demonstrativos trimestrais para {self.TICKER}")

        # Transpõe: row=trimestre, col=item financeiro
        balance_t = balance.T
        income_t = income.T
        cashflow_t = cashflow.T if not cashflow.empty else pd.DataFrame()

        # Remove timezone dos índices (datas de trimestre)
        for df in (balance_t, income_t, cashflow_t):
            if not df.empty and df.index.tz is not None:
                df.index = df.index.tz_convert(None)

        logger.debug(
            f"Demonstrativos baixados: {len(balance_t)} trimestres "
            f"({balance_t.index.min()} → {balance_t.index.max()})"
        )
        return balance_t, income_t, cashflow_t

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

        Preço é necessário para computar P/VP e EV/EBITDA.
        O FundamentalsFetcher busca preços diretamente para evitar
        dependência circular com MarketDataFetcher.

        Args:
            start_date: Data inicial.
            end_date: Data final.

        Returns:
            Tupla (price_series, shares_outstanding, dividends_series).
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
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cashflow: pd.DataFrame,
        price_series: pd.Series,
        shares_outstanding: float,
        dividends: pd.Series,
    ) -> pd.DataFrame:
        """Computa os ratios fundamentalistas por trimestre.

        Args:
            balance: Balance sheet transposto (row=trimestre).
            income: Income statement transposto.
            cashflow: Cash flow statement transposto.
            price_series: Série diária de preços ajustados.
            shares_outstanding: Número de ações em circulação.
            dividends: Série diária de dividendos por ação.

        Returns:
            DataFrame trimestral com colunas de ratios.
        """
        ratios: dict[str, pd.Series] = {}

        # --- Extrai itens do balanço ---
        equity = self._safe_get_column(balance, _BALANCE_CANDIDATES["stockholders_equity"])
        total_debt = self._safe_get_column(balance, _BALANCE_CANDIDATES["total_debt"])
        cash = self._safe_get_column(balance, _BALANCE_CANDIDATES["cash"])

        # --- Extrai itens do DRE ---
        revenue = self._safe_get_column(income, _INCOME_CANDIDATES["total_revenue"])
        gross_profit = self._safe_get_column(income, _INCOME_CANDIDATES["gross_profit"])
        ebit = self._safe_get_column(income, _INCOME_CANDIDATES["ebit"])
        net_income = self._safe_get_column(income, _INCOME_CANDIDATES["net_income"])

        # --- EBITDA = EBIT + D&A (yfinance não fornece EBITDA diretamente) ---
        if not cashflow.empty:
            depr = self._safe_get_column(cashflow, _CASHFLOW_CANDIDATES["depreciation"])
            ebitda = ebit + depr
        else:
            logger.warning("Cash flow vazio — usando EBIT como proxy de EBITDA")
            ebitda = ebit

        # --- Preço no final de cada trimestre ---
        # Usamos o preço do último dia do trimestre (ou o mais próximo anterior)
        quarter_dates = balance.index
        price_at_quarter_end = price_series.reindex(quarter_dates, method="nearest")

        # --- Dívida líquida ---
        net_debt = (total_debt - cash).fillna(0.0)

        # --- Market Cap no final do trimestre ---
        market_cap = price_at_quarter_end * shares_outstanding

        # --- P/VPA ---
        book_value_per_share = equity / shares_outstanding
        # np.where para proteger contra book_value <= 0
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

        # --- ROE ---
        ratios["roe"] = pd.Series(
            np.where(equity != 0, net_income / equity, np.nan),
            index=quarter_dates,
        )

        # --- Dívida Líquida / EBITDA ---
        ratios["net_debt_ebitda"] = pd.Series(
            np.where(ebitda > 0, net_debt / ebitda, np.nan),
            index=quarter_dates,
        )

        # --- Margem Bruta ---
        ratios["gross_margin"] = pd.Series(
            np.where(revenue > 0, gross_profit / revenue, np.nan),
            index=quarter_dates,
        )

        # --- Dividend Yield (trailing 12 meses) ---
        # Soma dividendos dos últimos 12 meses antes do final do trimestre
        dy_values: list[float] = []
        for q_date in quarter_dates:
            trailing_12m_start = q_date - pd.DateOffset(months=12)
            mask = (dividends.index >= trailing_12m_start) & (dividends.index <= q_date)
            divs_sum = dividends.loc[mask].sum()
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
        spine_df = pd.DataFrame(index=spine).reset_index()  # coluna "date"

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

    @staticmethod
    def _safe_get_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        """Retorna a primeira coluna encontrada dentre os candidatos.

        Robustez a mudanças nos nomes de colunas do yfinance entre versões.

        Args:
            df: DataFrame de demonstrativo financeiro.
            candidates: Nomes alternativos em ordem de preferência.

        Returns:
            Série correspondente.

        Raises:
            KeyError: Se nenhum candidato for encontrado.
        """
        for col in candidates:
            if col in df.columns:
                return df[col].astype(float)
        raise KeyError(
            f"Nenhum dos candidatos encontrado: {candidates}. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# Execução standalone: uv run python -m src.data_ingestion.fundamentals
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import pandas as pd

    cfg = load_pipeline_config()
    di_cfg = cfg["data_ingestion"]
    end = di_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    fetcher = FundamentalsFetcher(config=cfg)
    df = fetcher.fetch(start_date=di_cfg["start_date"], end_date=end)
    fetcher.save(df)

    print(f"\nAmostras:\n{df.tail()}")
    print(f"\nInfo:\n{df.dtypes}")
    print(f"\nNaN por coluna:\n{df.isna().sum()}")
