"""
Consolidador de dados — une os três DataFrames da Fase 1 em um spine único.

Os três fetchers da Fase 1 produzem Parquets independentes, todos ancorados
no spine semanal W-FRI. Este módulo carrega cada um via fetcher.load() e
executa um left-join usando market_df como spine mestre.

Por que left-join no spine de market?
    O market_df é o DataFrame mais denso e define as semanas efetivamente
    negociadas. Fundamentals começam com NaN (lag de 45 dias) e macro cobre
    o mesmo período — um left-join preserva a integridade do spine sem
    introduzir semanas extras.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data_ingestion.fundamentals import FundamentalsFetcher
from src.data_ingestion.macro_data import MacroDataFetcher
from src.data_ingestion.market_data import MarketDataFetcher
from src.utils.logger import get_logger
from src.utils.validators import validate_columns, validate_no_future_leakage

if TYPE_CHECKING:
    pass

logger: logging.Logger = get_logger(__name__)

_MARKET_REQUIRED = ["open_adj", "high_adj", "low_adj", "close_adj", "volume"]
_FUNDAMENTALS_REQUIRED = [
    "p_vpa",
    "ev_ebitda",
    "roe",
    "net_debt_ebitda",
    "gross_margin",
    "dividend_yield",
]
_MACRO_REQUIRED = [
    "cdi_rate",
    "usd_brl",
    "selic_rate",
    "igpm",
    "ipca",
    "soy_price_usd",
    "corn_price_usd",
    "selic_real",
]


class DataConsolidator:
    """Consolida os três DataFrames da Fase 1 em um único DataFrame semanal.

    Responsabilidades:
    - Carregar market, fundamentals e macro via fetcher.load()
    - Validar schemas de entrada
    - Alinhar ao spine de market via left-join no índice W-FRI
    - Filtrar pelo intervalo de datas solicitado
    - Validar ordenação e ausência de duplicatas no resultado

    O construtor aceita instâncias de fetcher injetadas, o que permite
    mockar dependências externas nos testes sem acesso a disco.

    Exemplo de uso:
        consolidator = DataConsolidator()
        df = consolidator.consolidate("2006-01-01", "2025-12-31")
    """

    def __init__(
        self,
        market_fetcher: MarketDataFetcher | None = None,
        fundamentals_fetcher: FundamentalsFetcher | None = None,
        macro_fetcher: MacroDataFetcher | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            market_fetcher: Fetcher de dados de mercado. Se None, instancia padrão.
            fundamentals_fetcher: Fetcher de fundamentos. Se None, instancia padrão.
            macro_fetcher: Fetcher de dados macro. Se None, instancia padrão.
            config: Dicionário de configuração (passado aos fetchers padrão).
        """
        self._market = market_fetcher or MarketDataFetcher(config=config)
        self._fundamentals = fundamentals_fetcher or FundamentalsFetcher(config=config)
        self._macro = macro_fetcher or MacroDataFetcher(config=config)

    def consolidate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Carrega, valida e consolida os três DataFrames da Fase 1.

        O spine é definido pelo market_df. Fundamentals e macro são
        left-joined ao spine de market, preservando NaN onde fundamentos
        ainda não estão disponíveis (primeiras semanas após o lag de CVM).

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD' para filtro.
            end_date: Data final no formato 'YYYY-MM-DD' para filtro.

        Returns:
            DataFrame consolidado com DatetimeIndex W-FRI, timezone-naive,
            contendo todas as colunas de market, fundamentals e macro.
            Primeiras semanas de fundamentals são NaN (comportamento esperado).

        Raises:
            FileNotFoundError: Se qualquer parquet da Fase 1 não existir.
            ValueError: Se qualquer DataFrame falhar na validação de schema
                ou se o resultado não estiver ordenado cronologicamente.
        """
        logger.info(
            f"Consolidando dados: {start_date} → {end_date}"
        )

        market = self._load_and_validate_market()
        fundamentals = self._load_and_validate_fundamentals()
        macro = self._load_and_validate_macro()

        consolidated = self._join_on_market_spine(market, fundamentals, macro)

        # Filtra pelo intervalo solicitado após o join para não perder
        # linhas necessárias para features que usam janelas passadas
        mask = (consolidated.index >= start_date) & (consolidated.index <= end_date)
        consolidated = consolidated.loc[mask]

        if consolidated.empty:
            raise ValueError(
                f"Nenhum dado encontrado para o intervalo {start_date} → {end_date}. "
                "Verifique se os parquets da Fase 1 cobrem este período."
            )

        validate_no_future_leakage(
            consolidated, "index", context="DataConsolidator.consolidate"
        )

        logger.info(
            f"Consolidação concluída: {consolidated.shape[0]} semanas, "
            f"{consolidated.shape[1]} colunas."
        )
        return consolidated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_validate_market(self) -> pd.DataFrame:
        """Carrega e valida o DataFrame de mercado.

        Returns:
            DataFrame com colunas _MARKET_REQUIRED e DatetimeIndex W-FRI.

        Raises:
            FileNotFoundError: Se o parquet não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        df = self._market.load()
        validate_columns(df, _MARKET_REQUIRED, context="DataConsolidator.market")
        logger.debug(f"Market carregado: {df.shape}")
        return df

    def _load_and_validate_fundamentals(self) -> pd.DataFrame:
        """Carrega e valida o DataFrame de fundamentos.

        Returns:
            DataFrame com colunas _FUNDAMENTALS_REQUIRED e DatetimeIndex W-FRI.
            Primeiras linhas podem ser NaN (lag de 45 dias — comportamento esperado).

        Raises:
            FileNotFoundError: Se o parquet não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        df = self._fundamentals.load()
        validate_columns(df, _FUNDAMENTALS_REQUIRED, context="DataConsolidator.fundamentals")
        logger.debug(f"Fundamentals carregados: {df.shape}")
        return df

    def _load_and_validate_macro(self) -> pd.DataFrame:
        """Carrega e valida o DataFrame macro.

        Returns:
            DataFrame com colunas _MACRO_REQUIRED e DatetimeIndex W-FRI.

        Raises:
            FileNotFoundError: Se o parquet não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        df = self._macro.load()
        validate_columns(df, _MACRO_REQUIRED, context="DataConsolidator.macro")
        logger.debug(f"Macro carregado: {df.shape}")
        return df

    @staticmethod
    def _join_on_market_spine(
        market: pd.DataFrame,
        fundamentals: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """Executa o left-join usando o índice de market como spine mestre.

        Um left-join preserva exatamente as linhas de market_df. Colunas de
        fundamentals e macro são alinhadas pelo índice DatetimeIndex. NaN em
        fundamentals (primeiras semanas) é preservado — não é preenchido aqui.

        Args:
            market: DataFrame de mercado (define o spine W-FRI).
            fundamentals: DataFrame de fundamentos (pode ter NaN iniciais).
            macro: DataFrame macro (mesmo spine que market).

        Returns:
            DataFrame consolidado. Shape: (len(market), 19).
        """
        result = market.join(fundamentals, how="left").join(macro, how="left")
        logger.debug(
            f"Join concluído: {result.shape[0]} linhas × {result.shape[1]} colunas"
        )
        return result
