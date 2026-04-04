"""
Módulo de ingestão de dados — Protocolo de interface compartilhado.

Todos os fetchers numéricos implementam DataFetcher.
PdfDownloader tem interface própria (semântica diferente).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["DataFetcher"]


@runtime_checkable
class DataFetcher(Protocol):
    """Interface comum a todos os fetchers de dados numéricos.

    Cada fetcher concreto (MarketDataFetcher, MacroDataFetcher,
    FundamentalsFetcher) deve implementar estes três métodos.

    O uso de @runtime_checkable permite isinstance(fetcher, DataFetcher)
    nos testes de integração sem necessidade de herança explícita.
    """

    def fetch(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """Busca dados da fonte externa e retorna DataFrame alinhado ao spine semanal.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'.
            end_date: Data final no formato 'YYYY-MM-DD'.

        Returns:
            DataFrame com DatetimeIndex semanal (freq='W-FRI'), timezone-naive,
            sem colunas com nome duplicado.

        Raises:
            ValueError: Se a fonte retornar dados vazios após todas as tentativas.
            FileNotFoundError: Se um arquivo local necessário não existir.
        """
        ...

    def save(self, df: "pd.DataFrame") -> None:
        """Persiste o DataFrame em formato Parquet no caminho canônico do módulo.

        Args:
            df: DataFrame retornado por fetch().
        """
        ...

    def load(self) -> "pd.DataFrame":
        """Carrega o DataFrame previamente salvo por save().

        Returns:
            DataFrame idêntico ao salvo por save().

        Raises:
            FileNotFoundError: Se o arquivo Parquet ainda não existir.
        """
        ...
