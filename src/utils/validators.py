"""Validadores de schemas para DataFrames ao longo do pipeline."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_columns(df: pd.DataFrame, required_cols: list[str], context: str = "") -> None:
    """
    Verifica se todas as colunas obrigatórias estão presentes no DataFrame.

    Args:
        df: DataFrame a ser validado.
        required_cols: Lista de colunas obrigatórias.
        context: Descrição do ponto do pipeline para mensagens de erro claras.

    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] Colunas obrigatórias ausentes no DataFrame: {sorted(missing)}"
        )


def validate_no_future_leakage(
    df: pd.DataFrame,
    timestamp_col: str,
    context: str = "",
) -> None:
    """
    Verifica se o índice temporal está ordenado e sem duplicatas — pré-condição
    para garantir ausência de data leakage nas janelas deslizantes.

    Args:
        df: DataFrame com coluna ou índice temporal.
        timestamp_col: Nome da coluna de timestamp ou 'index' para usar o índice.
        context: Descrição do ponto do pipeline.

    Raises:
        ValueError: Se o DataFrame não estiver ordenado ou contiver timestamps duplicados.
    """
    series = df.index if timestamp_col == "index" else df[timestamp_col]

    if not series.is_monotonic_increasing:
        raise ValueError(
            f"[{context}] DataFrame não está ordenado cronologicamente. "
            "Ordene por timestamp antes de construir janelas deslizantes."
        )

    if series.duplicated().any():
        n_dupes = series.duplicated().sum()
        raise ValueError(
            f"[{context}] {n_dupes} timestamp(s) duplicado(s) encontrado(s). "
            "Resolva duplicatas antes de prosseguir."
        )

    logger.debug(f"[{context}] Validação temporal OK — {len(df)} registros, sem leakage detectado.")


def validate_no_nulls(df: pd.DataFrame, cols: list[str], context: str = "") -> None:
    """
    Verifica ausência de valores nulos nas colunas especificadas.

    Args:
        df: DataFrame a ser validado.
        cols: Colunas onde nulos são inaceitáveis.
        context: Descrição do ponto do pipeline.

    Raises:
        ValueError: Se qualquer coluna contiver valores nulos.
    """
    null_counts = df[cols].isnull().sum()
    problematic = null_counts[null_counts > 0]

    if not problematic.empty:
        raise ValueError(
            f"[{context}] Valores nulos encontrados:\n{problematic.to_string()}"
        )
