"""Configuração centralizada de logging estruturado para o projeto."""

import logging
import sys


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Retorna um logger configurado com o formato padrão do projeto.

    Args:
        name: Nome do logger — use __name__ no módulo chamador.
        level: Nível de log. Se None, usa INFO como padrão.

    Returns:
        Logger configurado e pronto para uso.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False

    return logger
