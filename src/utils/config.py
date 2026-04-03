"""Carregamento e validação de arquivos de configuração YAML."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_filename: str) -> dict[str, Any]:
    """
    Carrega um arquivo YAML do diretório config/ na raiz do projeto.

    Args:
        config_filename: Nome do arquivo (ex: 'model_config.yaml').

    Returns:
        Dicionário com os valores de configuração.

    Raises:
        FileNotFoundError: Se o arquivo não existir no diretório config/.
        yaml.YAMLError: Se o arquivo não for um YAML válido.
    """
    config_path = _PROJECT_ROOT / "config" / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    logger.debug(f"Carregando configuração: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_model_config() -> dict[str, Any]:
    """Atalho para carregar model_config.yaml."""
    return load_config("model_config.yaml")


def load_pipeline_config() -> dict[str, Any]:
    """Atalho para carregar pipeline_config.yaml."""
    return load_config("pipeline_config.yaml")
