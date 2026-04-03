"""Testes unitários para os utilitários de configuração e validação."""

import pandas as pd
import pytest
from src.utils.config import load_model_config, load_pipeline_config
from src.utils.validators import (
    validate_columns,
    validate_no_future_leakage,
    validate_no_nulls,
)


class TestConfigLoader:
    def test_load_model_config_returns_dict(self) -> None:
        config = load_model_config()
        assert isinstance(config, dict)

    def test_model_config_has_required_sections(self) -> None:
        config = load_model_config()
        assert "model" in config
        assert "training" in config
        assert "target" in config

    def test_load_pipeline_config_returns_dict(self) -> None:
        config = load_pipeline_config()
        assert isinstance(config, dict)

    def test_pipeline_config_has_required_sections(self) -> None:
        config = load_pipeline_config()
        assert "data_ingestion" in config
        assert "feature_engineering" in config

    def test_load_config_raises_on_missing_file(self) -> None:
        from src.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("arquivo_que_nao_existe.yaml")


class TestValidators:
    def test_validate_columns_passes_when_all_present(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_columns(df, ["a", "b"], context="test")  # não deve levantar

    def test_validate_columns_raises_on_missing(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="ausentes"):
            validate_columns(df, ["a", "b", "c"], context="test")

    def test_validate_no_future_leakage_passes_on_sorted(self) -> None:
        df = pd.DataFrame(
            {"val": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        )
        validate_no_future_leakage(df, "index", context="test")  # não deve levantar

    def test_validate_no_future_leakage_raises_on_unsorted(self) -> None:
        df = pd.DataFrame(
            {"val": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-15", "2024-01-01", "2024-01-08"]),
        )
        with pytest.raises(ValueError, match="ordenado"):
            validate_no_future_leakage(df, "index", context="test")

    def test_validate_no_future_leakage_raises_on_duplicates(self) -> None:
        df = pd.DataFrame(
            {"val": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08"]),
        )
        with pytest.raises(ValueError, match="duplicado"):
            validate_no_future_leakage(df, "index", context="test")

    def test_validate_no_nulls_passes_on_clean_data(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_no_nulls(df, ["a", "b"], context="test")  # não deve levantar

    def test_validate_no_nulls_raises_when_nulls_present(self) -> None:
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="nulos"):
            validate_no_nulls(df, ["a", "b"], context="test")
