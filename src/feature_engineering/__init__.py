"""
Módulo de feature engineering — Fase 2.

Expõe as classes públicas do pipeline de transformação de features.
"""

from src.feature_engineering.consolidator import DataConsolidator
from src.feature_engineering.fundamental_features import FundamentalFeatureBuilder
from src.feature_engineering.pipeline import FeatureBuilder, FeatureEngineeringPipeline
from src.feature_engineering.target_builder import TargetBuilder
from src.feature_engineering.technical_features import TechnicalFeatureBuilder

__all__ = [
    "DataConsolidator",
    "FeatureBuilder",
    "FeatureEngineeringPipeline",
    "FundamentalFeatureBuilder",
    "TargetBuilder",
    "TechnicalFeatureBuilder",
]
