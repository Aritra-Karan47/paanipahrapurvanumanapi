"""Core prediction engine (baseline + optional CatBoost)."""

from .predictor import WaterDemandPredictor
from .baseline import HistoricalAverageBaseline
from .features import build_features, FeatureConfig

__all__ = [
    "WaterDemandPredictor",
    "HistoricalAverageBaseline",
    "build_features",
    "FeatureConfig",
]