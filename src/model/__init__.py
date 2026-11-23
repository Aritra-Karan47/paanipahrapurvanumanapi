"""Core prediction engine (baseline + optional CatBoost)."""

from .predictor import WaterDemandPredictor
from .baseline import HistoricalAverageBaseline
from .cache import PredictionCache, compute_temp_band
from .features import build_features, FeatureConfig

__all__ = [
    "WaterDemandPredictor",
    "HistoricalAverageBaseline",
    "PredictionCache",
    "compute_temp_band",
    "build_features",
    "FeatureConfig",
]