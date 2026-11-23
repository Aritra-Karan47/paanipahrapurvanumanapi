"""cooler-demand: SDK for School Water Cooler Demand Prediction service."""

from .client import CoolerDemandClient
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ForecastDayRequest,
    ForecastDayResponse,
    TemperatureTargetRequest,
    TemperatureTargetResponse,
    ShortageCheckRequest,
    ShortageCheckResponse,
    ValidateDataRequest,
    ValidateDataResponse,
    IntervalPrediction,
    ModelInfo,
    LiveWeatherRequest,
    LiveWeatherResponse,
)

__version__ = "0.1.0"
__all__ = [
    "CoolerDemandClient",
    "PredictionRequest",
    "PredictionResponse",
    "ForecastDayRequest",
    "ForecastDayResponse",
    "TemperatureTargetRequest",
    "TemperatureTargetResponse",
    "ShortageCheckRequest",
    "ShortageCheckResponse",
    "ValidateDataRequest",
    "ValidateDataResponse",
    "IntervalPrediction",
    "ModelInfo",
    "LiveWeatherRequest",
    "LiveWeatherResponse",
    "__version__",
]