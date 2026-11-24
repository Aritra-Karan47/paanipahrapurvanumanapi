"""cooler-demand: SDK for School Water Cooler Demand Prediction service."""

from .client import (
    CoolerDemandClient,
    WaterDemandClient,
    AsyncWaterDemandClient,
    AsyncCoolerDemandClient,
    CoolerDemandError,
)
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

__version__ = "0.1.1"
__all__ = [
    "CoolerDemandClient",
    "WaterDemandClient",
    "AsyncWaterDemandClient",
    "AsyncCoolerDemandClient",
    "CoolerDemandError",
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
