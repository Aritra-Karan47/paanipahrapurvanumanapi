"""Pydantic schemas for request/response contracts between SDK and HF service."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SlotType(str, Enum):
    PERIOD = "period"
    RECESS = "recess"
    GAMES = "games"


class GamesIntensity(str, Enum):
    NONE = "None"
    LIGHT = "Light"
    MODERATE = "Moderate"
    HIGH = "High"


class WeatherCondition(str, Enum):
    CLEAR = "Clear"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"
    HUMID = "Humid"
    UNKNOWN = "Unknown"


class CoolerInfo(BaseModel):
    cooler_id: str = Field(..., min_length=1, max_length=32)
    location: str = "unknown"
    capacity_liters: float = Field(..., gt=0)
    floor: int = 0
    near_games: bool = False
    current_level_liters: Optional[float] = Field(None, ge=0)


class IntervalContext(BaseModel):
    """One timetable interval (period / recess / games)."""

    period_slot: str = Field(..., description="e.g. P1, P2, Recess, Games")
    day_of_week: str
    slot_type: SlotType = SlotType.PERIOD
    games_intensity: GamesIntensity = GamesIntensity.NONE
    duration_minutes: int = Field(40, ge=5, le=120)
    subject: Optional[str] = None
    outdoor_temp_c: Optional[float] = Field(None, ge=-20, le=55, description="Slot-specific temperature override in °C")
    apparent_temp_c: Optional[float] = Field(None, ge=-20, le=70, description="Slot-specific apparent temperature override in °C")

    @field_validator("day_of_week")
    @classmethod
    def normalize_day(cls, v: str) -> str:
        v = v.strip().capitalize()
        valid = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        if v not in valid:
            raise ValueError(f"day_of_week must be one of {valid}")
        return v

    @model_validator(mode="after")
    def infer_slot_type(self) -> "IntervalContext":
        slot = self.period_slot.strip()
        if slot.lower() == "recess":
            self.slot_type = SlotType.RECESS
            self.period_slot = "Recess"
        elif slot.lower() == "games":
            self.slot_type = SlotType.GAMES
            self.period_slot = "Games"
        else:
            self.slot_type = SlotType.PERIOD
        return self


class WeatherContext(BaseModel):
    outdoor_temp_c: float = Field(..., ge=-20, le=55)
    weather: WeatherCondition = WeatherCondition.UNKNOWN
    humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    apparent_temp_c: Optional[float] = Field(None, ge=-20, le=70, description="Apparent temperature / Heat Index in °C")


class HistoricalPoint(BaseModel):
    date: str
    period_slot: str
    consumption_liters: float = Field(..., ge=0)
    outdoor_temp_c: Optional[float] = None
    weather: Optional[str] = None


class PredictionRequest(BaseModel):
    """Single-interval or multi-interval prediction for one cooler."""

    cooler: CoolerInfo
    intervals: List[IntervalContext] = Field(..., min_length=1)
    weather: WeatherContext
    historical: Optional[List[HistoricalPoint]] = None
    model_preference: str = Field("auto", description="auto | baseline | catboost")
    request_id: Optional[str] = None


class IntervalPrediction(BaseModel):
    period_slot: str
    expected_liters: float = Field(..., ge=0)
    p50_liters: Optional[float] = None
    p90_liters: Optional[float] = None
    conservative_liters: float = Field(..., ge=0, description="P90 or expected*1.25 fallback")
    model_used: str
    confidence: str = "low"  # low | medium | high


class PredictionResponse(BaseModel):
    request_id: Optional[str] = None
    cooler_id: str
    predictions: List[IntervalPrediction]
    total_expected_liters: float
    total_conservative_liters: float
    model_version: str
    fallback_used: bool = False
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForecastDayRequest(BaseModel):
    cooler: CoolerInfo
    day_of_week: str
    intervals: List[IntervalContext]
    weather: WeatherContext
    historical: Optional[List[HistoricalPoint]] = None
    current_level_liters: Optional[float] = None
    next_refill_after_slots: Optional[int] = Field(
        None, description="Number of intervals until next manual refill opportunity"
    )
    model_preference: str = "auto"
    request_id: Optional[str] = None


class ShortageRisk(BaseModel):
    risk_level: str  # none | low | medium | high | critical
    projected_level_after_day: float
    shortage_liters: float = 0.0
    message: str


class ForecastDayResponse(BaseModel):
    request_id: Optional[str] = None
    cooler_id: str
    day_of_week: str
    predictions: List[IntervalPrediction]
    total_expected_liters: float
    total_conservative_liters: float
    shortage: Optional[ShortageRisk] = None
    model_version: str
    fallback_used: bool = False
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TemperatureTargetRequest(BaseModel):
    outdoor_temp_c: float = Field(..., ge=-30, le=60)
    request_id: Optional[str] = None


class TemperatureTargetResponse(BaseModel):
    outdoor_temp_c: float
    target_water_temp_c: float
    rule_applied: str
    model_version: str = "temp-control-v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ShortageCheckRequest(BaseModel):
    cooler_id: str
    capacity_liters: float = Field(..., gt=0)
    current_level_liters: float = Field(..., ge=0)
    expected_demand_liters: float = Field(..., ge=0)
    conservative_demand_liters: float = Field(..., ge=0)
    next_refill_hours: Optional[float] = None
    request_id: Optional[str] = None


class ShortageCheckResponse(BaseModel):
    cooler_id: str
    risk_level: str
    projected_level: float
    shortage_liters: float
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidateDataRequest(BaseModel):
    cooler: Optional[CoolerInfo] = None
    intervals: Optional[List[IntervalContext]] = None
    weather: Optional[WeatherContext] = None
    historical: Optional[List[HistoricalPoint]] = None
    consumption_records: Optional[List[Dict[str, Any]]] = None


class ValidateDataResponse(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_version: str
    baseline_available: bool
    catboost_available: bool
    catboost_trained: bool
    supported_slots: List[str]
    temperature_rules_loaded: bool
    notes: List[str] = Field(default_factory=list)


class LiveWeatherRequest(BaseModel):
    city: Optional[str] = "New Delhi"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LiveWeatherResponse(BaseModel):
    success: bool
    outdoor_temp_c: float
    apparent_temp_c: Optional[float] = None
    heat_index_c: Optional[float] = None
    weather: str
    humidity_pct: float
    weather_code: int = 0
    location_name: str
    latitude: float
    longitude: float
    source: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
