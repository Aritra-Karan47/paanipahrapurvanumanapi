"""Schema validation and client construction tests (no live HF)."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cooler_demand.schemas import (
    CoolerInfo,
    IntervalContext,
    PredictionRequest,
    WeatherContext,
    ForecastDayRequest,
    TemperatureTargetRequest,
    ShortageCheckRequest,
    ValidateDataRequest,
)
from cooler_demand.client import (
    CoolerDemandClient,
    WaterDemandClient,
    AsyncWaterDemandClient,
    AsyncCoolerDemandClient,
)


def test_interval_infers_recess():
    iv = IntervalContext(period_slot="recess", day_of_week="monday")
    assert iv.slot_type.value == "recess"
    assert iv.period_slot == "Recess"
    assert iv.day_of_week == "Monday"


def test_interval_games():
    iv = IntervalContext(period_slot="Games", day_of_week="Tuesday", games_intensity="High")
    assert iv.slot_type.value == "games"


def test_prediction_request_ok():
    req = PredictionRequest(
        cooler=CoolerInfo(cooler_id="C01", capacity_liters=200),
        intervals=[IntervalContext(period_slot="P1", day_of_week="Monday")],
        weather=WeatherContext(outdoor_temp_c=28.5, weather="Clear"),
    )
    assert req.cooler.cooler_id == "C01"


def test_invalid_capacity():
    with pytest.raises(ValidationError):
        CoolerInfo(cooler_id="C01", capacity_liters=-10)


def test_client_construct():
    c = CoolerDemandClient(base_url="http://localhost:7860")
    assert c.base_url == "http://localhost:7860"
    c.close()

    # Verify alias
    wc = WaterDemandClient(base_url="http://localhost:7860")
    assert wc.base_url == "http://localhost:7860"
    wc.close()


def test_async_client_construct():
    import asyncio

    ac = AsyncWaterDemandClient(base_url="http://localhost:7860")
    assert ac.base_url == "http://localhost:7860"
    asyncio.run(ac.close())

    # Verify alias
    acc = AsyncCoolerDemandClient(base_url="http://localhost:7860")
    assert acc.base_url == "http://localhost:7860"
    asyncio.run(acc.close())



def test_shortage_schema():
    s = ShortageCheckRequest(
        cooler_id="C01",
        capacity_liters=200,
        current_level_liters=50,
        expected_demand_liters=80,
        conservative_demand_liters=100,
    )
    assert s.conservative_demand_liters == 100


def test_temp_request():
    t = TemperatureTargetRequest(outdoor_temp_c=31.0)
    assert t.outdoor_temp_c == 31.0