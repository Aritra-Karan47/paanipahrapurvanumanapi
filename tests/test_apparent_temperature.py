"""Tests for Apparent Temperature (Heat Index) calculation, feature engineering, and model inference."""

import math
from pathlib import Path
import pytest
import pandas as pd

from weather.service import (
    calculate_apparent_temperature,
    calculate_heat_index,
    diurnal_humidity_for_slot,
    get_slot_hourly_apparent_temperature,
    get_slot_hourly_humidity,
    map_period_hourly_apparent_temperatures,
    fetch_live_weather,
)
from model.features import (
    build_features,
    build_single_feature_row,
    feature_columns,
)
from model.baseline import HistoricalAverageBaseline
from model.predictor import WaterDemandPredictor
from app.app import service_predict, service_forecast_day
from cooler_demand.schemas import WeatherContext, LiveWeatherResponse


def test_calculate_apparent_temperature_steadman_formula():
    """Verify Steadman apparent temperature formula for humid Indian school summers.

    Specifically: 34°C with 75% humidity feels like ~42°C (as specified in Issue #24).
    """
    at_summer = calculate_apparent_temperature(temp_c=34.0, humidity_pct=75.0, wind_speed_ms=1.5)
    # 34°C + 75% RH should feel like 42°C (within +/- 0.5°C)
    assert 41.5 <= at_summer <= 42.5
    assert at_summer == 42.1

    # Comfortable room temperature: 25°C with 50% RH feels like ~25°C
    at_room = calculate_apparent_temperature(temp_c=25.0, humidity_pct=50.0, wind_speed_ms=1.5)
    assert 24.5 <= at_room <= 25.5
    assert at_room == 25.2

    # Dry heat vs humid heat at 36°C:
    at_dry = calculate_apparent_temperature(temp_c=36.0, humidity_pct=30.0, wind_speed_ms=1.5)
    at_humid = calculate_apparent_temperature(temp_c=36.0, humidity_pct=80.0, wind_speed_ms=1.5)
    assert at_humid > at_dry
    assert (at_humid - at_dry) > 7.0  # At least 7°C higher perceived heat


def test_calculate_apparent_temperature_wind_and_boundaries():
    """Verify wind cooling effect and boundary clamping for relative humidity."""
    # Higher wind provides evaporative cooling, lowering apparent temperature
    at_calm = calculate_apparent_temperature(32.0, 60.0, wind_speed_ms=0.0)
    at_breezy = calculate_apparent_temperature(32.0, 60.0, wind_speed_ms=3.0)
    assert at_calm > at_breezy
    assert round(at_calm - at_breezy, 1) == 2.1  # 0.7 * 3.0 m/s = 2.1°C difference

    # Boundary handling: clamping humidity between 0 and 100
    at_zero = calculate_apparent_temperature(30.0, -10.0)
    at_hundred = calculate_apparent_temperature(30.0, 110.0)
    assert at_zero < at_hundred


def test_calculate_heat_index_noaa_formula():
    """Verify NOAA / Rothfusz heat index calculation."""
    # Under 80°F (< 26.7°C): simple formula
    hi_mild = calculate_heat_index(22.0, 50.0)
    assert 21.0 <= hi_mild <= 23.0

    # Hot & humid conditions (>= 80°F, high RH):
    hi_hot = calculate_heat_index(34.0, 75.0)
    assert hi_hot > 42.0

    # Low humidity adjustment trigger
    hi_dry = calculate_heat_index(38.0, 10.0)
    assert hi_dry < 38.0


def test_diurnal_humidity_and_slot_hourly_apparent_temp():
    """Verify diurnal humidity and apparent temperature mapping across school periods."""
    base_humidity = 55.0
    rh_p1 = diurnal_humidity_for_slot(base_humidity, "P1")
    rh_p6 = diurnal_humidity_for_slot(base_humidity, "P6")

    # Morning humidity (P1, 8 AM) is higher than peak afternoon heat (P6, 1:30 PM)
    assert rh_p1 > rh_p6
    assert rh_p1 == 67.0  # 55 + 12
    assert rh_p6 == 49.0  # 55 - 6

    # Test get_slot_hourly_apparent_temperature with mock hourly forecast
    mock_hourly = {
        "time": ["2026-11-24T08:00", "2026-11-24T13:00"],
        "apparent_temperature": [21.0, 41.5],
    }
    at_p1 = get_slot_hourly_apparent_temperature("P1", base_temp_c=28.0, hourly_forecast=mock_hourly)
    at_p6 = get_slot_hourly_apparent_temperature("P6", base_temp_c=28.0, hourly_forecast=mock_hourly)
    assert at_p1 == 21.0
    assert at_p6 == 41.5

    # Fallback to diurnal calculation when hourly is None
    at_p1_fallback = get_slot_hourly_apparent_temperature("P1", base_temp_c=30.0, base_humidity_pct=50.0)
    at_p6_fallback = get_slot_hourly_apparent_temperature("P6", base_temp_c=30.0, base_humidity_pct=50.0)
    assert at_p1_fallback < at_p6_fallback


def test_feature_columns_and_row_generation():
    """Verify that apparent_temp_c is included in feature columns and row generation."""
    cols = feature_columns(include_target=False)
    assert "apparent_temp_c" in cols
    assert cols.index("apparent_temp_c") > 0

    # Explicit apparent temperature passed
    row1 = build_single_feature_row(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="P6",
        outdoor_temp_c=34.0,
        apparent_temp_c=42.1,
    )
    assert row1["apparent_temp_c"] == 42.1
    assert row1["outdoor_temp_c"] == 34.0

    # Humidity passed, apparent temperature computed
    row2 = build_single_feature_row(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="P6",
        outdoor_temp_c=34.0,
        humidity_pct=75.0,
    )
    assert row2["apparent_temp_c"] == 42.1

    # Neither passed: graceful fallback to outdoor_temp_c
    row3 = build_single_feature_row(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="P6",
        outdoor_temp_c=34.0,
    )
    assert row3["apparent_temp_c"] == 34.0


def test_baseline_scales_with_apparent_temperature():
    """Verify HistoricalAverageBaseline increases consumption when apparent temperature rises."""
    baseline = HistoricalAverageBaseline()
    df_sample = pd.DataFrame([
        {"cooler_id": "C01", "day_of_week": "Monday", "period_slot": "P6", "consumption_liters": 10.0},
    ])
    baseline.fit(df_sample)

    # Moderate temperature baseline prediction
    exp_mild, _, _ = baseline.predict_one("C01", "Monday", "P6", outdoor_temp_c=25.0)

    # Humid heatwave: outdoor 34°C, apparent temp 42.1°C
    exp_heatwave, _, _ = baseline.predict_one(
        "C01", "Monday", "P6", outdoor_temp_c=34.0, apparent_temp_c=42.1
    )

    # Heatwave thirst demand must be higher
    assert exp_heatwave > exp_mild
    assert exp_heatwave > 15.0  # Significant increase during humid heatwave


def test_predictor_responds_to_humidity():
    """Verify end-to-end predictor produces higher water demand under high humidity."""
    predictor = WaterDemandPredictor()

    intervals = [{"period_slot": "P6", "day_of_week": "Monday"}]

    # Predict at 34°C with low humidity (30%)
    preds_dry = predictor.predict_intervals(
        cooler_id="Location01A",
        intervals=intervals,
        outdoor_temp_c=34.0,
        humidity_pct=30.0,
        use_cache=False,
    )

    # Predict at 34°C with high humidity (80% - oppressive summer humid heatwave)
    preds_humid = predictor.predict_intervals(
        cooler_id="Location01A",
        intervals=intervals,
        outdoor_temp_c=34.0,
        humidity_pct=80.0,
        use_cache=False,
    )

    # High humidity heatwave should generate higher expected and conservative demand
    assert preds_humid[0]["expected_liters"] >= preds_dry[0]["expected_liters"]
    assert preds_humid[0]["conservative_liters"] >= preds_dry[0]["conservative_liters"]
    assert preds_humid[0]["conservative_liters"] >= preds_humid[0]["expected_liters"]


def test_service_predict_and_schemas_with_humidity():
    """Verify service_predict accepts humidity and returns valid forecast."""
    payload = {
        "cooler": {"cooler_id": "Location01A", "capacity_liters": 200},
        "intervals": [
            {"period_slot": "P1", "day_of_week": "Monday"},
            {"period_slot": "Recess", "day_of_week": "Monday"},
            {"period_slot": "P6", "day_of_week": "Monday"},
        ],
        "weather": {
            "outdoor_temp_c": 34.0,
            "humidity_pct": 75.0,
            "weather": "Humid",
        },
        "model_preference": "auto",
        "use_cache": False,
    }

    res = service_predict(payload, use_cache=False)
    assert res["cooler_id"] == "Location01A"
    assert len(res["predictions"]) == 3
    for p in res["predictions"]:
        assert p["conservative_liters"] >= p["expected_liters"]
        assert p["expected_liters"] > 0

    # Verify LiveWeatherResponse schema
    live = fetch_live_weather(city="New Delhi")
    schema_obj = LiveWeatherResponse.model_validate(live)
    assert schema_obj.outdoor_temp_c is not None
    assert schema_obj.apparent_temp_c is not None
    assert schema_obj.humidity_pct is not None
