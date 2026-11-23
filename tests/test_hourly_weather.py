"""Tests for period-level hourly weather mapping and diurnal progression."""

import pytest
from weather.service import (
    PERIOD_START_HOURS,
    PERIOD_START_TIMES,
    diurnal_temperature_for_slot,
    get_slot_hourly_temperature,
    map_period_hourly_temperatures,
)


def test_period_start_hours_mapping():
    assert PERIOD_START_HOURS["P1"] == 8    # 8:00 AM
    assert PERIOD_START_HOURS["P6"] == 13   # 1:30 PM peak window
    assert PERIOD_START_HOURS["P8"] == 15   # 3:00 PM
    assert "Recess" in PERIOD_START_HOURS


def test_diurnal_temperature_progression_p1_cooler_than_p6():
    """Verify Period 1 (8:00 AM) is cooler than Period 6 (1:30 PM)."""
    base_temp = 30.0
    temp_p1 = diurnal_temperature_for_slot(base_temp, "P1")
    temp_p4 = diurnal_temperature_for_slot(base_temp, "P4")
    temp_p6 = diurnal_temperature_for_slot(base_temp, "P6")

    # Morning is cooler than afternoon peak
    assert temp_p1 < temp_p4 < temp_p6
    assert temp_p1 == 24.0  # 30.0 - 6.0
    assert temp_p6 == 33.0  # 30.0 + 3.0
    assert (temp_p6 - temp_p1) == 9.0  # 9 degrees intra-day progression


def test_open_meteo_hourly_forecast_mapping():
    """Verify mapping slot start times to exact Open-Meteo hourly forecasted temperatures."""
    mock_hourly = {
        "time": [
            "2026-11-24T08:00",
            "2026-11-24T09:00",
            "2026-11-24T10:00",
            "2026-11-24T11:00",
            "2026-11-24T12:00",
            "2026-11-24T13:00",
            "2026-11-24T14:00",
            "2026-11-24T15:00",
        ],
        "temperature_2m": [22.0, 24.5, 27.0, 29.5, 32.0, 36.0, 35.5, 33.0],
    }

    # P1 (08:00) should be 22.0°C
    t_p1 = get_slot_hourly_temperature("P1", base_temp_c=28.0, hourly_forecast=mock_hourly)
    assert t_p1 == 22.0

    # P6 (13:00) should be 36.0°C (matches prompt requirement: 8 AM 22°C vs 1:30 PM 36°C)
    t_p6 = get_slot_hourly_temperature("P6", base_temp_c=28.0, hourly_forecast=mock_hourly)
    assert t_p6 == 36.0

    # P8 (15:00) should be 33.0°C
    t_p8 = get_slot_hourly_temperature("P8", base_temp_c=28.0, hourly_forecast=mock_hourly)
    assert t_p8 == 33.0


def test_map_period_hourly_temperatures_batch():
    mock_hourly = {
        "time": ["2026-11-24T08:00", "2026-11-24T13:00"],
        "temperature_2m": [21.5, 35.8],
    }
    mapped = map_period_hourly_temperatures(base_temp_c=28.0, hourly_forecast=mock_hourly, slots=["P1", "P6"])
    assert mapped["P1"] == 21.5
    assert mapped["P6"] == 35.8


def test_fallback_when_hourly_forecast_missing():
    """Verify graceful fallback to diurnal progression curve when hourly data is absent."""
    base_temp = 28.0
    t_p1 = get_slot_hourly_temperature("P1", base_temp_c=base_temp, hourly_forecast=None)
    t_p6 = get_slot_hourly_temperature("P6", base_temp_c=base_temp, hourly_forecast=None)

    assert t_p1 == 22.0  # 28.0 - 6.0
    assert t_p6 == 31.0  # 28.0 + 3.0
    assert t_p1 < t_p6
