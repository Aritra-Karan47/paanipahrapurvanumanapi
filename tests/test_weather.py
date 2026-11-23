"""Tests for Open-Meteo weather detection and 8-hour cache manager."""

import time
from pathlib import Path

import pytest

from weather.service import (
    WeatherCacheManager,
    _wmo_code_to_condition,
    fetch_live_weather,
)
from cooler_demand.schemas import LiveWeatherRequest, LiveWeatherResponse


def test_wmo_code_mapping():
    assert _wmo_code_to_condition(0, humidity=50) == "Clear"
    assert _wmo_code_to_condition(0, humidity=85) == "Humid"
    assert _wmo_code_to_condition(1, humidity=50) == "Cloudy"
    assert _wmo_code_to_condition(3, humidity=50) == "Cloudy"
    assert _wmo_code_to_condition(45) == "Cloudy"
    assert _wmo_code_to_condition(61) == "Rainy"
    assert _wmo_code_to_condition(95) == "Rainy"


def test_weather_cache_8h_ttl(tmp_path):
    cache_file = tmp_path / "test_weather_cache.json"
    mgr = WeatherCacheManager(cache_file=cache_file, ttl_seconds=8 * 3600)

    payload = {
        "outdoor_temp_c": 32.5,
        "weather": "Clear",
        "humidity_pct": 45.0,
        "station_info": "Test Base Station",
        "location_name": "Test City",
        "latitude": 28.61,
        "longitude": 77.20,
    }

    # Put in cache
    mgr.put("Test City", None, None, payload)

    # Immediate retrieval should be cached
    cached = mgr.get("Test City", None, None)
    assert cached is not None
    assert cached["cached"] is True
    assert cached["outdoor_temp_c"] == 32.5
    assert cached["cache_age_hours"] < 0.1
    assert cached["cache_ttl_remaining_hours"] <= 8.0

    # Simulate 9 hours later (should expire)
    cached_expired = mgr._memory_cache["city_test city"]
    cached_expired["timestamp"] = time.time() - (9 * 3600)  # 9 hours ago
    expired_result = mgr.get("Test City", None, None)
    assert expired_result is None


def test_fetch_live_weather_structure():
    # Should return a valid payload even when offline or using fallback
    res = fetch_live_weather(city="New Delhi")
    assert "outdoor_temp_c" in res
    assert "weather" in res
    assert "station_info" in res
    assert res["weather"] in ("Clear", "Cloudy", "Rainy", "Humid", "Unknown")
    assert -20 <= res["outdoor_temp_c"] <= 60


def test_live_weather_schemas():
    req = LiveWeatherRequest(city="Mumbai")
    assert req.city == "Mumbai"

    resp = LiveWeatherResponse(
        success=True,
        outdoor_temp_c=31.5,
        weather="Humid",
        humidity_pct=82.0,
        weather_code=0,
        location_name="Mumbai, India",
        latitude=19.07,
        longitude=72.87,
        source="Open-Meteo",
        message="OK",
    )
    assert resp.outdoor_temp_c == 31.5
    assert resp.weather == "Humid"
