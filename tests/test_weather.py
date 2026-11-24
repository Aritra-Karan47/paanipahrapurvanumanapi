"""Tests for Open-Meteo weather detection and 8-hour cache manager."""

import concurrent.futures
import json
import time
from pathlib import Path
from unittest.mock import patch

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


def test_weather_cache_concurrent_thread_writes(tmp_path):
    """Verify that multiple concurrent threads writing to the weather cache do not cause race conditions or corruption."""
    cache_file = tmp_path / "concurrent_weather_cache.json"
    mgr = WeatherCacheManager(cache_file=cache_file, ttl_seconds=8 * 3600)

    num_threads = 20
    cities = [f"City_{i}" for i in range(num_threads)]

    def worker(city_name: str, idx: int):
        payload = {
            "outdoor_temp_c": 20.0 + idx,
            "weather": "Clear",
            "humidity_pct": 50.0 + idx,
            "station_info": f"Station {city_name}",
            "location_name": city_name,
            "latitude": 20.0 + idx,
            "longitude": 70.0 + idx,
        }
        mgr.put(city_name, None, None, payload)
        # Verify immediate read
        cached = mgr.get(city_name, None, None)
        assert cached is not None
        assert cached["location_name"] == city_name

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, city, i) for i, city in enumerate(cities)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Disk cache should exist and be valid JSON
    assert cache_file.exists()
    with open(cache_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)

    # All cities should be in disk data
    assert len(disk_data) == num_threads
    for city in cities:
        key = f"city_{city.lower()}"
        assert key in disk_data
        assert disk_data[key]["location_name"] == city

    # Verify no temp files remain
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_weather_cache_atomic_replace_preserves_on_failure(tmp_path):
    """Verify that a disk write failure does not corrupt the existing cache and cleans up temp files."""
    cache_file = tmp_path / "fail_safe_weather_cache.json"
    mgr = WeatherCacheManager(cache_file=cache_file, ttl_seconds=8 * 3600)

    # Save an initial valid entry
    initial_payload = {
        "outdoor_temp_c": 25.0,
        "weather": "Clear",
        "humidity_pct": 50.0,
        "station_info": "Initial Station",
        "location_name": "Initial City",
        "latitude": 25.0,
        "longitude": 75.0,
    }
    mgr.put("Initial City", None, None, initial_payload)

    # Verify initial disk contents
    with open(cache_file, "r", encoding="utf-8") as f:
        data_before = json.load(f)
    assert "city_initial city" in data_before

    # Simulate a failure during disk sync/write
    with patch("os.fsync", side_effect=OSError("Simulated disk I/O error")):
        bad_payload = {
            "outdoor_temp_c": 99.0,
            "weather": "Rainy",
            "humidity_pct": 99.0,
            "station_info": "Bad Station",
            "location_name": "Bad City",
        }
        # put calls _save_disk_cache internally which catches Exception and cleans up
        mgr.put("Bad City", None, None, bad_payload)

    # Verify the original disk file was NOT corrupted or overwritten
    with open(cache_file, "r", encoding="utf-8") as f:
        data_after = json.load(f)
    assert "city_initial city" in data_after
    assert "city_bad city" not in data_after  # atomic rename never happened

    # Verify temp file was cleanly unlinked
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(tmp_files) == 0

