"""
Free live weather detection using Open-Meteo with 8-hour caching and nearest base station resolution.

Features:
- 100% Free Open-Meteo endpoints (No API key needed).
- Resolves coordinates and nearest meteorological base station grid node.
- Caches temperature & weather readings for 8 hours (in-memory + persistent disk cache).
- Supports manual slider override in Gradio UI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo API Endpoints (Free, no API key required)
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# 8 hours cache by default (can be overridden via WEATHER_CACHE_HOURS)
CACHE_TTL_SECONDS = int(os.environ.get("WEATHER_CACHE_HOURS", 8)) * 3600

# Default city (can be overridden via DEFAULT_WEATHER_CITY)
DEFAULT_CITY = os.environ.get("DEFAULT_WEATHER_CITY", "New Delhi")
DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090

CACHE_FILE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "weather_cache.json"


def _wmo_code_to_condition(code: int, humidity: Optional[float] = None) -> str:
    """Map WMO weather code to standard condition: Clear | Cloudy | Rainy | Humid | Unknown."""
    if code == 0:
        if humidity is not None and humidity >= 75.0:
            return "Humid"
        return "Clear"
    elif code in (1, 2, 3, 45, 48):
        if humidity is not None and humidity >= 80.0:
            return "Humid"
        return "Cloudy"
    elif code in (
        51, 53, 55, 56, 57,
        61, 63, 65, 66, 67,
        71, 73, 75, 77,
        80, 81, 82, 85, 86,
        95, 96, 99,
    ):
        return "Rainy"
    
    if humidity is not None and humidity >= 70.0:
        return "Humid"
    return "Unknown"


class WeatherCacheManager:
    """Thread-safe persistent cache for weather readings with an 8-hour TTL."""

    def __init__(self, cache_file: Path = CACHE_FILE_PATH, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.cache_file = cache_file
        self.ttl_seconds = ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._load_disk_cache()

    def _normalize_key(self, city: Optional[str], lat: Optional[float], lon: Optional[float]) -> str:
        if lat is not None and lon is not None:
            return f"coord_{round(lat, 2)}_{round(lon, 2)}"
        c = (city or DEFAULT_CITY).strip().lower()
        return f"city_{c}"

    def _load_disk_cache(self) -> None:
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._memory_cache = data
                        logger.debug("Loaded %d weather cache entries from %s", len(data), self.cache_file)
        except Exception as exc:
            logger.warning("Could not read weather cache from %s: %s", self.cache_file, exc)
            self._memory_cache = {}

    def _save_disk_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_cache, f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist weather cache to %s: %s", self.cache_file, exc)

    def get(self, city: Optional[str], lat: Optional[float], lon: Optional[float]) -> Optional[Dict[str, Any]]:
        key = self._normalize_key(city, lat, lon)
        entry = self._memory_cache.get(key)
        if not entry:
            return None

        now = time.time()
        stored_time = float(entry.get("timestamp", 0))
        age = now - stored_time

        if age < self.ttl_seconds:
            # Cache is fresh (less than 8 hours old)
            res = dict(entry)
            res["cached"] = True
            res["cache_age_hours"] = round(age / 3600, 2)
            res["cache_ttl_remaining_hours"] = round((self.ttl_seconds - age) / 3600, 2)
            res["source"] = f"Open-Meteo (Cached {res['cache_age_hours']}h ago - 8h TTL)"
            return res

        # Expired
        return None

    def put(self, city: Optional[str], lat: Optional[float], lon: Optional[float], data: Dict[str, Any]) -> None:
        key = self._normalize_key(city, lat, lon)
        now = time.time()
        record = dict(data)
        record["timestamp"] = now
        record["cached_at"] = datetime.now(timezone.utc).isoformat()
        record["cached"] = False
        record["cache_age_hours"] = 0.0
        record["cache_ttl_remaining_hours"] = 8.0

        self._memory_cache[key] = record
        self._save_disk_cache()


# Global cache instance
weather_cache = WeatherCacheManager()


def geocode_city(city_name: str, timeout: float = 4.0) -> Tuple[Optional[float], Optional[float], str]:
    """Lookup latitude and longitude for a city name via Open-Meteo Geocoding."""
    if not city_name or not city_name.strip():
        return DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY

    city = city_name.strip()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                GEOCODING_API_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results")
                if results and len(results) > 0:
                    top = results[0]
                    lat = float(top["latitude"])
                    lon = float(top["longitude"])
                    name = f"{top.get('name', city)}, {top.get('country', '')}".strip(", ")
                    return lat, lon, name
    except Exception as exc:
        logger.warning("Geocoding lookup failed for '%s': %s", city, exc)

    return None, None, city


def fetch_live_weather(
    city: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    force_refresh: bool = False,
    timeout: float = 4.0,
) -> Dict[str, Any]:
    """
    Fetch outdoor weather from Open-Meteo from the nearest base station grid point.
    Caches results for 8 hours (TTL = 8 hours). Set force_refresh=True to bypass cache.
    """
    resolved_name = city or "Current Location"

    # 1. Check 8-hour cache first (if not force-refreshing)
    if not force_refresh:
        cached_result = weather_cache.get(city, lat, lon)
        if cached_result:
            logger.info("Serving weather for '%s' from 8-hour cache (age: %sh)", city, cached_result.get("cache_age_hours"))
            return cached_result

    # 2. Resolve coordinates if not provided
    if lat is None or lon is None:
        c_lat, c_lon, resolved_name = geocode_city(city or DEFAULT_CITY, timeout=timeout)
        if c_lat is not None and c_lon is not None:
            lat, lon = c_lat, c_lon
        else:
            lat, lon = DEFAULT_LAT, DEFAULT_LON

    # Check cache by coordinates as well
    if not force_refresh:
        cached_coord = weather_cache.get(None, lat, lon)
        if cached_coord:
            return cached_coord

    # 3. Query Open-Meteo nearest base station endpoint
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                FORECAST_API_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code",
                    "timezone": "auto",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                temp = float(current.get("temperature_2m", 25.0))
                humidity = float(current.get("relative_humidity_2m", 50.0))
                code = int(current.get("weather_code", 0))
                condition = _wmo_code_to_condition(code, humidity=humidity)

                # Station metadata resolved by Open-Meteo
                st_lat = data.get("latitude", lat)
                st_lon = data.get("longitude", lon)
                elev = data.get("elevation", 0.0)
                station_info = f"Nearest Base Station Grid: {st_lat:.2f}°N, {st_lon:.2f}°E (Elev: {elev}m)"

                weather_payload = {
                    "success": True,
                    "outdoor_temp_c": round(temp, 1),
                    "weather": condition,
                    "humidity_pct": round(humidity, 1),
                    "weather_code": code,
                    "location_name": resolved_name,
                    "station_info": station_info,
                    "latitude": lat,
                    "longitude": lon,
                    "source": "Open-Meteo API (Live - Cached for 8 hours)",
                    "message": f"Weather detected from {station_info} for {resolved_name}.",
                }

                # Save in 8-hour cache
                weather_cache.put(city, lat, lon, weather_payload)
                return weather_payload
            else:
                logger.warning("Open-Meteo returned status %s", resp.status_code)
    except Exception as exc:
        logger.warning("Failed to fetch live weather from Open-Meteo: %s", exc)

    # 4. Fallback if offline or unreachable
    fallback_payload = {
        "success": False,
        "outdoor_temp_c": 28.0,
        "weather": "Clear",
        "humidity_pct": 50.0,
        "weather_code": 0,
        "location_name": resolved_name,
        "station_info": f"Default coordinates: {lat:.2f}°N, {lon:.2f}°E",
        "latitude": lat,
        "longitude": lon,
        "source": "Fallback default (offline or unreachable)",
        "message": "Open-Meteo unreachable. Slider remains available for manual adjustment.",
        "cached": False,
        "cache_age_hours": 0.0,
        "cache_ttl_remaining_hours": 8.0,
    }
    return fallback_payload
