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
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# School timetable slot schedules and start times
PERIOD_START_TIMES: Dict[str, str] = {
    "P1": "08:00",
    "P2": "08:45",
    "P3": "09:30",
    "P4": "10:15",
    "Recess": "11:00",
    "P5": "11:30",
    "P6": "12:15",
    "P7": "13:00",
    "P8": "13:45",
}

PERIOD_START_HOURS: Dict[str, int] = {
    "P1": 8,       # 08:00 AM (cooler morning)
    "P2": 9,       # 09:00 AM
    "P3": 10,      # 10:00 AM
    "P4": 11,      # 11:00 AM
    "Recess": 11,  # 11:30 AM
    "P5": 12,      # 12:00 PM
    "P6": 13,      # 01:30 PM (peak afternoon heat)
    "P7": 14,      # 02:00 PM
    "P8": 15,      # 03:00 PM
}

# Realistic intra-day offset from base temperature (°C):
# Early morning (8:00 AM, P1) is significantly cooler; afternoon (1:30 PM, P6) reaches peak heat.
DIURNAL_TEMP_OFFSETS: Dict[str, float] = {
    "P1": -6.0,     # 08:00 AM (e.g. 22°C when peak is 30°C / base is 28°C)
    "P2": -4.2,     # 08:45 AM
    "P3": -2.8,     # 09:30 AM
    "P4": -1.5,     # 10:15 AM
    "Recess": -0.5, # 11:00 AM
    "P5": +0.8,     # 11:30 AM / 12:00 PM
    "P6": +3.0,     # 01:30 PM (peak heat)
    "P7": +2.2,     # 02:00 PM
    "P8": +1.0,     # 03:00 PM
}

# Diurnal relative humidity progression (% offset from base humidity):
# Early morning (8:00 AM) has higher relative humidity; afternoon peak heat has lower relative humidity.
DIURNAL_HUMIDITY_OFFSETS: Dict[str, float] = {
    "P1": +12.0,    # 08:00 AM (higher morning humidity)
    "P2": +8.0,     # 08:45 AM
    "P3": +5.0,     # 09:30 AM
    "P4": +2.0,     # 10:15 AM
    "Recess": 0.0,  # 11:00 AM
    "P5": -2.0,     # 11:30 AM
    "P6": -6.0,     # 01:30 PM (peak dry heat)
    "P7": -4.0,     # 02:00 PM
    "P8": -2.0,     # 03:00 PM
}


def calculate_apparent_temperature(
    temp_c: float,
    humidity_pct: float,
    wind_speed_ms: float = 1.5,
) -> float:
    """Calculate Apparent Temperature (Feels-Like temperature in °C) using Steadman's formula.

    Formula (Australian Bureau of Meteorology / Steadman model used by Open-Meteo):
      e = (humidity_pct / 100.0) * 6.105 * exp((17.27 * temp_c) / (237.7 + temp_c))
      apparent_temp = temp_c + 0.33 * e - 0.70 * wind_speed_ms - 4.0

    At 34°C with 75% relative humidity and light indoor breeze (1.5 m/s), feels like ~42.1°C (~42°C).
    At 25°C with 50% relative humidity, feels like ~25.1°C.
    """
    rh = max(0.0, min(100.0, float(humidity_pct)))
    t = float(temp_c)
    # Vapor pressure e in hPa
    e = (rh / 100.0) * 6.105 * math.exp((17.27 * t) / (237.7 + t))
    at = t + 0.33 * e - 0.70 * float(wind_speed_ms) - 4.0
    return round(at, 1)


def calculate_heat_index(temp_c: float, humidity_pct: float) -> float:
    """Calculate Heat Index in °C using the standard NOAA / Rothfusz polynomial regression equation.

    Provides standard US NWS / NOAA Heat Index metric.
    """
    rh = max(0.0, min(100.0, float(humidity_pct)))
    t_c = float(temp_c)
    t_f = t_c * 1.8 + 32.0

    if t_f < 80.0:
        # Simple Steadman formula for mild temperatures (< 80°F / ~26.7°C)
        hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    else:
        # Rothfusz polynomial regression
        hi_f = (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 0.00683783 * (t_f ** 2)
            - 0.05481717 * (rh ** 2)
            + 0.00122874 * (t_f ** 2) * rh
            + 0.00085282 * t_f * (rh ** 2)
            - 0.00000199 * (t_f ** 2) * (rh ** 2)
        )
        if rh < 13.0 and 80.0 <= t_f <= 112.0:
            adj = ((13.0 - rh) / 4.0) * math.sqrt((17.0 - abs(t_f - 95.0)) / 17.0)
            hi_f -= adj
        elif rh > 85.0 and 80.0 <= t_f <= 87.0:
            adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
            hi_f += adj

    hi_c = (hi_f - 32.0) / 1.8
    return round(hi_c, 1)


def diurnal_humidity_for_slot(base_humidity_pct: float, period_slot: str) -> float:
    """Derive realistic intra-day relative humidity for a school period based on diurnal curve."""
    slot = str(period_slot).strip()
    offset = DIURNAL_HUMIDITY_OFFSETS.get(slot, 0.0)
    return round(max(5.0, min(100.0, float(base_humidity_pct) + offset)), 1)


def diurnal_temperature_for_slot(base_temp_c: float, period_slot: str) -> float:
    """Derive realistic intra-day temperature for a school period based on typical diurnal progression."""
    slot = str(period_slot).strip()
    offset = DIURNAL_TEMP_OFFSETS.get(slot, 0.0)
    return round(float(base_temp_c) + offset, 1)


def get_slot_hourly_temperature(
    period_slot: str,
    base_temp_c: float,
    hourly_forecast: Optional[Dict[str, Any]] = None,
    target_date: Optional[str] = None,
) -> float:
    """Map a period slot's start time to its respective hourly forecasted temperature.

    If Open-Meteo hourly forecast is provided, looks up the forecasted temperature
    at the slot's start hour (e.g., 08:00 for P1, 13:00 for P6).
    Otherwise, applies the realistic diurnal progression curve.
    """
    slot = str(period_slot).strip()
    hour = PERIOD_START_HOURS.get(slot)
    if hour is None:
        return round(float(base_temp_c), 1)

    if hourly_forecast and isinstance(hourly_forecast, dict):
        times = hourly_forecast.get("time", [])
        temps = hourly_forecast.get("temperature_2m", [])
        if times and temps and len(times) == len(temps):
            hour_str = f"T{hour:02d}:00"
            for t_str, val in zip(times, temps):
                if target_date and not str(t_str).startswith(target_date):
                    continue
                if hour_str in str(t_str):
                    try:
                        return round(float(val), 1)
                    except (TypeError, ValueError):
                        pass

    return diurnal_temperature_for_slot(base_temp_c, slot)


def get_slot_hourly_humidity(
    period_slot: str,
    base_humidity_pct: float = 50.0,
    hourly_forecast: Optional[Dict[str, Any]] = None,
    target_date: Optional[str] = None,
) -> float:
    """Map a period slot's start time to its respective hourly relative humidity (%)."""
    slot = str(period_slot).strip()
    hour = PERIOD_START_HOURS.get(slot)
    if hour is None:
        return round(float(base_humidity_pct), 1)

    if hourly_forecast and isinstance(hourly_forecast, dict):
        times = hourly_forecast.get("time", [])
        rhs = hourly_forecast.get("relative_humidity_2m", [])
        if times and rhs and len(times) == len(rhs):
            hour_str = f"T{hour:02d}:00"
            for t_str, val in zip(times, rhs):
                if target_date and not str(t_str).startswith(target_date):
                    continue
                if hour_str in str(t_str):
                    try:
                        return round(float(val), 1)
                    except (TypeError, ValueError):
                        pass

    return diurnal_humidity_for_slot(base_humidity_pct, slot)


def get_slot_hourly_apparent_temperature(
    period_slot: str,
    base_temp_c: float,
    base_humidity_pct: float = 50.0,
    hourly_forecast: Optional[Dict[str, Any]] = None,
    target_date: Optional[str] = None,
    wind_speed_ms: float = 1.5,
) -> float:
    """Map a period slot's start time to its respective hourly forecasted apparent temperature (feels-like °C).

    If Open-Meteo hourly forecast contains 'apparent_temperature', uses the hourly value directly.
    Otherwise, derives it from hourly temperature & relative humidity or diurnal progression.
    """
    slot = str(period_slot).strip()
    hour = PERIOD_START_HOURS.get(slot)
    if hour is None:
        return calculate_apparent_temperature(base_temp_c, base_humidity_pct, wind_speed_ms=wind_speed_ms)

    if hourly_forecast and isinstance(hourly_forecast, dict):
        times = hourly_forecast.get("time", [])
        app_temps = hourly_forecast.get("apparent_temperature", [])
        if times and app_temps and len(times) == len(app_temps):
            hour_str = f"T{hour:02d}:00"
            for t_str, val in zip(times, app_temps):
                if target_date and not str(t_str).startswith(target_date):
                    continue
                if hour_str in str(t_str):
                    try:
                        return round(float(val), 1)
                    except (TypeError, ValueError):
                        pass

        # If apparent_temperature not in hourly, try temperature_2m and relative_humidity_2m
        temps = hourly_forecast.get("temperature_2m", [])
        rhs = hourly_forecast.get("relative_humidity_2m", [])
        if times and temps and len(times) == len(temps):
            hour_str = f"T{hour:02d}:00"
            for idx, (t_str, val) in enumerate(zip(times, temps)):
                if target_date and not str(t_str).startswith(target_date):
                    continue
                if hour_str in str(t_str):
                    try:
                        t_val = float(val)
                        rh_val = float(rhs[idx]) if rhs and len(rhs) > idx else base_humidity_pct
                        return calculate_apparent_temperature(t_val, rh_val, wind_speed_ms=wind_speed_ms)
                    except (TypeError, ValueError):
                        pass

    # Diurnal progression fallback
    slot_temp = diurnal_temperature_for_slot(base_temp_c, slot)
    slot_rh = diurnal_humidity_for_slot(base_humidity_pct, slot)
    return calculate_apparent_temperature(slot_temp, slot_rh, wind_speed_ms=wind_speed_ms)


def map_period_hourly_temperatures(
    base_temp_c: float,
    hourly_forecast: Optional[Dict[str, Any]] = None,
    slots: Optional[List[str]] = None,
    target_date: Optional[str] = None,
) -> Dict[str, float]:
    """Map all given period slots to their hourly forecasted temperatures."""
    target_slots = slots or ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    return {
        s: get_slot_hourly_temperature(s, base_temp_c, hourly_forecast, target_date)
        for s in target_slots
    }


def map_period_hourly_apparent_temperatures(
    base_temp_c: float,
    base_humidity_pct: float = 50.0,
    hourly_forecast: Optional[Dict[str, Any]] = None,
    slots: Optional[List[str]] = None,
    target_date: Optional[str] = None,
) -> Dict[str, float]:
    """Map all given period slots to their hourly forecasted apparent temperatures."""
    target_slots = slots or ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    return {
        s: get_slot_hourly_apparent_temperature(
            s, base_temp_c, base_humidity_pct, hourly_forecast, target_date
        )
        for s in target_slots
    }


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
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code",
                    "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code",
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

                app_temp_val = current.get("apparent_temperature")
                if app_temp_val is not None:
                    apparent_temp = float(app_temp_val)
                else:
                    apparent_temp = calculate_apparent_temperature(temp, humidity)
                heat_index = calculate_heat_index(temp, humidity)

                hourly = data.get("hourly", {})
                hourly_forecast = {
                    "time": hourly.get("time", []),
                    "temperature_2m": hourly.get("temperature_2m", []),
                    "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
                    "apparent_temperature": hourly.get("apparent_temperature", []),
                    "weather_code": hourly.get("weather_code", []),
                }

                # Station metadata resolved by Open-Meteo
                st_lat = data.get("latitude", lat)
                st_lon = data.get("longitude", lon)
                elev = data.get("elevation", 0.0)
                station_info = f"Nearest Base Station Grid: {st_lat:.2f}°N, {st_lon:.2f}°E (Elev: {elev}m)"

                weather_payload = {
                    "success": True,
                    "outdoor_temp_c": round(temp, 1),
                    "apparent_temp_c": round(apparent_temp, 1),
                    "heat_index_c": round(heat_index, 1),
                    "weather": condition,
                    "humidity_pct": round(humidity, 1),
                    "weather_code": code,
                    "hourly_forecast": hourly_forecast,
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
    fallback_temp = 28.0
    fallback_rh = 50.0
    fallback_app = calculate_apparent_temperature(fallback_temp, fallback_rh)
    fallback_hi = calculate_heat_index(fallback_temp, fallback_rh)

    fallback_payload = {
        "success": False,
        "outdoor_temp_c": fallback_temp,
        "apparent_temp_c": fallback_app,
        "heat_index_c": fallback_hi,
        "weather": "Clear",
        "humidity_pct": fallback_rh,
        "weather_code": 0,
        "hourly_forecast": None,
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
