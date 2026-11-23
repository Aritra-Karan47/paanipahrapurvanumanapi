"""Weather auto-detection module."""

from .service import fetch_live_weather, geocode_city

__all__ = ["fetch_live_weather", "geocode_city"]
