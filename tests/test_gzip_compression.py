"""Tests for GZip compression middleware on FastAPI endpoints for low-bandwidth networks."""

import gzip
import json
import pytest
from starlette.testclient import TestClient
from fastapi.middleware.gzip import GZipMiddleware

from app.app import create_demo


@pytest.fixture(scope="module")
def api_client():
    demo = create_demo()
    return TestClient(demo.app)


def test_gzip_middleware_registered():
    """Verify that GZipMiddleware is registered in the FastAPI app's user middleware."""
    demo = create_demo()
    app = demo.app

    has_gzip = any(
        getattr(m, "cls", None) is GZipMiddleware or isinstance(m, GZipMiddleware)
        for m in getattr(app, "user_middleware", [])
    )
    assert has_gzip, "GZipMiddleware is not registered on demo.app"


def test_forecast_day_gzip_compression(api_client):
    """Verify full-day forecast response (/api/forecast_day) is gzipped when over 500 bytes."""
    payload = {
        "cooler": {"cooler_id": "Location01A", "capacity_liters": 200, "near_games": False},
        "intervals": [
            {"period_slot": s, "day_of_week": "Monday"}
            for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
        ],
        "weather": {"outdoor_temp_c": 32.0, "weather": "Clear"},
        "current_level_liters": 140,
        "model_preference": "auto",
        "use_cache": False,
    }

    # 1. Request with Accept-Encoding: gzip
    resp_gzip = api_client.post("/api/forecast_day", json=payload, headers={"Accept-Encoding": "gzip"})
    assert resp_gzip.status_code == 200
    assert resp_gzip.headers.get("content-encoding") == "gzip"

    # Verify response structure decodes correctly
    data = resp_gzip.json()
    assert data["cooler_id"] == "Location01A"
    assert "predictions" in data
    assert len(data["predictions"]) == 9
    assert "shortage" in data

    # 2. Request without compression (Accept-Encoding: identity)
    resp_raw = api_client.post("/api/forecast_day", json=payload, headers={"Accept-Encoding": "identity"})
    assert resp_raw.status_code == 200
    assert resp_raw.headers.get("content-encoding") is None

    # 3. Check compression ratio
    raw_size = len(resp_raw.content)
    compressed_size = len(gzip.compress(resp_raw.content))
    bandwidth_saved_pct = (1.0 - (compressed_size / raw_size)) * 100

    assert raw_size > 500, f"Expected response > 500 bytes, got {raw_size}"
    # Validates expected 60-75% data bandwidth savings for mobile/low-bandwidth clients
    assert 60.0 <= bandwidth_saved_pct <= 80.0, f"Expected 60-80% savings, got {bandwidth_saved_pct:.1f}%"


def test_predict_endpoint_gzip_compression(api_client):
    """Verify /api/predict responses (> 500 bytes) are compressed with gzip."""
    payload = {
        "cooler": {"cooler_id": "Location04A", "capacity_liters": 250, "near_games": True},
        "intervals": [
            {"period_slot": s, "day_of_week": "Wednesday"}
            for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
        ],
        "weather": {"outdoor_temp_c": 35.0, "weather": "Clear"},
        "model_preference": "auto",
        "use_cache": False,
    }

    resp = api_client.post("/api/predict", json=payload, headers={"Accept-Encoding": "gzip"})
    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") == "gzip"

    data = resp.json()
    assert len(data["predictions"]) == 9


def test_small_response_under_500_bytes_uncompressed(api_client):
    """Verify small responses below 500 bytes are not compressed to avoid gzip overhead."""
    resp = api_client.get("/api/cache_info", headers={"Accept-Encoding": "gzip"})
    assert resp.status_code == 200

    # Cache info JSON is ~150 bytes, well below the 500-byte threshold
    raw_len = len(resp.content)
    assert raw_len < 500, f"Cache info unexpectedly large: {raw_len} bytes"
    assert resp.headers.get("content-encoding") is None


def test_identity_encoding_bypass(api_client):
    """Verify client can opt out of gzip via Accept-Encoding: identity."""
    payload = {
        "cooler": {"cooler_id": "Location02A", "capacity_liters": 180, "near_games": False},
        "intervals": [{"period_slot": "P1", "day_of_week": "Friday"}],
        "weather": {"outdoor_temp_c": 25.0, "weather": "Clear"},
    }

    resp = api_client.post("/api/predict", json=payload, headers={"Accept-Encoding": "identity"})
    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") is None


def test_main_app_gzip_compression():
    """Verify that main.app (FastAPI cloud entrypoint) has GZipMiddleware and compresses responses."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from main import app as main_app

    has_gzip = any(
        getattr(m, "cls", None) is GZipMiddleware or isinstance(m, GZipMiddleware)
        for m in getattr(main_app, "user_middleware", [])
    )
    assert has_gzip, "GZipMiddleware is not registered on main.app"

    client = TestClient(main_app)
    payload = {
        "cooler": {"cooler_id": "Location01A", "capacity_liters": 200, "near_games": False},
        "intervals": [
            {"period_slot": s, "day_of_week": "Monday"}
            for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
        ],
        "weather": {"outdoor_temp_c": 32.0, "weather": "Clear"},
        "current_level_liters": 140,
        "model_preference": "auto",
        "use_cache": False,
    }
    resp = client.post("/api/forecast_day", json=payload, headers={"Accept-Encoding": "gzip"})
    assert resp.status_code == 200
    assert resp.headers.get("content-encoding") == "gzip"

