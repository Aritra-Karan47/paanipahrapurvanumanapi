"""In-process service function tests with real data bootstrap."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import app as app_module

service_predict = app_module.service_predict
service_forecast_day = app_module.service_forecast_day
service_temperature_target = app_module.service_temperature_target
service_check_shortage = app_module.service_check_shortage
service_validate_data = app_module.service_validate_data
service_model_info = app_module.service_model_info


def _base_payload(slots=None, cooler_id="Location01A", capacity=200, near=False):
    slots = slots or ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    return {
        "cooler": {
            "cooler_id": cooler_id,
            "capacity_liters": capacity,
            "near_games": near,
            "current_level_liters": capacity * 0.7,
        },
        "intervals": [
            {
                "period_slot": s,
                "day_of_week": "Tuesday",
                "games_intensity": "Moderate" if s == "P4" else "None",
                "duration_minutes": 20 if s == "Recess" else 40,
            }
            for s in slots
        ],
        "weather": {"outdoor_temp_c": 33.5, "weather": "Clear"},
        "day_of_week": "Tuesday",
        "current_level_liters": capacity * 0.7,
        "model_preference": "auto",
    }


def test_predict_includes_recess():
    out = service_predict(_base_payload())
    slots = [p["period_slot"] for p in out["predictions"]]
    assert "Recess" in slots
    assert out["total_expected_liters"] > 0


def test_forecast_day_shortage():
    out = service_forecast_day(_base_payload())
    assert "shortage" in out
    assert out["shortage"]["risk_level"] in ("none", "low", "medium", "high", "critical")


def test_temperature_service_is1475():
    r = service_temperature_target({"outdoor_temp_c": 33.5})
    assert r["target_water_temp_c"] == 14.0
    assert r["cooling_enabled"] is True
    r2 = service_temperature_target({"outdoor_temp_c": 15.0})
    assert r2["cooling_enabled"] is False


def test_check_shortage_service():
    r = service_check_shortage(
        {
            "cooler_id": "Location01A",
            "capacity_liters": 200,
            "current_level_liters": 40,
            "expected_demand_liters": 90,
            "conservative_demand_liters": 110,
        }
    )
    assert r["risk_level"] in ("high", "critical")
    assert r["shortage_liters"] > 0


def test_validate_good():
    r = service_validate_data(
        {
            "cooler": {"cooler_id": "Location01A", "capacity_liters": 200},
            "intervals": [{"period_slot": "Recess", "day_of_week": "Monday"}],
            "weather": {"outdoor_temp_c": 22},
        }
    )
    assert r["valid"] is True


def test_validate_bad_temp():
    r = service_validate_data({"weather": {"outdoor_temp_c": 99}})
    assert r["valid"] is False


def test_model_info():
    info = service_model_info()
    assert info["baseline_available"] is True


def test_all_eight_coolers_service():
    for cid, cap, near in [
        ("Location01A", 200, False),
        ("Location01B", 200, False),
        ("Location02A", 180, False),
        ("Location02B", 180, False),
        ("Location03A", 150, False),
        ("Location03B", 150, False),
        ("Location04A", 250, True),
        ("Location04B", 250, True),
    ]:
        out = service_forecast_day(_base_payload(cooler_id=cid, capacity=cap, near=near))
        assert out["cooler_id"] == cid
        assert len(out["predictions"]) == 9


def test_api_predict_and_forecast_day_endpoints():
    from fastapi.testclient import TestClient

    client = TestClient(app_module.demo.app)
    payload = _base_payload()

    # /api/predict wrapped in threadpool
    resp_predict = client.post("/api/predict", json=payload)
    assert resp_predict.status_code == 200
    pred_data = resp_predict.json()
    assert "predictions" in pred_data
    assert pred_data["total_expected_liters"] > 0
    assert pred_data["cooler_id"] == "Location01A"

    # /api/forecast_day wrapped in threadpool
    resp_forecast = client.post("/api/forecast_day", json=payload)
    assert resp_forecast.status_code == 200
    forecast_data = resp_forecast.json()
    assert "shortage" in forecast_data
    assert forecast_data["shortage"]["risk_level"] in ("none", "low", "medium", "high", "critical")


def test_api_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient

    client = TestClient(app_module.demo.app)
    payload = _base_payload()

    def run_predict(i):
        res = client.post("/api/predict", json=payload)
        return res.status_code, res.json()["cooler_id"]

    def run_forecast(i):
        res = client.post("/api/forecast_day", json=payload)
        return res.status_code, res.json()["shortage"]["risk_level"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        predict_futures = list(executor.map(run_predict, range(5)))
        forecast_futures = list(executor.map(run_forecast, range(5)))

    for status, cooler_id in predict_futures:
        assert status == 200
        assert cooler_id == "Location01A"

    for status, risk_level in forecast_futures:
        assert status == 200
        assert risk_level in ("none", "low", "medium", "high", "critical")

