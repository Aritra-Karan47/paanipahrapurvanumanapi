"""Tests for AsyncWaterDemandClient high-concurrency async SDK client."""

import asyncio
import time
from pathlib import Path
import pytest
import httpx

from cooler_demand import (
    AsyncWaterDemandClient,
    AsyncCoolerDemandClient,
    WaterDemandClient,
    CoolerDemandClient,
    CoolerDemandError,
    PredictionRequest,
    PredictionResponse,
    ForecastDayRequest,
    ForecastDayResponse,
    TemperatureTargetRequest,
    TemperatureTargetResponse,
    ShortageCheckRequest,
    ShortageCheckResponse,
    ValidateDataRequest,
    ValidateDataResponse,
    ModelInfo,
    LiveWeatherResponse,
)
from app.app import create_demo


@pytest.fixture(scope="module")
def fastapi_app():
    """Create and return the FastAPI app from demo."""
    demo = create_demo()
    return demo.app


@pytest.fixture
def async_client(fastapi_app):
    """Fixture providing an AsyncWaterDemandClient connected to the in-memory FastAPI app."""
    transport = httpx.ASGITransport(app=fastapi_app)
    client = AsyncWaterDemandClient(base_url="http://testserver", transport=transport)
    yield client
    asyncio.run(client.aclose())


def _make_forecast_payload(cooler_id: str = "Location01A", capacity: int = 200):
    return {
        "cooler": {
            "cooler_id": cooler_id,
            "capacity_liters": capacity,
            "near_games": False,
            "current_level_liters": capacity * 0.7,
        },
        "intervals": [
            {"period_slot": s, "day_of_week": "Tuesday"}
            for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
        ],
        "weather": {"outdoor_temp_c": 33.5, "weather": "Clear"},
        "day_of_week": "Tuesday",
        "current_level_liters": capacity * 0.7,
        "model_preference": "auto",
        "use_cache": False,
    }


def _make_prediction_payload(cooler_id: str = "Location01A", capacity: int = 200):
    return {
        "cooler": {
            "cooler_id": cooler_id,
            "capacity_liters": capacity,
            "near_games": False,
        },
        "intervals": [
            {"period_slot": "P1", "day_of_week": "Tuesday"},
            {"period_slot": "Recess", "day_of_week": "Tuesday"},
            {"period_slot": "P6", "day_of_week": "Tuesday"},
        ],
        "weather": {"outdoor_temp_c": 33.5, "weather": "Clear"},
        "day_of_week": "Tuesday",
        "model_preference": "auto",
        "use_cache": False,
    }


def test_async_client_lifecycle(fastapi_app):
    """Verify async client context manager and alias resolution."""
    async def _test():
        transport = httpx.ASGITransport(app=fastapi_app)
        async with AsyncWaterDemandClient(base_url="http://testserver", transport=transport) as client:
            assert client.base_url == "http://testserver"
            assert client._owns_client is True

        # Test aliases
        assert AsyncCoolerDemandClient is AsyncWaterDemandClient
        assert WaterDemandClient is CoolerDemandClient

    asyncio.run(_test())


def test_async_predict(async_client):
    """Verify async prediction request returns a valid PredictionResponse."""
    async def _test():
        payload = _make_prediction_payload()
        resp = await async_client.predict(payload)
        assert isinstance(resp, PredictionResponse)
        assert resp.cooler_id == "Location01A"
        assert len(resp.predictions) == 3
        assert resp.total_expected_liters > 0
        assert resp.total_conservative_liters >= resp.total_expected_liters

    asyncio.run(_test())


def test_async_forecast_day(async_client):
    """Verify async full-day forecast returns a valid ForecastDayResponse with shortage analysis."""
    async def _test():
        payload = _make_forecast_payload()
        resp = await async_client.forecast_day(payload)
        assert isinstance(resp, ForecastDayResponse)
        assert resp.cooler_id == "Location01A"
        assert len(resp.predictions) == 9
        assert resp.shortage is not None
        assert resp.shortage.risk_level in ("none", "low", "medium", "high", "critical")

    asyncio.run(_test())


def test_async_temperature_target(async_client):
    """Verify async water temperature target mapping."""
    async def _test():
        # Float shortcut
        resp1 = await async_client.get_temperature_target(34.0)
        assert isinstance(resp1, TemperatureTargetResponse)
        assert resp1.target_water_temp_c in (12.0, 14.0, 16.0, 18.0, 20.0, 22.0)

        # Schema request
        req = TemperatureTargetRequest(outdoor_temp_c=22.0)
        resp2 = await async_client.get_temperature_target(req)
        assert isinstance(resp2, TemperatureTargetResponse)
        assert resp2.target_water_temp_c == 20.0

    asyncio.run(_test())


def test_async_check_shortage(async_client):
    """Verify async shortage evaluation."""
    async def _test():
        req = ShortageCheckRequest(
            cooler_id="Location01A",
            capacity_liters=200,
            current_level_liters=30,
            expected_demand_liters=80,
            conservative_demand_liters=100,
        )
        resp = await async_client.check_shortage(req)
        assert isinstance(resp, ShortageCheckResponse)
        assert resp.risk_level in ("high", "critical")
        assert resp.shortage_liters > 0

    asyncio.run(_test())


def test_async_validate_data(async_client):
    """Verify async data validation endpoint."""
    async def _test():
        req = ValidateDataRequest(
            cooler={"cooler_id": "Location01A", "capacity_liters": 200},
            weather={"outdoor_temp_c": 32.0, "weather": "Clear"},
        )
        resp = await async_client.validate_data(req)
        assert isinstance(resp, ValidateDataResponse)
        assert resp.valid is True

    asyncio.run(_test())


def test_async_model_info(async_client):
    """Verify async model info returns valid metadata."""
    async def _test():
        info = await async_client.model_info()
        assert isinstance(info, ModelInfo)
        assert info.baseline_available is True
        assert len(info.supported_slots) > 0
        assert info.temperature_rules_loaded is True

    asyncio.run(_test())


def test_async_live_weather(async_client):
    """Verify async live weather detection."""
    async def _test():
        weather = await async_client.get_live_weather(city="New Delhi")
        assert isinstance(weather, LiveWeatherResponse)
        assert weather.outdoor_temp_c is not None

    asyncio.run(_test())


def test_async_concurrent_all_eight_coolers(async_client):
    """Verify high-concurrency parallel queries for all 8 coolers.
    
    Demonstrates querying all 8 coolers simultaneously in parallel via
    client.forecast_all_coolers() and client.predict_all_coolers().
    """
    all_8_coolers = [
        "Location01A", "Location01B",
        "Location02A", "Location02B",
        "Location03A", "Location03B",
        "Location04A", "Location04B",
    ]

    async def _test():
        forecast_requests = [_make_forecast_payload(c) for c in all_8_coolers]
        predict_requests = [_make_prediction_payload(c) for c in all_8_coolers]

        t0 = time.perf_counter()
        # 1. Parallel forecast for all 8 coolers via forecast_all_coolers
        forecast_results = await async_client.forecast_all_coolers(forecast_requests)
        t_forecast = time.perf_counter() - t0

        assert len(forecast_results) == 8
        for i, res in enumerate(forecast_results):
            assert isinstance(res, ForecastDayResponse)
            assert res.cooler_id == all_8_coolers[i]
            assert res.total_expected_liters > 0
            assert res.shortage.risk_level in ("none", "low", "medium", "high", "critical")

        # 2. Parallel predict for all 8 coolers via predict_all_coolers
        t1 = time.perf_counter()
        predict_results = await async_client.predict_all_coolers(predict_requests)
        t_predict = time.perf_counter() - t1

        assert len(predict_results) == 8
        for i, res in enumerate(predict_results):
            assert isinstance(res, PredictionResponse)
            assert res.cooler_id == all_8_coolers[i]
            assert res.total_expected_liters > 0

        # Concurrent parallel batching should execute all 8 in well under 2 seconds
        assert t_forecast < 2.0, f"Forecast all 8 coolers took {t_forecast:.2f}s"
        assert t_predict < 2.0, f"Predict all 8 coolers took {t_predict:.2f}s"

    asyncio.run(_test())


def test_async_client_error_handling(fastapi_app):
    """Verify CoolerDemandError is raised on 422 validation failure."""
    async def _test():
        transport = httpx.ASGITransport(app=fastapi_app)
        async with AsyncWaterDemandClient(base_url="http://testserver", transport=transport) as client:
            with pytest.raises(CoolerDemandError) as exc_info:
                # Missing required intervals will trigger 422
                await client._post("/api/predict", {"invalid_key": "bad_data"})
            assert exc_info.value.status_code == 422

    asyncio.run(_test())
