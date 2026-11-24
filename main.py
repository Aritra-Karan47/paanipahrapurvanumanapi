"""
Main entrypoint for FastAPI Cloud deployment.

Exposes REST endpoints under /api/* and mounts the Gradio UI on /.
"""
from __future__ import annotations

import logging
import os
import sys

# Disable Gradio telemetry for instant cold boots and fast testing
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr

# Ensure src is on python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from app.app import (
    bootstrap,
    create_demo,
    service_predict,
    service_forecast_day,
    service_temperature_target,
    service_check_shortage,
    service_validate_data,
    service_model_info,
    service_live_weather,
)

logger = logging.getLogger("water-demand-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Pre-warms and loads data files & baseline model into memory on server startup,
    ensuring ultra-fast, zero-overhead imports.
    """
    logger.info("FastAPI lifespan startup: Pre-warming models and loading institute data...")
    status = bootstrap()
    logger.info("FastAPI lifespan startup complete: %s", status)
    yield
    logger.info("FastAPI lifespan shutdown: Cleaning up resources...")


app = FastAPI(
    title="PaniPaharaPurvanuman API",
    description="School Water Cooler Demand Prediction & IS 1475:2024 Temperature Control REST API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for external SDKs and dashboard callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "PaniPaharaPurvanuman API"}


@app.post("/api/predict", tags=["Prediction"])
async def api_predict(request: Request):
    """Predict single interval cooler demand (P50/P90 quantiles)."""
    body = await request.json()
    result = await run_in_threadpool(service_predict, body)
    return JSONResponse(result)


@app.post("/api/forecast_day", tags=["Prediction"])
async def api_forecast(request: Request):
    """Generate 9-interval school day demand profile."""
    body = await request.json()
    result = await run_in_threadpool(service_forecast_day, body)
    return JSONResponse(result)


@app.post("/api/temperature_target", tags=["Temperature Control"])
async def api_temp(request: Request):
    """Get IS 1475:2024 compliant dispenser setpoints and cooling recommendations."""
    body = await request.json()
    result = await run_in_threadpool(service_temperature_target, body)
    return JSONResponse(result)


@app.post("/api/check_shortage", tags=["Shortage Evaluation"])
async def api_shortage(request: Request):
    """Evaluate reserve capacity risk and shortage alert."""
    body = await request.json()
    result = await run_in_threadpool(service_check_shortage, body)
    return JSONResponse(result)


@app.post("/api/validate_data", tags=["Validation"])
async def api_validate(request: Request):
    """Validate institution data files."""
    body = await request.json()
    result = await run_in_threadpool(service_validate_data, body)
    return JSONResponse(result)


@app.get("/api/model_info", tags=["Metadata"])
async def api_info():
    """Retrieve model training status, feature weights, and evaluation metrics."""
    result = await run_in_threadpool(service_model_info)
    return JSONResponse(result)


@app.post("/api/live_weather", tags=["Live Weather"])
async def api_weather_post(request: Request):
    """Fetch live ambient temperature and base station telemetry via Open-Meteo."""
    body = await request.json()
    result = await run_in_threadpool(service_live_weather, body)
    return JSONResponse(result)


@app.get("/api/live_weather", tags=["Live Weather"])
async def api_weather_get(city: str = "New Delhi", force_refresh: bool = False):
    """Fetch live ambient temperature for a given city with 8-hour cache."""
    result = await run_in_threadpool(service_live_weather, {"city": city, "force_refresh": force_refresh})
    return JSONResponse(result)


# Mount Gradio UI on root "/" (skipped automatically during unit tests for speed and test isolation)
if (
    "pytest" not in sys.modules
    and not os.environ.get("PYTEST_CURRENT_TEST")
    and os.environ.get("MOUNT_GRADIO", "True").lower() == "true"
):
    app = gr.mount_gradio_app(app, create_demo(), path="/", theme=gr.themes.Base())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
