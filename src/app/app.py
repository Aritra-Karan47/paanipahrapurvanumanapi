"""
Gradio application for Hugging Face Space.

Exposes both a friendly UI and REST-style JSON endpoints under /api/*
so the cooler-demand SDK can call it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

try:
    from fastapi import Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
except ImportError:
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
        import asyncio
        run_in_threadpool = asyncio.to_thread  # type: ignore
    except ImportError:
        Request = Any  # type: ignore
        run_in_threadpool = None  # type: ignore
        JSONResponse = None  # type: ignore

# Ensure src is on path when running as Space
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model.predictor import WaterDemandPredictor
from model.preprocessing import (
    load_consumption_record,
    load_cooler_config,
    load_temperature_table,
    load_timetable,
    extract_day_intervals,
)
from temp_control.controller import TemperatureController
from weather.service import fetch_live_weather

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("water-demand-app")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global service state (stateless per request; models loaded once)
# ---------------------------------------------------------------------------
predictor = WaterDemandPredictor(model_dir=MODEL_DIR)
temp_ctrl = TemperatureController()


def bootstrap() -> None:
    """Load institute data and fit baseline at startup."""
    try:
        cooler_path = DATA_DIR / "cooler_config.csv"
        if cooler_path.exists():
            meta = load_cooler_config(cooler_path)
            predictor.load_cooler_meta(meta)
            logger.info("Loaded %d cooler configs", len(meta))

        cons_path = DATA_DIR / "Cooler_Water_Consumption_Record.csv"
        if not cons_path.exists():
            cons_path = DATA_DIR / "Cooler_Water_Consumption_Record.xlsx"
        if cons_path.exists():
            df = load_consumption_record(cons_path)
            # Real multi-week data is still modest; keep CatBoost optional
            info = predictor.fit_from_consumption(df, train_catboost=False)
            logger.info("Baseline fit: %s", info)
        else:
            logger.warning("No consumption record found – using engineered defaults only")

        temp_path = DATA_DIR / "Water_Temperature_Control_Table.xlsx"
        if temp_path.exists():
            temp_ctrl.load_from_file(temp_path)
            logger.info("Temperature control table loaded from %s", temp_path)
    except Exception as exc:
        logger.exception("Bootstrap warning: %s", exc)


bootstrap()


# ---------------------------------------------------------------------------
# Core service functions (used by both Gradio UI and API wrappers)
# ---------------------------------------------------------------------------

def service_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    cooler = payload.get("cooler", {})
    cooler_id = cooler.get("cooler_id", "C01")
    intervals = payload.get("intervals", [])
    weather = payload.get("weather", {})
    outdoor = float(weather.get("outdoor_temp_c", 25.0))
    weather_name = weather.get("weather", "Unknown")
    if hasattr(weather_name, "value"):
        weather_name = weather_name.value
    pref = payload.get("model_preference", "auto")

    # Ensure cooler meta is present for this request
    if cooler_id not in predictor.cooler_meta and cooler.get("capacity_liters"):
        predictor.cooler_meta[cooler_id] = {
            "capacity_liters": float(cooler["capacity_liters"]),
            "floor": int(cooler.get("floor", 0)),
            "near_games": bool(cooler.get("near_games", False)),
            "location": cooler.get("location", "unknown"),
        }

    iv_list = []
    for iv in intervals:
        iv_list.append(
            {
                "period_slot": iv.get("period_slot", "P1"),
                "day_of_week": iv.get("day_of_week", "Monday"),
                "games_intensity": iv.get("games_intensity", "None"),
                "duration_minutes": iv.get("duration_minutes", 40),
            }
        )

    preds = predictor.predict_intervals(
        cooler_id=cooler_id,
        intervals=iv_list,
        outdoor_temp_c=outdoor,
        weather=str(weather_name),
        model_preference=pref,
    )

    results = []
    total_exp = 0.0
    total_cons = 0.0
    fallback = False
    for iv, p in zip(iv_list, preds):
        results.append(
            {
                "period_slot": iv["period_slot"],
                "expected_liters": p["expected_liters"],
                "p50_liters": p.get("p50_liters"),
                "p90_liters": p.get("p90_liters"),
                "conservative_liters": p["conservative_liters"],
                "model_used": p["model_used"],
                "confidence": p.get("confidence", "low"),
            }
        )
        total_exp += p["expected_liters"]
        total_cons += p["conservative_liters"]
        if "baseline" in p["model_used"] or "default" in p["model_used"]:
            fallback = True

    return {
        "request_id": payload.get("request_id"),
        "cooler_id": cooler_id,
        "predictions": results,
        "total_expected_liters": round(total_exp, 2),
        "total_conservative_liters": round(total_cons, 2),
        "model_version": predictor.version,
        "fallback_used": fallback,
        "warnings": [],
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }


def service_forecast_day(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = service_predict(payload)
    cooler = payload.get("cooler", {})
    capacity = float(cooler.get("capacity_liters", 200))
    current = payload.get("current_level_liters")
    if current is None:
        current = cooler.get("current_level_liters")
    if current is None:
        current = capacity * 0.7  # assume 70% full if unknown

    shortage = predictor.compute_shortage(
        capacity_liters=capacity,
        current_level_liters=float(current),
        expected_demand=base["total_expected_liters"],
        conservative_demand=base["total_conservative_liters"],
        next_refill_hours=None,
    )
    base["day_of_week"] = payload.get("day_of_week", payload.get("intervals", [{}])[0].get("day_of_week", "Monday"))
    base["shortage"] = shortage
    return base


def service_temperature_target(payload: Dict[str, Any]) -> Dict[str, Any]:
    outdoor = float(payload.get("outdoor_temp_c", 25.0))
    result = temp_ctrl.get_target(outdoor)
    return {
        "outdoor_temp_c": result["outdoor_temp_c"],
        "target_water_temp_c": result["target_water_temp_c"],
        "cooling_enabled": result.get("cooling_enabled"),
        "setpoint_min_c": result.get("setpoint_min_c"),
        "setpoint_max_c": result.get("setpoint_max_c"),
        "rule_applied": result["rule_applied"],
        "action": result.get("action"),
        "notes": result.get("notes"),
        "model_version": "temp-control-v1",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }


def service_check_shortage(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = predictor.compute_shortage(
        capacity_liters=float(payload["capacity_liters"]),
        current_level_liters=float(payload["current_level_liters"]),
        expected_demand=float(payload["expected_demand_liters"]),
        conservative_demand=float(payload["conservative_demand_liters"]),
        next_refill_hours=payload.get("next_refill_hours"),
    )
    return {
        "cooler_id": payload.get("cooler_id", "unknown"),
        "risk_level": result["risk_level"],
        "projected_level": result["projected_level_after_day"],
        "shortage_liters": result["shortage_liters"],
        "message": result["message"],
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }


def service_validate_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    cooler = payload.get("cooler")
    if cooler:
        if not cooler.get("cooler_id"):
            errors.append("cooler.cooler_id is required")
        cap = cooler.get("capacity_liters")
        if cap is None or float(cap) <= 0:
            errors.append("cooler.capacity_liters must be > 0")
        summary["cooler_id"] = cooler.get("cooler_id")

    intervals = payload.get("intervals") or []
    if intervals:
        slots = [iv.get("period_slot") for iv in intervals]
        summary["n_intervals"] = len(intervals)
        summary["slots"] = slots
        if "Recess" not in slots and "recess" not in [str(s).lower() for s in slots]:
            warnings.append("No Recess interval found – timetable may be incomplete")
    else:
        warnings.append("No intervals provided")

    weather = payload.get("weather")
    if weather:
        t = weather.get("outdoor_temp_c")
        if t is None:
            errors.append("weather.outdoor_temp_c is required")
        elif not (-20 <= float(t) <= 55):
            errors.append("outdoor_temp_c out of plausible range [-20, 55]")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def service_model_info() -> Dict[str, Any]:
    return predictor.model_info()


def service_live_weather(payload: Dict[str, Any]) -> Dict[str, Any]:
    city = payload.get("city")
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    force = bool(payload.get("force_refresh", False))
    return fetch_live_weather(city=city, lat=lat, lon=lon, force_refresh=force)


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------

def ui_detect_weather(city: str, force_refresh: bool = False):
    res = fetch_live_weather(city=city, force_refresh=force_refresh)
    temp = res.get("outdoor_temp_c", 28.0)
    cond = res.get("weather", "Clear")
    st_info = res.get("station_info", "")
    loc = res.get("location_name", city or "Location")

    if res.get("cached"):
        age_str = f"Cached ({res.get('cache_age_hours', 0)}h old, valid for {res.get('cache_ttl_remaining_hours', 8)}h)"
    else:
        age_str = "Freshly fetched from Open-Meteo (8-hour cache initialized)"

    status_md = (
        f"**Live Weather Detected**: **{temp} °C**, **{cond}** ({res.get('humidity_pct', 50)}% humidity)\n\n"
        f"• **Location**: {loc}\n"
        f"• **Source**: {res.get('source', 'Open-Meteo')}\n"
        f"• **Base Station**: {st_info}\n"
        f"• **Cache**: {age_str}\n\n"
        f"_Slider below is auto-updated, but you can still drag it manually to simulate what-if scenarios._"
    )
    return temp, cond, status_md


def ui_predict(
    cooler_id: str,
    capacity: float,
    near_games: bool,
    day: str,
    slots_csv: str,
    outdoor_temp: float,
    weather: str,
    current_level: float,
):
    slots = [s.strip() for s in slots_csv.split(",") if s.strip()]
    intervals = [
        {
            "period_slot": s,
            "day_of_week": day,
            "games_intensity": "Moderate" if s == "Games" else "None",
            "duration_minutes": 20 if s == "Recess" else (45 if s == "Games" else 40),
        }
        for s in slots
    ]
    payload = {
        "cooler": {
            "cooler_id": cooler_id,
            "capacity_liters": capacity,
            "near_games": near_games,
            "current_level_liters": current_level,
        },
        "intervals": intervals,
        "weather": {"outdoor_temp_c": outdoor_temp, "weather": weather},
        "day_of_week": day,
        "current_level_liters": current_level,
        "model_preference": "auto",
    }
    result = service_forecast_day(payload)
    lines = [
        f"**Cooler** {result['cooler_id']}  |  Model: `{result['model_version']}`",
        f"Total expected: **{result['total_expected_liters']:.1f} L**  |  Conservative: **{result['total_conservative_liters']:.1f} L**",
        "",
        "| Slot | Expected (L) | Conservative (L) | Model | Confidence |",
        "|------|-------------:|-----------------:|-------|------------|",
    ]
    for p in result["predictions"]:
        lines.append(
            f"| {p['period_slot']} | {p['expected_liters']:.2f} | {p['conservative_liters']:.2f} "
            f"| {p['model_used']} | {p['confidence']} |"
        )
    if result.get("shortage"):
        s = result["shortage"]
        lines += [
            "",
            f"### Shortage Risk: **{s['risk_level'].upper()}**",
            s["message"],
            f"Projected level after day: {s['projected_level_after_day']:.1f} L",
        ]
    return "\n".join(lines), json.dumps(result, indent=2)


def ui_temp(outdoor: float):
    r = service_temperature_target({"outdoor_temp_c": outdoor})
    return f"Target water temperature: **{r['target_water_temp_c']} °C**\n\nRule: `{r['rule_applied']}`"


def ui_info():
    return json.dumps(service_model_info(), indent=2)


# ---------------------------------------------------------------------------
# Gradio Blocks + API endpoints
# ---------------------------------------------------------------------------

with gr.Blocks(title="School Water Demand Prediction") as demo:
    gr.Markdown(
        """
        # School Water Cooler Demand Prediction
        Period-wise water demand (including **Recess** after P4 and **Games**).
        Uses a robust historical-average baseline; CatBoost quantiles activate only after sufficient real data.
        Temperature control is deterministic and separate from ML.
        """
    )

    with gr.Tab("Forecast"):
        with gr.Row():
            cooler_id = gr.Textbox(value="Location01A", label="Cooler ID")
            capacity = gr.Number(value=200, label="Capacity (L)")
            near_games = gr.Checkbox(value=False, label="Near Games area")
            current_level = gr.Number(value=140, label="Current level (L)")
        with gr.Row():
            day = gr.Dropdown(
                choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                value="Monday",
                label="Day",
            )
            slots = gr.Textbox(
                value="P1,P2,P3,P4,Recess,P5,P6,P7,P8",
                label="Period slots (comma-separated)",
            )
        with gr.Accordion("Auto-Detect Live Weather (Open-Meteo · Nearest Base Station · 8h Cache)", open=True):
            with gr.Row():
                weather_city = gr.Textbox(value="New Delhi", label="School City / Location", scale=3)
                detect_btn = gr.Button("Detect Weather (8h Cache)", variant="secondary", scale=2)
                force_btn = gr.Button("Force Refresh API", variant="outline", scale=1)
            weather_status = gr.Markdown(
                "_Click **Detect Weather** to auto-set the slider from Open-Meteo's nearest base station, or adjust manually below._"
            )

        with gr.Row():
            outdoor = gr.Slider(-5, 45, value=28, step=0.5, label="Outdoor temp (°C)")
            weather = gr.Dropdown(
                choices=["Clear", "Cloudy", "Rainy", "Humid", "Unknown"],
                value="Clear",
                label="Weather",
            )

        detect_btn.click(
            fn=lambda c: ui_detect_weather(c, force_refresh=False),
            inputs=[weather_city],
            outputs=[outdoor, weather, weather_status],
        )
        force_btn.click(
            fn=lambda c: ui_detect_weather(c, force_refresh=True),
            inputs=[weather_city],
            outputs=[outdoor, weather, weather_status],
        )
        btn = gr.Button("Predict", variant="primary")
        md_out = gr.Markdown()
        json_out = gr.Code(language="json", label="Raw JSON response")
        btn.click(
            ui_predict,
            inputs=[cooler_id, capacity, near_games, day, slots, outdoor, weather, current_level],
            outputs=[md_out, json_out],
        )

    with gr.Tab("Temperature Target"):
        with gr.Row():
            t_city = gr.Textbox(value="New Delhi", label="School Location", scale=3)
            t_detect_btn = gr.Button("Apply Live Temp (Open-Meteo)", scale=2)
        t_in = gr.Slider(-10, 50, value=30, step=0.5, label="Outdoor temperature (°C)")
        t_btn = gr.Button("Get target water temperature", variant="primary")
        t_out = gr.Markdown()
        t_btn.click(ui_temp, inputs=[t_in], outputs=[t_out])

        def _apply_live_temp_to_target(city: str):
            res = fetch_live_weather(city=city)
            t = res.get("outdoor_temp_c", 30.0)
            st = res.get("station_info", "")
            return t, ui_temp(t) + f"\n\n_Auto-detected from {st}_"

        t_detect_btn.click(_apply_live_temp_to_target, inputs=[t_city], outputs=[t_in, t_out])

    with gr.Tab("Model Info"):
        info_btn = gr.Button("Refresh model info")
        info_out = gr.Code(language="json")
        info_btn.click(ui_info, outputs=[info_out])
        demo.load(ui_info, outputs=[info_out])

    gr.Markdown(
        """
        ---
        **API** (for `cooler-demand` SDK):  
        `POST /api/predict`, `/api/forecast_day`, `/api/temperature_target`,  
        `/api/check_shortage`, `/api/validate_data`, `GET /api/model_info`
        """
    )

# Mount simple JSON API routes via Gradio's underlying FastAPI app when available.
# Gradio 4+ exposes .app (Starlette/FastAPI). We attach routes after launch in
# Spaces; for local we also register them here if possible.

def _attach_api(blocks: gr.Blocks) -> None:
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
        try:
            from fastapi.concurrency import run_in_threadpool
        except ImportError:
            import asyncio
            run_in_threadpool = asyncio.to_thread  # type: ignore

        app = blocks.app

        @app.post("/api/predict")
        async def api_predict(request: Request):
            body = await request.json()
            result = await run_in_threadpool(service_predict, body)
            return JSONResponse(result)

        @app.post("/api/forecast_day")
        async def api_forecast(request: Request):
            body = await request.json()
            result = await run_in_threadpool(service_forecast_day, body)
            return JSONResponse(result)

        @app.post("/api/temperature_target")
        async def api_temp(request: Request):
            body = await request.json()
            return JSONResponse(service_temperature_target(body))

        @app.post("/api/check_shortage")
        async def api_shortage(request: Request):
            body = await request.json()
            return JSONResponse(service_check_shortage(body))

        @app.post("/api/validate_data")
        async def api_validate(request: Request):
            body = await request.json()
            return JSONResponse(service_validate_data(body))

        @app.get("/api/model_info")
        async def api_info():
            return JSONResponse(service_model_info())

        @app.post("/api/live_weather")
        async def api_weather(request: Request):
            body = await request.json()
            return JSONResponse(service_live_weather(body))

        @app.get("/api/live_weather")
        async def api_weather_get(city: str = "New Delhi", force_refresh: bool = False):
            return JSONResponse(service_live_weather({"city": city, "force_refresh": force_refresh}))

        logger.info("REST API routes attached under /api/*")
    except Exception as exc:
        logger.warning("Could not attach FastAPI routes (UI-only mode): %s", exc)


_attach_api(demo)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))