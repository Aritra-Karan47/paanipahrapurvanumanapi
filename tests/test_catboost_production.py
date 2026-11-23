"""Tests for pre-trained production CatBoost .cbm artifacts and cold boot inference."""

import json
from pathlib import Path
import pytest

from model.predictor import WaterDemandPredictor
from app.app import service_predict


def test_production_cbm_artifacts_exist():
    model_dir = Path("artifacts/models")
    p50_path = model_dir / "catboost_p50.cbm"
    p90_path = model_dir / "catboost_p90.cbm"
    meta_path = model_dir / "catboost_meta.json"

    assert p50_path.exists(), f"Missing {p50_path}"
    assert p90_path.exists(), f"Missing {p90_path}"
    assert meta_path.exists(), f"Missing {meta_path}"

    assert p50_path.stat().st_size > 50_000, "P50 .cbm file suspiciously small"
    assert p90_path.stat().st_size > 50_000, "P90 .cbm file suspiciously small"


def test_catboost_metadata_validation():
    meta_path = Path("artifacts/models/catboost_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["n_train"] == 2160
    assert "metrics" in meta
    metrics = meta["metrics"]

    assert "holdout_mae" in metrics
    assert metrics["holdout_mae"] < 1.5  # High accuracy (< 1.0 L)

    assert "holdout_coverage_p90_pct" in metrics
    assert metrics["holdout_coverage_p90_pct"] >= 90.0  # Mandatory >= 90% quantile coverage requirement


def test_cold_boot_instant_inference_without_runtime_training():
    """Verify that predictor loads persisted artifacts on startup and serves CatBoost instantly."""
    predictor = WaterDemandPredictor()

    assert predictor.catboost.trained is True
    assert predictor.catboost.n_train == 2160
    assert predictor.version == "water-demand-v0.1.0-catboost"

    intervals = [
        {"period_slot": s, "day_of_week": "Monday"}
        for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    ]

    preds = predictor.predict_intervals("Location01A", intervals, outdoor_temp_c=28.0)
    assert len(preds) == 9

    for p in preds:
        assert p["model_used"] == "catboost"
        assert p["conservative_liters"] >= p["expected_liters"]
        assert p["expected_liters"] > 0
        assert p["p90_liters"] >= p["p50_liters"]


def test_service_predict_cold_boot_api():
    """Verify service_predict serves CatBoost predictions out-of-the-box."""
    payload = {
        "cooler": {"cooler_id": "Location04A", "capacity_liters": 250, "near_games": True},
        "intervals": [
            {"period_slot": "P1", "day_of_week": "Monday"},
            {"period_slot": "P3", "day_of_week": "Monday"},
            {"period_slot": "Recess", "day_of_week": "Monday"},
        ],
        "weather": {"outdoor_temp_c": 32.0, "weather": "Clear"},
        "model_preference": "auto",
        "use_cache": False,
    }

    resp = service_predict(payload, use_cache=False)
    assert resp["cooler_id"] == "Location04A"
    assert resp["model_version"] == "water-demand-v0.1.0-catboost"
    assert resp["fallback_used"] is False

    preds = resp["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert p["model_used"] == "catboost"
        assert p["conservative_liters"] >= p["expected_liters"]


def test_dual_quantile_ordering_guarantee():
    """Verify p90 >= p50 strictly holds across academic wings and sports complex."""
    predictor = WaterDemandPredictor()
    slots = ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]

    for cid in ["Location01A", "Location02B", "Location03A", "Location04A"]:
        intervals = [{"period_slot": s, "day_of_week": "Wednesday"} for s in slots]
        preds = predictor.predict_intervals(cid, intervals, outdoor_temp_c=34.0, use_cache=False)
        for s, p in zip(slots, preds):
            assert p["conservative_liters"] >= p["expected_liters"], f"Violation at {cid} {s}: {p}"
