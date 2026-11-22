"""Baseline, features, predictor, shortage, temperature tests against real data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.baseline import HistoricalAverageBaseline
from model.features import build_features, build_single_feature_row
from model.predictor import WaterDemandPredictor
from model.preprocessing import load_consumption_record, load_cooler_config
from model.catboost_model import CatBoostQuantileModel
from temp_control.controller import TemperatureController

DATA = ROOT / "data"


@pytest.fixture(scope="module")
def cons_df():
    return load_consumption_record(DATA / "Cooler_Water_Consumption_Record.csv")


@pytest.fixture(scope="module")
def cooler_meta():
    return load_cooler_config(DATA / "cooler_config.csv")


def test_feature_row_recess_and_games():
    r = build_single_feature_row(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="Recess",
        outdoor_temp_c=30.0,
    )
    assert r["is_recess"] == 1
    assert r["is_games"] == 0
    g = build_single_feature_row(
        cooler_id="Location04A",
        day_of_week="Tuesday",
        period_slot="P4",
        outdoor_temp_c=28.0,
        games_intensity="High",
        near_games=True,
    )
    assert g["games_intensity"] == 3
    assert g["games_x_near"] == 3


def test_baseline_fit_and_predict(cons_df, cooler_meta):
    bl = HistoricalAverageBaseline()
    bl.fit(cons_df, cooler_meta=cooler_meta)
    assert bl.fitted
    exp, src, cnt = bl.predict_one("Location01A", "Monday", "Recess", outdoor_temp_c=28.0)
    assert exp > 0
    assert src in ("cooler_day_slot", "cooler_slot", "global_slot", "engineered_default")
    assert cnt >= 1


def test_baseline_fallback_unknown_cooler(cooler_meta):
    bl = HistoricalAverageBaseline()
    bl._cooler_meta = cooler_meta
    exp, src, cnt = bl.predict_one("UNKNOWN", "Monday", "P1", 25.0)
    assert exp > 0
    assert src == "engineered_default"


def test_predictor_all_eight_coolers(cons_df, cooler_meta):
    pred = WaterDemandPredictor()
    pred.load_cooler_meta(cooler_meta)
    pred.fit_from_consumption(cons_df, train_catboost=False)
    intervals = [
        {"period_slot": s, "day_of_week": "Monday", "games_intensity": "None", "duration_minutes": 40}
        for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    ]
    for cid in cooler_meta:
        out = pred.predict_intervals(cid, intervals, outdoor_temp_c=27.0, weather="Clear")
        assert len(out) == 9
        assert all(p["expected_liters"] >= 0 for p in out)


def test_catboost_refuses_under_threshold(cons_df, cooler_meta):
    model = CatBoostQuantileModel()
    feat = build_features(cons_df.to_dict(orient="records"), cooler_meta=cooler_meta, historical_df=cons_df)
    ok = model.fit(feat, min_rows=50000)
    assert ok is False


def test_catboost_trains_when_forced(cons_df, cooler_meta):
    model = CatBoostQuantileModel()
    feat = build_features(cons_df.to_dict(orient="records"), cooler_meta=cooler_meta, historical_df=cons_df)
    ok = model.fit(feat, min_rows=50)
    if not model.available:
        return
    assert ok is True
    preds = model.predict(feat.head(5))
    assert len(preds) == 5
    assert preds[0]["p90_liters"] >= preds[0]["p50_liters"]


def test_shortage_levels():
    pred = WaterDemandPredictor()
    r = pred.compute_shortage(200, 180, 20, 25)
    assert r["risk_level"] == "none"
    r2 = pred.compute_shortage(200, 30, 80, 100)
    assert r2["risk_level"] in ("high", "critical")
    assert r2["shortage_liters"] > 0


def test_temperature_boundaries_is1475():
    ctrl = TemperatureController()
    ctrl.load_from_file(DATA / "Water_Temperature_Control_Table.xlsx")
    # < 20 → cooling off
    r = ctrl.get_target(15)
    assert r["cooling_enabled"] is False
    assert r["target_water_temp_c"] is None
    # 33.5 → band 4, target 14
    r2 = ctrl.get_target(33.5)
    assert r2["cooling_enabled"] is True
    assert r2["target_water_temp_c"] == 14.0
    # 43 → band 6, target 20
    r3 = ctrl.get_target(43)
    assert r3["target_water_temp_c"] == 20.0
    # 50 → outside, target 21
    r4 = ctrl.get_target(50)
    assert r4["target_water_temp_c"] == 21.0


def test_malformed_missing_inputs(cons_df):
    bl = HistoricalAverageBaseline()
    bl.fit(cons_df)
    exp, src, _ = bl.predict_one("Location01A", "Holiday", "P1", 25.0)
    assert exp > 0
