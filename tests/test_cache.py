"""Tests for in-memory TTL/LRU prediction cache and latency validation (< 2 ms)."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.cache import PredictionCache, compute_temp_band, make_lru_cached_predictor
from model.predictor import WaterDemandPredictor
from app import app as app_module


def test_temp_band_mapping():
    assert compute_temp_band(-5.0) == 0
    assert compute_temp_band(9.9) == 0
    assert compute_temp_band(10.0) == 1
    assert compute_temp_band(17.9) == 1
    assert compute_temp_band(18.0) == 2
    assert compute_temp_band(24.9) == 2
    assert compute_temp_band(25.0) == 3
    assert compute_temp_band(31.9) == 3
    assert compute_temp_band(32.0) == 4
    assert compute_temp_band(45.0) == 4


def test_cache_key_construction():
    key = PredictionCache.build_key(
        cooler_id=" Location01A ",
        day_of_week="tuesday",
        temp_band=3,
        slot_tuple=["P1", "P2", "Recess"],
    )
    assert key == ("Location01A", "Tuesday", 3, ("P1", "P2", "Recess"))

    # Passing float temp computes temp_band automatically
    key2 = PredictionCache.build_key(
        cooler_id="Location01A",
        day_of_week="Tuesday",
        temp_band=28.5,  # maps to band 3
        slot_tuple=("P1", "P2", "Recess"),
    )
    assert key2 == ("Location01A", "Tuesday", 3, ("P1", "P2", "Recess"))


def test_cache_hit_and_miss_tracking():
    cache = PredictionCache(maxsize=10, ttl_seconds=60.0)
    key = ("C01", "Monday", 3, ("P1", "P2"))

    # Initial miss
    assert cache.get(key) is None
    assert cache.misses == 1
    assert cache.hits == 0

    # Populate and hit
    cache.set(key, [{"expected_liters": 15.0}])
    result = cache.get(key)
    assert result == [{"expected_liters": 15.0}]
    assert cache.hits == 1
    assert cache.misses == 1

    info = cache.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["currsize"] == 1
    assert info["maxsize"] == 10


def test_cache_lru_eviction():
    cache = PredictionCache(maxsize=3, ttl_seconds=60.0)
    k1 = ("C01", "Monday", 1, ("P1",))
    k2 = ("C01", "Monday", 2, ("P1",))
    k3 = ("C01", "Monday", 3, ("P1",))
    k4 = ("C01", "Monday", 4, ("P1",))

    cache.set(k1, "v1")
    cache.set(k2, "v2")
    cache.set(k3, "v3")

    # Access k1 to make it most recently used
    assert cache.get(k1) == "v1"

    # Add k4 -> k2 should be evicted (as k2 was least recently used)
    cache.set(k4, "v4")
    assert len(cache) == 3
    assert cache.get(k2) is None  # Evicted
    assert cache.get(k1) == "v1"  # Retained
    assert cache.get(k3) == "v3"
    assert cache.get(k4) == "v4"


def test_cache_ttl_expiration():
    cache = PredictionCache(maxsize=10, ttl_seconds=0.1)  # 100ms TTL
    key = ("C01", "Monday", 2, ("P1",))
    cache.set(key, "data")

    # Immediate retrieval
    assert cache.get(key) == "data"

    # Wait for TTL to expire
    time.sleep(0.15)
    assert cache.get(key) is None
    assert key not in cache


def test_cache_thread_safety():
    cache = PredictionCache(maxsize=50, ttl_seconds=30.0)

    def worker(idx: int):
        key = (f"C0{idx % 5}", "Monday", idx % 3, ("P1", f"P{idx % 4}"))
        cache.set(key, f"val_{idx}")
        val = cache.get(key)
        return val is not None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(100)))

    assert all(results)
    assert len(cache) <= 50


def test_functools_lru_wrapper():
    calls = 0

    def mock_predict(cooler_id, day, band, slots):
        nonlocal calls
        calls += 1
        return [{"cooler_id": cooler_id, "band": band, "slots": slots}]

    lru_fn = make_lru_cached_predictor(mock_predict, maxsize=16)

    # First call - cache miss
    res1 = lru_fn("Location01A", "Monday", 3, ("P1", "P2"))
    assert calls == 1

    # Second call with same tuple - cache hit
    res2 = lru_fn("Location01A", "Monday", 3, ("P1", "P2"))
    assert calls == 1
    assert res1 == res2


def test_predictor_cache_latency_under_2ms():
    """Verify duplicate queries achieve instant response times (< 2 ms)."""
    predictor = WaterDemandPredictor()
    predictor.cooler_meta["Location01A"] = {"capacity_liters": 200, "floor": 1, "near_games": False}

    intervals = [
        {"period_slot": s, "day_of_week": "Monday"}
        for s in ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    ]

    # Warm-up / First call (populates cache)
    t0 = time.perf_counter()
    res1 = predictor.predict_intervals(
        cooler_id="Location01A",
        intervals=intervals,
        outdoor_temp_c=28.0,
        weather="Clear",
    )
    t_first = (time.perf_counter() - t0) * 1000

    # Second call (same temperature band: 28.5C is also band 3 [25..32))
    t1 = time.perf_counter()
    res2 = predictor.predict_intervals(
        cooler_id="Location01A",
        intervals=intervals,
        outdoor_temp_c=28.5,  # Same temp band 3
        weather="Clear",
    )
    t_cached = (time.perf_counter() - t1) * 1000

    assert len(res1) == len(res2) == 9
    assert predictor.cache.hits >= 1

    # The cached response time MUST be strictly < 2 ms (requirement)
    print(f"\n[Latency benchmark] First call: {t_first:.3f} ms | Cached duplicate: {t_cached:.4f} ms")
    assert t_cached < 2.0, f"Cached query took {t_cached:.3f} ms, expected < 2.0 ms"


def test_bypasses_dataframe_and_feature_transformation():
    """Verify that on cache hit, feature building and model inference are completely bypassed."""
    predictor = WaterDemandPredictor()
    predictor.cooler_meta["Location01A"] = {"capacity_liters": 200, "floor": 0, "near_games": False}

    intervals = [{"period_slot": "P1", "day_of_week": "Monday"}]

    # First call stores in cache
    predictor.predict_intervals("Location01A", intervals, outdoor_temp_c=30.0)

    # Patch build_features and predict_batch to ensure neither is called on cache hit
    with patch("model.predictor.build_features") as mock_feat, \
         patch.object(predictor.baseline, "predict_batch") as mock_base:
        cached_res = predictor.predict_intervals("Location01A", intervals, outdoor_temp_c=30.0)
        assert cached_res is not None
        assert mock_feat.call_count == 0
        assert mock_base.call_count == 0


def test_cache_cleared_on_metadata_or_refit():
    predictor = WaterDemandPredictor()
    intervals = [{"period_slot": "P1", "day_of_week": "Monday"}]
    predictor.predict_intervals("Location01A", intervals, outdoor_temp_c=28.0)
    assert len(predictor.cache) == 1

    # Invalidate via load_cooler_meta
    predictor.load_cooler_meta({"Location01A": {"capacity_liters": 250}})
    assert len(predictor.cache) == 0


def test_service_predict_and_cache_info_api():
    """Test /api/predict cache behavior and cache_info metrics."""
    payload = {
        "cooler": {"cooler_id": "Location01A", "capacity_liters": 200},
        "intervals": [
            {"period_slot": "P1", "day_of_week": "Monday"},
            {"period_slot": "Recess", "day_of_week": "Monday"},
        ],
        "weather": {"outdoor_temp_c": 26.5, "weather": "Clear"},
    }

    # Initial call
    r1 = app_module.service_predict(payload)
    assert r1.get("cooler_id") == "Location01A"

    # Duplicate call within same temperature band (26.5 and 29.0 are both band 3)
    payload_dup = dict(payload)
    payload_dup["weather"] = {"outdoor_temp_c": 29.0, "weather": "Clear"}

    t0 = time.perf_counter()
    r2 = app_module.service_predict(payload_dup)
    t_dup = (time.perf_counter() - t0) * 1000
    assert r2.get("cooler_id") == "Location01A"

    # Check cache info
    info_data = app_module.predictor.cache.cache_info()
    assert info_data["hits"] >= 1
    assert info_data["currsize"] >= 1
