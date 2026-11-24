"""Tests for temporal weekly cross-validation and leakage prevention."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from model.validation import (
    assign_school_weeks,
    TemporalWeeklySplit,
    WeeklyGroupKFold,
    temporal_weekly_cross_validate,
)
from model.catboost_model import CatBoostQuantileModel
from model.features import build_features


def test_assign_school_weeks():
    """Verify 432-row school week assignment for 2,160 rows."""
    weeks = assign_school_weeks(2160, rows_per_week=432)
    assert len(weeks) == 2160
    assert np.min(weeks) == 1
    assert np.max(weeks) == 5

    # Each week should have exactly 432 rows
    for w in range(1, 6):
        assert np.sum(weeks == w) == 432

    # Honors explicit week column if present
    df = pd.DataFrame({"school_week": [1, 1, 2, 2, 3]})
    custom_weeks = assign_school_weeks(df)
    assert np.array_equal(custom_weeks, [1, 1, 2, 2, 3])


def test_temporal_weekly_split_strictly_forward_chaining():
    """Verify TemporalWeeklySplit train indices strictly precede test indices with zero leakage."""
    n_samples = 2160
    X = np.zeros((n_samples, 2))
    weeks = assign_school_weeks(n_samples, rows_per_week=432)

    cv = TemporalWeeklySplit(min_train_weeks=2, rows_per_week=432)
    assert cv.get_n_splits(X, groups=weeks) == 3

    folds = list(cv.split(X, groups=weeks))
    assert len(folds) == 3

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # 1. No overlap between train and test
        assert len(set(train_idx) & set(test_idx)) == 0

        # 2. Strict temporal forward chaining: all training weeks precede test week
        train_weeks = weeks[train_idx]
        test_weeks = weeks[test_idx]
        assert np.max(train_weeks) < np.min(test_weeks), (
            f"Leakage detected in fold {fold_idx}: train weeks {np.unique(train_weeks)} "
            f"not strictly prior to test weeks {np.unique(test_weeks)}"
        )

        # 3. Exactly one full week in test set (432 samples)
        assert len(test_idx) == 432

        # 4. Expanding training window
        expected_test_week = fold_idx + 3  # Weeks 3, 4, 5
        assert np.unique(test_weeks).tolist() == [expected_test_week]
        assert np.unique(train_weeks).tolist() == list(range(1, expected_test_week))


def test_weekly_group_kfold():
    """Verify WeeklyGroupKFold tests every week exactly once without cross-group leakage."""
    n_samples = 2160
    X = np.zeros((n_samples, 2))
    weeks = assign_school_weeks(n_samples, rows_per_week=432)

    cv = WeeklyGroupKFold(n_splits=5, rows_per_week=432)
    assert cv.get_n_splits(X, groups=weeks) == 5

    folds = list(cv.split(X, groups=weeks))
    assert len(folds) == 5

    tested_weeks = []
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        test_week = np.unique(weeks[test_idx])
        assert len(test_week) == 1
        tested_weeks.append(int(test_week[0]))

    # Every week 1 to 5 evaluated as an unseen test block
    assert sorted(tested_weeks) == [1, 2, 3, 4, 5]


def test_temporal_cross_validation_runner():
    """Verify temporal_weekly_cross_validate executes on synthetic dataset."""
    # Create 3-week synthetic feature matrix
    rows_per_week = 60
    total_rows = rows_per_week * 3
    weeks = assign_school_weeks(total_rows, rows_per_week=rows_per_week)

    np.random.seed(42)
    X = pd.DataFrame({
        "day_idx": np.random.randint(0, 6, total_rows),
        "slot_idx": np.random.randint(0, 9, total_rows),
        "is_recess": np.zeros(total_rows),
        "is_games": np.zeros(total_rows),
        "is_period": np.ones(total_rows),
        "games_intensity": np.zeros(total_rows),
        "duration_minutes": np.full(total_rows, 40),
        "outdoor_temp_c": np.random.uniform(20, 35, total_rows),
        "apparent_temp_c": np.random.uniform(20, 35, total_rows),
        "temp_band": np.random.randint(0, 5, total_rows),
        "weather_code": np.zeros(total_rows),
        "capacity_liters": np.full(total_rows, 200.0),
        "floor": np.zeros(total_rows),
        "near_games": np.zeros(total_rows),
        "active_students_in_zone": np.full(total_rows, 240),
        "hist_mean": np.random.uniform(5, 8, total_rows),
        "hist_std": np.random.uniform(0.5, 1.5, total_rows),
        "hist_count": np.full(total_rows, 5),
        "games_x_near": np.zeros(total_rows),
        "temp_x_recess": np.zeros(total_rows),
    })
    y = np.random.uniform(2.0, 9.0, total_rows)

    res = temporal_weekly_cross_validate(
        X,
        y,
        weeks=weeks,
        strategy="time_series",
        min_train_weeks=2,
        iterations=50,
        target_coverage=90.0,
    )

    assert "folds" in res
    assert res["n_folds"] == 1  # 3 weeks with min_train=2 -> 1 test fold
    assert res["mean_mae"] > 0
    assert res["mean_calibrated_p90_coverage_pct"] >= 90.0


def test_catboost_model_cross_validate_temporal():
    """Verify CatBoostQuantileModel.cross_validate_temporal executes and populates metrics."""
    model = CatBoostQuantileModel()

    # Synthetic 3-week data with consumption_liters
    rows = 180
    np.random.seed(42)
    df = pd.DataFrame({
        "day_idx": np.random.randint(0, 6, rows),
        "slot_idx": np.random.randint(0, 9, rows),
        "is_recess": np.zeros(rows),
        "is_games": np.zeros(rows),
        "is_period": np.ones(rows),
        "games_intensity": np.zeros(rows),
        "duration_minutes": np.full(rows, 40),
        "outdoor_temp_c": np.random.uniform(20, 35, rows),
        "apparent_temp_c": np.random.uniform(20, 35, rows),
        "temp_band": np.random.randint(0, 5, rows),
        "weather_code": np.zeros(rows),
        "capacity_liters": np.full(rows, 200.0),
        "floor": np.zeros(rows),
        "near_games": np.zeros(rows),
        "active_students_in_zone": np.full(rows, 240),
        "hist_mean": np.random.uniform(5, 8, rows),
        "hist_std": np.random.uniform(0.5, 1.5, rows),
        "hist_count": np.full(rows, 5),
        "games_x_near": np.zeros(rows),
        "temp_x_recess": np.zeros(rows),
        "consumption_liters": np.random.uniform(3.0, 8.0, rows),
    })

    cv = model.cross_validate_temporal(df, strategy="time_series", min_train_weeks=2, rows_per_week=60)
    assert "temporal_cv" in model.metrics
    assert cv["n_folds"] >= 1
    assert "mean_mae" in cv


def test_production_metadata_has_temporal_cv():
    """Verify artifacts/models/catboost_meta.json persists temporal CV results."""
    meta_path = Path("artifacts/models/catboost_meta.json")
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "temporal_cv" in meta
    t_cv = meta["temporal_cv"]
    assert t_cv["strategy"] == "time_series"
    assert t_cv["n_folds"] == 3
    assert t_cv["mean_mae"] < 1.5
    assert t_cv["mean_calibrated_p90_coverage_pct"] >= 90.0

    # Ensure each fold records honest evaluation on unseen future weeks
    for fold in t_cv["folds"]:
        assert max(fold["train_weeks"]) < min(fold["test_weeks"]), "Temporal leakage in persisted fold"
        assert fold["calibrated_p90_coverage_pct"] >= 90.0
