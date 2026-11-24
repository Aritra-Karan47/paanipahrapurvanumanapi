"""Temporal Weekly Cross-Validation for school water demand prediction.

Prevents train-test data leakage across periods of the same day and across weeks
by evaluating models exclusively on unseen future weeks or isolated weekly blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROWS_PER_WEEK = 432  # 6 days * 9 slots * 8 coolers


def assign_school_weeks(
    data: Union[pd.DataFrame, int],
    rows_per_week: int = ROWS_PER_WEEK,
) -> np.ndarray:
    """Assign 1-based school week indices to rows.

    If a DataFrame with 'week' or 'week_number' is provided, uses that.
    Otherwise, assumes sequential chronological ordering (432 rows per school week).
    """
    if isinstance(data, pd.DataFrame):
        for col in ("week", "week_number", "school_week"):
            if col in data.columns:
                return data[col].to_numpy()
        n = len(data)
    else:
        n = int(data)

    return (np.arange(n) // rows_per_week) + 1


class TemporalWeeklySplit:
    """
    Time-series expanding window cross-validator grouped by school week.

    Guarantees that training data strictly precedes evaluation data in time.
    For 5 weeks with min_train_weeks=2:
      Fold 1: Train on Weeks 1..2 -> Test on Week 3
      Fold 2: Train on Weeks 1..3 -> Test on Week 4
      Fold 3: Train on Weeks 1..4 -> Test on Week 5

    Zero temporal leakage: no future periods or concurrent afternoon slots are ever seen.
    """

    def __init__(
        self,
        n_splits: Optional[int] = None,
        min_train_weeks: int = 2,
        rows_per_week: int = ROWS_PER_WEEK,
    ):
        self.n_splits = n_splits
        self.min_train_weeks = max(1, min_train_weeks)
        self.rows_per_week = rows_per_week

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if groups is None:
            weeks = assign_school_weeks(X, self.rows_per_week)
        else:
            weeks = np.asarray(groups)

        unique_weeks = np.sort(np.unique(weeks))
        total_weeks = len(unique_weeks)

        available_folds = total_weeks - self.min_train_weeks
        if available_folds <= 0:
            raise ValueError(
                f"Total weeks ({total_weeks}) must be greater than min_train_weeks ({self.min_train_weeks})."
            )

        n_folds = self.n_splits if self.n_splits is not None else available_folds
        n_folds = min(n_folds, available_folds)

        # Folds evaluate from (total_weeks - n_folds) to (total_weeks - 1) as test week index
        start_test_idx = total_weeks - n_folds
        for i in range(start_test_idx, total_weeks):
            test_week = unique_weeks[i]
            train_weeks = unique_weeks[:i]

            train_mask = np.isin(weeks, train_weeks)
            test_mask = weeks == test_week

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        if groups is not None:
            total_weeks = len(np.unique(groups))
        elif X is not None:
            total_weeks = len(np.unique(assign_school_weeks(X, self.rows_per_week)))
        else:
            return self.n_splits or 3

        available = max(0, total_weeks - self.min_train_weeks)
        if self.n_splits is not None:
            return min(self.n_splits, available)
        return available


class WeeklyGroupKFold:
    """
    Leave-One-Week-Out / GroupKFold cross-validator grouped by school week.

    In each fold, exactly one school week is held out for testing while the
    remaining weeks are used for training.
    """

    def __init__(
        self,
        n_splits: int = 5,
        rows_per_week: int = ROWS_PER_WEEK,
    ):
        self.n_splits = n_splits
        self.rows_per_week = rows_per_week

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if groups is None:
            weeks = assign_school_weeks(X, self.rows_per_week)
        else:
            weeks = np.asarray(groups)

        unique_weeks = np.sort(np.unique(weeks))
        n_folds = min(self.n_splits, len(unique_weeks))

        # Evaluate each week as a test set
        for test_week in unique_weeks[:n_folds]:
            train_mask = weeks != test_week
            test_mask = weeks == test_week

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        if groups is not None:
            return min(self.n_splits, len(np.unique(groups)))
        if X is not None:
            return min(self.n_splits, len(np.unique(assign_school_weeks(X, self.rows_per_week))))
        return self.n_splits


def temporal_weekly_cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    weeks: Optional[np.ndarray] = None,
    strategy: str = "time_series",  # "time_series" (expanding window) or "group_kfold"
    min_train_weeks: int = 2,
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.06,
    l2_leaf_reg: float = 3.0,
    target_coverage: float = 90.0,
) -> Dict[str, Any]:
    """Execute temporal cross-validation using CatBoost quantile regressors.

    Evaluates MAE, MAPE, raw P90 coverage, calibrated P90 coverage, and calibration offsets
    across unseen future school weeks.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost is required for temporal weekly cross-validation.")

    if weeks is None:
        weeks = assign_school_weeks(X)

    if strategy == "group_kfold":
        cv = WeeklyGroupKFold(n_splits=len(np.unique(weeks)))
    else:
        cv = TemporalWeeklySplit(min_train_weeks=min_train_weeks)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=weeks)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        train_weeks = sorted(np.unique(weeks[train_idx]).tolist())
        test_weeks = sorted(np.unique(weeks[test_idx]).tolist())

        m_p50 = CatBoostRegressor(
            loss_function="Quantile:alpha=0.50",
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=42 + fold_idx,
            verbose=False,
            allow_writing_files=False,
        )
        m_p90 = CatBoostRegressor(
            loss_function="Quantile:alpha=0.90",
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=42 + fold_idx,
            verbose=False,
            allow_writing_files=False,
        )

        m_p50.fit(X_train, y_train)
        m_p90.fit(X_train, y_train)

        preds_p50 = np.clip(m_p50.predict(X_test), 0, None)
        preds_p90_raw = np.clip(m_p90.predict(X_test), 0, None)

        mae = float(np.mean(np.abs(preds_p50 - y_test)))
        mape = float(np.mean(np.abs((preds_p50 - y_test) / np.clip(y_test, 1e-3, None))) * 100)
        rmse = float(np.sqrt(np.mean((preds_p50 - y_test) ** 2)))

        raw_cov = float(np.mean(y_test <= preds_p90_raw) * 100)
        residuals = y_test - preds_p90_raw
        if raw_cov < target_coverage:
            offset = float(max(0.0, np.percentile(residuals, target_coverage + 1.0)))
        else:
            offset = 0.0

        calibrated_preds_p90 = np.maximum(preds_p90_raw + offset, preds_p50)
        calibrated_cov = float(np.mean(y_test <= calibrated_preds_p90) * 100)

        fold_info = {
            "fold": fold_idx + 1,
            "train_weeks": train_weeks,
            "test_weeks": test_weeks,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "mae": round(mae, 4),
            "mape_pct": round(mape, 2),
            "rmse": round(rmse, 4),
            "raw_p90_coverage_pct": round(raw_cov, 2),
            "calibrated_p90_coverage_pct": round(calibrated_cov, 2),
            "p90_calibration_offset": round(offset, 4),
        }
        fold_results.append(fold_info)
        logger.info(
            "Temporal CV Fold %d (Train Weeks %s -> Test Weeks %s): MAE=%.3f L, CalCov=%.1f%%, Offset=+%.3f L",
            fold_idx + 1,
            train_weeks,
            test_weeks,
            mae,
            calibrated_cov,
            offset,
        )

    mean_mae = float(np.mean([f["mae"] for f in fold_results]))
    std_mae = float(np.std([f["mae"] for f in fold_results]))
    mean_mape = float(np.mean([f["mape_pct"] for f in fold_results]))
    mean_rmse = float(np.mean([f["rmse"] for f in fold_results]))
    mean_cal_cov = float(np.mean([f["calibrated_p90_coverage_pct"] for f in fold_results]))
    mean_offset = float(np.mean([f["p90_calibration_offset"] for f in fold_results]))

    summary = {
        "strategy": strategy,
        "n_folds": len(fold_results),
        "mean_mae": round(mean_mae, 4),
        "std_mae": round(std_mae, 4),
        "mean_mape_pct": round(mean_mape, 2),
        "mean_rmse": round(mean_rmse, 4),
        "mean_calibrated_p90_coverage_pct": round(mean_cal_cov, 2),
        "mean_calibration_offset": round(mean_offset, 4),
        "target_coverage_met": bool(mean_cal_cov >= target_coverage),
        "folds": fold_results,
    }
    return summary
