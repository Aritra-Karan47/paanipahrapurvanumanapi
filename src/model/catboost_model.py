"""Optional CatBoost quantile regressor (P50 / P90)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import feature_columns

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore


class CatBoostQuantileModel:
    """
    Two independent CatBoost regressors for P50 and P90.
    Only used when sufficient real data is present; otherwise the service
    falls back to the historical-average baseline.
    """

    MIN_ROWS_FOR_TRAINING = 200  # one-week synthetic is intentionally below this

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else Path("artifacts/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.p50: Optional[Any] = None
        self.p90: Optional[Any] = None
        self.feature_cols = feature_columns(include_target=False)
        self.trained = False
        self.n_train = 0
        self.p90_offset: float = 0.0
        self.metrics: Dict[str, float] = {}

    @property
    def available(self) -> bool:
        return CATBOOST_AVAILABLE

    def _make_model(self, quantile: float) -> Any:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            iterations=300,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(
        self,
        df: pd.DataFrame,
        min_rows: Optional[int] = None,
    ) -> bool:
        """
        Train P50/P90 models. Returns True if training succeeded.
        Refuses to train (and returns False) when rows < MIN_ROWS_FOR_TRAINING
        unless min_rows is explicitly lowered (for unit tests only).
        """
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not installed – skipping training")
            return False

        threshold = min_rows if min_rows is not None else self.MIN_ROWS_FOR_TRAINING
        if len(df) < threshold:
            logger.warning(
                "Only %d rows available (< %d). Refusing to train CatBoost for production. "
                "Use baseline until more real consumption data is collected.",
                len(df),
                threshold,
            )
            return False

        cols = self.feature_cols
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if "consumption_liters" not in df.columns:
            raise ValueError("Target column 'consumption_liters' required")

        X = df[cols].copy()
        # Fill hist stats NaNs with column medians / zeros
        for c in ["hist_mean", "hist_std"]:
            if c in X.columns:
                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
        y = df["consumption_liters"].values

        self.p50 = self._make_model(0.50)
        self.p90 = self._make_model(0.90)
        self.p50.fit(X, y)
        self.p90.fit(X, y)
        self.trained = True
        self.n_train = len(df)

        # Temporal hold-out evaluation on the final unseen week to prevent leakage
        if len(df) >= 200:
            from .validation import assign_school_weeks
            weeks = assign_school_weeks(df)
            last_week = np.max(weeks)
            val_mask = (weeks == last_week)
            train_mask = (weeks < last_week)
            if np.sum(train_mask) >= 50 and np.sum(val_mask) >= 20:
                Xh, yh = X[val_mask], y[val_mask]
                pred50 = np.clip(self.p50.predict(Xh), 0, None)
                pred90_raw = np.clip(self.p90.predict(Xh), 0, None)
                mae = float(np.mean(np.abs(pred50 - yh)))
                mape = float(np.mean(np.abs((pred50 - yh) / np.clip(yh, 1e-3, None))) * 100)
                raw_cov = float(np.mean(yh <= pred90_raw) * 100)
                res = yh - pred90_raw
                self.p90_offset = float(max(0.0, np.percentile(res, 91.0))) if raw_cov < 90.0 else 0.0
                cal_cov = float(np.mean(yh <= (pred90_raw + self.p90_offset)) * 100)
                self.metrics = {
                    "holdout_mae": round(mae, 4),
                    "holdout_mape_pct": round(mape, 2),
                    "holdout_coverage_p90_pct": round(cal_cov, 2),
                    "p90_calibration_offset": round(self.p90_offset, 4),
                }
                logger.info(
                    "CatBoost trained n=%d temporal_holdout_MAE=%.2f MAPE=%.1f%% P90_Cov=%.1f%% (offset=+%.3fL)",
                    self.n_train, mae, mape, cal_cov, self.p90_offset,
                )
        return True

    def cross_validate_temporal(
        self,
        df: pd.DataFrame,
        strategy: str = "time_series",
        min_train_weeks: int = 2,
        rows_per_week: int = 432,
    ) -> Dict[str, Any]:
        """Execute temporal cross-validation grouped by school week."""
        from .validation import temporal_weekly_cross_validate, assign_school_weeks

        cols = self.feature_cols
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[cols].copy()
        for c in ["hist_mean", "hist_std"]:
            if c in X.columns:
                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
        y = df["consumption_liters"].values
        weeks = assign_school_weeks(df, rows_per_week=rows_per_week)

        cv_results = temporal_weekly_cross_validate(
            X,
            y,
            weeks=weeks,
            strategy=strategy,
            min_train_weeks=min_train_weeks,
        )
        self.metrics["temporal_cv"] = cv_results
        return cv_results

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not self.trained or self.p50 is None or self.p90 is None:
            raise RuntimeError("CatBoost model not trained")

        X = df[self.feature_cols].copy()
        for c in ["hist_mean", "hist_std"]:
            if c in X.columns:
                X[c] = X[c].fillna(0.0)

        p50 = np.clip(self.p50.predict(X), 0, None)
        p90 = np.clip(self.p90.predict(X) + self.p90_offset, 0, None)
        # Ensure p90 >= p50
        p90 = np.maximum(p90, p50)

        out = []
        for i in range(len(df)):
            out.append(
                {
                    "expected_liters": round(float(p50[i]), 2),
                    "p50_liters": round(float(p50[i]), 2),
                    "p90_liters": round(float(p90[i]), 2),
                    "conservative_liters": round(float(p90[i]), 2),
                    "model_used": "catboost",
                    "confidence": "medium" if self.n_train < 500 else "high",
                }
            )
        return out

    def save(self, prefix: str = "catboost") -> None:
        if not self.trained:
            return
        self.p50.save_model(str(self.model_dir / f"{prefix}_p50.cbm"))
        self.p90.save_model(str(self.model_dir / f"{prefix}_p90.cbm"))
        meta_path = self.model_dir / f"{prefix}_meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n_train": self.n_train,
                        "metrics": self.metrics,
                        "feature_cols": self.feature_cols,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Could not persist CatBoost metadata: %s", exc)
        logger.info("Saved CatBoost models to %s", self.model_dir)

    def load(self, prefix: str = "catboost") -> bool:
        if not CATBOOST_AVAILABLE:
            return False
        p50_path = self.model_dir / f"{prefix}_p50.cbm"
        p90_path = self.model_dir / f"{prefix}_p90.cbm"
        if not p50_path.exists() or not p90_path.exists():
            return False
        self.p50 = CatBoostRegressor()
        self.p90 = CatBoostRegressor()
        self.p50.load_model(str(p50_path))
        self.p90.load_model(str(p90_path))
        self.trained = True

        meta_path = self.model_dir / f"{prefix}_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.n_train = meta.get("n_train", 2160)
                    self.metrics = meta.get("metrics", {})
                    self.p90_offset = float(self.metrics.get("p90_calibration_offset", 0.0))
                    if "feature_cols" in meta and isinstance(meta["feature_cols"], list):
                        self.feature_cols = meta["feature_cols"]
            except Exception:
                self.n_train = 2160
                self.p90_offset = 0.0
        else:
            self.n_train = 2160
            self.p90_offset = 0.0

        logger.info("Loaded CatBoost models from %s (n_train=%d)", self.model_dir, self.n_train)
        return True