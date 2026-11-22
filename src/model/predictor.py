"""Orchestrates baseline + optional CatBoost and shortage logic."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .baseline import HistoricalAverageBaseline
from .catboost_model import CatBoostQuantileModel
from .features import build_features, build_single_feature_row

logger = logging.getLogger(__name__)

MODEL_VERSION = "water-demand-v0.1.0-baseline"


class WaterDemandPredictor:
    """
    Production predictor.

    - Always has a fitted baseline (or engineered defaults).
    - CatBoost is used only when explicitly preferred AND a trained model exists
      with sufficient data history. One-week synthetic data is NOT treated as
      sufficient for production CatBoost.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else Path("artifacts/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = HistoricalAverageBaseline()
        self.catboost = CatBoostQuantileModel(model_dir=self.model_dir)
        self.cooler_meta: Dict[str, Dict[str, Any]] = {}
        self.historical_df: Optional[pd.DataFrame] = None
        self.version = MODEL_VERSION

    def load_cooler_meta(self, meta: Dict[str, Dict[str, Any]]) -> None:
        self.cooler_meta = meta
        self.baseline._cooler_meta = meta

    def fit_from_consumption(
        self,
        df: pd.DataFrame,
        train_catboost: bool = False,
        catboost_min_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit baseline always; CatBoost only if train_catboost and enough rows."""
        self.historical_df = df.copy()
        self.baseline.fit(df, cooler_meta=self.cooler_meta)
        info = {
            "baseline_fitted": True,
            "baseline_n": len(df),
            "catboost_trained": False,
            "catboost_n": 0,
        }
        if train_catboost:
            feat_df = build_features(
                df.to_dict(orient="records"),
                cooler_meta=self.cooler_meta,
                historical_df=df,
            )
            ok = self.catboost.fit(feat_df, min_rows=catboost_min_rows)
            info["catboost_trained"] = ok
            info["catboost_n"] = self.catboost.n_train
            if ok:
                self.catboost.save()
                self.version = "water-demand-v0.1.0-catboost"
        return info

    def _resolve_model_preference(self, preference: str) -> str:
        pref = (preference or "auto").lower()
        if pref == "baseline":
            return "baseline"
        if pref == "catboost":
            if self.catboost.trained:
                return "catboost"
            logger.info("CatBoost requested but not trained – falling back to baseline")
            return "baseline"
        # auto
        if self.catboost.trained and self.catboost.n_train >= CatBoostQuantileModel.MIN_ROWS_FOR_TRAINING:
            return "catboost"
        return "baseline"

    def predict_intervals(
        self,
        cooler_id: str,
        intervals: List[Dict[str, Any]],
        outdoor_temp_c: float,
        weather: str = "Unknown",
        model_preference: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Predict for a list of interval dicts (period_slot, day_of_week, ...)."""
        model_kind = self._resolve_model_preference(model_preference)
        meta = self.cooler_meta.get(cooler_id, {})
        records = []
        for iv in intervals:
            rec = {
                "cooler_id": cooler_id,
                "day_of_week": iv.get("day_of_week", "Monday"),
                "period_slot": iv["period_slot"],
                "outdoor_temp_c": outdoor_temp_c,
                "weather": weather,
                "games_intensity": iv.get("games_intensity", "None"),
                "duration_minutes": iv.get("duration_minutes", 40),
                "capacity_liters": meta.get("capacity_liters", 200),
                "floor": meta.get("floor", 0),
                "near_games": meta.get("near_games", False),
            }
            records.append(rec)

        if model_kind == "catboost" and self.catboost.trained:
            feat_df = build_features(
                records,
                cooler_meta=self.cooler_meta,
                historical_df=self.historical_df,
            )
            preds = self.catboost.predict(feat_df)
            for p in preds:
                p["model_used"] = "catboost"
            return preds

        # Baseline path
        return self.baseline.predict_batch(records)

    def compute_shortage(
        self,
        capacity_liters: float,
        current_level_liters: float,
        expected_demand: float,
        conservative_demand: float,
        next_refill_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Deterministic shortage risk assessment."""
        # Use conservative demand for risk
        demand = conservative_demand
        projected = current_level_liters - demand
        shortage = max(0.0, -projected)

        if projected >= capacity_liters * 0.25:
            level = "none"
            msg = "Sufficient water for the forecast horizon."
        elif projected >= capacity_liters * 0.10:
            level = "low"
            msg = "Level will drop but remain above 10% of capacity."
        elif projected >= 0:
            level = "medium"
            msg = "Level will approach empty; plan refill soon."
        elif shortage < capacity_liters * 0.15:
            level = "high"
            msg = f"Projected shortfall of {shortage:.1f} L before next refill opportunity."
        else:
            level = "critical"
            msg = f"Critical shortfall of {shortage:.1f} L. Immediate refill recommended."

        if next_refill_hours is not None and next_refill_hours < 2 and level in ("high", "critical"):
            msg += f" Next refill window in ~{next_refill_hours:.1f} h."

        return {
            "risk_level": level,
            "projected_level_after_day": round(projected, 2),
            "shortage_liters": round(shortage, 2),
            "message": msg,
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "model_version": self.version,
            "baseline_available": True,
            "catboost_available": self.catboost.available,
            "catboost_trained": self.catboost.trained,
            "supported_slots": list(self.baseline._tables.get("s", {}).keys())
            or ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8", "Games"],
            "temperature_rules_loaded": True,  # handled by temp_control module
            "notes": [
                "One-week synthetic data is insufficient for reliable CatBoost production training.",
                "Baseline (historical average + engineered defaults) is the primary production path.",
                "Retrain CatBoost only after collecting multi-week real consumption telemetry.",
            ],
        }