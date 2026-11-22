"""Historical-average baseline predictor with robust fallbacks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default per-slot multipliers relative to a "normal period" base rate (liters)
DEFAULT_SLOT_MULT = {
    "P1": 1.15,
    "P2": 1.0,
    "P3": 1.0,
    "P4": 1.05,
    "Recess": 1.85,
    "P5": 1.20,
    "P6": 1.0,
    "P7": 0.95,
    "P8": 0.90,
    "Games": 2.40,
}

# Rough default base rates by cooler capacity tier
def _default_base_rate(capacity: float) -> float:
    if capacity >= 250:
        return 12.0
    if capacity >= 180:
        return 8.5
    if capacity >= 140:
        return 6.5
    return 5.0


class HistoricalAverageBaseline:
    """
    Lightweight, production-safe baseline.

    Hierarchy of lookup (most specific → least):
      1. cooler + day_of_week + period_slot
      2. cooler + period_slot
      3. period_slot global
      4. engineered default using capacity / slot multiplier / temperature factor
    """

    def __init__(self):
        self._tables: Dict[str, Dict[Tuple, float]] = {
            "c_d_s": {},
            "c_s": {},
            "s": {},
        }
        self._counts: Dict[str, Dict[Tuple, int]] = {
            "c_d_s": defaultdict(int),
            "c_s": defaultdict(int),
            "s": defaultdict(int),
        }
        self._cooler_meta: Dict[str, Dict[str, Any]] = {}
        self.fitted = False
        self.n_samples = 0

    def fit(
        self,
        df: pd.DataFrame,
        cooler_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "HistoricalAverageBaseline":
        """
        Fit from a consumption DataFrame with columns:
          cooler_id, day_of_week, period_slot, consumption_liters
        """
        required = {"cooler_id", "day_of_week", "period_slot", "consumption_liters"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for baseline fit: {missing}")

        self._cooler_meta = cooler_meta or {}
        self.n_samples = len(df)

        # Level 1
        g1 = df.groupby(["cooler_id", "day_of_week", "period_slot"])["consumption_liters"].agg(["mean", "count"])
        for (c, d, s), row in g1.iterrows():
            key = (c, d, s)
            self._tables["c_d_s"][key] = float(row["mean"])
            self._counts["c_d_s"][key] = int(row["count"])

        # Level 2
        g2 = df.groupby(["cooler_id", "period_slot"])["consumption_liters"].agg(["mean", "count"])
        for (c, s), row in g2.iterrows():
            key = (c, s)
            self._tables["c_s"][key] = float(row["mean"])
            self._counts["c_s"][key] = int(row["count"])

        # Level 3
        g3 = df.groupby(["period_slot"])["consumption_liters"].agg(["mean", "count"])
        for s, row in g3.iterrows():
            self._tables["s"][s] = float(row["mean"])
            self._counts["s"][s] = int(row["count"])

        self.fitted = True
        logger.info(
            "Baseline fitted on %d rows | unique coolers=%d | slots=%s",
            self.n_samples,
            df["cooler_id"].nunique(),
            sorted(df["period_slot"].unique().tolist()),
        )
        return self

    def _temp_factor(self, outdoor_temp_c: float) -> float:
        # Mild positive slope above 20 °C
        return 1.0 + max(0.0, (outdoor_temp_c - 20.0) * 0.025)

    def _default_prediction(
        self,
        cooler_id: str,
        period_slot: str,
        outdoor_temp_c: float,
        games_intensity: str = "None",
        near_games: bool = False,
    ) -> float:
        meta = self._cooler_meta.get(cooler_id, {})
        capacity = float(meta.get("capacity_liters", 200))
        base = _default_base_rate(capacity)
        mult = DEFAULT_SLOT_MULT.get(period_slot, 1.0)
        if period_slot == "Games":
            intensity_boost = {"None": 1.0, "Light": 1.1, "Moderate": 1.25, "High": 1.45}.get(
                games_intensity, 1.2
            )
            mult *= intensity_boost
            if near_games:
                mult *= 1.15
        return round(base * mult * self._temp_factor(outdoor_temp_c), 2)

    def predict_one(
        self,
        cooler_id: str,
        day_of_week: str,
        period_slot: str,
        outdoor_temp_c: float = 25.0,
        games_intensity: str = "None",
        near_games: bool = False,
    ) -> Tuple[float, str, int]:
        """
        Returns (expected_liters, source_level, support_count).
        """
        day = day_of_week.capitalize()
        slot = period_slot

        # 1
        key1 = (cooler_id, day, slot)
        if key1 in self._tables["c_d_s"] and self._counts["c_d_s"][key1] >= 1:
            val = self._tables["c_d_s"][key1] * self._temp_factor(outdoor_temp_c)
            return round(val, 2), "cooler_day_slot", self._counts["c_d_s"][key1]

        # 2
        key2 = (cooler_id, slot)
        if key2 in self._tables["c_s"] and self._counts["c_s"][key2] >= 1:
            val = self._tables["c_s"][key2] * self._temp_factor(outdoor_temp_c)
            return round(val, 2), "cooler_slot", self._counts["c_s"][key2]

        # 3
        if slot in self._tables["s"] and self._counts["s"][slot] >= 1:
            val = self._tables["s"][slot] * self._temp_factor(outdoor_temp_c)
            return round(val, 2), "global_slot", self._counts["s"][slot]

        # 4 engineered default
        val = self._default_prediction(
            cooler_id, slot, outdoor_temp_c, games_intensity, near_games
        )
        return val, "engineered_default", 0

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for rec in records:
            exp, src, cnt = self.predict_one(
                cooler_id=rec["cooler_id"],
                day_of_week=rec.get("day_of_week", "Monday"),
                period_slot=rec["period_slot"],
                outdoor_temp_c=float(rec.get("outdoor_temp_c", 25.0)),
                games_intensity=rec.get("games_intensity", "None"),
                near_games=bool(rec.get("near_games", False)),
            )
            # Conservative = expected * 1.25 when no quantiles
            out.append(
                {
                    "expected_liters": exp,
                    "conservative_liters": round(exp * 1.25, 2),
                    "p50_liters": exp,
                    "p90_liters": round(exp * 1.25, 2),
                    "model_used": f"baseline:{src}",
                    "support_count": cnt,
                    "confidence": "high" if cnt >= 5 else ("medium" if cnt >= 2 else "low"),
                }
            )
        return out