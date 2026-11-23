"""Feature engineering for cooler water demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .cache import compute_temp_band


DAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

SLOT_ORDER = ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8", "Games"]
SLOT_INDEX = {s: i for i, s in enumerate(SLOT_ORDER)}

GAMES_INTENSITY_MAP = {"None": 0, "Light": 1, "Moderate": 2, "High": 3}
WEATHER_MAP = {"Clear": 0, "Cloudy": 1, "Rainy": 2, "Humid": 3, "Unknown": -1}


@dataclass
class FeatureConfig:
    slot_order: List[str] = field(default_factory=lambda: list(SLOT_ORDER))
    use_historical_stats: bool = True


def _slot_type_flags(slot: str) -> Dict[str, int]:
    s = slot.strip()
    return {
        "is_recess": int(s.lower() == "recess"),
        "is_games": int(s.lower() == "games"),
        "is_period": int(s.lower() not in ("recess", "games")),
        "slot_idx": SLOT_INDEX.get(s, -1),
    }


def build_single_feature_row(
    *,
    cooler_id: str,
    day_of_week: str,
    period_slot: str,
    outdoor_temp_c: float,
    weather: str = "Unknown",
    games_intensity: str = "None",
    duration_minutes: int = 40,
    capacity_liters: float = 200.0,
    floor: int = 0,
    near_games: bool = False,
    hist_mean: Optional[float] = None,
    hist_std: Optional[float] = None,
    hist_count: int = 0,
) -> Dict[str, Any]:
    """Build a flat feature dict for one prediction row."""
    flags = _slot_type_flags(period_slot)
    day_idx = DAY_MAP.get(day_of_week.capitalize(), -1)
    intensity = GAMES_INTENSITY_MAP.get(games_intensity, 0)
    weather_code = WEATHER_MAP.get(weather, -1)

    # Temperature bands (same spirit as control table)
    temp_band = compute_temp_band(outdoor_temp_c)

    row = {
        "cooler_id": cooler_id,
        "day_idx": day_idx,
        "slot_idx": flags["slot_idx"],
        "is_recess": flags["is_recess"],
        "is_games": flags["is_games"],
        "is_period": flags["is_period"],
        "games_intensity": intensity,
        "duration_minutes": duration_minutes,
        "outdoor_temp_c": outdoor_temp_c,
        "temp_band": temp_band,
        "weather_code": weather_code,
        "capacity_liters": capacity_liters,
        "floor": floor,
        "near_games": int(near_games),
        "hist_mean": hist_mean if hist_mean is not None else np.nan,
        "hist_std": hist_std if hist_std is not None else np.nan,
        "hist_count": hist_count,
        # Interaction helpers
        "games_x_near": intensity * int(near_games),
        "temp_x_recess": outdoor_temp_c * flags["is_recess"],
    }
    return row


def build_features(
    records: List[Dict[str, Any]],
    cooler_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    historical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame from a list of interval records.

    Each record should contain at least:
      cooler_id, day_of_week, period_slot, outdoor_temp_c
    Optional: weather, games_intensity, duration_minutes, capacity_liters, floor, near_games
    """
    cooler_meta = cooler_meta or {}
    rows = []

    # Pre-compute historical stats if provided
    hist_stats: Dict[tuple, Dict[str, float]] = {}
    if historical_df is not None and len(historical_df) > 0:
        grouped = historical_df.groupby(["cooler_id", "day_of_week", "period_slot"])["consumption_liters"]
        for key, series in grouped:
            hist_stats[key] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
                "count": int(len(series)),
            }

    for rec in records:
        cid = rec["cooler_id"]
        meta = cooler_meta.get(cid, {})
        day = rec.get("day_of_week", "Monday")
        slot = rec.get("period_slot", "P1")
        key = (cid, day, slot)
        hs = hist_stats.get(key, {})
        row = build_single_feature_row(
            cooler_id=cid,
            day_of_week=day,
            period_slot=slot,
            outdoor_temp_c=float(rec.get("outdoor_temp_c", 25.0)),
            weather=rec.get("weather", "Unknown"),
            games_intensity=rec.get("games_intensity", "None"),
            duration_minutes=int(rec.get("duration_minutes", 40)),
            capacity_liters=float(rec.get("capacity_liters", meta.get("capacity_liters", 200))),
            floor=int(rec.get("floor", meta.get("floor", 0))),
            near_games=bool(rec.get("near_games", meta.get("near_games", False))),
            hist_mean=hs.get("mean"),
            hist_std=hs.get("std"),
            hist_count=hs.get("count", 0),
        )
        if "consumption_liters" in rec:
            row["consumption_liters"] = float(rec["consumption_liters"])
        rows.append(row)

    return pd.DataFrame(rows)


def feature_columns(include_target: bool = False) -> List[str]:
    """Ordered list of numeric feature columns used by models."""
    cols = [
        "day_idx",
        "slot_idx",
        "is_recess",
        "is_games",
        "is_period",
        "games_intensity",
        "duration_minutes",
        "outdoor_temp_c",
        "temp_band",
        "weather_code",
        "capacity_liters",
        "floor",
        "near_games",
        "hist_mean",
        "hist_std",
        "hist_count",
        "games_x_near",
        "temp_x_recess",
    ]
    if include_target:
        cols.append("consumption_liters")
    return cols