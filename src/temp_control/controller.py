"""Deterministic outdoor-temperature → target water-temperature mapping.

Parses Water_Temperature_Control_Table.xlsx (Rules sheet) with bands:
  ambient_min_c (>=), ambient_max_c (<), cooling_enabled,
  setpoint_min_c, setpoint_max_c, target_c
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class TemperatureController:
    """
    Deterministic mapping. Completely separate from ML.

    Band semantics: [ambient_min_c, ambient_max_c) — last band closed on the right.
    When cooling_enabled is False, target is None / ambient-neutral.
    """

    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.source: str = "builtin_defaults"
        self._load_defaults()

    def _load_defaults(self) -> None:
        # Mirrors the shipped IS 1475-inspired table
        self.rules = [
            {"ambient_min_c": -50.0, "ambient_max_c": 20.0, "cooling_enabled": False,
             "setpoint_min_c": 20.0, "setpoint_max_c": 25.0, "target_c": None,
             "notes": "Cool day, chilled water not needed"},
            {"ambient_min_c": 20.0, "ambient_max_c": 26.0, "cooling_enabled": True,
             "setpoint_min_c": 18.0, "setpoint_max_c": 22.0, "target_c": 20.0,
             "notes": "Mildly cool is enough"},
            {"ambient_min_c": 26.0, "ambient_max_c": 31.0, "cooling_enabled": True,
             "setpoint_min_c": 15.0, "setpoint_max_c": 18.0, "target_c": 16.5, "notes": ""},
            {"ambient_min_c": 31.0, "ambient_max_c": 36.0, "cooling_enabled": True,
             "setpoint_min_c": 12.0, "setpoint_max_c": 16.0, "target_c": 14.0,
             "notes": "IS 1475:2024 - 16 C or below at 35 C ambient"},
            {"ambient_min_c": 36.0, "ambient_max_c": 41.0, "cooling_enabled": True,
             "setpoint_min_c": 16.0, "setpoint_max_c": 19.0, "target_c": 17.5,
             "notes": "Cooler may not reach colder in heat"},
            {"ambient_min_c": 41.0, "ambient_max_c": 44.0, "cooling_enabled": True,
             "setpoint_min_c": 19.0, "setpoint_max_c": 21.0, "target_c": 20.0,
             "notes": "IS 1475:2024 - 21 C at 43 C ambient"},
            {"ambient_min_c": 44.0, "ambient_max_c": 100.0, "cooling_enabled": True,
             "setpoint_min_c": 21.0, "setpoint_max_c": 25.0, "target_c": 21.0,
             "notes": "Outside rated range; alert staff"},
        ]
        self.source = "builtin_defaults"

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        rules = []
        for _, row in df.iterrows():
            target = row.get("target_c")
            if pd.isna(target):
                target = None
            else:
                target = float(target)
            rules.append(
                {
                    "ambient_min_c": float(row["ambient_min_c"]),
                    "ambient_max_c": float(row["ambient_max_c"]),
                    "cooling_enabled": bool(row.get("cooling_enabled", True)),
                    "setpoint_min_c": float(row["setpoint_min_c"]) if pd.notna(row.get("setpoint_min_c")) else None,
                    "setpoint_max_c": float(row["setpoint_max_c"]) if pd.notna(row.get("setpoint_max_c")) else None,
                    "target_c": target,
                    "notes": str(row.get("notes", "") or ""),
                }
            )
        rules.sort(key=lambda r: r["ambient_min_c"])
        self.rules = rules
        self.source = "loaded_table"
        logger.info("Loaded %d temperature control rules", len(rules))

    def load_from_file(self, path: Union[str, Path]) -> None:
        from model.preprocessing import load_temperature_table
        df = load_temperature_table(path)
        self.load_from_dataframe(df)
        self.source = str(path)

    def get_target(self, outdoor_temp_c: float, hard_min_c: float = 10.0) -> Dict[str, Any]:
        if not self.rules:
            self._load_defaults()

        t = float(outdoor_temp_c)
        matched = None
        for i, rule in enumerate(self.rules):
            lo, hi = rule["ambient_min_c"], rule["ambient_max_c"]
            if i == len(self.rules) - 1:
                if lo <= t <= hi:
                    matched = rule
                    break
            else:
                if lo <= t < hi:
                    matched = rule
                    break

        if matched is None:
            # clamp
            if t < self.rules[0]["ambient_min_c"]:
                matched = self.rules[0]
            else:
                matched = self.rules[-1]

        target = matched["target_c"]
        if target is not None and target < hard_min_c:
            target = hard_min_c

        cooling = matched["cooling_enabled"]
        action = "Cooling OFF" if not cooling else (
            f"Cooling ON, hold target {target}°C" if target is not None else "Cooling ON"
        )

        return {
            "outdoor_temp_c": t,
            "target_water_temp_c": target,
            "cooling_enabled": cooling,
            "setpoint_min_c": matched.get("setpoint_min_c"),
            "setpoint_max_c": matched.get("setpoint_max_c"),
            "rule_applied": (
                f"[{matched['ambient_min_c']}, {matched['ambient_max_c']}) "
                f"cooling={cooling} target={target}"
            ),
            "notes": matched.get("notes", ""),
            "action": action,
            "source": self.source,
        }
