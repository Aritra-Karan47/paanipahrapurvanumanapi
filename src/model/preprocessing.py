"""Excel/CSV ingestion for institute upload formats.

Supported institute uploads
---------------------------
1. Cooler_Water_Consumption_Record.csv
   Columns: Day, Period_Number, Cooler_Number, Consumption_Value

2. School_Timetable_With_Recess.xlsx  (sheet: Timetable)
   Columns: Class_section, day, period, subject
   day values: Mon/Tue/Wed/Thu/Fri/Sat
   period: 1..8 or Recess; Games appears as subject in a period slot

3. Water_Temperature_Control_Table.xlsx  (sheet: Rules)
   Columns: id, ambient_min_c (>=), ambient_max_c (<), cooling_enabled,
            setpoint_min_c, setpoint_max_c, target_c, notes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

DAY_FULL = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

DAY_ABBR = {
    "Monday": "Mon",
    "Tuesday": "Tue",
    "Wednesday": "Wed",
    "Thursday": "Thu",
    "Friday": "Fri",
    "Saturday": "Sat",
    "Sunday": "Sun",
}


def normalize_day(value: Any) -> str:
    s = str(value).strip()
    key = s.lower()
    if key in DAY_FULL:
        return DAY_FULL[key]
    cap = s.capitalize()
    if cap in DAY_ABBR:
        return cap
    return s


def normalize_period(value: Any) -> str:
    s = str(value).strip()
    if s.lower() == "recess":
        return "Recess"
    try:
        n = int(float(s))
        return f"P{n}"
    except (TypeError, ValueError):
        return s


def _read_table(path: Union[str, Path], sheet: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


REQUIRED_CONSUMPTION_ALIASES = {
    "day": ["Day", "day", "day_of_week", "Day_of_Week"],
    "period": ["Period_Number", "period", "period_slot", "Period", "Period_Slot"],
    "cooler": ["Cooler_Number", "cooler_id", "Cooler_ID", "cooler", "Cooler"],
    "value": ["Consumption_Value", "consumption_liters", "consumption", "value", "liters"],
}


def _resolve_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> Dict[str, str]:
    colmap = {}
    lower = {c.lower(): c for c in df.columns}
    for key, opts in aliases.items():
        found = None
        for o in opts:
            if o in df.columns:
                found = o
                break
            if o.lower() in lower:
                found = lower[o.lower()]
                break
        if found is None:
            raise ValueError(
                f"Could not find column for '{key}'. Tried: {opts}. Have: {list(df.columns)}"
            )
        colmap[key] = found
    return colmap


def load_consumption_record(path: Union[str, Path]) -> pd.DataFrame:
    """
    Canonical form: day_of_week, period_slot (P1..P8|Recess), cooler_id, consumption_liters
    """
    df = _read_table(path)
    cmap = _resolve_columns(df, REQUIRED_CONSUMPTION_ALIASES)
    out = pd.DataFrame(
        {
            "day_of_week": df[cmap["day"]].map(normalize_day),
            "period_raw": df[cmap["period"]].astype(str).str.strip(),
            "cooler_id": df[cmap["cooler"]].astype(str).str.strip(),
            "consumption_liters": pd.to_numeric(df[cmap["value"]], errors="coerce"),
        }
    )
    out["period_slot"] = out["period_raw"].map(normalize_period)
    n_before = len(out)
    out = out.dropna(subset=["consumption_liters"])
    out["consumption_liters"] = out["consumption_liters"].clip(lower=0)
    if len(out) < n_before:
        logger.warning("Dropped %d rows with invalid consumption", n_before - len(out))
    logger.info(
        "Loaded consumption: %d rows, coolers=%s, slots=%s",
        len(out),
        sorted(out["cooler_id"].unique().tolist()),
        sorted(out["period_slot"].unique().tolist()),
    )
    return out


REQUIRED_TIMETABLE_ALIASES = {
    "class": ["Class_section", "class_name", "class", "Class", "section"],
    "day": ["day", "Day", "day_of_week"],
    "period": ["period", "Period", "period_slot", "Period_Number"],
    "subject": ["subject", "Subject"],
}


def load_timetable(path: Union[str, Path], sheet: str = "Timetable") -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)

    cmap = _resolve_columns(df, REQUIRED_TIMETABLE_ALIASES)
    out = pd.DataFrame(
        {
            "class_name": df[cmap["class"]].astype(str).str.strip(),
            "day_of_week": df[cmap["day"]].map(normalize_day),
            "period_raw": df[cmap["period"]].astype(str).str.strip(),
            "subject": df[cmap["subject"]].astype(str).str.strip(),
        }
    )
    out["period_slot"] = out["period_raw"].map(normalize_period)
    out["games_intensity"] = out["subject"].apply(
        lambda s: "Moderate" if str(s).strip().lower() == "games" else "None"
    )
    out["is_games_period"] = out["subject"].str.lower() == "games"
    out["duration_minutes"] = out["period_slot"].apply(
        lambda s: 20 if s == "Recess" else 40
    )
    return out


def extract_day_intervals(
    timetable: pd.DataFrame,
    day_of_week: str,
    class_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    day = normalize_day(day_of_week)
    sub = timetable[timetable["day_of_week"] == day]
    if class_name:
        sub = sub[sub["class_name"] == class_name]

    order = ["P1", "P2", "P3", "P4", "Recess", "P5", "P6", "P7", "P8"]
    seen = set()
    intervals = []
    for slot in order:
        rows = sub[sub["period_slot"] == slot]
        if rows.empty:
            if slot == "Recess":
                intervals.append(
                    {
                        "period_slot": "Recess",
                        "day_of_week": day,
                        "subject": "Recess",
                        "games_intensity": "None",
                        "duration_minutes": 20,
                        "is_games_period": False,
                    }
                )
            continue
        if slot in seen:
            continue
        seen.add(slot)
        r = rows.iloc[0]
        intervals.append(
            {
                "period_slot": slot,
                "day_of_week": day,
                "subject": r["subject"],
                "games_intensity": r["games_intensity"],
                "duration_minutes": int(r["duration_minutes"]),
                "is_games_period": bool(r["is_games_period"]),
            }
        )
    return intervals


def games_load_by_day_period(timetable: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    g = timetable[timetable["is_games_period"]]
    return g.groupby(["day_of_week", "period_slot"]).size().to_dict()


def load_temperature_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, sheet_name="Rules")
        except ValueError:
            df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)

    col_renames = {}
    for c in df.columns:
        cl = str(c).lower().replace(" ", "_")
        if "ambient_min" in cl:
            col_renames[c] = "ambient_min_c"
        elif "ambient_max" in cl:
            col_renames[c] = "ambient_max_c"
        elif "cooling" in cl:
            col_renames[c] = "cooling_enabled"
        elif "setpoint_min" in cl:
            col_renames[c] = "setpoint_min_c"
        elif "setpoint_max" in cl:
            col_renames[c] = "setpoint_max_c"
        elif "target" in cl:
            col_renames[c] = "target_c"
        elif cl == "id":
            col_renames[c] = "id"
        elif "note" in cl:
            col_renames[c] = "notes"
    df = df.rename(columns=col_renames)

    if "ambient_min_c" in df.columns:
        df = df[pd.to_numeric(df["ambient_min_c"], errors="coerce").notna()].copy()
        for c in ["ambient_min_c", "ambient_max_c", "setpoint_min_c", "setpoint_max_c", "target_c"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "cooling_enabled" in df.columns:
            df["cooling_enabled"] = df["cooling_enabled"].astype(bool)
    return df.reset_index(drop=True)


def load_cooler_config(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    df = _read_table(path)
    id_col = "cooler_id" if "cooler_id" in df.columns else df.columns[0]
    meta = {}
    for _, row in df.iterrows():
        cid = str(row[id_col]).strip()
        meta[cid] = {
            "location": str(row.get("location", "unknown")),
            "capacity_liters": float(row.get("capacity_liters", 200)),
            "floor": int(row.get("floor", 0)),
            "near_games": bool(row.get("near_games", False)),
        }
    return meta


def validate_consumption_schema(records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors = []
    if not records:
        errors.append("No consumption records provided")
        return False, errors
    for i, rec in enumerate(records[:50]):
        for col in ("cooler_id", "period_slot", "consumption_liters"):
            if col not in rec:
                errors.append(f"Record {i}: missing '{col}'")
        if "consumption_liters" in rec:
            try:
                v = float(rec["consumption_liters"])
                if v < 0:
                    errors.append(f"Record {i}: negative consumption")
            except (TypeError, ValueError):
                errors.append(f"Record {i}: non-numeric consumption_liters")
    return len(errors) == 0, errors
