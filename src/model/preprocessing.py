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


def save_table_as_parquet(df: pd.DataFrame, target_path: Union[str, Path]) -> Optional[Path]:
    """Safely convert and save a DataFrame to Parquet format for fast binary reads (< 20 ms).

    Ensures mixed object types are cleanly cast to strings and directory exists.
    """
    try:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = df.copy()

        # If ambient_min_c exists, filter out non-data explanation footer rows
        col_lower = {str(c).lower(): c for c in df_to_save.columns}
        for amb in ["ambient_min_c", "ambient_min_c (>=)"]:
            if amb in col_lower:
                real_col = col_lower[amb]
                df_to_save = df_to_save[pd.to_numeric(df_to_save[real_col], errors="coerce").notna()].copy()
                break

        # Cast all object/mixed columns to strings to prevent PyArrow type mismatch
        for col in df_to_save.columns:
            if df_to_save[col].dtype == "object":
                df_to_save[col] = df_to_save[col].astype(str)

        df_to_save.to_parquet(target, index=False)
        logger.info("Saved binary Parquet table: %s (%d bytes)", target, target.stat().st_size)
        return target
    except Exception as exc:
        logger.warning("Could not auto-convert DataFrame to Parquet at %s: %s", target_path, exc)
        return None


def convert_excel_to_parquet(
    excel_path: Union[str, Path],
    target_path: Optional[Union[str, Path]] = None,
    sheet: Optional[str] = None,
) -> Path:
    """Convert an Excel file to Parquet format."""
    src = Path(excel_path)
    if not src.exists():
        raise FileNotFoundError(f"Source Excel file not found: {src}")
    dst = Path(target_path) if target_path else src.with_suffix(".parquet")

    # Pick appropriate sheet if not specified
    if sheet is None:
        sheet_name_lower = src.stem.lower()
        if "timetable" in sheet_name_lower:
            sheet = "Timetable"
        elif "temperature" in sheet_name_lower:
            sheet = "Rules"
        else:
            sheet = 0

    try:
        df = pd.read_excel(src, sheet_name=sheet)
    except ValueError:
        df = pd.read_excel(src, sheet_name=0)

    saved = save_table_as_parquet(df, dst)
    if saved is None:
        raise RuntimeError(f"Failed to convert {src} to {dst}")
    return saved


def ensure_binary_data(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Ensure Parquet binary formats exist for all Excel files in data directory.

    Converts any missing or stale Parquet files (where .xlsx is newer than .parquet).
    """
    d = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"
    results = {}
    if not d.exists():
        return results

    for xlsx in d.glob("*.xlsx"):
        pq = xlsx.with_suffix(".parquet")
        needs_convert = False
        if not pq.exists():
            needs_convert = True
        else:
            if xlsx.stat().st_mtime > pq.stat().st_mtime:
                needs_convert = True

        if needs_convert:
            try:
                converted = convert_excel_to_parquet(xlsx, pq)
                results[xlsx.name] = converted
            except Exception as exc:
                logger.warning("Build-time/first-run conversion failed for %s: %s", xlsx, exc)
        else:
            results[xlsx.name] = pq
    return results


def _read_table(path: Union[str, Path], sheet: Optional[str] = None, prefer_binary: bool = True) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in (".csv", ".tsv"):
        return pd.read_csv(path)

    if suffix in (".xlsx", ".xls"):
        if prefer_binary:
            pq_path = path.with_suffix(".parquet")
            if pq_path.exists():
                return pd.read_parquet(pq_path)
            feather_path = path.with_suffix(".feather")
            if feather_path.exists():
                return pd.read_feather(feather_path)

        if not path.exists():
            pq_path = path.with_suffix(".parquet")
            if pq_path.exists():
                return pd.read_parquet(pq_path)
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_excel(path, sheet_name=sheet or 0)
        if prefer_binary:
            save_table_as_parquet(df, path.with_suffix(".parquet"))
        return df

    if not path.exists():
        pq_path = path.with_suffix(".parquet")
        if pq_path.exists():
            return pd.read_parquet(pq_path)
        raise FileNotFoundError(f"File not found: {path}")

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


def load_timetable(
    path: Union[str, Path],
    sheet: str = "Timetable",
    prefer_binary: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    df = None
    if prefer_binary:
        try:
            if suffix in (".parquet", ".feather"):
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_feather(path)
            else:
                pq_path = path.with_suffix(".parquet")
                feather_path = path.with_suffix(".feather")
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                elif feather_path.exists():
                    df = pd.read_feather(feather_path)
        except (ImportError, Exception) as exc:
            logger.debug("Binary table read skipped (pyarrow/fastparquet not available): %s", exc)
            df = None

    if df is None:
        if suffix in (".xlsx", ".xls"):
            if not path.exists():
                pq_path = path.with_suffix(".parquet")
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                else:
                    raise FileNotFoundError(f"File not found: {path}")
            else:
                try:
                    df = pd.read_excel(path, sheet_name=sheet)
                except ValueError:
                    df = pd.read_excel(path, sheet_name=0)
                if prefer_binary:
                    save_table_as_parquet(df, path.with_suffix(".parquet"))
        elif suffix in (".parquet", ".feather"):
            try:
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_feather(path)
            except (ImportError, Exception):
                xlsx_fallback = path.with_suffix(".xlsx")
                if xlsx_fallback.exists():
                    try:
                        df = pd.read_excel(xlsx_fallback, sheet_name=sheet)
                    except ValueError:
                        df = pd.read_excel(xlsx_fallback, sheet_name=0)
                else:
                    raise
        else:
            df = pd.read_csv(path)

    # If canonical columns already exist, ensure dtypes and return
    canonical_cols = {"class_name", "day_of_week", "period_raw", "subject", "period_slot", "is_games_period"}
    if canonical_cols.issubset(df.columns):
        out = df.copy()
        out["day_of_week"] = out["day_of_week"].map(normalize_day)
        out["period_slot"] = out["period_slot"].map(normalize_period)
        out["is_games_period"] = out["is_games_period"].astype(bool)
        out["duration_minutes"] = out["period_slot"].apply(lambda s: 20 if s == "Recess" else 40)
        return out

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


def load_temperature_table(
    path: Union[str, Path],
    prefer_binary: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    df = None
    if prefer_binary:
        try:
            if suffix in (".parquet", ".feather"):
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_feather(path)
            else:
                pq_path = path.with_suffix(".parquet")
                feather_path = path.with_suffix(".feather")
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                elif feather_path.exists():
                    df = pd.read_feather(feather_path)
        except (ImportError, Exception) as exc:
            logger.debug("Binary table read skipped (pyarrow/fastparquet not available): %s", exc)
            df = None

    if df is None:
        if suffix in (".xlsx", ".xls"):
            if not path.exists():
                pq_path = path.with_suffix(".parquet")
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                else:
                    raise FileNotFoundError(f"File not found: {path}")
            else:
                try:
                    df = pd.read_excel(path, sheet_name="Rules")
                except ValueError:
                    df = pd.read_excel(path, sheet_name=0)
                if prefer_binary:
                    save_table_as_parquet(df, path.with_suffix(".parquet"))
        elif suffix in (".parquet", ".feather"):
            try:
                df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_feather(path)
            except (ImportError, Exception):
                xlsx_fallback = path.with_suffix(".xlsx")
                if xlsx_fallback.exists():
                    try:
                        df = pd.read_excel(xlsx_fallback, sheet_name="Rules")
                    except ValueError:
                        df = pd.read_excel(xlsx_fallback, sheet_name=0)
                else:
                    raise
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
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
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
