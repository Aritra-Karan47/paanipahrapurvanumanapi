"""Tests for real institute upload formats."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.preprocessing import (
    load_consumption_record,
    load_timetable,
    load_temperature_table,
    load_cooler_config,
    extract_day_intervals,
    validate_consumption_schema,
    normalize_day,
    normalize_period,
)

DATA = ROOT / "data"


def test_normalize_day_abbr():
    assert normalize_day("Mon") == "Monday"
    assert normalize_day("tuesday") == "Tuesday"
    assert normalize_day("Saturday") == "Saturday"


def test_normalize_period():
    assert normalize_period(1) == "P1"
    assert normalize_period("4") == "P4"
    assert normalize_period("Recess") == "Recess"
    assert normalize_period("recess") == "Recess"


def test_load_real_consumption():
    df = load_consumption_record(DATA / "Cooler_Water_Consumption_Record.csv")
    assert len(df) >= 2000
    assert set(df["cooler_id"].unique()) == {
        "Location01A", "Location01B", "Location02A", "Location02B",
        "Location03A", "Location03B", "Location04A", "Location04B",
    }
    assert "Recess" in df["period_slot"].values
    assert "P1" in df["period_slot"].values
    assert (df["consumption_liters"] >= 0).all()


def test_load_timetable_has_recess_and_games():
    df = load_timetable(DATA / "School_Timetable_With_Recess.xlsx")
    assert "Recess" in df["period_slot"].values
    assert df["is_games_period"].any()
    assert set(df["day_of_week"].unique()) >= {"Monday", "Tuesday", "Wednesday"}


def test_extract_day_intervals_order():
    tt = load_timetable(DATA / "School_Timetable_With_Recess.xlsx")
    intervals = extract_day_intervals(tt, "Monday", class_name="1-A")
    slots = [i["period_slot"] for i in intervals]
    assert "Recess" in slots
    if "P4" in slots and "Recess" in slots:
        assert slots.index("P4") < slots.index("Recess")


def test_temperature_table_is1475():
    df = load_temperature_table(DATA / "Water_Temperature_Control_Table.xlsx")
    assert "ambient_min_c" in df.columns
    assert "target_c" in df.columns
    assert len(df) >= 5
    # band covering 33.5 C should exist
    row = df[(df["ambient_min_c"] <= 33.5) & (df["ambient_max_c"] > 33.5)]
    assert len(row) == 1
    assert float(row.iloc[0]["target_c"]) == 14.0


def test_cooler_config_eight():
    meta = load_cooler_config(DATA / "cooler_config.csv")
    assert len(meta) == 8
    assert "Location01A" in meta


def test_validate_schema():
    ok, errs = validate_consumption_schema(
        [{"cooler_id": "Location01A", "period_slot": "P1", "consumption_liters": 5.0}]
    )
    assert ok
    ok2, errs2 = validate_consumption_schema([{"cooler_id": "Location01A"}])
    assert not ok2
