"""Tests for Parquet/Arrow binary tables, build-time/first-run conversion, and latency (< 20 ms)."""

import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.preprocessing import (
    convert_excel_to_parquet,
    ensure_binary_data,
    load_temperature_table,
    load_timetable,
    save_table_as_parquet,
)

DATA = ROOT / "data"


def test_parquet_files_exist_and_readable():
    tt_pq = DATA / "School_Timetable_With_Recess.parquet"
    temp_pq = DATA / "Water_Temperature_Control_Table.parquet"

    assert tt_pq.exists(), "School_Timetable_With_Recess.parquet should exist in data/"
    assert temp_pq.exists(), "Water_Temperature_Control_Table.parquet should exist in data/"

    df_tt = pd.read_parquet(tt_pq)
    assert len(df_tt) == 2322
    assert "period" in df_tt.columns

    df_temp = pd.read_parquet(temp_pq)
    assert len(df_temp) == 7
    assert "id" in df_temp.columns


def test_parquet_read_latency_under_20ms():
    """Verify that reading timetable and temperature tables via Parquet takes < 20 ms."""
    tt_pq = DATA / "School_Timetable_With_Recess.parquet"
    temp_pq = DATA / "Water_Temperature_Control_Table.parquet"

    # Pre-warm pyarrow engine
    _ = pd.read_parquet(temp_pq)

    # Benchmark Timetable Parquet read
    times_tt = []
    for _ in range(5):
        t0 = time.perf_counter()
        df = pd.read_parquet(tt_pq)
        times_tt.append((time.perf_counter() - t0) * 1000)
    avg_tt = sum(times_tt) / len(times_tt)

    # Benchmark Temperature Table Parquet read
    times_temp = []
    for _ in range(5):
        t0 = time.perf_counter()
        df = pd.read_parquet(temp_pq)
        times_temp.append((time.perf_counter() - t0) * 1000)
    avg_temp = sum(times_temp) / len(times_temp)

    print(f"\n[Read Latency] Timetable Parquet avg: {avg_tt:.2f} ms | Temp table avg: {avg_temp:.2f} ms")
    assert avg_tt < 20.0, f"Timetable read latency {avg_tt:.2f} ms exceeds 20 ms limit"
    assert avg_temp < 20.0, f"Temperature table read latency {avg_temp:.2f} ms exceeds 20 ms limit"


def test_data_equivalence_xlsx_and_parquet():
    """Verify DataFrames produced from Parquet and Excel are 100% equivalent."""
    tt_xlsx = DATA / "School_Timetable_With_Recess.xlsx"
    tt_pq = DATA / "School_Timetable_With_Recess.parquet"
    temp_xlsx = DATA / "Water_Temperature_Control_Table.xlsx"
    temp_pq = DATA / "Water_Temperature_Control_Table.parquet"

    # Compare Timetable loading
    df_tt_from_xlsx = load_timetable(tt_xlsx, prefer_binary=False)
    df_tt_from_pq = load_timetable(tt_pq, prefer_binary=True)
    pd.testing.assert_frame_equal(df_tt_from_xlsx, df_tt_from_pq)

    # Compare Temperature table loading
    df_temp_from_xlsx = load_temperature_table(temp_xlsx, prefer_binary=False)
    df_temp_from_pq = load_temperature_table(temp_pq, prefer_binary=True)
    pd.testing.assert_frame_equal(df_temp_from_xlsx, df_temp_from_pq)


def test_prefer_binary_transparently_uses_parquet():
    """Verify that calling load_timetable with .xlsx path transparently uses .parquet when present."""
    tt_xlsx = DATA / "School_Timetable_With_Recess.xlsx"

    # Spy on read_excel to verify it is NOT called when parquet exists
    with patch("pandas.read_excel") as mock_read_excel:
        df = load_timetable(tt_xlsx, prefer_binary=True)
        assert len(df) == 2322
        assert mock_read_excel.call_count == 0


def test_first_run_autoconversion(tmp_path):
    """Verify first-run conversion step creates Parquet binary automatically if missing."""
    temp_excel = tmp_path / "School_Timetable_With_Recess.xlsx"
    shutil.copy(DATA / "School_Timetable_With_Recess.xlsx", temp_excel)

    expected_pq = tmp_path / "School_Timetable_With_Recess.parquet"
    assert not expected_pq.exists()

    # First run: should read Excel and auto-generate Parquet companion
    df1 = load_timetable(temp_excel, prefer_binary=True)
    assert len(df1) == 2322
    assert expected_pq.exists(), "Parquet file should have been generated on first run"

    # Second run: should read from Parquet without calling read_excel
    with patch("pandas.read_excel") as mock_read_excel:
        df2 = load_timetable(temp_excel, prefer_binary=True)
        assert len(df2) == 2322
        assert mock_read_excel.call_count == 0


def test_ensure_binary_data(tmp_path):
    """Test ensure_binary_data batch conversion helper."""
    shutil.copy(DATA / "School_Timetable_With_Recess.xlsx", tmp_path / "School_Timetable_With_Recess.xlsx")
    shutil.copy(DATA / "Water_Temperature_Control_Table.xlsx", tmp_path / "Water_Temperature_Control_Table.xlsx")

    results = ensure_binary_data(tmp_path)
    assert len(results) == 2
    assert (tmp_path / "School_Timetable_With_Recess.parquet").exists()
    assert (tmp_path / "Water_Temperature_Control_Table.parquet").exists()
