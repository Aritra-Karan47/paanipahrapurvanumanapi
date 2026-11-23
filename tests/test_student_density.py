"""Tests for timetable student-density aggregation per cooler zone."""

import pytest
import pandas as pd

from model.preprocessing import (
    compute_active_students_in_zone,
    get_default_timetable,
    get_zone_student_density_map,
    DEFAULT_COOLER_SECTIONS,
)
from model.features import build_features, build_single_feature_row, feature_columns


def test_feature_columns_includes_active_students():
    cols = feature_columns(include_target=False)
    assert "active_students_in_zone" in cols
    idx = cols.index("active_students_in_zone")
    assert idx >= 0


def test_timetable_loaded_correctly():
    tt = get_default_timetable()
    assert tt is not None
    assert len(tt) > 0
    assert "class_name" in tt.columns
    assert "day_of_week" in tt.columns
    assert "period_slot" in tt.columns
    assert "is_games_period" in tt.columns


def test_sports_complex_student_density_on_games_periods():
    """Verify that coolers near games aggregate sections having Games across the school."""
    # On Monday Period 3, 3 sections have Games: 4-A, 12-A, 12-D
    active_students = compute_active_students_in_zone(
        cooler_id="Location04A",
        day_of_week="Monday",
        period_slot="P3",
        students_per_section=40,
    )
    # 3 sections * 40 students = 120 students
    assert active_students == 120

    # On Monday Period 1, 0 sections have Games
    active_p1 = compute_active_students_in_zone(
        cooler_id="Location04A",
        day_of_week="Monday",
        period_slot="P1",
        students_per_section=40,
    )
    assert active_p1 == 0


def test_academic_wing_student_density_deducts_games_classes():
    """Verify that resident classroom sections attending games are deducted from wing footfall."""
    # Location01A (Block 1 Wing A) has 8 assigned sections: 1-A, 1-B, 2-A, 2-B, 3-A, 3-B, 4-A, 4-B.
    # On Monday Period 3, section 4-A has Games.
    # Active sections = 8 - 1 = 7 sections -> 7 * 40 = 280 students.
    active_students_p3 = compute_active_students_in_zone(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="P3",
        students_per_section=40,
    )
    assert active_students_p3 == 280

    # On Monday Period 1, no sections from Block 1 Wing A have games -> full 8 sections = 320 students.
    active_students_p1 = compute_active_students_in_zone(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="P1",
        students_per_section=40,
    )
    assert active_students_p1 == 320


def test_recess_footfall_density():
    """Verify that Recess models full wing presence for classrooms and outdoor surge for sports complex."""
    # Academic wing during Recess: all assigned sections on break in wing (8 * 40 = 320)
    recess_block1 = compute_active_students_in_zone(
        cooler_id="Location01A",
        day_of_week="Monday",
        period_slot="Recess",
        students_per_section=40,
    )
    assert recess_block1 == 320

    # Sports complex during Recess: outdoor activity surge (~20% of student body)
    recess_sports = compute_active_students_in_zone(
        cooler_id="Location04A",
        day_of_week="Monday",
        period_slot="Recess",
        students_per_section=40,
    )
    assert recess_sports > 200


def test_zone_density_precomputation_map():
    density_map = get_zone_student_density_map()
    assert isinstance(density_map, dict)
    assert len(density_map) >= 8 * 6 * 9  # 8 coolers x 6 days x 9 slots = 432
    assert ("Location01A", "Monday", "P1") in density_map
    assert density_map[("Location01A", "Monday", "P1")] == 320
    assert density_map[("Location04A", "Monday", "P3")] == 120


def test_build_features_incorporates_student_density():
    records = [
        {"cooler_id": "Location01A", "day_of_week": "Monday", "period_slot": "P1", "outdoor_temp_c": 22.0},
        {"cooler_id": "Location04A", "day_of_week": "Monday", "period_slot": "P3", "outdoor_temp_c": 28.0},
    ]
    df = build_features(records)
    assert "active_students_in_zone" in df.columns
    assert df.loc[0, "active_students_in_zone"] == 320
    assert df.loc[1, "active_students_in_zone"] == 120
