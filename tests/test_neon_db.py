"""Tests for Neon Serverless Postgres integration, connection pooling, and CSV fallback."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from db.connection import (
    DatabaseNotConfiguredError,
    check_db_connection,
    get_database_url,
    is_db_configured,
)
from db.repository import CoolerRepository
from app.app import bootstrap, service_db_status, service_record_consumption


def test_database_url_normalization():
    """Verify that postgres:// is normalized to postgresql:// and sslmode=require is injected for cloud."""
    with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@ep-cool-fog-123.eu-central-1.aws.neon.tech/neondb"}):
        url = get_database_url()
        assert url is not None
        assert url.startswith("postgresql://")
        assert "sslmode=require" in url

    # Query params preserved
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@ep-cool-fog-123.neon.tech/neondb?channel_binding=require"}):
        url2 = get_database_url()
        assert "sslmode=require" in url2
        assert "channel_binding=require" in url2

    # Localhost does not force sslmode
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost:5432/testdb"}):
        url_local = get_database_url()
        assert "sslmode=require" not in url_local


def test_database_url_from_decomposed_pg_vars():
    """Verify that decomposed PG* environment variables reconstruct a valid Neon URL."""
    env_vars = {
        "PGHOST": "ep-cool-fog-123.us-east-2.aws.neon.tech",
        "PGDATABASE": "neondb",
        "PGUSER": "neondb_owner",
        "PGPASSWORD": "secretpassword",
        "PGPORT": "5432",
        "PGSSLMODE": "require",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("NEON_DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)

        url = get_database_url()
        assert url is not None
        assert "neondb_owner:secretpassword@ep-cool-fog-123.us-east-2.aws.neon.tech:5432/neondb" in url
        assert "sslmode=require" in url


def test_is_db_configured_and_health_unconfigured():
    """Verify behavior when no DATABASE_URL is set (default dev & CI mode)."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("NEON_DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)

        assert is_db_configured() is False
        ok, msg = check_db_connection()
        assert ok is False
        assert "DATABASE_URL is not configured" in msg

        repo = CoolerRepository()
        health = repo.get_db_health()
        assert health["configured"] is False
        assert health["connected"] is False
        assert "local" in health["source"].lower()


def test_repository_mocked_fetch():
    """Verify repository methods with simulated Neon connection cursor."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = [
        # coolers
        [
            {
                "cooler_id": "Location01A",
                "location": "Wing A Ground Floor",
                "capacity_liters": 200.0,
                "floor": 0,
                "near_games": False,
                "current_level_liters": 150.0,
            }
        ],
        # consumption_records
        [
            {
                "cooler_id": "Location01A",
                "day_of_week": "Monday",
                "period_slot": "P1",
                "consumption_liters": 3.43,
                "recorded_at": "2026-11-24T08:00:00Z",
            }
        ],
        # school_timetable
        [
            {
                "class_name": "1-A",
                "day_of_week": "Monday",
                "period_slot": "P1",
                "subject": "Mathematics",
                "is_games_period": False,
            }
        ],
        # temperature_control_rules
        [
            {
                "outdoor_temp_min": 31.0,
                "outdoor_temp_max": 36.0,
                "target_water_temp_c": 14.0,
                "cooling_enabled": True,
                "rule_name": "IS 1475:2024 Hot Weather Rule",
            }
        ],
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    @contextmanager
    def mock_factory():
        yield mock_conn

    repo = CoolerRepository(connection_factory=mock_factory)

    # 1. Fetch coolers
    coolers = repo.fetch_cooler_config()
    assert coolers is not None
    assert "Location01A" in coolers
    assert coolers["Location01A"]["capacity_liters"] == 200.0

    # 2. Fetch consumption
    cons_df = repo.fetch_consumption_records()
    assert cons_df is not None
    assert len(cons_df) == 1
    assert cons_df.iloc[0]["consumption_liters"] == 3.43

    # 3. Fetch timetable
    tt_df = repo.fetch_timetable()
    assert tt_df is not None
    assert len(tt_df) == 1
    assert tt_df.iloc[0]["class_name"] == "1-A"

    # 4. Fetch temperature rules
    rules = repo.fetch_temperature_rules()
    assert rules is not None
    assert len(rules) == 1
    assert rules[0]["target_c"] == 14.0


def test_repository_insert_telemetry():
    """Verify live IoT consumption insert returns generated id."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"id": 1042}

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    @contextmanager
    def mock_factory():
        yield mock_conn

    repo = CoolerRepository(connection_factory=mock_factory)
    rec_id = repo.insert_consumption_record("Location01A", "Monday", "P2", 6.85)
    assert rec_id == 1042


def test_bootstrap_graceful_local_fallback():
    """Verify bootstrap cleanly loads local CSV files and fits models when DB is offline."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("DATABASE_URL", None)
        status = bootstrap(force=True)
        assert status["status"] == "success"
        assert status["cooler_configs_loaded"] == 8
        assert status["baseline_fitted"] is True
        assert status["temperature_rules_loaded"] > 0
        assert status["data_source"] == "local_files"


def test_service_db_status_and_telemetry_functions():
    """Verify service functions for db_status and telemetry logging."""
    # Status
    status = service_db_status()
    assert "configured" in status
    assert "connected" in status

    # Telemetry invalid
    bad_res = service_record_consumption({"consumption_liters": -1.0})
    assert bad_res["success"] is False
    assert "error" in bad_res

    # Missing cooler_id
    bad_res2 = service_record_consumption({"consumption_liters": 10.0})
    assert bad_res2["success"] is False

    # Valid telemetry (in offline mode returns graceful message)
    valid_res = service_record_consumption({
        "cooler_id": "Location01A",
        "day_of_week": "Tuesday",
        "period_slot": "P3",
        "consumption_liters": 8.5,
    })
    assert "success" in valid_res


def test_schema_sql_exists_and_contains_required_tables():
    """Verify schema.sql file defines all 4 required relational tables."""
    schema_path = Path("src/db/schema.sql")
    assert schema_path.exists()
    content = schema_path.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS coolers" in content
    assert "CREATE TABLE IF NOT EXISTS consumption_records" in content
    assert "CREATE TABLE IF NOT EXISTS school_timetable" in content
    assert "CREATE TABLE IF NOT EXISTS temperature_control_rules" in content
