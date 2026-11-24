"""Repository layer for fetching and storing school water cooler data in Neon Postgres."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .connection import (
    DatabaseNotConfiguredError,
    check_db_connection,
    get_db_connection,
    is_db_configured,
)

logger = logging.getLogger(__name__)


class CoolerRepository:
    """Repository providing typed query and insertion methods for Neon Postgres."""

    def __init__(self, connection_factory: Optional[Callable[[], Any]] = None):
        self._conn_factory = connection_factory or get_db_connection

    def _can_connect(self) -> bool:
        """Check if database query can be attempted (or custom factory injected)."""
        if self._conn_factory != get_db_connection:
            return True
        return is_db_configured()

    def is_connected(self) -> bool:
        """Verify if database connection is available."""
        if not self._can_connect():
            return False
        ok, _ = check_db_connection(timeout=3.0)
        return ok

    def fetch_cooler_config(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetch all cooler configurations from 'coolers' table.

        Returns:
            Dictionary indexed by cooler_id, or None if database is unavailable or table empty.
        """
        if not self._can_connect():
            return None
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT cooler_id, location, capacity_liters, floor, near_games, current_level_liters
                        FROM coolers
                        ORDER BY cooler_id
                        """
                    )
                    rows = cur.fetchall()
                    if not rows:
                        logger.warning("Neon DB 'coolers' table is empty.")
                        return None

                    config: Dict[str, Dict[str, Any]] = {}
                    for r in rows:
                        cid = str(r["cooler_id"]).strip()
                        config[cid] = {
                            "cooler_id": cid,
                            "location": str(r.get("location") or "unknown"),
                            "capacity_liters": float(r["capacity_liters"]),
                            "floor": int(r.get("floor") or 0),
                            "near_games": bool(r.get("near_games", False)),
                            "current_level_liters": (
                                float(r["current_level_liters"])
                                if r.get("current_level_liters") is not None
                                else None
                            ),
                        }
                    logger.info("Fetched %d coolers from Neon Postgres", len(config))
                    return config
        except Exception as exc:
            logger.warning("Could not fetch cooler config from database: %s", exc)
            return None

    def fetch_consumption_records(self) -> Optional[pd.DataFrame]:
        """Fetch historical consumption records from 'consumption_records' table.

        Returns:
            pandas DataFrame with columns [cooler_id, day_of_week, period_slot, consumption_liters],
            or None on failure.
        """
        if not self._can_connect():
            return None
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT cooler_id, day_of_week, period_slot, consumption_liters, recorded_at
                        FROM consumption_records
                        ORDER BY id
                        """
                    )
                    rows = cur.fetchall()
                    if not rows:
                        logger.warning("Neon DB 'consumption_records' table is empty.")
                        return None

                    data = []
                    for r in rows:
                        data.append(
                            {
                                "cooler_id": str(r["cooler_id"]).strip(),
                                "day_of_week": str(r["day_of_week"]).strip().capitalize(),
                                "period_slot": str(r["period_slot"]).strip(),
                                "consumption_liters": float(r["consumption_liters"]),
                                "recorded_at": r.get("recorded_at"),
                            }
                        )
                    df = pd.DataFrame(data)
                    logger.info("Fetched %d consumption records from Neon Postgres", len(df))
                    return df
        except Exception as exc:
            logger.warning("Could not fetch consumption records from database: %s", exc)
            return None

    def fetch_timetable(self) -> Optional[pd.DataFrame]:
        """Fetch weekly section timetable from 'school_timetable' table.

        Returns:
            pandas DataFrame with columns [class_name, day_of_week, period_slot, subject, is_games_period],
            or None on failure.
        """
        if not self._can_connect():
            return None
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT class_name, day_of_week, period_slot, subject, is_games_period
                        FROM school_timetable
                        ORDER BY id
                        """
                    )
                    rows = cur.fetchall()
                    if not rows:
                        logger.warning("Neon DB 'school_timetable' table is empty.")
                        return None

                    data = []
                    for r in rows:
                        data.append(
                            {
                                "class_name": str(r["class_name"]).strip(),
                                "day_of_week": str(r["day_of_week"]).strip().capitalize(),
                                "period_slot": str(r["period_slot"]).strip(),
                                "subject": str(r.get("subject") or "General"),
                                "is_games_period": bool(r.get("is_games_period", False)),
                            }
                        )
                    df = pd.DataFrame(data)
                    logger.info("Fetched %d timetable rows from Neon Postgres", len(df))
                    return df
        except Exception as exc:
            logger.warning("Could not fetch timetable from database: %s", exc)
            return None

    def fetch_temperature_rules(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch IS 1475:2024 water dispenser temperature control setpoint rules."""
        if not self._can_connect():
            return None
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT outdoor_temp_min, outdoor_temp_max, target_water_temp_c, cooling_enabled, rule_name
                        FROM temperature_control_rules
                        ORDER BY outdoor_temp_min
                        """
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return None

                    rules = []
                    for r in rows:
                        rules.append(
                            {
                                "temp_min": float(r["outdoor_temp_min"]),
                                "temp_max": float(r["outdoor_temp_max"]),
                                "target_c": float(r["target_water_temp_c"]),
                                "cooling_enabled": bool(r.get("cooling_enabled", True)),
                                "rule_name": str(r["rule_name"]),
                            }
                        )
                    logger.info("Fetched %d temperature control rules from Neon Postgres", len(rules))
                    return rules
        except Exception as exc:
            logger.warning("Could not fetch temperature rules from database: %s", exc)
            return None

    def insert_consumption_record(
        self,
        cooler_id: str,
        day_of_week: str,
        period_slot: str,
        consumption_liters: float,
    ) -> Optional[int]:
        """Record live IoT water meter consumption telemetry into Neon Postgres.

        Returns:
            The generated primary key ID, or None on failure.
        """
        if not self._can_connect():
            return None
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO consumption_records (cooler_id, day_of_week, period_slot, consumption_liters)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            str(cooler_id).strip(),
                            str(day_of_week).strip().capitalize(),
                            str(period_slot).strip(),
                            float(consumption_liters),
                        ),
                    )
                    row = cur.fetchone()
                    record_id = int(row["id"]) if row else None
                    logger.info(
                        "Logged live consumption telemetry into Neon DB: cooler=%s, slot=%s, liters=%.2f (id=%s)",
                        cooler_id,
                        period_slot,
                        consumption_liters,
                        record_id,
                    )
                    return record_id
        except Exception as exc:
            logger.error("Failed to insert consumption record: %s", exc)
            return None

    def update_cooler_level(self, cooler_id: str, current_level_liters: float) -> bool:
        """Update current water reserve level for a cooler."""
        if not self._can_connect():
            return False
        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE coolers
                        SET current_level_liters = %s, updated_at = NOW()
                        WHERE cooler_id = %s
                        """,
                        (float(current_level_liters), str(cooler_id).strip()),
                    )
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error("Failed to update cooler level for %s: %s", cooler_id, exc)
            return False

    def get_db_health(self) -> Dict[str, Any]:
        """Return connectivity state and table row counts."""
        if not is_db_configured():
            return {
                "configured": False,
                "connected": False,
                "source": "Local files (CSV/Parquet)",
                "counts": {},
            }

        try:
            with self._conn_factory() as conn:
                with conn.cursor() as cur:
                    counts = {}
                    for table in ("coolers", "consumption_records", "school_timetable", "temperature_control_rules"):
                        try:
                            cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")  # noqa: S608
                            row = cur.fetchone()
                            counts[table] = int(row["cnt"]) if row else 0
                        except Exception:
                            counts[table] = 0

                    return {
                        "configured": True,
                        "connected": True,
                        "source": "Neon Postgres Cloud",
                        "counts": counts,
                    }
        except Exception as exc:
            return {
                "configured": True,
                "connected": False,
                "source": "Local files (CSV/Parquet fallback)",
                "error": str(exc),
                "counts": {},
            }
