"""One-click seeding script to migrate school water cooler dataset into Neon Serverless Postgres.

Usage:
    python scripts/seed_neon.py [--force-reseed] [--database-url postgresql://...]

Environment Variables:
    DATABASE_URL or NEON_DATABASE_URL
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db.connection import get_database_url, get_db_connection
from model.preprocessing import (
    load_consumption_record,
    load_cooler_config,
    load_timetable,
)
from temp_control.controller import TemperatureController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed-neon")


def run_migration_ddl(conn) -> None:
    """Execute schema.sql DDL to create tables if they do not exist."""
    schema_path = ROOT / "src" / "db" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    logger.info("Applying DDL from %s...", schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    with conn.cursor() as cur:
        cur.execute(ddl)
    logger.info("DDL applied successfully.")


def seed_coolers(conn, data_dir: Path, force: bool = False) -> int:
    """Seed 'coolers' table from cooler_config.csv / cooler_config.xlsx."""
    cooler_path = data_dir / "cooler_config.csv"
    if not cooler_path.exists():
        cooler_path = data_dir / "cooler_config.xlsx"
    if not cooler_path.exists():
        logger.warning("No cooler config file found in %s", data_dir)
        return 0

    meta = load_cooler_config(cooler_path)
    logger.info("Parsed %d coolers from %s", len(meta), cooler_path.name)

    with conn.cursor() as cur:
        # Check if locations table exists (IoT schema)
        cur.execute("SELECT to_regclass('public.locations') AS loc_tbl")
        has_locations = bool(cur.fetchone().get("loc_tbl"))

        inserted = 0
        if has_locations:
            for cid, cfg in meta.items():
                cap = float(cfg.get("capacity_liters", 200))
                floor = int(cfg.get("floor", 0))
                near_games = bool(cfg.get("near_games", False))
                loc_desc = cfg.get("location", "unknown")

                cur.execute(
                    """
                    UPDATE coolers c
                    SET 
                        tank_capacity_litre = %s,
                        capacity_liters = %s,
                        floor = %s,
                        near_games = %s,
                        location = %s,
                        last_updated_at = NOW(),
                        updated_at = NOW()
                    FROM locations l
                    WHERE c.location_id = l.location_id AND l.location_name = %s
                    """,
                    (cap, cap, floor, near_games, loc_desc, cid),
                )
                if cur.rowcount > 0:
                    inserted += cur.rowcount
        else:
            if force:
                cur.execute("DELETE FROM coolers")

            for cid, cfg in meta.items():
                cur.execute(
                    """
                    INSERT INTO coolers (cooler_id, location, capacity_liters, floor, near_games, current_level_liters)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cooler_id) DO UPDATE SET
                        location = EXCLUDED.location,
                        capacity_liters = EXCLUDED.capacity_liters,
                        floor = EXCLUDED.floor,
                        near_games = EXCLUDED.near_games,
                        current_level_liters = EXCLUDED.current_level_liters,
                        updated_at = NOW()
                    """,
                    (
                        cid,
                        cfg.get("location", "unknown"),
                        float(cfg.get("capacity_liters", 200)),
                        int(cfg.get("floor", 0)),
                        bool(cfg.get("near_games", False)),
                        cfg.get("current_level_liters"),
                    ),
                )
                inserted += 1

    logger.info("Seeded/Updated %d coolers.", inserted)
    return inserted


def seed_consumption(conn, data_dir: Path, force: bool = False) -> int:
    """Seed 'consumption_records' table from Cooler_Water_Consumption_Record."""
    cons_path = data_dir / "Cooler_Water_Consumption_Record.csv"
    if not cons_path.exists():
        cons_path = data_dir / "Cooler_Water_Consumption_Record.xlsx"
    if not cons_path.exists():
        logger.warning("No consumption file found in %s", data_dir)
        return 0

    df = load_consumption_record(cons_path)
    logger.info("Parsed %d consumption records from %s", len(df), cons_path.name)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM consumption_records")
        existing_cnt = cur.fetchone()["cnt"]

        if existing_cnt > 0 and not force:
            logger.info("consumption_records already contains %d rows (skipping, use --force-reseed to overwrite)", existing_cnt)
            return existing_cnt

        if force:
            cur.execute("DELETE FROM consumption_records")

        # Batch insert
        records = [
            (
                str(row["cooler_id"]).strip(),
                str(row["day_of_week"]).strip().capitalize(),
                str(row["period_slot"]).strip(),
                float(row["consumption_liters"]),
            )
            for _, row in df.iterrows()
        ]

        cur.executemany(
            """
            INSERT INTO consumption_records (cooler_id, day_of_week, period_slot, consumption_liters)
            VALUES (%s, %s, %s, %s)
            """,
            records,
        )

    logger.info("Seeded %d consumption records.", len(records))
    return len(records)


def seed_timetable(conn, data_dir: Path, force: bool = False) -> int:
    """Seed 'school_timetable' table from School_Timetable.csv / xlsx."""
    tt_path = data_dir / "School_Timetable_With_Recess.xlsx"
    if not tt_path.exists():
        tt_path = data_dir / "School_Timetable_With_Recess.csv"
    if not tt_path.exists():
        tt_path = data_dir / "School_Timetable.xlsx"
    if not tt_path.exists():
        tt_path = data_dir / "School_Timetable.csv"
    if not tt_path.exists():
        logger.warning("No timetable file found in %s", data_dir)
        return 0

    df = load_timetable(tt_path)
    logger.info("Parsed %d timetable entries from %s", len(df), tt_path.name)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM school_timetable")
        existing_cnt = cur.fetchone()["cnt"]

        if existing_cnt > 0 and not force:
            logger.info("school_timetable already contains %d rows (skipping, use --force-reseed to overwrite)", existing_cnt)
            return existing_cnt

        if force:
            cur.execute("DELETE FROM school_timetable")

        records = [
            (
                str(row["class_name"]).strip(),
                str(row["day_of_week"]).strip().capitalize(),
                str(row["period_slot"]).strip(),
                str(row.get("subject", "General")).strip(),
                bool(row.get("is_games_period", False)),
            )
            for _, row in df.iterrows()
        ]

        cur.executemany(
            """
            INSERT INTO school_timetable (class_name, day_of_week, period_slot, subject, is_games_period)
            VALUES (%s, %s, %s, %s, %s)
            """,
            records,
        )

    logger.info("Seeded %d timetable rows.", len(records))
    return len(records)


def seed_temperature_rules(conn, force: bool = False) -> int:
    """Seed IS 1475:2024 water dispenser temperature control setpoint rules."""
    ctrl = TemperatureController()
    rules = ctrl.rules

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM temperature_control_rules")
        existing_cnt = cur.fetchone()["cnt"]

        if existing_cnt > 0 and not force:
            logger.info("temperature_control_rules already contains %d rows (skipping, use --force-reseed to overwrite)", existing_cnt)
            return existing_cnt

        if force:
            cur.execute("DELETE FROM temperature_control_rules")

        records = [
            (
                float(r["ambient_min_c"]),
                float(r["ambient_max_c"]),
                float(r["target_c"] if r.get("target_c") is not None else 20.0),
                bool(r.get("cooling_enabled", True)),
                str(r.get("notes") or f"Rule {r['ambient_min_c']} to {r['ambient_max_c']} C"),
            )
            for r in rules
        ]

        cur.executemany(
            """
            INSERT INTO temperature_control_rules (outdoor_temp_min, outdoor_temp_max, target_water_temp_c, cooling_enabled, rule_name)
            VALUES (%s, %s, %s, %s, %s)
            """,
            records,
        )

    logger.info("Seeded %d temperature control rules.", len(records))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Neon Postgres database from local data files.")
    parser.add_argument("--database-url", type=str, default=None, help="PostgreSQL connection string (overrides DATABASE_URL)")
    parser.add_argument("--force-reseed", action="store_true", help="Truncate and reseed all tables")
    args = parser.parse_args()

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    url = get_database_url()
    if not url:
        logger.error(
            "DATABASE_URL or NEON_DATABASE_URL environment variable is required.\n"
            "Usage: python scripts/seed_neon.py --database-url postgresql://user:pass@ep-xyz.neon.tech/neondb?sslmode=require"
        )
        sys.exit(1)

    logger.info("Connecting to Neon Postgres...")
    conn = get_db_connection()

    try:
        data_dir = ROOT / "data"
        run_migration_ddl(conn)
        n_coolers = seed_coolers(conn, data_dir, force=args.force_reseed)
        n_cons = seed_consumption(conn, data_dir, force=args.force_reseed)
        n_tt = seed_timetable(conn, data_dir, force=args.force_reseed)
        n_rules = seed_temperature_rules(conn, force=args.force_reseed)

        logger.info("=" * 60)
        logger.info("Neon Postgres Database Seeding Complete:")
        logger.info("  Coolers:                   %d rows", n_coolers)
        logger.info("  Consumption Telemetry:     %d rows", n_cons)
        logger.info("  Timetable & Footfall:      %d rows", n_tt)
        logger.info("  IS 1475 Temperature Rules: %d rows", n_rules)
        logger.info("=" * 60)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
