-- PaniPaharaPurvanuman Neon Postgres Schema
-- School Water Cooler Demand & IS 1475:2024 Temperature Control Tables

-- 1. Ensure Cooler metadata columns exist on coolers table
CREATE TABLE IF NOT EXISTS coolers (
    cooler_id BIGINT PRIMARY KEY,
    location VARCHAR(128) NOT NULL DEFAULT 'unknown',
    capacity_liters NUMERIC(8, 2) NOT NULL DEFAULT 200.0,
    floor INTEGER NOT NULL DEFAULT 0,
    near_games BOOLEAN NOT NULL DEFAULT FALSE,
    current_level_liters NUMERIC(8, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE coolers ADD COLUMN IF NOT EXISTS location VARCHAR(128) DEFAULT 'unknown';
ALTER TABLE coolers ADD COLUMN IF NOT EXISTS capacity_liters NUMERIC(8, 2);
ALTER TABLE coolers ADD COLUMN IF NOT EXISTS floor INTEGER DEFAULT 0;
ALTER TABLE coolers ADD COLUMN IF NOT EXISTS near_games BOOLEAN DEFAULT FALSE;
ALTER TABLE coolers ADD COLUMN IF NOT EXISTS current_level_liters NUMERIC(8, 2);
ALTER TABLE coolers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- 2. Time-Series Water Consumption Telemetry
CREATE TABLE IF NOT EXISTS consumption_records (
    id BIGSERIAL PRIMARY KEY,
    cooler_id VARCHAR(32) NOT NULL,
    day_of_week VARCHAR(16) NOT NULL,
    period_slot VARCHAR(16) NOT NULL,
    consumption_liters NUMERIC(8, 2) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consumption_lookup 
    ON consumption_records(cooler_id, day_of_week, period_slot);

-- 3. School Timetable & Student Density Footfall
CREATE TABLE IF NOT EXISTS school_timetable (
    id SERIAL PRIMARY KEY,
    class_name VARCHAR(32) NOT NULL,
    day_of_week VARCHAR(16) NOT NULL,
    period_slot VARCHAR(16) NOT NULL,
    subject VARCHAR(64),
    is_games_period BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_timetable_lookup 
    ON school_timetable(day_of_week, period_slot);

-- 4. IS 1475:2024 Water Temperature Control Rules
CREATE TABLE IF NOT EXISTS temperature_control_rules (
    id SERIAL PRIMARY KEY,
    outdoor_temp_min NUMERIC(5, 1) NOT NULL,
    outdoor_temp_max NUMERIC(5, 1) NOT NULL,
    target_water_temp_c NUMERIC(5, 1) NOT NULL,
    cooling_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    rule_name VARCHAR(64) NOT NULL
);
