-- PaniPaharaPurvanuman Neon Postgres Schema
-- School Water Cooler Demand & IS 1475:2024 Temperature Control Tables

-- 1. Cooler Metadata
CREATE TABLE IF NOT EXISTS coolers (
    cooler_id VARCHAR(32) PRIMARY KEY,
    location VARCHAR(128) NOT NULL DEFAULT 'unknown',
    capacity_liters NUMERIC(8, 2) NOT NULL,
    floor INTEGER NOT NULL DEFAULT 0,
    near_games BOOLEAN NOT NULL DEFAULT FALSE,
    current_level_liters NUMERIC(8, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Time-Series Water Consumption Telemetry
CREATE TABLE IF NOT EXISTS consumption_records (
    id BIGSERIAL PRIMARY KEY,
    cooler_id VARCHAR(32) NOT NULL REFERENCES coolers(cooler_id) ON DELETE CASCADE,
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
