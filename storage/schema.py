import sqlite3

SCHEMA = """
-- Raw data table
CREATE TABLE IF NOT EXISTS raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timeframe TEXT NOT NULL,
    datetime TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    symbol TEXT,
    interval TEXT,
    collected_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Feature data table
CREATE TABLE IF NOT EXISTS feature_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timeframe TEXT NOT NULL,
    datetime TEXT NOT NULL,
    close REAL,
    candle_status TEXT,
    -- Add all feature columns dynamically, but we'll use JSON for flexibility
    features_json TEXT,
    collected_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Dataset table (training data)
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT NOT NULL,
    X_json TEXT,
    y REAL,
    feature_names_json TEXT,
    collected_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_time TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    entry REAL,
    stop_loss REAL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    close_price REAL,
    confidence REAL,
    model_used TEXT,
    executed INTEGER DEFAULT 0,
    execution_time TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT,
    accuracy REAL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
    feature_count INTEGER,
    model_file_path TEXT,
    hyperparameters_json TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_raw_timeframe ON raw_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_raw_datetime ON raw_data(datetime);
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time);
"""
