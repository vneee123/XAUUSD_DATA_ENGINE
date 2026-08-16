import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "xauusd.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

class Database:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    symbol TEXT,
                    interval TEXT,
                    timeframe TEXT,
                    candle_close_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS features_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT,
                    timeframe TEXT,
                    candle_status TEXT,
                    close REAL,
                    feature_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TEXT,
                    timeframe TEXT,
                    signal TEXT,
                    entry REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    confidence REAL,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_raw_datetime ON raw_data(datetime);
                CREATE INDEX IF NOT EXISTS idx_features_datetime ON features_data(datetime);
                CREATE INDEX IF NOT EXISTS idx_signals_datetime ON signals(datetime);
            ''')

    def insert_raw(self, df, timeframe):
        if df.empty:
            return
        df = df.copy()
        df["timeframe"] = timeframe
        # Ensure required columns
        required = ["datetime", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                df[col] = None
        # Convert datetime to string
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        # Select columns that exist in the table
        cols = ["datetime", "open", "high", "low", "close", "volume", "symbol", "interval", "timeframe", "candle_close_time"]
        existing_cols = [c for c in cols if c in df.columns]
        with sqlite3.connect(self.db_path) as conn:
            df[existing_cols].to_sql("raw_data", conn, if_exists="append", index=False)

    def insert_features(self, df, timeframe):
        if df.empty:
            return
        df = df.copy()
        df["timeframe"] = timeframe
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        # Store all features as JSON
        feature_cols = [col for col in df.columns if col not in ["datetime", "timeframe", "candle_status", "close"]]
        df["feature_json"] = df[feature_cols].to_json(orient="records")
        with sqlite3.connect(self.db_path) as conn:
            df[["datetime", "timeframe", "candle_status", "close", "feature_json"]].to_sql(
                "features_data", conn, if_exists="append", index=False
            )

    def insert_signal(self, signal_data, timeframe, model_version="v1.8"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (datetime, timeframe, signal, entry, stop_loss, take_profit_1, take_profit_2, confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                timeframe,
                signal_data.get("signal"),
                signal_data.get("entry"),
                signal_data.get("stop_loss"),
                signal_data.get("take_profit_1"),
                signal_data.get("take_profit_2"),
                signal_data.get("confidence"),
                model_version
            ))
            conn.commit()

    def get_raw_data(self, timeframe, limit=100):
        query = "SELECT * FROM raw_data WHERE timeframe = ? ORDER BY datetime DESC LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(timeframe, limit))

    def get_features_data(self, timeframe, limit=100):
        query = "SELECT datetime, timeframe, candle_status, close, feature_json FROM features_data WHERE timeframe = ? ORDER BY datetime DESC LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(timeframe, limit))

    def get_recent_signals(self, limit=20):
        query = "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(limit,))
