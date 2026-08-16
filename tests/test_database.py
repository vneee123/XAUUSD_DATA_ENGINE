import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.database import Database
import pandas as pd

def test_database():
    db = Database()
    # Test insert raw with timeframe
    test_df = pd.DataFrame({
        "datetime": ["2026-08-16 17:00:00"],
        "open": [4375.0],
        "high": [4376.0],
        "low": [4374.0],
        "close": [4375.5],
        "volume": [100],
        "symbol": ["XAUUSD"],
        "interval": ["1h"],
        "candle_close_time": ["2026-08-16 18:00:00"]
    })
    db.insert_raw(test_df, "H1")
    result = db.get_raw_data("H1", limit=1)
    assert not result.empty, "Database insert/retrieve failed"
    assert "timeframe" in result.columns, "timeframe column missing"
    print("Database test passed")

if __name__ == "__main__":
    test_database()
