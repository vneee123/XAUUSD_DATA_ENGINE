import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector


def main():

    settings_path = PROJECT_ROOT / "config" / "settings.json"

    with open(
        settings_path,
        "r",
        encoding="utf-8-sig"
    ) as f:
        settings = json.load(f)

    collector = MarketDataCollector()

    data = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    for timeframe, df in data.items():

        print()
        print("=" * 60)
        print(timeframe)
        print("=" * 60)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))

        print()
        print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
