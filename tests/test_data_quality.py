import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.data_validator import DataValidator
from core.candle_status import CandleStatus


def main():

    settings_path = (
        PROJECT_ROOT /
        "config" /
        "settings.json"
    )

    with open(
        settings_path,
        "r",
        encoding="utf-8-sig"
    ) as f:
        settings = json.load(f)

    collector = MarketDataCollector()

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    for timeframe, raw_df in datasets.items():

        print()
        print("=" * 70)
        print(f"DATA QUALITY: {timeframe}")
        print("=" * 70)

        clean_df = DataNormalizer.normalize(
            raw_df,
            source_timezone="UTC",
            target_timezone="Asia/Jakarta"
        )

        clean_df = CandleStatus.add_status(
            clean_df
        )

        report = DataValidator.validate(
            clean_df
        )

        print(json.dumps(
            report,
            indent=4
        ))

        print()
        print("LATEST CANDLES:")

        columns = [
            "datetime",
            "datetime_local",
            "open",
            "high",
            "low",
            "close",
            "candle_close_time",
            "candle_status"
        ]

        print(
            clean_df[
                columns
            ].tail(5).to_string(index=False)
        )

        print()
        print("STATUS COUNT:")

        print(
            clean_df[
                "candle_status"
            ].value_counts().to_string()
        )


if __name__ == "__main__":
    main()
