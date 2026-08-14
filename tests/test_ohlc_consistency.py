import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus


def check_ohlc(df, timeframe):

    errors = []

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        errors.append(
            f"Missing columns: {missing}"
        )
        return errors

    # NULL
    null_counts = df[required].isna().sum()

    for column, count in null_counts.items():
        if count > 0:
            errors.append(
                f"{column}: {count} NULL values"
            )

    # Positive prices
    for column in ["open", "high", "low", "close"]:

        invalid = (
            df[column] <= 0
        ).sum()

        if invalid > 0:
            errors.append(
                f"{column}: {invalid} non-positive values"
            )

    # High must contain open and close
    invalid_high = (
        df["high"]
        < df[["open", "close"]].max(axis=1)
    ).sum()

    if invalid_high > 0:
        errors.append(
            f"high < max(open, close): {invalid_high}"
        )

    # Low must contain open and close
    invalid_low = (
        df["low"]
        > df[["open", "close"]].min(axis=1)
    ).sum()

    if invalid_low > 0:
        errors.append(
            f"low > min(open, close): {invalid_low}"
        )

    # High >= Low
    invalid_range = (
        df["high"] < df["low"]
    ).sum()

    if invalid_range > 0:
        errors.append(
            f"high < low: {invalid_range}"
        )

    # Timestamp sorted
    if not df["datetime"].is_monotonic_increasing:
        errors.append(
            "datetime is not sorted ascending"
        )

    # Duplicate timestamp
    duplicate_count = (
        df["datetime"].duplicated().sum()
    )

    if duplicate_count > 0:
        errors.append(
            f"duplicate timestamps: {duplicate_count}"
        )

    # Expected interval
    interval_minutes = {
        "1h": 60,
        "15min": 15,
        "5min": 5
    }

    expected = interval_minutes.get(
        timeframe.lower()
    )

    if expected is not None and len(df) > 1:

        delta = (
            df["datetime"]
            .diff()
            .dt.total_seconds()
            / 60
        )

        gaps = (
            delta.iloc[1:] != expected
        ).sum()

        if gaps > 0:
            errors.append(
                f"unexpected timestamp gaps: {gaps}"
            )

    return errors


def run_interval(
    timeframe,
    raw_df
):

    print()
    print("=" * 70)
    print(
        f"OHLC CONSISTENCY TEST: {timeframe}"
    )
    print("=" * 70)

    df = DataNormalizer.normalize(
        raw_df,
        source_timezone="UTC",
        target_timezone="Asia/Jakarta"
    )

    df = CandleStatus.add_status(df)

    print(
        "ROWS:",
        len(df)
    )

    errors = check_ohlc(
        df,
        timeframe
    )

    if errors:

        print()
        print("RESULT: FAILED")

        for error in errors:
            print(
                "ERROR:",
                error
            )

        return False

    print()
    print("RESULT: PASSED")

    print()
    print("LATEST CANDLE:")

    columns = [
        "datetime",
        "datetime_local",
        "open",
        "high",
        "low",
        "close",
        "candle_status"
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df.tail(1)[available]
        .to_string(index=False)
    )

    return True


def main():

    settings_path = (
        PROJECT_ROOT
        / "config"
        / "settings.json"
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

    results = {}

    for timeframe, raw_df in datasets.items():

        try:

            results[timeframe] = run_interval(
                timeframe,
                raw_df
            )

        except Exception as e:

            print()
            print(
                f"ERROR {timeframe}: "
                f"{type(e).__name__}"
            )

            print(e)

            results[timeframe] = False

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_passed = True

    for timeframe, passed in results.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{timeframe.upper():8} : {status}"
        )

        if not passed:
            all_passed = False

    print()

    if all_passed:

        print(
            "OHLC CONSISTENCY: ALL PASSED"
        )

    else:

        print(
            "OHLC CONSISTENCY: FAILED"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
