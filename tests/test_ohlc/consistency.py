import sys
from pathlib import Path
import json
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus


# ============================================================
# EXPECTED TIMEFRAME
# ============================================================

EXPECTED_INTERVAL_MINUTES = {
    "H1": 60,
    "M15": 15,
    "M5": 5,
}


# ============================================================
# SETTINGS
# ============================================================

def load_settings():

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

        return json.load(f)


# ============================================================
# OHLC CHECK
# ============================================================

def check_ohlc(df):

    report = {}

    required_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required_columns:

        if column not in df.columns:

            report[f"{column}_missing"] = len(df)

            continue

        report[f"{column}_null"] = int(
            df[column].isna().sum()
        )

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        report[f"{column}_non_numeric"] = int(
            numeric.isna().sum()
            - df[column].isna().sum()
        )

    # --------------------------------------------------------
    # Convert OHLC to numeric
    # --------------------------------------------------------

    ohlc = df.copy()

    for column in required_columns:

        if column in ohlc.columns:

            ohlc[column] = pd.to_numeric(
                ohlc[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # High / Low consistency
    # --------------------------------------------------------

    if all(
        column in ohlc.columns
        for column in required_columns
    ):

        invalid_high = (
            ohlc["high"]
            < ohlc[
                ["open", "close"]
            ].max(axis=1)
        )

        invalid_low = (
            ohlc["low"]
            > ohlc[
                ["open", "close"]
            ].min(axis=1)
        )

        report["invalid_high"] = int(
            invalid_high.sum()
        )

        report["invalid_low"] = int(
            invalid_low.sum()
        )

        report["non_positive_prices"] = int(
            (
                ohlc[
                    required_columns
                ] <= 0
            )
            .any(axis=1)
            .sum()
        )

        candle_range = (
            ohlc["high"] -
            ohlc["low"]
        )

        report["negative_range"] = int(
            (candle_range < 0).sum()
        )

        report["zero_range"] = int(
            (candle_range == 0).sum()
        )

    return report


# ============================================================
# TIMESTAMP CHECK
# ============================================================

def check_timestamps(df):

    report = {
        "invalid_timestamps": 0,
        "duplicate_timestamps": 0,
        "timestamp_not_sorted": False,
    }

    if "datetime" not in df.columns:

        report["invalid_timestamps"] = len(df)

        return report

    timestamps = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    )

    report["invalid_timestamps"] = int(
        timestamps.isna().sum()
    )

    valid_timestamps = timestamps.dropna()

    report["duplicate_timestamps"] = int(
        valid_timestamps.duplicated().sum()
    )

    if len(valid_timestamps) > 1:

        report["timestamp_not_sorted"] = not (
            valid_timestamps.is_monotonic_increasing
        )

    return report


# ============================================================
# INTERVAL CHECK
# ============================================================

def check_interval(
    df,
    timeframe
):

    expected_minutes = (
        EXPECTED_INTERVAL_MINUTES
        .get(timeframe.upper())
    )

    if expected_minutes is None:

        return {
            "checked": False,
            "reason": f"Timeframe tidak dikenal: {timeframe}",
        }

    if "datetime" not in df.columns:

        return {
            "checked": False,
            "reason": "Kolom datetime tidak ditemukan",
        }

    timestamps = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    ).dropna()

    if len(timestamps) < 2:

        return {
            "checked": False,
            "reason": "Data tidak cukup untuk interval check",
        }

    differences = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Jangan menganggap semua gap harus sama.
    # Market dapat mempunyai gap/session break.
    #
    # Yang kita anggap invalid adalah:
    #
    # 1. interval <= 0
    # 2. interval lebih kecil dari expected
    #
    # Gap yang lebih besar dari expected dicatat sebagai
    # market/session gap, bukan langsung corruption.
    # --------------------------------------------------------

    invalid_intervals = (
        differences <= 0
    )

    too_short_intervals = (
        differences < expected_minutes
    )

    valid_or_gap = (
        differences >= expected_minutes
    )

    market_gaps = (
        differences > expected_minutes
    )

    report = {
        "checked": True,
        "expected_minutes": expected_minutes,
        "min_observed_minutes": (
            float(differences.min())
            if len(differences) > 0
            else None
        ),
        "max_observed_minutes": (
            float(differences.max())
            if len(differences) > 0
            else None
        ),
        "invalid_intervals": int(
            invalid_intervals.sum()
        ),
        "too_short_intervals": int(
            too_short_intervals.sum()
        ),
        "market_session_gaps": int(
            market_gaps.sum()
        ),
    }

    return report


# ============================================================
# CANDLE STATUS CHECK
# ============================================================

def check_candle_status(df):

    report = {
        "status_column_missing": False,
        "closed_count": 0,
        "open_count": 0,
        "future_count": 0,
    }

    if "candle_status" not in df.columns:

        report["status_column_missing"] = True

        return report

    counts = (
        df["candle_status"]
        .value_counts()
        .to_dict()
    )

    report["closed_count"] = int(
        counts.get("CLOSED", 0)
    )

    report["open_count"] = int(
        counts.get("OPEN", 0)
    )

    report["future_count"] = int(
        counts.get("FUTURE_DATA", 0)
    )

    return report


# ============================================================
# CANDLE CLOSE TIME CHECK
# ============================================================

def check_candle_close_time(
    df,
    timeframe
):

    report = {
        "column_missing": False,
        "invalid_close_time": 0,
        "close_time_before_open": 0,
    }

    if "candle_close_time" not in df.columns:

        report["column_missing"] = True

        return report

    if "datetime" not in df.columns:

        report["invalid_close_time"] = len(df)

        return report

    datetime_values = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    )

    close_time_values = pd.to_datetime(
        df["candle_close_time"],
        errors="coerce",
        utc=True
    )

    report["invalid_close_time"] = int(
        close_time_values.isna().sum()
    )

    valid = (
        datetime_values.notna()
        &
        close_time_values.notna()
    )

    report["close_time_before_open"] = int(
        (
            close_time_values[valid]
            <=
            datetime_values[valid]
        )
        .sum()
    )

    return report


# ============================================================
# REPORT VALIDATION
# ============================================================

def determine_result(
    ohlc,
    timestamps,
    interval,
    status,
    close_time
):

    failures = []

    # OHLC
    for key, value in ohlc.items():

        if key.endswith("_missing") and value > 0:
            failures.append(key)

        elif key.endswith("_null") and value > 0:
            failures.append(key)

        elif key.endswith("_non_numeric") and value > 0:
            failures.append(key)

        elif key in [
            "invalid_high",
            "invalid_low",
            "non_positive_prices",
            "negative_range",
            "zero_range",
        ] and value > 0:

            failures.append(key)

    # Timestamp
    if timestamps["invalid_timestamps"] > 0:
        failures.append("invalid_timestamps")

    if timestamps["duplicate_timestamps"] > 0:
        failures.append("duplicate_timestamps")

    if timestamps["timestamp_not_sorted"]:
        failures.append("timestamp_not_sorted")

    # Interval
    if not interval.get("checked", False):
        failures.append("interval_not_checked")

    if interval.get("invalid_intervals", 0) > 0:
        failures.append("invalid_intervals")

    if interval.get("too_short_intervals", 0) > 0:
        failures.append("too_short_intervals")

    # Candle status
    if status["status_column_missing"]:
        failures.append("status_column_missing")

    if status["closed_count"] == 0:
        failures.append("no_closed_candles")

    # Close time
    if close_time["column_missing"]:
        failures.append("candle_close_time_missing")

    if close_time["invalid_close_time"] > 0:
        failures.append("invalid_close_time")

    if close_time["close_time_before_open"] > 0:
        failures.append("close_time_before_open")

    return failures


# ============================================================
# RUN ONE TIMEFRAME
# ============================================================

def run_timeframe(
    timeframe,
    raw_df
):

    print()
    print("=" * 70)
    print(f"OHLC CONSISTENCY: {timeframe}")
    print("=" * 70)

    print()
    print("RAW ROWS:", len(raw_df))

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = DataNormalizer.normalize(
        raw_df,
        source_timezone="UTC",
        target_timezone="Asia/Jakarta"
    )

    # --------------------------------------------------------
    # Candle status
    # --------------------------------------------------------

    df = CandleStatus.add_status(df)

    # --------------------------------------------------------
    # OHLC
    # --------------------------------------------------------

    ohlc_report = check_ohlc(df)

    print()
    print("OHLC CHECK:")
    print(
        json.dumps(
            ohlc_report,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_report = check_timestamps(df)

    print()
    print("TIMESTAMP CHECK:")
    print(
        json.dumps(
            timestamp_report,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Interval
    # --------------------------------------------------------

    interval_report = check_interval(
        df,
        timeframe
    )

    print()
    print("INTERVAL CHECK:")
    print(
        json.dumps(
            interval_report,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Candle status
    # --------------------------------------------------------

    status_report = check_candle_status(df)

    print()
    print("CANDLE STATUS:")
    print(
        json.dumps(
            status_report,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Candle close time
    # --------------------------------------------------------

    close_time_report = check_candle_close_time(
        df,
        timeframe
    )

    print()
    print("CANDLE CLOSE TIME CHECK:")
    print(
        json.dumps(
            close_time_report,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Latest candles
    # --------------------------------------------------------

    display_columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "candle_close_time",
        "candle_status",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    print()
    print("LATEST CANDLES:")

    print(
        df[
            available_columns
        ]
        .tail(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    failures = determine_result(
        ohlc_report,
        timestamp_report,
        interval_report,
        status_report,
        close_time_report
    )

    print()
    print("-" * 70)

    if failures:

        print("RESULT: FAIL")

        print()
        print("FAILURES:")

        for failure in failures:

            print(
                f"  - {failure}"
            )

        return False

    print("RESULT: PASS")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OHLC CONSISTENCY TEST")
    print("=" * 70)

    print()
    print(
        "PROJECT ROOT:",
        PROJECT_ROOT
    )

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    settings = load_settings()

    print()
    print("COLLECTING MARKET DATA...")

    print(
        "TIMEFRAMES:",
        settings["timeframes"]
    )

    # --------------------------------------------------------
    # Collector
    # --------------------------------------------------------

    collector = MarketDataCollector()

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    results = {}

    # --------------------------------------------------------
    # Test each timeframe
    # --------------------------------------------------------

    for timeframe, raw_df in datasets.items():

        results[timeframe] = run_timeframe(
            timeframe,
            raw_df
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    for timeframe, passed in results.items():

        print(
            f"{timeframe:<10}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if all(results.values()):

        print(
            "OHLC CONSISTENCY TEST: ALL PASS"
        )

        return 0

    print(
        "OHLC CONSISTENCY TEST: FAILED"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
