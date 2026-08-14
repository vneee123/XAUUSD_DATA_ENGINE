import sys
from pathlib import Path
import json
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine


REQUIRED_BASE_COLUMNS = {
    "datetime",
    "datetime_local",
    "open",
    "high",
    "low",
    "close",
    "symbol",
    "interval",
    "candle_close_time",
    "candle_status",
}

REQUIRED_FEATURE_COLUMNS = {
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "range",
    "body",
    "body_abs",
    "upper_wick",
    "lower_wick",
    "body_ratio",
    "close_position",
    "sma_20",
    "ema_20",
    "ema_50",
    "ema_200",
    "trend_score",
    "rsi_14",
    "atr_14",
    "volatility_20",
    "macd",
    "macd_signal",
    "macd_histogram",
    "stoch_k_14",
    "williams_r_14",
    "cci_14",
    "plus_di_14",
    "minus_di_14",
    "adx_14",
}


def check(condition, label, details=""):
    if condition:
        print(f"OK   {label}")
        return True

    print(f"FAIL {label}")

    if details:
        print(f"     {details}")

    return False


def run_interval(timeframe, raw_df):
    print()
    print("=" * 70)
    print(f"PIPELINE TEST: {timeframe}")
    print("=" * 70)

    passed = True

    # ------------------------------------------------------------
    # 1. RAW DATA
    # ------------------------------------------------------------
    print()
    print("[1] RAW DATA")

    passed &= check(
        raw_df is not None,
        "Collector returned dataframe"
    )

    if raw_df is None or len(raw_df) == 0:
        print("     Raw dataframe kosong.")
        return False

    print(f"     RAW ROWS: {len(raw_df)}")

    passed &= check(
        len(raw_df) > 0,
        "Raw dataframe contains rows"
    )

    # ------------------------------------------------------------
    # 2. NORMALIZER
    # ------------------------------------------------------------
    print()
    print("[2] DATA NORMALIZER")

    try:
        normalized = DataNormalizer.normalize(
            raw_df,
            source_timezone="UTC",
            target_timezone="Asia/Jakarta"
        )

        passed &= check(
            normalized is not None,
            "Normalizer returned dataframe"
        )

        passed &= check(
            len(normalized) > 0,
            "Normalized dataframe contains rows"
        )

        required_normalizer_columns = {
            "datetime",
            "datetime_local",
            "open",
            "high",
            "low",
            "close",
        }

        missing = required_normalizer_columns - set(
            normalized.columns
        )

        passed &= check(
            not missing,
            "Normalizer required columns",
            f"Missing: {sorted(missing)}" if missing else ""
        )

        print(f"     NORMALIZED ROWS: {len(normalized)}")

    except Exception as exc:
        passed = False
        print(f"FAIL Normalizer execution")
        print(f"     {type(exc).__name__}: {exc}")
        return False

    # ------------------------------------------------------------
    # 3. CANDLE STATUS
    # ------------------------------------------------------------
    print()
    print("[3] CANDLE STATUS")

    try:
        status_df = CandleStatus.add_status(normalized)

        passed &= check(
            "candle_status" in status_df.columns,
            "candle_status column exists"
        )

        if "candle_status" in status_df.columns:

            status_counts = (
                status_df["candle_status"]
                .value_counts()
                .to_dict()
            )

            print(
                f"     STATUS: {status_counts}"
            )

            passed &= check(
                status_df["candle_status"].notna().all(),
                "All candles have candle_status"
            )

            passed &= check(
                "CLOSED" in status_counts,
                "CLOSED candles detected"
            )

    except Exception as exc:
        passed = False
        print(f"FAIL CandleStatus execution")
        print(f"     {type(exc).__name__}: {exc}")
        return False

    # ------------------------------------------------------------
    # 4. CLOSED-ONLY DATA
    # ------------------------------------------------------------
    print()
    print("[4] CLOSED-ONLY FILTER")

    closed_df = status_df[
        status_df["candle_status"] == "CLOSED"
    ].copy()

    print(
        f"     CLOSED ROWS: {len(closed_df)}"
    )

    passed &= check(
        len(closed_df) > 0,
        "Closed candle dataset contains rows"
    )

    passed &= check(
        not closed_df.empty,
        "Closed candle dataframe is not empty"
    )

    # ------------------------------------------------------------
    # 5. FEATURE ENGINE
    # ------------------------------------------------------------
    print()
    print("[5] FEATURE ENGINE")

    try:
        features = FeatureEngine.calculate(
            status_df,
            closed_only=True
        )

        passed &= check(
            features is not None,
            "FeatureEngine returned dataframe"
        )

        if features is None:
            return False

        passed &= check(
            len(features) > 0,
            "Feature dataframe contains rows"
        )

        print(
            f"     FEATURE ROWS: {len(features)}"
        )

        print(
            f"     FEATURE COLUMNS: {len(features.columns)}"
        )

    except Exception as exc:
        passed = False
        print(f"FAIL FeatureEngine execution")
        print(f"     {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------
    # 6. BASE COLUMNS
    # ------------------------------------------------------------
    print()
    print("[6] BASE COLUMNS")

    missing_base = (
        REQUIRED_BASE_COLUMNS
        - set(features.columns)
    )

    passed &= check(
        not missing_base,
        "Required base columns",
        f"Missing: {sorted(missing_base)}"
        if missing_base else ""
    )

    # ------------------------------------------------------------
    # 7. FEATURE COLUMNS
    # ------------------------------------------------------------
    print()
    print("[7] FEATURE COLUMNS")

    missing_features = (
        REQUIRED_FEATURE_COLUMNS
        - set(features.columns)
    )

    passed &= check(
        not missing_features,
        "Required feature columns",
        f"Missing: {sorted(missing_features)}"
        if missing_features else ""
    )

    # ------------------------------------------------------------
    # 8. CLOSED-ONLY GUARANTEE
    # ------------------------------------------------------------
    print()
    print("[8] CLOSED-ONLY GUARANTEE")

    if "candle_status" in features.columns:

        non_closed = features[
            features["candle_status"] != "CLOSED"
        ]

        passed &= check(
            len(non_closed) == 0,
            "Feature output contains CLOSED candles only",
            f"Non-closed rows: {len(non_closed)}"
            if len(non_closed) > 0 else ""
        )

    # ------------------------------------------------------------
    # 9. FUTURE DATA PROTECTION
    # ------------------------------------------------------------
    print()
    print("[9] FUTURE DATA PROTECTION")

    if "is_future_data" in features.columns:

        future_rows = features[
            features["is_future_data"] == True
        ]

        passed &= check(
            len(future_rows) == 0,
            "No FUTURE_DATA rows in feature output",
            f"Future rows: {len(future_rows)}"
            if len(future_rows) > 0 else ""
        )

    # ------------------------------------------------------------
    # 10. TIMESTAMP VALIDATION
    # ------------------------------------------------------------
    print()
    print("[10] TIMESTAMP")

    if "datetime" in features.columns:

        dt = features["datetime"]

        passed &= check(
            dt.notna().all(),
            "Feature timestamps contain no NULL"
        )

        passed &= check(
            dt.is_monotonic_increasing,
            "Feature timestamps are sorted"
        )

        passed &= check(
            not dt.duplicated().any(),
            "Feature timestamps are unique"
        )

    # ------------------------------------------------------------
    # 11. FEATURE NaN STRUCTURE
    # ------------------------------------------------------------
    print()
    print("[11] FEATURE NaN STRUCTURE")

    feature_columns = [
        c for c in features.columns
        if c not in REQUIRED_BASE_COLUMNS
    ]

    null_counts = (
        features[feature_columns]
        .isna()
        .sum()
    )

    nonzero_nulls = null_counts[
        null_counts > 0
    ]

    print(
        f"     FEATURE COLUMNS: {len(feature_columns)}"
    )

    print(
        f"     COLUMNS WITH NaN: {len(nonzero_nulls)}"
    )

    if len(nonzero_nulls) > 0:
        print(
            nonzero_nulls.to_string()
        )

    # NaN pada bagian awal rolling indicators
    # diperbolehkan. Yang penting baris terakhir
    # tidak memiliki NaN pada feature utama.

    latest = features.tail(1)

    critical_latest = [
        c for c in REQUIRED_FEATURE_COLUMNS
        if c in features.columns
    ]

    latest_nulls = (
        latest[critical_latest]
        .isna()
        .sum()
    )

    latest_bad = latest_nulls[
        latest_nulls > 0
    ]

    passed &= check(
        len(latest_bad) == 0,
        "Latest critical features contain no NaN",
        f"NaN: {latest_bad.to_dict()}"
        if len(latest_bad) > 0 else ""
    )

    # ------------------------------------------------------------
    # 12. FINAL RESULT
    # ------------------------------------------------------------
    print()
    print("-" * 70)

    if passed:
        print(f"{timeframe:<10}: PASS")
    else:
        print(f"{timeframe:<10}: FAILED")

    return bool(passed)


def main():

    print("=" * 70)
    print("PIPELINE INTEGRATION TEST")
    print("=" * 70)

    print()
    print(f"PROJECT ROOT: {PROJECT_ROOT}")

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

    print()
    print("COLLECTING MARKET DATA...")
    print(
        "TIMEFRAMES:",
        settings["timeframes"]
    )

    collector = MarketDataCollector()

    try:
        datasets = collector.collect(
            symbol=settings["symbol"],
            timeframes=settings["timeframes"]
        )
    except Exception as exc:
        print()
        print("COLLECTOR FAILED")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)

    results = {}

    for timeframe in settings["timeframes"]:

        if timeframe not in datasets:
            print()
            print(
                f"{timeframe:<10}: FAILED"
            )
            print(
                "Reason: timeframe tidak dikembalikan Collector."
            )
            results[timeframe] = False
            continue

        results[timeframe] = run_interval(
            timeframe,
            datasets[timeframe]
        )

    print()
    print("=" * 70)
    print("FINAL PIPELINE RESULT")
    print("=" * 70)

    for timeframe, result in results.items():
        print(
            f"{timeframe:<10}: "
            f"{'PASS' if result else 'FAILED'}"
        )

    all_pass = all(results.values())

    print()

    if all_pass:
        print(
            "PIPELINE INTEGRATION TEST: ALL PASS"
        )
        sys.exit(0)

    print(
        "PIPELINE INTEGRATION TEST: FAILED"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
