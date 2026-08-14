import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine


BASE_COLUMNS = {
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

METADATA_COLUMNS = {
    "candle_closed",
    "interval_minutes",
    "is_current_candle",
    "is_future_data",
    "source_timezone",
    "target_timezone",
}


def assert_series_close(actual, expected, name, tolerance=1e-8):
    actual = pd.to_numeric(actual, errors="coerce")
    expected = pd.to_numeric(expected, errors="coerce")

    mask = actual.notna() & expected.notna()

    if not mask.any():
        raise AssertionError(
            f"{name}: tidak ada nilai valid untuk dibandingkan"
        )

    if not np.allclose(
        actual[mask].to_numpy(dtype=float),
        expected[mask].to_numpy(dtype=float),
        rtol=1e-6,
        atol=tolerance,
        equal_nan=True,
    ):
        diff = (
            actual[mask].to_numpy(dtype=float)
            - expected[mask].to_numpy(dtype=float)
        )

        max_diff = np.nanmax(np.abs(diff))

        raise AssertionError(
            f"{name}: mismatch. MAX ABS DIFF={max_diff}"
        )


def calculate_independent_features(df):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    expected = {}

    # ---------------------------------------------------------
    # RETURNS
    # ---------------------------------------------------------

    expected["return_1"] = close.pct_change(1)
    expected["return_3"] = close.pct_change(3)
    expected["return_5"] = close.pct_change(5)
    expected["return_10"] = close.pct_change(10)

    # ---------------------------------------------------------
    # GAP
    # ---------------------------------------------------------

    expected["gap"] = open_ - close.shift(1)

    previous_close = close.shift(1)

    expected["gap_pct"] = (
        (open_ - previous_close)
        / previous_close
    )

    # ---------------------------------------------------------
    # RANGE
    # ---------------------------------------------------------

    candle_range = high - low

    expected["range_mean_5"] = (
        candle_range.rolling(5).mean()
    )

    expected["range_std_5"] = (
        candle_range.rolling(5).std()
    )

    expected["close_position_mean_5"] = (
        ((close - low) / candle_range.replace(0, np.nan))
        .rolling(5)
        .mean()
    )

    expected["range_mean_10"] = (
        candle_range.rolling(10).mean()
    )

    expected["range_std_10"] = (
        candle_range.rolling(10).std()
    )

    expected["close_position_mean_10"] = (
        ((close - low) / candle_range.replace(0, np.nan))
        .rolling(10)
        .mean()
    )

    expected["range_mean_20"] = (
        candle_range.rolling(20).mean()
    )

    expected["range_std_20"] = (
        candle_range.rolling(20).std()
    )

    expected["close_position_mean_20"] = (
        ((close - low) / candle_range.replace(0, np.nan))
        .rolling(20)
        .mean()
    )

    # ---------------------------------------------------------
    # ROLLING HIGH / LOW
    # ---------------------------------------------------------

    expected["rolling_high_5"] = (
        high.rolling(5).max()
    )

    expected["rolling_low_5"] = (
        low.rolling(5).min()
    )

    expected["rolling_high_10"] = (
        high.rolling(10).max()
    )

    expected["rolling_low_10"] = (
        low.rolling(10).min()
    )

    expected["rolling_high_20"] = (
        high.rolling(20).max()
    )

    expected["rolling_low_20"] = (
        low.rolling(20).min()
    )

    # ---------------------------------------------------------
    # SMA
    # ---------------------------------------------------------

    expected["sma_5"] = close.rolling(5).mean()
    expected["sma_10"] = close.rolling(10).mean()
    expected["sma_20"] = close.rolling(20).mean()
    expected["sma_50"] = close.rolling(50).mean()
    expected["sma_100"] = close.rolling(100).mean()
    expected["sma_200"] = close.rolling(200).mean()

    # ---------------------------------------------------------
    # ROC
    # ---------------------------------------------------------

    expected["roc_5"] = (
        (close / close.shift(5) - 1) * 100
    )

    expected["roc_10"] = (
        (close / close.shift(10) - 1) * 100
    )

    expected["roc_20"] = (
        (close / close.shift(20) - 1) * 100
    )

    # ---------------------------------------------------------
    # MOMENTUM
    # ---------------------------------------------------------

    expected["momentum_5"] = (
        close - close.shift(5)
    )

    expected["momentum_10"] = (
        close - close.shift(10)
    )

    expected["momentum_20"] = (
        close - close.shift(20)
    )

    return expected


def main():

    print("=" * 70)
    print("INDEPENDENT FEATURE MATHEMATICAL VALIDATION")
    print("=" * 70)

    print()
    print("PROJECT ROOT:", PROJECT_ROOT)

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

    print()
    print("COLLECTING MARKET DATA...")
    print("TIMEFRAMES:", list(settings["timeframes"].keys()))

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    results = []

    for timeframe, raw_df in datasets.items():

        print()
        print("=" * 70)
        print(f"FEATURE MATH VALIDATION: {timeframe}")
        print("=" * 70)

        normalized = DataNormalizer.normalize(
            raw_df,
            source_timezone="UTC",
            target_timezone="Asia/Jakarta"
        )

        normalized = CandleStatus.add_status(
            normalized
        )

        closed_df = normalized[
            normalized["candle_status"] == "CLOSED"
        ].copy()

        features = FeatureEngine.calculate(
            normalized,
            closed_only=True
        )

        print("RAW ROWS:", len(raw_df))
        print("CLOSED ROWS:", len(closed_df))
        print("FEATURE ROWS:", len(features))

        expected = calculate_independent_features(
            closed_df
        )

        mismatches = 0

        print()
        print("[1] INDEPENDENT FORMULA CHECK")

        for name, expected_series in expected.items():

            if name not in features.columns:
                print(
                    f"FAIL {name}: feature tidak ditemukan"
                )
                mismatches += 1
                continue

            try:
                assert_series_close(
                    features[name],
                    expected_series,
                    name
                )

                print(
                    f"OK   {name}"
                )

            except AssertionError as exc:

                print(
                    f"FAIL {exc}"
                )

                mismatches += 1

        print()
        print("[2] FEATURE ALIGNMENT")

        if len(features) != len(closed_df):

            print(
                "FAIL Feature rows != closed rows"
            )

            mismatches += 1

        else:

            print(
                "OK   Feature rows aligned with closed candles"
            )

        print()
        print("[3] TIMESTAMP ALIGNMENT")

        if not features["datetime"].reset_index(
            drop=True
        ).equals(
            closed_df["datetime"].reset_index(
                drop=True
            )
        ):

            print(
                "FAIL Feature timestamps tidak align"
            )

            mismatches += 1

        else:

            print(
                "OK   Feature timestamps aligned"
            )

        if mismatches == 0:

            print()
            print(
                f"{timeframe:<10}: PASS"
            )

            results.append(True)

        else:

            print()
            print(
                f"{timeframe:<10}: FAIL"
            )

            results.append(False)

    print()
    print("=" * 70)
    print("FINAL FEATURE MATH RESULT")
    print("=" * 70)

    for timeframe, result in zip(
        datasets.keys(),
        results
    ):

        print(
            f"{timeframe:<10}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    if all(results):

        print()
        print(
            "FEATURE MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    else:

        print()
        print(
            "FEATURE MATHEMATICAL TEST: FAIL"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
