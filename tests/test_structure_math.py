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


def compare(actual, expected, name, rtol=1e-5, atol=1e-8):

    a = pd.to_numeric(actual, errors="coerce")
    e = pd.to_numeric(expected, errors="coerce")

    mask = a.notna() & e.notna()

    if not mask.any():
        raise AssertionError(
            f"{name}: tidak ada data valid"
        )

    if not np.allclose(
        a[mask].to_numpy(float),
        e[mask].to_numpy(float),
        rtol=rtol,
        atol=atol,
    ):
        diff = np.abs(
            a[mask].to_numpy(float)
            - e[mask].to_numpy(float)
        )

        raise AssertionError(
            f"{name}: mismatch, "
            f"MAX_DIFF={np.nanmax(diff)}"
        )


def independent_structure(df):

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    result = {}

    # =========================================================
    # RETURNS
    # =========================================================

    result["return_1"] = close.pct_change()
    result["return_3"] = close.pct_change(3)
    result["return_5"] = close.pct_change(5)
    result["return_10"] = close.pct_change(10)

    # =========================================================
    # CANDLE STRUCTURE
    # =========================================================

    candle_range = high - low

    result["range"] = candle_range

    result["body"] = close - open_

    result["body_abs"] = result["body"].abs()

    result["upper_wick"] = (
        high
        - pd.concat(
            [open_, close],
            axis=1
        ).max(axis=1)
    )

    result["lower_wick"] = (
        pd.concat(
            [open_, close],
            axis=1
        ).min(axis=1)
        - low
    )

    # Gunakan Series, bukan np.ndarray
    result["body_ratio"] = (
        result["body_abs"]
        .div(candle_range.replace(0, np.nan))
        .fillna(0)
    )

    result["upper_wick_ratio"] = (
        result["upper_wick"]
        .div(candle_range.replace(0, np.nan))
        .fillna(0)
    )

    result["lower_wick_ratio"] = (
        result["lower_wick"]
        .div(candle_range.replace(0, np.nan))
        .fillna(0)
    )

    result["close_position"] = (
        (close - low)
        .div(candle_range.replace(0, np.nan))
        .fillna(0.5)
    )

    result["candle_direction"] = (
        np.sign(close - open_)
    )

    # =========================================================
    # GAP
    # =========================================================

    previous_close = close.shift(1)

    result["gap"] = (
        open_ - previous_close
    )

    result["gap_pct"] = (
        (result["gap"])
        .div(previous_close.replace(0, np.nan))
        .fillna(0)
    )

    # =========================================================
    # ROLLING RANGE
    # =========================================================

    for period in [5, 10, 20, 50]:

        result[f"range_mean_{period}"] = (
            candle_range
            .rolling(period)
            .mean()
        )

        result[f"range_std_{period}"] = (
            candle_range
            .rolling(period)
            .std()
        )

        result[f"close_position_mean_{period}"] = (
            result["close_position"]
            .rolling(period)
            .mean()
        )

    # =========================================================
    # ROLLING HIGH / LOW
    # =========================================================

    for period in [5, 10, 20]:

        result[f"rolling_high_{period}"] = (
            high
            .rolling(period)
            .max()
        )

        result[f"rolling_low_{period}"] = (
            low
            .rolling(period)
            .min()
        )

    return result


def main():

    print("=" * 70)
    print("INDEPENDENT STRUCTURE MATHEMATICAL VALIDATION")
    print("=" * 70)

    print()
    print("PROJECT ROOT:", PROJECT_ROOT)

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

    print()
    print("COLLECTING MARKET DATA...")
    print(
        "TIMEFRAMES:",
        list(settings["timeframes"].keys())
    )

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    results = []

    for timeframe, raw_df in datasets.items():

        print()
        print("=" * 70)
        print(
            f"STRUCTURE VALIDATION: {timeframe}"
        )
        print("=" * 70)

        df = DataNormalizer.normalize(
            raw_df,
            source_timezone="UTC",
            target_timezone="Asia/Jakarta"
        )

        df = CandleStatus.add_status(df)

        closed_df = df[
            df["candle_status"] == "CLOSED"
        ].copy()

        features = FeatureEngine.calculate(
            df,
            closed_only=True
        )

        print("RAW ROWS:", len(raw_df))
        print("CLOSED ROWS:", len(closed_df))
        print("FEATURE ROWS:", len(features))

        expected = independent_structure(
            closed_df
        )

        failures = 0

        # =====================================================
        # FORMULA CHECK
        # =====================================================

        print()
        print("[1] INDEPENDENT STRUCTURE CHECK")

        for name, expected_series in expected.items():

            if name not in features.columns:

                print(
                    f"FAIL {name}: "
                    f"kolom tidak ditemukan"
                )

                failures += 1
                continue

            try:

                compare(
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

                failures += 1

        # =====================================================
        # RANGE CHECK
        # =====================================================

        print()
        print("[2] STRUCTURE RANGE CHECK")

        bounded_ranges = {
            "body_ratio": (0, 1),
            "upper_wick_ratio": (0, 1),
            "lower_wick_ratio": (0, 1),
            "close_position": (0, 1),
        }

        for name, (
            minimum,
            maximum
        ) in bounded_ranges.items():

            series = pd.to_numeric(
                features[name],
                errors="coerce"
            ).dropna()

            if len(series) == 0:

                print(
                    f"FAIL {name}: "
                    f"tidak ada nilai"
                )

                failures += 1
                continue

            if (
                series.min() < minimum
                or
                series.max() > maximum
            ):

                print(
                    f"FAIL {name}: "
                    f"range {series.min()} "
                    f"to {series.max()}"
                )

                failures += 1

            else:

                print(
                    f"OK   {name} "
                    f"range {minimum}..{maximum}"
                )

        # =====================================================
        # CANDLE DIRECTION
        # =====================================================

        print()
        print("[3] CANDLE DIRECTION CHECK")

        direction = pd.to_numeric(
            features["candle_direction"],
            errors="coerce"
        ).dropna()

        if direction.isin([-1, 0, 1]).all():

            print(
                "OK   candle_direction "
                "contains only -1, 0, 1"
            )

        else:

            print(
                "FAIL candle_direction "
                "contains invalid values"
            )

            failures += 1

        # =====================================================
        # TIMESTAMP ALIGNMENT
        # =====================================================

        print()
        print("[4] TIMESTAMP ALIGNMENT")

        feature_time = (
            features["datetime"]
            .reset_index(drop=True)
        )

        closed_time = (
            closed_df["datetime"]
            .reset_index(drop=True)
        )

        if feature_time.equals(closed_time):

            print(
                "OK   Structure timestamps aligned"
            )

        else:

            print(
                "FAIL Structure timestamps "
                "tidak align"
            )

            failures += 1

        # =====================================================
        # CLOSED ONLY
        # =====================================================

        print()
        print("[5] CLOSED-ONLY GUARANTEE")

        if (
            "candle_closed" in features.columns
            and
            features["candle_closed"].astype(bool).all()
        ):

            print(
                "OK   Feature output "
                "contains CLOSED candles only"
            )

        else:

            print(
                "FAIL Feature output "
                "contains non-CLOSED candle"
            )

            failures += 1

        # =====================================================
        # SORTING
        # =====================================================

        print()
        print("[6] TIMESTAMP SORTING")

        if features["datetime"].is_monotonic_increasing:

            print(
                "OK   Feature timestamps sorted"
            )

        else:

            print(
                "FAIL Feature timestamps "
                "not sorted"
            )

            failures += 1

        # =====================================================
        # DETERMINISM
        # =====================================================

        print()
        print("[7] STRUCTURE DETERMINISM")

        features_again = FeatureEngine.calculate(
            df,
            closed_only=True
        )

        structure_columns = list(
            expected.keys()
        )

        mismatch_count = 0

        for column in structure_columns:

            if column not in features_again.columns:
                continue

            a = pd.to_numeric(
                features[column],
                errors="coerce"
            )

            b = pd.to_numeric(
                features_again[column],
                errors="coerce"
            )

            mask = a.notna() & b.notna()

            if mask.any():

                if not np.allclose(
                    a[mask].to_numpy(float),
                    b[mask].to_numpy(float),
                    rtol=1e-5,
                    atol=1e-8
                ):

                    mismatch_count += 1

        print(
            "STRUCTURE FEATURES COMPARED:",
            len(structure_columns)
        )

        print(
            "VALUE MISMATCHES:",
            mismatch_count
        )

        if mismatch_count == 0:

            print(
                "OK   Structure calculation deterministic"
            )

        else:

            print(
                "FAIL Structure calculation "
                "not deterministic"
            )

            failures += 1

        passed = failures == 0

        print()
        print(
            f"{timeframe:<10}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        results.append(passed)

    print()
    print("=" * 70)
    print("FINAL STRUCTURE MATH RESULT")
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
            "STRUCTURE MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    print()
    print(
        "STRUCTURE MATHEMATICAL TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
