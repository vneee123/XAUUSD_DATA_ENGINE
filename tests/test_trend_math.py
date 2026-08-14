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
        atol=atol
    ):

        diff = np.abs(
            a[mask].to_numpy(float)
            - e[mask].to_numpy(float)
        )

        raise AssertionError(
            f"{name}: mismatch, "
            f"MAX_DIFF={np.nanmax(diff)}"
        )


def independent_trend(df):

    close = df["close"].astype(float)

    result = {}

    # =========================================================
    # SMA
    # =========================================================

    for period in [5, 10, 20, 50, 100, 200]:

        result[f"sma_{period}"] = (
            close
            .rolling(period)
            .mean()
        )

    # =========================================================
    # EMA
    # =========================================================

    for period in [5, 10, 20, 50, 100, 200]:

        result[f"ema_{period}"] = (
            close
            .ewm(
                span=period,
                adjust=False
            )
            .mean()
        )

    # =========================================================
    # EMA SLOPE
    # =========================================================

    for period in [10, 20, 50, 200]:

        ema = result[f"ema_{period}"]

        result[f"ema_{period}_slope"] = (
            ema.diff()
        )

    # =========================================================
    # EMA DISTANCE %
    # =========================================================

    for period in [10, 20, 50, 200]:

        ema = result[f"ema_{period}"]

        result[f"ema_{period}_distance_pct"] = (
            (close - ema)
            / ema
        )

    # =========================================================
    # EMA ALIGNMENT
    # =========================================================

    result["ema_10_above_20"] = (
        result["ema_10"]
        >
        result["ema_20"]
    ).astype(int)

    result["ema_20_above_50"] = (
        result["ema_20"]
        >
        result["ema_50"]
    ).astype(int)

    result["ema_50_above_200"] = (
        result["ema_50"]
        >
        result["ema_200"]
    ).astype(int)

    # =========================================================
    # TREND SCORE
    # =========================================================

    result["trend_score"] = (
        result["ema_10_above_20"]
        +
        result["ema_20_above_50"]
        +
        result["ema_50_above_200"]
    )

    return result


def main():

    print("=" * 70)
    print("INDEPENDENT TREND MATHEMATICAL VALIDATION")
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
            f"TREND VALIDATION: {timeframe}"
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

        expected = independent_trend(
            closed_df
        )

        failures = 0

        # =====================================================
        # 1. INDEPENDENT TREND CHECK
        # =====================================================

        print()
        print("[1] INDEPENDENT TREND CHECK")

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
        # 2. TREND SCORE RANGE
        # =====================================================

        print()
        print("[2] TREND SCORE RANGE")

        score = pd.to_numeric(
            features["trend_score"],
            errors="coerce"
        ).dropna()

        if len(score) == 0:

            print(
                "FAIL trend_score: "
                "tidak ada nilai"
            )

            failures += 1

        elif (
            score.min() < 0
            or
            score.max() > 3
        ):

            print(
                f"FAIL trend_score range "
                f"{score.min()}..{score.max()}"
            )

            failures += 1

        else:

            print(
                f"OK   trend_score range "
                f"0..3"
            )

        # =====================================================
        # 3. EMA ALIGNMENT RANGE
        # =====================================================

        print()
        print("[3] EMA ALIGNMENT RANGE")

        alignment_columns = [
            "ema_10_above_20",
            "ema_20_above_50",
            "ema_50_above_200",
        ]

        for name in alignment_columns:

            values = pd.to_numeric(
                features[name],
                errors="coerce"
            ).dropna()

            unique_values = set(
                values.astype(int).unique()
            )

            if not unique_values.issubset({0, 1}):

                print(
                    f"FAIL {name}: "
                    f"nilai={sorted(unique_values)}"
                )

                failures += 1

            else:

                print(
                    f"OK   {name} "
                    f"contains only 0/1"
                )

        # =====================================================
        # 4. TIMESTAMP ALIGNMENT
        # =====================================================

        print()
        print("[4] TIMESTAMP ALIGNMENT")

        feature_datetime = (
            features["datetime"]
            .reset_index(drop=True)
        )

        closed_datetime = (
            closed_df["datetime"]
            .reset_index(drop=True)
        )

        if not feature_datetime.equals(
            closed_datetime
        ):

            print(
                "FAIL Timestamp tidak align"
            )

            failures += 1

        else:

            print(
                "OK   Trend timestamps aligned"
            )

        # =====================================================
        # 5. CLOSED-ONLY GUARANTEE
        # =====================================================

        print()
        print("[5] CLOSED-ONLY GUARANTEE")

        if "candle_status" in features.columns:

            statuses = (
                features["candle_status"]
                .astype(str)
                .unique()
            )

            if set(statuses) != {"CLOSED"}:

                print(
                    "FAIL Feature output "
                    "mengandung non-CLOSED candle"
                )

                failures += 1

            else:

                print(
                    "OK   Feature output "
                    "contains CLOSED candles only"
                )

        # =====================================================
        # 6. TIMESTAMP SORTING
        # =====================================================

        print()
        print("[6] TIMESTAMP SORTING")

        if not features["datetime"].is_monotonic_increasing:

            print(
                "FAIL Feature timestamps "
                "tidak sorted"
            )

            failures += 1

        else:

            print(
                "OK   Feature timestamps sorted"
            )

        # =====================================================
        # 7. TREND DETERMINISM
        # =====================================================

        print()
        print("[7] TREND DETERMINISM")

        features_again = FeatureEngine.calculate(
            df,
            closed_only=True
        )

        trend_columns = list(
            expected.keys()
        )

        mismatches = 0

        for name in trend_columns:

            a = pd.to_numeric(
                features[name],
                errors="coerce"
            )

            b = pd.to_numeric(
                features_again[name],
                errors="coerce"
            )

            mask = a.notna() & b.notna()

            if not np.allclose(
                a[mask].to_numpy(float),
                b[mask].to_numpy(float),
                rtol=1e-10,
                atol=1e-12
            ):

                mismatches += 1

        print(
            "TREND FEATURES COMPARED:",
            len(trend_columns)
        )

        print(
            "VALUE MISMATCHES:",
            mismatches
        )

        if mismatches == 0:

            print(
                "OK   Trend calculation deterministic"
            )

        else:

            print(
                "FAIL Trend calculation "
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
    print("FINAL TREND MATH RESULT")
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
            "TREND MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    print()
    print(
        "TREND MATHEMATICAL TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
