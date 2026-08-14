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

    a = pd.Series(actual).reset_index(drop=True)
    e = pd.Series(expected).reset_index(drop=True)

    a = pd.to_numeric(a, errors="coerce")
    e = pd.to_numeric(e, errors="coerce")

    mask = a.notna() & e.notna()

    if not mask.any():
        raise AssertionError(
            f"{name}: tidak ada data valid"
        )

    av = a[mask].to_numpy(float)
    ev = e[mask].to_numpy(float)

    if not np.allclose(
        av,
        ev,
        rtol=rtol,
        atol=atol
    ):

        diff = np.abs(av - ev)

        raise AssertionError(
            f"{name}: mismatch, "
            f"MAX_DIFF={np.nanmax(diff)}"
        )


def independent_volatility(df):

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    result = {}

    # =========================================================
    # TRUE RANGE
    # =========================================================

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    result["true_range"] = tr

    # =========================================================
    # ATR - Wilder / RMA
    # =========================================================

    for period in [7, 14, 21]:

        atr = tr.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        result[f"atr_{period}"] = atr

        result[f"atr_pct_{period}"] = (
            atr / close * 100
        )

    # =========================================================
    # LOG RETURN
    # =========================================================

    log_return = np.log(
        close / close.shift(1)
    )

    result["log_return"] = log_return

    # =========================================================
    # HISTORICAL VOLATILITY
    # =========================================================

    for period in [10, 20, 50]:

        result[f"volatility_{period}"] = (
            log_return
            .rolling(period)
            .std()
            * np.sqrt(period)
        )

    # =========================================================
    # BOLLINGER BANDS
    # =========================================================

    period = 20

    middle = (
        close
        .rolling(period)
        .mean()
    )

    std = (
        close
        .rolling(period)
        .std()
    )

    upper = middle + (2 * std)
    lower = middle - (2 * std)

    result["bb_middle_20"] = middle
    result["bb_upper_20"] = upper
    result["bb_lower_20"] = lower

    # =========================================================
    # BOLLINGER WIDTH
    # =========================================================
    #
    # Canonical definition:
    # BB Width = Upper Band - Lower Band
    #
    # Must match:
    # features/volatility.py
    # =========================================================

    result["bb_width_20"] = (
        upper - lower
    )

    # =========================================================
    # BOLLINGER POSITION
    # =========================================================

    denominator = upper - lower

    result["bb_position_20"] = pd.Series(
        np.where(
            denominator != 0,
            (close - lower) / denominator,
            0.5
        ),
        index=df.index
    )

    return result


def main():

    print("=" * 70)
    print("INDEPENDENT VOLATILITY MATHEMATICAL VALIDATION")
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
            f"VOLATILITY VALIDATION: {timeframe}"
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

        print(
            "RAW ROWS:",
            len(raw_df)
        )

        print(
            "CLOSED ROWS:",
            len(closed_df)
        )

        print(
            "FEATURE ROWS:",
            len(features)
        )

        expected = independent_volatility(
            closed_df
        )

        failures = 0

        # =====================================================
        # 1. MATHEMATICAL CHECK
        # =====================================================

        print()
        print("[1] INDEPENDENT VOLATILITY CHECK")

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
        # 2. RANGE CHECK
        # =====================================================

        print()
        print("[2] VOLATILITY RANGE CHECK")

        for name in [
            "true_range",
            "atr_7",
            "atr_14",
            "atr_21",
            "atr_pct_7",
            "atr_pct_14",
            "atr_pct_21",
        ]:

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

            if (series < 0).any():

                print(
                    f"FAIL {name}: "
                    f"terdapat nilai negatif"
                )

                failures += 1

            else:

                print(
                    f"OK   {name} >= 0"
                )

        # =====================================================
        # 3. BOLLINGER POSITION
        # =====================================================

        print()
        print("[3] BOLLINGER POSITION RANGE")

        series = pd.to_numeric(
            features["bb_position_20"],
            errors="coerce"
        ).dropna()

        if len(series) == 0:

            print(
                "FAIL bb_position_20: "
                "tidak ada nilai"
            )

            failures += 1

        elif not np.isfinite(series.to_numpy(dtype=float)).all():

            print(
                "FAIL bb_position_20: "
                "terdapat nilai non-finite"
            )

            failures += 1

        else:

            print(
                "OK   bb_position_20 finite "
                f"range {series.min()} "
                f"to {series.max()}"
            )

        # =====================================================
        # 4. TIMESTAMP ALIGNMENT
        # =====================================================

        print()
        print("[4] TIMESTAMP ALIGNMENT")

        if not features[
            "datetime"
        ].reset_index(
            drop=True
        ).equals(
            closed_df[
                "datetime"
            ].reset_index(
                drop=True
            )
        ):

            print(
                "FAIL Timestamp tidak align"
            )

            failures += 1

        else:

            print(
                "OK   Volatility timestamps aligned"
            )

        # =====================================================
        # 5. CLOSED ONLY
        # =====================================================

        print()
        print("[5] CLOSED-ONLY GUARANTEE")

        statuses = (
            features["candle_status"]
            .dropna()
            .astype(str)
            .unique()
        )

        if all(
            status == "CLOSED"
            for status in statuses
        ):

            print(
                "OK   Feature output contains "
                "CLOSED candles only"
            )

        else:

            print(
                "FAIL Feature output contains "
                "non-CLOSED candles"
            )

            failures += 1

        # =====================================================
        # 6. TIMESTAMP SORTING
        # =====================================================

        print()
        print("[6] TIMESTAMP SORTING")

        if features[
            "datetime"
        ].is_monotonic_increasing:

            print(
                "OK   Volatility timestamps sorted"
            )

        else:

            print(
                "FAIL Volatility timestamps not sorted"
            )

            failures += 1

        # =====================================================
        # 7. DETERMINISM
        # =====================================================

        print()
        print("[7] VOLATILITY DETERMINISM")

        second_run = FeatureEngine.calculate(
            df,
            closed_only=True
        )

        volatility_columns = [
            name
            for name in expected
            if name in features.columns
        ]

        mismatches = 0

        for name in volatility_columns:

            a = pd.Series(
                features[name]
            ).reset_index(drop=True)

            b = pd.Series(
                second_run[name]
            ).reset_index(drop=True)

            a = pd.to_numeric(
                a,
                errors="coerce"
            )

            b = pd.to_numeric(
                b,
                errors="coerce"
            )

            mask = a.notna() & b.notna()

            if not np.allclose(
                a[mask].to_numpy(float),
                b[mask].to_numpy(float),
                rtol=1e-5,
                atol=1e-8
            ):

                mismatches += 1

        print(
            "VOLATILITY FEATURES COMPARED:",
            len(volatility_columns)
        )

        print(
            "VALUE MISMATCHES:",
            mismatches
        )

        if mismatches == 0:

            print(
                "OK   Volatility calculation deterministic"
            )

        else:

            print(
                "FAIL Volatility calculation "
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
    print("FINAL VOLATILITY MATH RESULT")
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
            "VOLATILITY MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    print()
    print(
        "VOLATILITY MATHEMATICAL TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()


