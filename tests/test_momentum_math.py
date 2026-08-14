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


def independent_momentum(df):

    close = df["close"].astype(float)

    result = {}

    # =========================================================
    # RSI - Wilder / RMA
    # =========================================================

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for period in [7, 14, 21]:

        avg_gain = gain.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        rs = (
            avg_gain
            /
            avg_loss.replace(0, np.nan)
        )

        result[f"rsi_{period}"] = (
            100
            -
            (
                100
                /
                (1 + rs)
            )
        )

    # =========================================================
    # ROC
    # =========================================================

    for period in [5, 10, 20]:

        result[f"roc_{period}"] = (
            close.pct_change(period)
            * 100
        )

    # =========================================================
    # MOMENTUM
    # =========================================================

    for period in [5, 10, 20]:

        result[f"momentum_{period}"] = (
            close
            -
            close.shift(period)
        )

    return result


def main():

    print("=" * 70)
    print("INDEPENDENT MOMENTUM MATHEMATICAL VALIDATION")
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
            f"MOMENTUM VALIDATION: {timeframe}"
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

        expected = independent_momentum(
            closed_df
        )

        failures = 0

        print()
        print("[1] INDEPENDENT MOMENTUM CHECK")

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

        print()
        print("[2] RSI RANGE CHECK")

        for name in [
            "rsi_7",
            "rsi_14",
            "rsi_21"
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

            if (
                series.min() < 0
                or
                series.max() > 100
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
                    f"range 0..100"
                )

        print()
        print("[3] TIMESTAMP ALIGNMENT")

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
                "OK   Momentum timestamps aligned"
            )

        print()
        print("[4] CLOSED-ONLY GUARANTEE")

        if (
            len(features)
            !=
            len(closed_df)
        ):

            print(
                "FAIL Feature output "
                "tidak sama dengan CLOSED candles"
            )

            failures += 1

        else:

            print(
                "OK   Feature output contains "
                "CLOSED candles only"
            )

        print()
        print("[5] TIMESTAMP SORTING")

        if not features[
            "datetime"
        ].is_monotonic_increasing:

            print(
                "FAIL Feature timestamps "
                "tidak sorted"
            )

            failures += 1

        else:

            print(
                "OK   Momentum timestamps sorted"
            )

        passed = failures == 0

        print()
        print(
            f"{timeframe:<10}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        results.append(passed)

    print()
    print("=" * 70)
    print("FINAL MOMENTUM MATH RESULT")
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
            "MOMENTUM MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    print()
    print(
        "MOMENTUM MATHEMATICAL TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
