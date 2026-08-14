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
        raise AssertionError(f"{name}: tidak ada data valid")

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


def independent_indicators(df):

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    result = {}

    # =========================================================
    # EMA
    # =========================================================

    result["ema_10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    result["ema_20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    result["ema_50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    result["ema_200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # =========================================================
    # RSI - Wilder / RMA
    # =========================================================

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result["rsi_14"] = (
        100 - (100 / (1 + rs))
    )

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

    # =========================================================
    # ATR - Wilder / RMA
    # =========================================================

    result["atr_14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    # =========================================================
    # ATR %
    # =========================================================

    result["atr_pct_14"] = (
        result["atr_14"] / close * 100
    )

    # =========================================================
    # BOLLINGER BANDS
    # =========================================================

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    result["bb_middle_20"] = middle
    result["bb_upper_20"] = middle + (2 * std)
    result["bb_lower_20"] = middle - (2 * std)

    result["bb_width_20"] = (
        result["bb_upper_20"]
        - result["bb_lower_20"]
    )

    result["bb_position_20"] = (
        (close - result["bb_lower_20"])
        /
        (
            result["bb_upper_20"]
            - result["bb_lower_20"]
        ).replace(0, np.nan)
    )

    # =========================================================
    # STOCHASTIC %K
    # =========================================================

    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()

    result["stoch_k_14"] = (
        100
        * (close - lowest_14)
        /
        (highest_14 - lowest_14).replace(
            0,
            np.nan
        )
    )

    result["stoch_d_14"] = (
        result["stoch_k_14"]
        .rolling(3)
        .mean()
    )

    # =========================================================
    # WILLIAMS %R
    # =========================================================

    result["williams_r_14"] = (
        -100
        *
        (
            highest_14 - close
        )
        /
        (highest_14 - lowest_14).replace(
            0,
            np.nan
        )
    )

    return result


def main():

    print("=" * 70)
    print("INDEPENDENT INDICATOR MATHEMATICAL VALIDATION")
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
            f"INDICATOR VALIDATION: {timeframe}"
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

        expected = independent_indicators(
            closed_df
        )

        failures = 0

        print()
        print("[1] INDEPENDENT INDICATOR CHECK")

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
        print("[2] INDICATOR RANGE CHECK")

        ranges = {
            "rsi_14": (0, 100),
            "stoch_k_14": (0, 100),
            "stoch_d_14": (0, 100),
            "williams_r_14": (-100, 0),
        }

        for name, (minimum, maximum) in ranges.items():

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
                "OK   Indicator timestamps aligned"
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
    print("FINAL INDICATOR MATH RESULT")
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
            "INDICATOR MATHEMATICAL TEST: ALL PASS"
        )

        sys.exit(0)

    print()
    print(
        "INDICATOR MATHEMATICAL TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
