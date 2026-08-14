import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collectors.market_data import MarketDataCollector
from features.engine import FeatureEngine
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from core.data_validator import DataValidator


# ============================================================
# CONFIGURATION
# ============================================================

SETTINGS_PATH = (
    PROJECT_ROOT /
    "config" /
    "settings.json"
)


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


# ============================================================
# REQUIRED MATHEMATICAL FEATURES
# ============================================================

REQUIRED_MATHEMATICAL_FEATURES = {
    "rsi_7",
    "rsi_14",
    "rsi_21",

    "stoch_k_9",
    "stoch_d_9",
    "stoch_k_14",
    "stoch_d_14",
    "stoch_k_21",
    "stoch_d_21",

    "williams_r_14",
    "williams_r_21",

    "bb_width_20",

    "atr_7",
    "atr_14",
    "atr_21",

    "atr_pct_7",
    "atr_pct_14",
    "atr_pct_21",

    "volatility_10",
    "volatility_20",
    "volatility_50",

    "adx_14",
    "adx_21",
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ok(message):

    print(
        f"OK   {message}"
    )


def fail(message):

    print(
        f"FAIL {message}"
    )


# ============================================================
# SETTINGS
# ============================================================

def load_settings():

    if not SETTINGS_PATH.exists():

        raise FileNotFoundError(
            f"Settings tidak ditemukan: {SETTINGS_PATH}"
        )

    with open(
        SETTINGS_PATH,
        "r",
        encoding="utf-8-sig"
    ) as f:

        return json.load(f)


# ============================================================
# MATHEMATICAL COLUMN DETECTION
# ============================================================

def get_mathematical_columns(df):

    excluded = (
        BASE_COLUMNS
        |
        METADATA_COLUMNS
    )

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# ============================================================
# INDICATOR RANGE CHECK
# ============================================================

def _check_indicator_ranges(features):

    print_header(
        "[6] INDICATOR RANGE TEST"
    )

    range_rules = {

        "rsi_7": (0, 100),
        "rsi_14": (0, 100),
        "rsi_21": (0, 100),

        "stoch_k_9": (0, 100),
        "stoch_d_9": (0, 100),

        "stoch_k_14": (0, 100),
        "stoch_d_14": (0, 100),

        "stoch_k_21": (0, 100),
        "stoch_d_21": (0, 100),

        "williams_r_14": (-100, 0),
        "williams_r_21": (-100, 0),
    }

    for column, (minimum, maximum) in range_rules.items():

        if column not in features.columns:

            fail(
                f"{column} missing"
            )

            return False

        series = pd.to_numeric(
            features[column],
            errors="coerce"
        ).dropna()

        if series.empty:

            fail(
                f"{column} contains no valid numeric values"
            )

            return False

        actual_min = series.min()
        actual_max = series.max()

        if (
            actual_min < minimum - 1e-10
            or
            actual_max > maximum + 1e-10
        ):

            fail(
                f"{column} outside range "
                f"{minimum}..{maximum}"
            )

            print(
                f"     ACTUAL RANGE: "
                f"{actual_min}..{actual_max}"
            )

            return False

        ok(
            f"{column} range {minimum}..{maximum}"
        )

    return True


# ============================================================
# NON-NEGATIVE CHECK
# ============================================================

def _check_non_negative(features):

    print_header(
        "[7] NON-NEGATIVE FEATURE TEST"
    )

    columns = [

        "bb_width_20",

        "atr_7",
        "atr_14",
        "atr_21",

        "atr_pct_7",
        "atr_pct_14",
        "atr_pct_21",

        "volatility_10",
        "volatility_20",
        "volatility_50",

        "adx_14",
        "adx_21",
    ]

    for column in columns:

        if column not in features.columns:

            fail(
                f"{column} missing"
            )

            return False

        series = pd.to_numeric(
            features[column],
            errors="coerce"
        ).dropna()

        if series.empty:

            fail(
                f"{column} contains no valid numeric values"
            )

            return False

        minimum = series.min()

        if minimum < -1e-10:

            fail(
                f"{column} contains negative values"
            )

            print(
                f"     MIN VALUE: {minimum}"
            )

            return False

        ok(
            f"{column} is non-negative"
        )

    return True


# ============================================================
# FEATURE STRUCTURE CHECK
# ============================================================

def _check_structure(features):

    print_header(
        "[1] FEATURE STRUCTURE"
    )

    if not isinstance(
        features,
        pd.DataFrame
    ):

        fail(
            "FeatureEngine did not return pandas DataFrame"
        )

        return False

    ok(
        "FeatureEngine returned pandas DataFrame"
    )

    if features.empty:

        fail(
            "Feature dataframe is empty"
        )

        return False

    ok(
        "Feature dataframe contains rows"
    )

    required_base = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "candle_status",
    }

    missing_base = (
        required_base
        -
        set(features.columns)
    )

    if missing_base:

        fail(
            f"Missing required base columns: "
            f"{sorted(missing_base)}"
        )

        return False

    ok(
        "Required base columns"
    )

    missing_features = (
        REQUIRED_MATHEMATICAL_FEATURES
        -
        set(features.columns)
    )

    if missing_features:

        fail(
            "Missing mathematical features: "
            f"{sorted(missing_features)}"
        )

        return False

    ok(
        "Mathematical feature columns exist"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    metadata_present = [
        column
        for column in features.columns
        if column in METADATA_COLUMNS
    ]

    print(
        f"     TOTAL COLUMNS      : "
        f"{len(features.columns)}"
    )

    print(
        f"     MATHEMATICAL COLS  : "
        f"{len(mathematical_columns)}"
    )

    print(
        f"     METADATA COLS      : "
        f"{len(metadata_present)}"
    )

    return True


# ============================================================
# DATA TYPE CHECK
# ============================================================

def _check_data_types(features):

    print_header(
        "[2] MATHEMATICAL FEATURE DATA TYPES"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    non_numeric = []

    for column in mathematical_columns:

        if not pd.api.types.is_numeric_dtype(
            features[column]
        ):

            non_numeric.append(
                column
            )

    if non_numeric:

        fail(
            "Non-numeric mathematical features: "
            f"{non_numeric}"
        )

        return False

    ok(
        "All mathematical feature columns are numeric"
    )

    print()
    print(
        "METADATA COLUMNS EXCLUDED FROM NUMERIC TEST:"
    )

    for column in METADATA_COLUMNS:

        if column in features.columns:

            print(
                f"- {column}"
            )

    return True


# ============================================================
# FINITE VALUE CHECK
# ============================================================

def _check_finite_values(features):

    print_header(
        "[3] FINITE VALUE TEST"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    positive_inf = 0
    negative_inf = 0

    for column in mathematical_columns:

        series = pd.to_numeric(
            features[column],
            errors="coerce"
        )

        positive_inf += int(
            np.isposinf(series.to_numpy()).sum()
        )

        negative_inf += int(
            np.isneginf(series.to_numpy()).sum()
        )

    if positive_inf > 0:

        fail(
            f"Found +INF values: {positive_inf}"
        )

        return False

    ok(
        "No +INF values"
    )

    if negative_inf > 0:

        fail(
            f"Found -INF values: {negative_inf}"
        )

        return False

    ok(
        "No -INF values"
    )

    return True


# ============================================================
# ALL NAN CHECK
# ============================================================

def _check_all_nan(features):

    print_header(
        "[4] ALL-NaN FEATURE TEST"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    all_nan = []

    for column in mathematical_columns:

        if features[column].isna().all():

            all_nan.append(
                column
            )

    if all_nan:

        fail(
            "All-NaN mathematical features: "
            f"{all_nan}"
        )

        return False

    ok(
        "No mathematical feature is entirely NaN"
    )

    return True


# ============================================================
# CONSTANT FEATURE CHECK
# ============================================================

def _check_constant_features(features):

    print_header(
        "[5] CONSTANT MATHEMATICAL FEATURE TEST"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    constant = []

    for column in mathematical_columns:

        series = pd.to_numeric(
            features[column],
            errors="coerce"
        ).dropna()

        if series.empty:

            continue

        if series.nunique(
            dropna=True
        ) <= 1:

            constant.append(
                column
            )

    if constant:

        fail(
            "Constant mathematical features: "
            f"{constant}"
        )

        return False

    ok(
        "No mathematical feature is constant"
    )

    return True


# ============================================================
# BASE COLUMN INTEGRITY
# ============================================================

def _check_base_columns(features):

    print_header(
        "[8] BASE COLUMN INTEGRITY"
    )

    columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "candle_status",
    ]

    for column in columns:

        if column not in features.columns:

            fail(
                f"Base column missing: {column}"
            )

            return False

        ok(
            f"Base column exists: {column}"
        )

    return True


# ============================================================
# TIMESTAMP INTEGRITY
# ============================================================

def _check_timestamp_integrity(features):

    print_header(
        "[9] FEATURE TIMESTAMP INTEGRITY"
    )

    datetime_series = pd.to_datetime(
        features["datetime"],
        errors="coerce"
    )

    if datetime_series.isna().any():

        fail(
            "Feature datetime contains NULL"
        )

        return False

    ok(
        "Feature datetime has no NULL"
    )

    if datetime_series.duplicated().any():

        fail(
            "Feature datetime is not unique"
        )

        return False

    ok(
        "Feature datetime is unique"
    )

    if not datetime_series.is_monotonic_increasing:

        fail(
            "Feature datetime is not sorted"
        )

        return False

    ok(
        "Feature datetime is sorted"
    )

    return True


# ============================================================
# CLOSED ONLY GUARANTEE
# ============================================================

def _check_closed_only(features):

    print_header(
        "[10] CLOSED-ONLY GUARANTEE"
    )

    if "candle_status" not in features.columns:

        fail(
            "candle_status column missing"
        )

        return False

    statuses = (
        features["candle_status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    non_closed = (
        statuses != "CLOSED"
    )

    if non_closed.any():

        fail(
            "Feature output contains "
            "non-CLOSED candles"
        )

        print(
            "     INVALID STATUSES:",
            statuses[
                non_closed
            ].unique().tolist()
        )

        return False

    ok(
        "Feature output contains CLOSED candles only"
    )

    return True


# ============================================================
# DETERMINISM CHECK
# ============================================================

def _check_determinism(
    raw_df
):

    print_header(
        "[11] FEATURE DETERMINISM"
    )

    # Pastikan raw_df sudah melalui pipeline yang sama
    # Kita gunakan raw_df yang sudah dinormalisasi dan punya candle_status
    first = FeatureEngine.calculate(
        raw_df,
        closed_only=True
    )

    second = FeatureEngine.calculate(
        raw_df,
        closed_only=True
    )

    first_columns = set(
        get_mathematical_columns(first)
    )

    second_columns = set(
        get_mathematical_columns(second)
    )

    common_columns = sorted(
        first_columns
        &
        second_columns
    )

    print(
        f"     MATHEMATICAL FEATURES "
        f"COMPARED: {len(common_columns)} columns"
    )

    mismatch_count = 0

    for column in common_columns:

        a = pd.to_numeric(
            first[column],
            errors="coerce"
        )

        b = pd.to_numeric(
            second[column],
            errors="coerce"
        )

        equal = (
            np.isclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        )

        mismatch_count += int(
            (~equal).sum()
        )

    print(
        f"     VALUE MISMATCHES: "
        f"{mismatch_count}"
    )

    if mismatch_count > 0:

        fail(
            "Feature calculation is not deterministic"
        )

        return False

    ok(
        "Feature calculation is deterministic"
    )

    return True


# ============================================================
# TIMEFRAME TEST
# ============================================================

def run_timeframe(
    timeframe,
    raw_df
):

    print_header(
        f"FEATURE ENGINE VALIDATION: {timeframe}"
    )

    print(
        f"RAW ROWS: {len(raw_df)}"
    )

    # =========================================================
    # INTEGRASI PIPELINE: Normalisasi + CandleStatus + Validasi
    # =========================================================
    normalized = DataNormalizer.normalize(
        raw_df,
        source_timezone="UTC",
        target_timezone="Asia/Jakarta"
    )

    with_status = CandleStatus.add_status(normalized)

    # Validasi (opsional, hanya untuk laporan)
    validator_report = DataValidator.validate(with_status)
    print(f"VALIDATION REPORT: {validator_report}")

    if not validator_report["valid"]:
        print("WARNING: Data validation failed. Proceeding anyway for testing.")

    # Sekarang beri ke FeatureEngine
    features = FeatureEngine.calculate(
        with_status,
        closed_only=True
    )

    print(
        f"CLOSED ROWS: "
        f"{int((with_status['candle_status'] == 'CLOSED').sum())}"
    )

    print(
        f"FEATURE ROWS: {len(features)}"
    )

    mathematical_columns = (
        get_mathematical_columns(features)
    )

    print(
        f"MATHEMATICAL FEATURE COLUMNS: "
        f"{len(mathematical_columns)}"
    )

    checks = [

        _check_structure(features),

        _check_data_types(features),

        _check_finite_values(features),

        _check_all_nan(features),

        _check_constant_features(features),

        _check_indicator_ranges(features),

        _check_non_negative(features),

        _check_base_columns(features),

        _check_timestamp_integrity(features),

        _check_closed_only(features),

        # determinism butuh raw_df yang sudah melalui pipeline
        _check_determinism(with_status),
    ]

    result = all(checks)

    print()
    print(
        "-" * 70
    )

    print(
        f"{timeframe:<10}: "
        f"{'PASS' if result else 'FAIL'}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "FEATURE ENGINE MATHEMATICAL / STRUCTURAL TEST"
    )

    print(
        f"PROJECT ROOT: {PROJECT_ROOT}"
    )

    settings = load_settings()

    print()
    print(
        "COLLECTING MARKET DATA..."
    )

    collector = MarketDataCollector()

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    print(
        "TIMEFRAMES:",
        list(datasets.keys())
    )

    results = {}

    for timeframe, raw_df in datasets.items():

        try:

            results[timeframe] = run_timeframe(
                timeframe,
                raw_df
            )

        except Exception as exc:

            print()
            print(
                f"ERROR {timeframe}: "
                f"{type(exc).__name__}: {exc}"
            )

            results[timeframe] = False

    print()
    print("=" * 70)
    print(
        "FINAL FEATURE ENGINE RESULT"
    )
    print("=" * 70)

    for timeframe, result in results.items():

        print(
            f"{timeframe:<10}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    all_pass = (
        bool(results)
        and
        all(results.values())
    )

    print()

    if all_pass:

        print(
            "FEATURE ENGINE TEST: ALL PASS"
        )

        return True

    print(
        "FEATURE ENGINE TEST: FAIL"
    )

    return False


# ============================================================
# PYTEST ENTRY POINT
# ============================================================

def test_feature_engine_all_timeframes():

    """
    Pytest entry point for FeatureEngine validation.
    """

    assert main() is True


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    sys.exit(
        0
        if main()
        else 1
    )
