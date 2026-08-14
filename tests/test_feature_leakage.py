import sys
import re
import json
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine


# ============================================================
# CONFIGURATION
# ============================================================

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

FEATURE_DIR = PROJECT_ROOT / "features"


# ============================================================
# STATIC LEAKAGE PATTERNS
# ============================================================

STATIC_PATTERNS = [
    {
        "name": "NEGATIVE SHIFT",
        "severity": "FAIL",
        "patterns": [
            r"\.shift\s*\(\s*-\s*\d+",
            r"\.shift\s*\(\s*-\s*[A-Za-z_][A-Za-z0-9_]*",
        ],
        "reason": (
            "shift negatif dapat mengambil nilai candle masa depan."
        ),
    },

    {
        "name": "NEGATIVE DIFF",
        "severity": "FAIL",
        "patterns": [
            r"\.diff\s*\(\s*-\s*\d+",
            r"\.diff\s*\(\s*-\s*[A-Za-z_][A-Za-z0-9_]*",
        ],
        "reason": (
            "diff negatif dapat menggunakan perubahan dari candle masa depan."
        ),
    },

    {
        "name": "BACKFILL",
        "severity": "FAIL",
        "patterns": [
            r"\.bfill\s*\(",
            r"\.backfill\s*\(",
        ],
        "reason": (
            "bfill mengisi nilai historis menggunakan observasi berikutnya."
        ),
    },

    {
        "name": "CENTERED ROLLING",
        "severity": "FAIL",
        "patterns": [
            r"rolling\s*\([^)]*center\s*=\s*True",
        ],
        "reason": (
            "center=True dapat memasukkan candle setelah timestamp saat ini."
        ),
    },

    {
        "name": "NEGATIVE ILOC",
        "severity": "WARNING",
        "patterns": [
            r"\.iloc\s*\[\s*-\s*\d+",
            r"\.iloc\s*\[\s*-\s*[A-Za-z_][A-Za-z0-9_]*",
        ],
        "reason": (
            "akses iloc negatif perlu ditinjau karena dapat mengambil data "
            "dari bagian akhir dataframe."
        ),
    },

    {
        "name": "REVERSE SERIES",
        "severity": "WARNING",
        "patterns": [
            r"\[\s*::\s*-\s*1\s*\]",
            r"\.iloc\s*\[\s*::\s*-\s*1\s*\]",
        ],
        "reason": (
            "pembalikan urutan time-series perlu diaudit agar feature "
            "tidak dihitung dalam arah waktu yang salah."
        ),
    },

    {
        "name": "FUTURE KEYWORD",
        "severity": "WARNING",
        "patterns": [
            r"\bfuture\b",
            r"\blead\b",
            r"\bnext\b",
            r"\bforward\b",
        ],
        "reason": (
            "nama/identifier yang mengindikasikan data masa depan ditemukan."
        ),
    },
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


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
# STATIC SOURCE SCAN
# ============================================================

def scan_source_file(path):
    findings = []

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        text = path.read_text(
            encoding="utf-8-sig"
        )

    lines = text.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1
    ):
        stripped = line.strip()

        # Abaikan komentar penuh
        if stripped.startswith("#"):
            continue

        for rule in STATIC_PATTERNS:

            for pattern in rule["patterns"]:

                if re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE
                ):

                    findings.append({
                        "file": str(
                            path.relative_to(PROJECT_ROOT)
                        ),
                        "line": line_number,
                        "severity": rule["severity"],
                        "rule": rule["name"],
                        "reason": rule["reason"],
                        "code": stripped,
                    })

                    break

    return findings


def run_static_leakage_scan():

    print_header(
        "STATIC FEATURE LEAKAGE SCAN"
    )

    print(
        f"FEATURE DIRECTORY: {FEATURE_DIR}"
    )

    if not FEATURE_DIR.exists():
        print(
            "ERROR: directory features tidak ditemukan."
        )
        return False

    python_files = sorted(
        FEATURE_DIR.rglob("*.py")
    )

    if not python_files:
        print(
            "ERROR: tidak ditemukan file Python "
            "di directory features."
        )
        return False

    print(
        f"SOURCE FILES SCANNED: {len(python_files)}"
    )

    all_findings = []

    for path in python_files:

        findings = scan_source_file(path)

        all_findings.extend(
            findings
        )

    print()

    if not all_findings:

        print(
            "OK   Tidak ditemukan pola static "
            "yang berpotensi menyebabkan leakage."
        )

        print()
        print(
            "STATIC SCAN RESULT: PASS"
        )

        return True

    print(
        f"FINDINGS: {len(all_findings)}"
    )

    print()

    for finding in all_findings:

        print(
            f"[{finding['severity']}] "
            f"{finding['rule']}"
        )

        print(
            f"FILE : {finding['file']}"
        )

        print(
            f"LINE : {finding['line']}"
        )

        print(
            f"CODE : {finding['code']}"
        )

        print(
            f"WHY  : {finding['reason']}"
        )

        print()

    fail_count = sum(
        1
        for x in all_findings
        if x["severity"] == "FAIL"
    )

    warning_count = sum(
        1
        for x in all_findings
        if x["severity"] == "WARNING"
    )

    print(
        f"FAILURES : {fail_count}"
    )

    print(
        f"WARNINGS : {warning_count}"
    )

    if fail_count > 0:

        print()
        print(
            "STATIC SCAN RESULT: FAIL"
        )

        print(
            "Potential future-data access ditemukan."
        )

        return False

    print()
    print(
        "STATIC SCAN RESULT: PASS WITH WARNINGS"
    )

    print(
        "Tidak ditemukan pola leakage fatal, "
        "tetapi beberapa bagian perlu review manual."
    )

    return True


# ============================================================
# DYNAMIC PREFIX LEAKAGE TEST
# ============================================================

def compare_feature_prefix(
    full_features,
    prefix_features
):
    """
    Membandingkan feature pada prefix historis.

    Jika feature dihitung menggunakan future data,
    nilai feature historis pada full dataset dapat berubah
    ketika candle masa depan ditambahkan.

    Kita membandingkan hanya kolom feature numerik.
    """

    base_columns = {
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

    feature_columns = [
        column
        for column in full_features.columns
        if column not in base_columns
    ]

    common_columns = [
        column
        for column in feature_columns
        if column in prefix_features.columns
    ]

    if not common_columns:
        return {
            "checked": 0,
            "mismatches": 0,
            "details": []
        }

    full_prefix = full_features[
        common_columns
    ].iloc[
        :len(prefix_features)
    ].reset_index(drop=True)

    test_prefix = prefix_features[
        common_columns
    ].reset_index(drop=True)

    details = []

    mismatch_count = 0

    for column in common_columns:

        a = pd.to_numeric(
            full_prefix[column],
            errors="coerce"
        )

        b = pd.to_numeric(
            test_prefix[column],
            errors="coerce"
        )

        comparison = (
            a.fillna(float("nan"))
        )

        # NaN == NaN dianggap sama
        equal = (
            (a == b)
            | (a.isna() & b.isna())
        )

        mismatches = (
            ~equal
        )

        count = int(
            mismatches.sum()
        )

        if count > 0:

            mismatch_count += count

            mismatch_indices = (
                mismatches[
                    mismatches
                ].index
                .tolist()
            )

            details.append({
                "column": column,
                "mismatches": count,
                "sample_indices": mismatch_indices[:5],
            })

    return {
        "checked": len(common_columns),
        "mismatches": mismatch_count,
        "details": details,
    }


def run_dynamic_leakage_test(
    timeframe,
    raw_df
):

    print_header(
        f"DYNAMIC LEAKAGE TEST: {timeframe}"
    )

    normalized = DataNormalizer.normalize(
        raw_df,
        source_timezone="UTC",
        target_timezone="Asia/Jakarta"
    )

    normalized = CandleStatus.add_status(
        normalized
    )

    closed = normalized[
        normalized["candle_status"] == "CLOSED"
    ].copy()

    closed = closed.reset_index(
        drop=True
    )

    if len(closed) < 20:

        print(
            "ERROR: closed candle terlalu sedikit "
            "untuk prefix leakage test."
        )

        return False

    # Gunakan sekitar 70% historical prefix.
    prefix_size = max(
        10,
        int(len(closed) * 0.70)
    )

    prefix = closed.iloc[
        :prefix_size
    ].copy()

    cutoff_datetime = prefix[
        "datetime"
    ].iloc[-1]

    print(
        f"RAW CLOSED ROWS : {len(closed)}"
    )

    print(
        f"PREFIX ROWS      : {len(prefix)}"
    )

    print(
        f"CUTOFF DATETIME   : {cutoff_datetime}"
    )

    full_features = FeatureEngine.calculate(
        closed,
        closed_only=True
    )

    prefix_features = FeatureEngine.calculate(
        prefix,
        closed_only=True
    )

    print(
        f"FULL FEATURES    : {len(full_features)}"
    )

    print(
        f"PREFIX FEATURES  : {len(prefix_features)}"
    )

    result = compare_feature_prefix(
        full_features,
        prefix_features
    )

    print()

    print(
        f"FEATURE COLUMNS CHECKED : "
        f"{result['checked']}"
    )

    print(
        f"FEATURE VALUE MISMATCHES : "
        f"{result['mismatches']}"
    )

    if result["mismatches"] > 0:

        print()
        print(
            "LEAKAGE DETAILS:"
        )

        for detail in result["details"]:

            print(
                f"- {detail['column']}: "
                f"{detail['mismatches']} mismatch"
            )

            print(
                f"  sample rows: "
                f"{detail['sample_indices']}"
            )

        print()
        print(
            "RESULT: FAIL"
        )

        print(
            "Feature historis berubah setelah "
            "data masa depan ditambahkan."
        )

        return False

    print()
    print(
        "RESULT: PASS"
    )

    print(
        "Tidak ditemukan indikasi "
        "future-data leakage pada feature historis."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "FEATURE LEAKAGE TEST"
    )

    print(
        f"PROJECT ROOT: {PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # 1. STATIC SCAN
    # --------------------------------------------------------

    static_pass = run_static_leakage_scan()

    # --------------------------------------------------------
    # 2. MARKET DATA
    # --------------------------------------------------------

    print_header(
        "COLLECTING MARKET DATA"
    )

    settings = load_settings()

    print(
        "TIMEFRAMES:",
        list(
            settings["timeframes"].keys()
        )
    )

    collector = MarketDataCollector()

    datasets = collector.collect(
        symbol=settings["symbol"],
        timeframes=settings["timeframes"]
    )

    # --------------------------------------------------------
    # 3. DYNAMIC TEST
    # --------------------------------------------------------

    dynamic_results = {}

    for timeframe, raw_df in datasets.items():

        try:

            dynamic_results[
                timeframe
            ] = run_dynamic_leakage_test(
                timeframe,
                raw_df
            )

        except Exception as exc:

            print()
            print(
                f"ERROR {timeframe}: "
                f"{type(exc).__name__}"
            )

            print(
                str(exc)
            )

            dynamic_results[
                timeframe
            ] = False

    # --------------------------------------------------------
    # 4. FINAL RESULT
    # --------------------------------------------------------

    print_header(
        "FINAL FEATURE LEAKAGE RESULT"
    )

    print(
        "STATIC SCAN :",
        "PASS" if static_pass else "FAIL"
    )

    for timeframe, result in dynamic_results.items():

        print(
            f"{timeframe:<10}:",
            "PASS" if result else "FAIL"
        )

    dynamic_pass = all(
        dynamic_results.values()
    )

    print()

    if static_pass and dynamic_pass:

        print(
            "FEATURE LEAKAGE TEST: ALL PASS"
        )

        sys.exit(0)

    if dynamic_pass and not static_pass:

        print(
            "FEATURE LEAKAGE TEST: "
            "DYNAMIC PASS / STATIC REVIEW REQUIRED"
        )

        sys.exit(1)

    print(
        "FEATURE LEAKAGE TEST: FAIL"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()

