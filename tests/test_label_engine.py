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
from labels.label_engine import LabelEngine

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

def test_label_engine():
    """Test label generation on M5 data."""
    with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
        settings = json.load(f)

    symbol = settings["symbol"]
    timeframes = settings["timeframes"]

    collector = MarketDataCollector()
    raw_dfs = collector.collect(symbol, timeframes)

    # Use M5 only for label test
    df = raw_dfs["M5"]

    # Normalize
    normalized = DataNormalizer.normalize(df)
    with_status = CandleStatus.add_status(normalized)

    # FeatureEngine just for consistency (not required for labels)
    features = FeatureEngine.calculate(with_status, closed_only=True)

    # Now add labels
    labeled = LabelEngine.add_forward_returns(features, horizons=[1, 2, 3, 5, 10])

    # Check columns exist
    for h in [1, 2, 3, 5, 10]:
        col = f"fwd_return_{h}"
        assert col in labeled.columns, f"Missing {col}"

    # Check that some rows have values (not all NaN). At least first few rows should be non-NaN.
    # Because we have enough data (outputsize ~1000), row index 100 should have value for horizon 1.
    if len(labeled) > 100:
        assert not pd.isna(labeled["fwd_return_1"].iloc[100]), "Row 100 should have non-NaN for h=1"
    # Last row should be NaN for any horizon
    assert pd.isna(labeled["fwd_return_1"].iloc[-1]), "Last row should be NaN for h=1"

    # Binary labels
    labeled_binary = LabelEngine.add_binary_labels(features, horizon=3)
    assert "label_binary_3" in labeled_binary.columns
    # Should have 0, 1, and NaN (for missing future data)
    unique_vals = set(labeled_binary["label_binary_3"].dropna().unique())
    assert unique_vals.issubset({0, 1}), "Binary labels should be 0 or 1"

    # Multiclass labels
    labeled_multi = LabelEngine.add_multiclass_labels(
        features, horizon=5,
        bins=[-np.inf, -0.3, 0.3, np.inf],
        labels=[0, 1, 2]
    )
    assert "label_multiclass_5" in labeled_multi.columns
    # Should have 0, 1, 2 and NaN
    unique_multi = set(labeled_multi["label_multiclass_5"].dropna().unique())
    assert unique_multi.issubset({0.0, 1.0, 2.0}), "Multiclass labels should be 0, 1, 2 (as float)"

    # Forward high/low
    labeled_hl = LabelEngine.add_forward_high_low(features, horizon=5)
    assert "fwd_high_5" in labeled_hl.columns
    assert "fwd_low_5" in labeled_hl.columns

    print("LabelEngine tests PASSED")
