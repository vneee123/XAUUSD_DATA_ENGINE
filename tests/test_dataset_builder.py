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
from dataset.builder import DatasetBuilder

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

def test_dataset_builder():
    """Test dataset builder: align, label, and produce X, y."""
    with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
        settings = json.load(f)

    symbol = settings["symbol"]
    timeframes = settings["timeframes"]

    collector = MarketDataCollector()
    raw_dfs = collector.collect(symbol, timeframes)

    # Process each timeframe
    processed = {}
    for tf, raw_df in raw_dfs.items():
        normalized = DataNormalizer.normalize(raw_df)
        with_status = CandleStatus.add_status(normalized)
        features = FeatureEngine.calculate(with_status, closed_only=True)
        processed[tf] = features

    # Build dataset with M5 as target, H1 and M15 as higher
    X, y, feature_names = DatasetBuilder.build(
        target_df=processed["M5"],
        higher_dfs={tf: processed[tf] for tf in ["H1", "M15"]},
        higher_timeframes=["H1", "M15"],
        label_horizon=5,
        label_type="binary",
        label_threshold=0.0,
        dropna=True
    )

    # Assertions
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array"
    assert len(X) == len(y), "X and y lengths mismatch"
    assert len(X) > 0, "Dataset should not be empty"
    assert len(feature_names) > 0, "No features selected"

    # Check feature names include prefixed columns
    assert any(col.startswith("H1_") for col in feature_names), "Missing H1 features"
    assert any(col.startswith("M15_") for col in feature_names), "Missing M15 features"

    # Check data types
    assert X.dtype == np.float32, "X should be float32"
    assert y.dtype == np.float32, "y should be float32"

    # Check no NaNs or infs
    assert not np.isnan(X).any(), "X contains NaN"
    assert not np.isinf(X).any(), "X contains Inf"

    print("DatasetBuilder tests PASSED")
