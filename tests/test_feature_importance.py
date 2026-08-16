import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine
from dataset.builder import DatasetBuilder
from models.trainer import ModelTrainer
from models.predictor import Predictor

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

def test_feature_importance():
    """Test feature importance analysis on trained model."""
    with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
        settings = json.load(f)

    symbol = settings["symbol"]
    timeframes = settings["timeframes"]

    collector = MarketDataCollector()
    raw_dfs = collector.collect(symbol, timeframes)

    processed = {}
    for tf, raw_df in raw_dfs.items():
        normalized = DataNormalizer.normalize(raw_df)
        with_status = CandleStatus.add_status(normalized)
        features = FeatureEngine.calculate(with_status, closed_only=True)
        processed[tf] = features

    if "M5" in processed:
        higher_timeframes = ["H1", "M15"]
        higher_dfs = {tf: processed[tf] for tf in higher_timeframes if tf in processed}
        X, y, feature_names = DatasetBuilder.build(
            target_df=processed["M5"],
            higher_dfs=higher_dfs,
            higher_timeframes=higher_timeframes,
            label_horizon=5,
            label_type="binary",
            label_threshold=0.0,
            dropna=True
        )
    else:
        assert False, "M5 data not available"

    # Train model
    model, acc, report = ModelTrainer.train(X, y, model_type="random_forest")

    # Extract feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Plot top 20
        indices = np.argsort(importances)[-20:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=100)
        plt.close()
    else:
        assert False, "Model does not have feature_importances_"

    # Ensure we have at least some features with importance > 0
    assert len(importances) > 0, "No feature importances"
    assert np.sum(importances) > 0, "All importances zero"

    print("Feature importance test passed. Plot saved to feature_importance.png")
