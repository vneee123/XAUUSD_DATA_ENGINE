import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import numpy as np
from optimization.hyperopt import HyperparameterOptimizer
from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine
from dataset.builder import DatasetBuilder

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"

def test_hyperopt():
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

    # Quick test with few trials
    model, best_params, best_score, study = HyperparameterOptimizer.run_optimization(
        X, y, model_type="random_forest", n_trials=5
    )

    assert best_score > 0.5, f"Best score too low: {best_score}"
    assert len(best_params) > 0, "No parameters found"
    print(f"Hyperparameter optimization test passed! Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")

if __name__ == "__main__":
    test_hyperopt()
