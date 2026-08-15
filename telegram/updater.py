import pandas as pd
from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine
from dataset.builder import DatasetBuilder
from telegram.cache import DataCache
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"

def update_all_cache():
    with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
        settings = json.load(f)

    symbol = settings["symbol"]
    timeframes = settings["timeframes"]

    collector = MarketDataCollector()
    raw_dfs = collector.collect(symbol, timeframes)

    cache = DataCache()
    processed = {}

    for tf, raw_df in raw_dfs.items():
        cache.update_raw(tf, raw_df)
        normalized = DataNormalizer.normalize(raw_df)
        with_status = CandleStatus.add_status(normalized)
        features = FeatureEngine.calculate(with_status, closed_only=True)
        cache.update_features(tf, features)
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
        cache.update_dataset(X, y, feature_names)

    print("Cache updated successfully.")
