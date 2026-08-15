import pandas as pd
from threading import Lock

class DataCache:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataCache, cls).__new__(cls)
                    cls._instance._init_cache()
        return cls._instance

    def _init_cache(self):
        self.cache = {
            "raw": {},
            "features": {},
            "dataset": {}
        }
        self.last_update = None

    def update_raw(self, timeframe, df):
        self.cache["raw"][timeframe] = df.copy()
        self.last_update = pd.Timestamp.now()

    def update_features(self, timeframe, df):
        self.cache["features"][timeframe] = df.copy()
        self.last_update = pd.Timestamp.now()

    def update_dataset(self, X, y, feature_names):
        self.cache["dataset"] = {
            "X": X,
            "y": y,
            "feature_names": feature_names
        }
        self.last_update = pd.Timestamp.now()

    def get_raw(self, timeframe=None):
        if timeframe:
            return self.cache["raw"].get(timeframe)
        return self.cache["raw"]

    def get_features(self, timeframe=None):
        if timeframe:
            return self.cache["features"].get(timeframe)
        return self.cache["features"]

    def get_dataset(self):
        return self.cache["dataset"]
