import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class Predictor:
    @staticmethod
    def save_model(model, name="random_forest"):
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, path)
        return path

    @staticmethod
    def load_model(name="random_forest"):
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model {name} not found at {path}")
        return joblib.load(path)

    @staticmethod
    def predict(model, X):
        """X must be 2D array (n_samples, n_features)"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return model.predict(X), model.predict_proba(X)
