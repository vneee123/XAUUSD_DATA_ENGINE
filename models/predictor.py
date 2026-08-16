import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class Predictor:
    @staticmethod
    def save_model(model, feature_names, name="random_forest"):
        path = MODEL_DIR / f"{name}.pkl"
        data = {"model": model, "feature_names": feature_names}
        joblib.dump(data, path)
        return path

    @staticmethod
    def load_model(name="random_forest"):
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model {name} not found at {path}")
        data = joblib.load(path)
        return data["model"], data["feature_names"]

    @staticmethod
    def predict(model, X):
        """Predict class and probabilities for given X."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        pred = model.predict(X)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        return pred, proba
