import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTrainer:
    @staticmethod
    def train(X, y, model_type="random_forest", test_size=0.3, random_state=42):
        """
        Train classifier with CHRONOLOGICAL SPLIT (time series aware).
        No look-ahead leakage.
        """
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss')
        else:
            raise ValueError("model_type must be 'random_forest', 'logistic', or 'xgboost'")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return model, acc, report
