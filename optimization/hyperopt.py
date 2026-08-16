import sys
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.trainer import ModelTrainer
from models.predictor import Predictor

class HyperparameterOptimizer:
    @staticmethod
    def optimize_random_forest(X, y, n_trials=50, cv=5):
        """Optimize Random Forest hyperparameters using Optuna."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize", study_name="random_forest_opt")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        # Train final model with best params
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X, y)

        return model, best_params, best_score, study

    @staticmethod
    def optimize_xgboost(X, y, n_trials=50, cv=5):
        """Optimize XGBoost hyperparameters using Optuna."""
        import xgboost as xgb

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
            }
            model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize", study_name="xgboost_opt")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
        model.fit(X, y)

        return model, best_params, best_score, study

    @staticmethod
    def optimize_logistic(X, y, n_trials=50, cv=5):
        """Optimize Logistic Regression hyperparameters."""
        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 10, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 500, 2000, step=500),
            }
            # l1 penalty only works with certain solvers
            if params["penalty"] == "l1" and params["solver"] not in ["liblinear", "saga"]:
                params["solver"] = "liblinear"
            model = LogisticRegression(**params, random_state=42, max_iter=params["max_iter"])
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize", study_name="logistic_opt")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        model = LogisticRegression(**best_params, random_state=42, max_iter=best_params.get("max_iter", 1000))
        model.fit(X, y)

        return model, best_params, best_score, study

    @staticmethod
    def run_optimization(X, y, model_type="random_forest", n_trials=30):
        if model_type == "random_forest":
            return HyperparameterOptimizer.optimize_random_forest(X, y, n_trials)
        elif model_type == "xgboost":
            return HyperparameterOptimizer.optimize_xgboost(X, y, n_trials)
        elif model_type == "logistic":
            return HyperparameterOptimizer.optimize_logistic(X, y, n_trials)
        else:
            raise ValueError("model_type must be random_forest, xgboost, or logistic")
