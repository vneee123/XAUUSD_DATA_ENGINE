import pandas as pd
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FeatureImportanceAnalyzer:
    @staticmethod
    def analyze(model, feature_names, top_n=20):
        """
        Extract feature importance from trained model.
        Supports RandomForest, XGBoost, and LogisticRegression.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")

        # Create dataframe
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # Calculate cumulative importance
        importance_df["cumulative_importance"] = importance_df["importance"].cumsum()
        importance_df["cumulative_percent"] = importance_df["cumulative_importance"] / importance_df["importance"].sum() * 100

        top_features = importance_df.head(top_n).copy()
        return {
            "all": importance_df,
            "top": top_features,
            "top_n": top_n
        }

    @staticmethod
    def save_result(result, filename="feature_importance.csv"):
        output_path = PROJECT_ROOT / "analysis" / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        result["all"].to_csv(filepath, index=False)
        return filepath

    @staticmethod
    def format_report(result, model_type="random_forest"):
        df = result["top"]
        report = f"Feature Importance Report ({model_type})\n"
        report += "=" * 50 + "\n"
        report += f"Top {result['top_n']} features:\n\n"
        for i, row in df.iterrows():
            idx = i + 1
            report += f"{idx:2d}. {row['feature']:<30} {row['importance']:.4f}\n"
        report += "\n"
        report += f"Cumulative importance of top {result['top_n']}: {df['cumulative_importance'].iloc[-1]:.2%}\n"
        return report
