import pandas as pd
import numpy as np

from features.multi_timeframe import MultiTimeframeAligner
from labels.label_engine import LabelEngine


class DatasetBuilder:
    """
    Build ML-ready datasets by combining features from multiple timeframes
    and adding labels.
    """

    @staticmethod
    def build(
        target_df: pd.DataFrame,
        higher_dfs: dict,
        higher_timeframes: list,
        label_horizon: int = 5,
        label_type: str = "binary",  # "binary", "multiclass", "regression"
        label_threshold: float = 0.0,
        label_bins: list = None,
        label_labels: list = None,
        exclude_features: list = None,
        dropna: bool = True
    ) -> tuple:
        """
        Build dataset: align features, add labels, return X (features) and y (labels).

        Parameters
        ----------
        target_df : pd.DataFrame
            Target timeframe DataFrame (e.g., M5) after FeatureEngine.
        higher_dfs : dict
            Dictionary of higher timeframe DataFrames.
        higher_timeframes : list
            List of timeframe keys to align.
        label_horizon : int
            Number of periods forward for label.
        label_type : str
            'binary', 'multiclass', or 'regression'.
        label_threshold : float
            For binary: threshold for positive class.
        label_bins : list
            For multiclass: bin edges.
        label_labels : list
            For multiclass: label values.
        exclude_features : list
            Column names to exclude from features (e.g., base columns).
        dropna : bool
            Whether to drop rows where label is NaN.

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label vector.
        feature_names : list
            Names of feature columns.
        """
        if exclude_features is None:
            exclude_features = []

        # 1. Align multi-timeframe features
        aligned = MultiTimeframeAligner.align(
            target_df=target_df,
            higher_dfs=higher_dfs,
            higher_timeframes=higher_timeframes
        )

        # 2. Add labels
        if label_type == "binary":
            labeled = LabelEngine.add_binary_labels(
                aligned,
                horizon=label_horizon,
                threshold=label_threshold
            )
            label_col = f"label_binary_{label_horizon}"
        elif label_type == "multiclass":
            labeled = LabelEngine.add_multiclass_labels(
                aligned,
                horizon=label_horizon,
                bins=label_bins,
                labels=label_labels
            )
            label_col = f"label_multiclass_{label_horizon}"
        elif label_type == "regression":
            labeled = LabelEngine.add_forward_returns(
                aligned,
                horizons=[label_horizon]
            )
            label_col = f"fwd_return_{label_horizon}"
        else:
            raise ValueError(f"Unknown label_type: {label_type}")

        # 3. Separate features and label
        # Get all columns except base/metadata and label
        base_columns = {
            "datetime", "datetime_local", "open", "high", "low", "close",
            "symbol", "interval", "candle_close_time", "candle_status",
            "candle_closed", "interval_minutes", "is_current_candle",
            "is_future_data", "source_timezone", "target_timezone"
        }

        # All columns except base and label
        feature_cols = [
            col for col in labeled.columns
            if col not in base_columns and col != label_col and col not in exclude_features
        ]

        # 4. Extract X and y
        X = labeled[feature_cols].copy()
        y = labeled[label_col].copy()

        # 5. Drop rows where label is NaN (if requested)
        if dropna:
            valid = y.notna()
            X = X[valid]
            y = y[valid]

        # Convert to numpy
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # Remove any infinite values
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        return X_np, y_np, feature_cols
