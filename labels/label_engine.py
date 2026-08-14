import numpy as np
import pandas as pd


class LabelEngine:
    """
    Generate labels for supervised learning using future data.
    Labels are based on forward returns (percentage change) over specified horizons.

    IMPORTANT: Labels are allowed to use future information (forward-looking).
    This is acceptable for labeling because labels are the target variable in supervised learning.
    """

    @staticmethod
    def add_forward_returns(
        df: pd.DataFrame,
        horizons: list = [1, 2, 3, 5, 10],
        close_col: str = "close"
    ) -> pd.DataFrame:
        """
        Add forward return labels for given horizons.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data, must have 'close' column.
            Assumed to be sorted chronologically.
        horizons : list of int
            Number of periods forward for return calculation.
        close_col : str
            Name of close price column (default: 'close').

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional columns:
            fwd_return_{h} for each horizon h (percentage return).
        """
        if df.empty:
            return df.copy()

        data = df.copy()

        # Ensure sorted
        data = data.sort_values("datetime").reset_index(drop=True)

        for h in horizons:
            # Future close
            fwd_close = data[close_col].shift(-h)
            # Return as percentage
            data[f"fwd_return_{h}"] = (fwd_close - data[close_col]) / data[close_col] * 100

        return data

    @staticmethod
    def add_binary_labels(
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0,
        close_col: str = "close"
    ) -> pd.DataFrame:
        """
        Add binary classification labels: 1 if forward return > threshold, else 0.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with forward returns already computed (or will compute).
        horizon : int
            Which horizon to use.
        threshold : float
            Minimum return percentage to be considered positive (e.g., 0.0 for neutral).
        close_col : str

        Returns
        -------
        pd.DataFrame
            Additional column 'label_binary_{horizon}' with 0/1.
        """
        if df.empty:
            return df.copy()

        data = df.copy()

        # Ensure forward return exists
        ret_col = f"fwd_return_{horizon}"
        if ret_col not in data.columns:
            # Compute it if missing
            data = LabelEngine.add_forward_returns(data, horizons=[horizon], close_col=close_col)

        # For rows where ret_col is NaN, keep NaN in label; otherwise assign 0/1
        data[f"label_binary_{horizon}"] = np.where(
            data[ret_col].isna(),
            np.nan,
            np.where(data[ret_col] > threshold, 1, 0)
        )

        return data

    @staticmethod
    def add_multiclass_labels(
        df: pd.DataFrame,
        horizon: int = 1,
        bins: list = None,
        labels: list = None,
        close_col: str = "close"
    ) -> pd.DataFrame:
        """
        Add multi-class labels by discretizing forward returns.

        Parameters
        ----------
        df : pd.DataFrame
        horizon : int
        bins : list of float, e.g. [-np.inf, -0.5, 0.5, np.inf]
        labels : list of int or str, same length as bins-1.
        close_col : str

        Returns
        -------
        pd.DataFrame
            Additional column 'label_multiclass_{horizon}' with float values (NaN for missing).
        """
        if df.empty:
            return df.copy()

        if bins is None:
            # Default: down (< -0.5%), neutral (-0.5% to +0.5%), up (> +0.5%)
            bins = [-np.inf, -0.5, 0.5, np.inf]
        if labels is None:
            labels = [0, 1, 2]  # 0=down, 1=neutral, 2=up

        data = df.copy()

        ret_col = f"fwd_return_{horizon}"
        if ret_col not in data.columns:
            data = LabelEngine.add_forward_returns(data, horizons=[horizon], close_col=close_col)

        # Use pd.cut; it returns a Categorical with NaN for missing
        cat_labels = pd.cut(
            data[ret_col],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Convert to float to keep NaN; also convert labels to numeric if needed
        # If labels are integers, we want float with NaN preserved.
        data[f"label_multiclass_{horizon}"] = cat_labels.astype(float)

        return data

    @staticmethod
    def add_forward_high_low(
        df: pd.DataFrame,
        horizon: int = 5,
        close_col: str = "close"
    ) -> pd.DataFrame:
        """
        Add future high and low over next 'horizon' periods as additional features.
        (These are also future-based, can be used for labeling or custom targets.)

        Returns
        -------
        df with columns: fwd_high_{h}, fwd_low_{h}, fwd_high_close_ratio_{h}, fwd_low_close_ratio_{h}
        """
        if df.empty:
            return df.copy()

        data = df.copy()
        data = data.sort_values("datetime").reset_index(drop=True)

        # For each h, create a list of shifted high/low series and take max/min
        for h in [horizon]:
            highs = pd.concat([data["high"].shift(-i) for i in range(1, h+1)], axis=1)
            lows = pd.concat([data["low"].shift(-i) for i in range(1, h+1)], axis=1)
            data[f"fwd_high_{h}"] = highs.max(axis=1)
            data[f"fwd_low_{h}"] = lows.min(axis=1)
            data[f"fwd_high_close_ratio_{h}"] = data[f"fwd_high_{h}"] / data[close_col]
            data[f"fwd_low_close_ratio_{h}"] = data[f"fwd_low_{h}"] / data[close_col]

        return data
