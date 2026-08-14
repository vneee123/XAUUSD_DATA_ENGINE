import pandas as pd

# ============================================================
# BASE & METADATA COLUMNS
# ============================================================

BASE_COLUMNS = {
    "datetime",
    "datetime_local",
    "open",
    "high",
    "low",
    "close",
    "symbol",
    "interval",
    "candle_close_time",
    "candle_status",
}

METADATA_COLUMNS = {
    "candle_closed",
    "interval_minutes",
    "is_current_candle",
    "is_future_data",
    "source_timezone",
    "target_timezone",
}


def get_mathematical_columns(df: pd.DataFrame) -> list:
    """Return all columns that are mathematical features (exclude base/metadata)."""
    excluded = BASE_COLUMNS | METADATA_COLUMNS
    return [col for col in df.columns if col not in excluded]


class MultiTimeframeAligner:
    """
    Align higher‑timeframe features to a target timeframe using as-of merge.
    Only uses data that is available at or before each target candle close time.
    """

    @staticmethod
    def align(
        target_df: pd.DataFrame,
        higher_dfs: dict,
        higher_timeframes: list,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        target_df : pd.DataFrame
            DataFrame for the target timeframe (e.g., M5).
            Must contain 'datetime' and must be sorted.
        higher_dfs : dict
            Dictionary mapping timeframe name (e.g., 'H1', 'M15') to its DataFrame.
            Each DataFrame must contain 'datetime' and mathematical features.
        higher_timeframes : list
            List of timeframe keys from higher_dfs to align (order is preserved).

        Returns
        -------
        pd.DataFrame
            Copy of target_df with additional columns named like:
            {timeframe}_{feature_name}
            for each aligned feature.
        """
        if target_df.empty:
            return target_df.copy()

        # Ensure target is sorted by datetime
        target_sorted = target_df.sort_values("datetime").reset_index(drop=True)

        # Prepare result: start with all columns from target
        result = target_sorted.copy()

        # Collect aligned feature DataFrames
        aligned_frames = []

        for tf in higher_timeframes:
            if tf not in higher_dfs:
                continue

            higher_df = higher_dfs[tf]
            if higher_df.empty:
                continue

            # Extract mathematical columns only
            math_cols = get_mathematical_columns(higher_df)
            if not math_cols:
                continue

            # Ensure higher is sorted by datetime
            higher_sorted = higher_df.sort_values("datetime").reset_index(drop=True)

            # Merge as-of: for each target datetime, take the latest higher row
            # with datetime <= target datetime (backward direction).
            # Both DataFrames have 'datetime' as a regular column (not index).
            merged = pd.merge_asof(
                target_sorted[["datetime"]],   # left side (target)
                higher_sorted[["datetime"] + math_cols],  # right side (higher)
                on="datetime",
                direction="backward",          # <= target datetime (no look-ahead)
                allow_exact_matches=True,
            )

            # Rename columns with prefix to avoid collisions
            prefixed = merged[math_cols].rename(
                columns={col: f"{tf}_{col}" for col in math_cols}
            )
            aligned_frames.append(prefixed)

        # Combine all aligned features
        if aligned_frames:
            aligned_all = pd.concat(aligned_frames, axis=1)
            # Add them to result
            result = pd.concat([result, aligned_all], axis=1)

        return result
