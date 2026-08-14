import numpy as np
import pandas as pd


class PriceActionFeatures:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        # Basic return
        data["return_1"] = data["close"].pct_change()

        data["return_3"] = (
            data["close"].pct_change(3)
        )

        data["return_5"] = (
            data["close"].pct_change(5)
        )

        data["return_10"] = (
            data["close"].pct_change(10)
        )

        # Candle structure
        data["range"] = (
            data["high"] - data["low"]
        )

        data["body"] = (
            data["close"] - data["open"]
        )

        data["body_abs"] = (
            data["body"].abs()
        )

        data["upper_wick"] = (
            data["high"]
            - data[["open", "close"]].max(axis=1)
        )

        data["lower_wick"] = (
            data[["open", "close"]].min(axis=1)
            - data["low"]
        )

        # Candle ratios
        data["body_ratio"] = np.where(
            data["range"] != 0,
            data["body_abs"] / data["range"],
            0
        )

        data["upper_wick_ratio"] = np.where(
            data["range"] != 0,
            data["upper_wick"] / data["range"],
            0
        )

        data["lower_wick_ratio"] = np.where(
            data["range"] != 0,
            data["lower_wick"] / data["range"],
            0
        )

        # Position inside candle
        data["close_position"] = np.where(
            data["range"] != 0,
            (data["close"] - data["low"])
            / data["range"],
            0.5
        )

        # Direction
        data["candle_direction"] = np.sign(
            data["close"] - data["open"]
        )

        # Gap
        data["gap"] = (
            data["open"]
            - data["close"].shift(1)
        )

        data["gap_pct"] = np.where(
            data["close"].shift(1) != 0,
            data["gap"]
            / data["close"].shift(1),
            0
        )

        # Rolling ranges
        for period in [5, 10, 20, 50]:

            data[f"range_mean_{period}"] = (
                data["range"]
                .rolling(period)
                .mean()
            )

            data[f"range_std_{period}"] = (
                data["range"]
                .rolling(period)
                .std()
            )

            data[f"close_position_mean_{period}"] = (
                data["close_position"]
                .rolling(period)
                .mean()
            )

        # Rolling highs/lows
        for period in [5, 10, 20]:

            data[f"rolling_high_{period}"] = (
                data["high"]
                .rolling(period)
                .max()
            )

            data[f"rolling_low_{period}"] = (
                data["low"]
                .rolling(period)
                .min()
            )

        return data
