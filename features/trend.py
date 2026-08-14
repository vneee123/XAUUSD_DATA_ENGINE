import numpy as np
import pandas as pd


class TrendFeatures:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        periods = [5, 10, 20, 50, 100, 200]

        for period in periods:

            data[f"sma_{period}"] = (
                data["close"]
                .rolling(period)
                .mean()
            )

            data[f"ema_{period}"] = (
                data["close"]
                .ewm(
                    span=period,
                    adjust=False
                )
                .mean()
            )

        # EMA slopes
        for period in [10, 20, 50, 200]:

            ema_col = f"ema_{period}"

            data[f"{ema_col}_slope"] = (
                data[ema_col]
                .diff()
            )

            data[f"{ema_col}_distance_pct"] = (
                (data["close"] - data[ema_col])
                / data[ema_col]
            )

        # Trend alignment
        data["ema_10_above_20"] = (
            data["ema_10"] >
            data["ema_20"]
        ).astype(int)

        data["ema_20_above_50"] = (
            data["ema_20"] >
            data["ema_50"]
        ).astype(int)

        data["ema_50_above_200"] = (
            data["ema_50"] >
            data["ema_200"]
        ).astype(int)

        # Simple trend score
        data["trend_score"] = (
            data["ema_10_above_20"]
            + data["ema_20_above_50"]
            + data["ema_50_above_200"]
        )

        return data
