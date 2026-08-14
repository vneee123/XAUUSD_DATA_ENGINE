import numpy as np
import pandas as pd


class OscillatorFeatures:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        # Stochastic
        for period in [9, 14, 21]:

            lowest_low = (
                data["low"]
                .rolling(period)
                .min()
            )

            highest_high = (
                data["high"]
                .rolling(period)
                .max()
            )

            denominator = (
                highest_high - lowest_low
            )

            data[f"stoch_k_{period}"] = np.where(
                denominator != 0,
                (
                    (data["close"] - lowest_low)
                    / denominator
                    * 100
                ),
                50
            )

            data[f"stoch_d_{period}"] = (
                pd.Series(
                    data[f"stoch_k_{period}"],
                    index=data.index
                )
                .rolling(3)
                .mean()
            )

        # Williams %R
        for period in [14, 21]:

            highest_high = (
                data["high"]
                .rolling(period)
                .max()
            )

            lowest_low = (
                data["low"]
                .rolling(period)
                .min()
            )

            denominator = (
                highest_high - lowest_low
            )

            data[f"williams_r_{period}"] = np.where(
                denominator != 0,
                (
                    (highest_high - data["close"])
                    / denominator
                    * -100
                ),
                -50
            )

        # CCI
        typical_price = (
            data["high"]
            + data["low"]
            + data["close"]
        ) / 3

        for period in [14, 20]:

            sma = (
                typical_price
                .rolling(period)
                .mean()
            )

            mean_deviation = (
                typical_price
                .rolling(period)
                .apply(
                    lambda x: np.mean(
                        np.abs(x - np.mean(x))
                    ),
                    raw=True
                )
            )

            data[f"cci_{period}"] = (
                (typical_price - sma)
                / (0.015 * mean_deviation)
            )

        return data
