import pandas as pd


class MACDFeatures:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        ema_fast = (
            data["close"]
            .ewm(
                span=12,
                adjust=False
            )
            .mean()
        )

        ema_slow = (
            data["close"]
            .ewm(
                span=26,
                adjust=False
            )
            .mean()
        )

        data["macd"] = (
            ema_fast - ema_slow
        )

        data["macd_signal"] = (
            data["macd"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        data["macd_histogram"] = (
            data["macd"]
            - data["macd_signal"]
        )

        data["macd_above_signal"] = (
            data["macd"]
            > data["macd_signal"]
        ).astype(int)

        data["macd_histogram_change"] = (
            data["macd_histogram"]
            .diff()
        )

        return data
