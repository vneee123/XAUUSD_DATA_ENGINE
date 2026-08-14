import numpy as np
import pandas as pd


class ADXFeatures:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        )

        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        )

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        for period in [14, 21]:

            atr = tr.rolling(period).mean()

            plus_di = (
                100
                * pd.Series(
                    plus_dm,
                    index=data.index
                )
                .rolling(period)
                .mean()
                / atr
            )

            minus_di = (
                100
                * pd.Series(
                    minus_dm,
                    index=data.index
                )
                .rolling(period)
                .mean()
                / atr
            )

            dx = (
                100
                * (plus_di - minus_di).abs()
                / (plus_di + minus_di)
            )

            data[f"plus_di_{period}"] = plus_di
            data[f"minus_di_{period}"] = minus_di

            data[f"adx_{period}"] = (
                dx.rolling(period).mean()
            )

            data[f"adx_direction_{period}"] = np.where(
                plus_di > minus_di,
                1,
                -1
            )

        return data
