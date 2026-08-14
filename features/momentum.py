import numpy as np
import pandas as pd


class MomentumFeatures:

    @staticmethod
    def calculate(
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        # =====================================================
        # RSI - Wilder / RMA
        # =====================================================

        delta = data["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        for period in [7, 14, 21]:

            avg_gain = gain.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period
            ).mean()

            avg_loss = loss.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period
            ).mean()

            rs = (
                avg_gain
                /
                avg_loss.replace(
                    0,
                    np.nan
                )
            )

            data[f"rsi_{period}"] = (
                100
                -
                (
                    100
                    /
                    (1 + rs)
                )
            )

        # =====================================================
        # ROC
        # =====================================================

        for period in [5, 10, 20]:

            data[f"roc_{period}"] = (
                data["close"]
                .pct_change(period)
                * 100
            )

        # =====================================================
        # MOMENTUM
        # =====================================================

        for period in [5, 10, 20]:

            data[f"momentum_{period}"] = (
                data["close"]
                -
                data["close"].shift(period)
            )

        return data
