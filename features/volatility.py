import numpy as np
import pandas as pd


class VolatilityFeatures:

    @staticmethod
    def calculate(
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        # =====================================================
        # TRUE RANGE
        # =====================================================

        prev_close = data["close"].shift(1)

        tr1 = (
            data["high"]
            - data["low"]
        )

        tr2 = (
            data["high"]
            - prev_close
        ).abs()

        tr3 = (
            data["low"]
            - prev_close
        ).abs()

        data["true_range"] = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        # =====================================================
        # ATR - Wilder / RMA
        # =====================================================

        for period in [7, 14, 21]:

            data[f"atr_{period}"] = (
                data["true_range"]
                .ewm(
                    alpha=1 / period,
                    adjust=False,
                    min_periods=period
                )
                .mean()
            )

            data[f"atr_pct_{period}"] = (
                data[f"atr_{period}"]
                / data["close"]
                * 100
            )

        # =====================================================
        # HISTORICAL VOLATILITY
        # =====================================================

        data["log_return"] = np.log(
            data["close"]
            / data["close"].shift(1)
        )

        for period in [10, 20, 50]:

            data[f"volatility_{period}"] = (
                data["log_return"]
                .rolling(period)
                .std()
                * np.sqrt(period)
            )

        # =====================================================
        # BOLLINGER BANDS
        # =====================================================

        for period in [20]:

            middle = (
                data["close"]
                .rolling(period)
                .mean()
            )

            std = (
                data["close"]
                .rolling(period)
                .std()
            )

            upper = (
                middle
                + (2 * std)
            )

            lower = (
                middle
                - (2 * std)
            )

            data[f"bb_middle_{period}"] = middle
            data[f"bb_upper_{period}"] = upper
            data[f"bb_lower_{period}"] = lower

            # =================================================
            # BOLLINGER WIDTH
            #
            # Canonical definition:
            #
            # BB Width = Upper Band - Lower Band
            #
            # Do NOT normalize by middle.
            # =================================================

            data[f"bb_width_{period}"] = (
                upper - lower
            )

            # =================================================
            # BOLLINGER POSITION
            #
            # 0   = lower band
            # 0.5 = middle band
            # 1   = upper band
            #
            # Can legitimately be < 0 or > 1 when price
            # moves outside the Bollinger Bands.
            # =================================================

            denominator = (
                upper - lower
            )

            data[f"bb_position_{period}"] = np.where(
                denominator != 0,
                (
                    data["close"]
                    - lower
                )
                /
                denominator,
                0.5
            )

        return data
