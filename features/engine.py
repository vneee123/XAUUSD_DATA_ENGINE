import pandas as pd

from features.price_action import PriceActionFeatures
from features.trend import TrendFeatures
from features.momentum import MomentumFeatures
from features.volatility import VolatilityFeatures
from features.macd import MACDFeatures
from features.oscillators import OscillatorFeatures
from features.adx import ADXFeatures


class FeatureEngine:

    @staticmethod
    def calculate(
        df: pd.DataFrame,
        closed_only: bool = False
    ) -> pd.DataFrame:

        if df.empty:
            return df.copy()

        data = df.copy()

        # Pastikan urutan waktu benar
        data = data.sort_values(
            "datetime"
        ).reset_index(drop=True)

        # Simpan status candle sebelum filtering
        if closed_only and "candle_status" in data.columns:
            data = data[
                data["candle_status"] == "CLOSED"
            ].copy()

            data = data.reset_index(drop=True)

        # Feature groups
        data = PriceActionFeatures.calculate(data)

        data = TrendFeatures.calculate(data)

        data = MomentumFeatures.calculate(data)

        data = VolatilityFeatures.calculate(data)

        data = MACDFeatures.calculate(data)

        data = OscillatorFeatures.calculate(data)

        data = ADXFeatures.calculate(data)

        return data
