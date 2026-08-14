import pandas as pd

from providers.twelve_data import TwelveDataProvider


class MarketDataCollector:

    def __init__(self):
        self.provider = TwelveDataProvider()

    def collect(
        self,
        symbol: str,
        timeframes: dict
    ) -> dict[str, pd.DataFrame]:

        result = {}

        for timeframe, config in timeframes.items():

            interval = config["interval"]
            outputsize = config["outputsize"]

            df = self.provider.get_candles(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize
            )

            result[timeframe] = df

        return result
