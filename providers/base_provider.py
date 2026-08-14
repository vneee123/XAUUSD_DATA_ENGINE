from abc import ABC, abstractmethod
import pandas as pd


class BaseMarketDataProvider(ABC):

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        interval: str,
        outputsize: int
    ) -> pd.DataFrame:
        """Return normalized OHLCV candle data."""
        raise NotImplementedError
