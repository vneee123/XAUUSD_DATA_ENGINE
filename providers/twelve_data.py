import os
import requests
import pandas as pd

from dotenv import load_dotenv

from .base_provider import BaseMarketDataProvider


class TwelveDataProvider(BaseMarketDataProvider):

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "TWELVE_DATA_API_KEY tidak ditemukan di .env"
            )

    def get_candles(
        self,
        symbol: str,
        interval: str,
        outputsize: int
    ) -> pd.DataFrame:

        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") == "error":
            raise RuntimeError(
                f"Twelve Data API error: {data}"
            )

        values = data.get("values", [])

        if not values:
            raise RuntimeError(
                f"Tidak ada data untuk {symbol} {interval}"
            )

        df = pd.DataFrame(values)

        # Numeric conversion
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # Timestamp
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        # Remove invalid rows
        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close"
            ]
        )

        # Sort oldest → newest
        df = df.sort_values(
            "datetime"
        ).reset_index(drop=True)

        # Metadata
        df["symbol"] = symbol
        df["interval"] = interval

        return df
