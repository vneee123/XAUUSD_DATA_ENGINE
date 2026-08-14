import pandas as pd


class CandleStatus:

    @staticmethod
    def add_status(df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        now_utc = pd.Timestamp.now(tz="UTC")

        # Data candle berada di masa depan
        data["is_future_data"] = (
            data["datetime"] > now_utc
        )

        # Candle sudah selesai
        data["candle_closed"] = (
            (data["candle_close_time"] <= now_utc)
            & (~data["is_future_data"])
        )

        # Candle sedang berjalan
        data["is_current_candle"] = (
            (data["datetime"] <= now_utc)
            & (data["candle_close_time"] > now_utc)
        )

        # Status final
        data["candle_status"] = "UNKNOWN"

        data.loc[
            data["candle_closed"],
            "candle_status"
        ] = "CLOSED"

        data.loc[
            data["is_current_candle"],
            "candle_status"
        ] = "OPEN"

        data.loc[
            data["is_future_data"],
            "candle_status"
        ] = "FUTURE_DATA"

        return data
