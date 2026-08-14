import pandas as pd


REQUIRED_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close"
]


class DataNormalizer:

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        source_timezone: str = "UTC",
        target_timezone: str = "Asia/Jakarta"
    ) -> pd.DataFrame:

        data = df.copy()

        missing = [
            col
            for col in REQUIRED_COLUMNS
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # Numeric OHLC
        for column in [
            "open",
            "high",
            "low",
            "close"
        ]:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        # Parse timestamp
        data["datetime"] = pd.to_datetime(
            data["datetime"],
            errors="coerce"
        )

        # Assume provider timestamp is UTC
        if data["datetime"].dt.tz is None:
            data["datetime"] = (
                data["datetime"]
                .dt.tz_localize(source_timezone)
            )

        # Convert to local timezone
        data["datetime_local"] = (
            data["datetime"]
            .dt.tz_convert(target_timezone)
        )

        # Remove invalid rows
        data = data.dropna(
            subset=REQUIRED_COLUMNS
        )

        # Remove duplicate timestamps
        data = data.drop_duplicates(
            subset=["datetime"],
            keep="last"
        )

        # Sort oldest → newest
        data = data.sort_values(
            "datetime"
        ).reset_index(drop=True)

        # Candle interval in minutes
        interval_minutes = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1h": 60,
            "4h": 240,
            "1day": 1440
        }

        if "interval" in data.columns:
            data["interval_minutes"] = (
                data["interval"]
                .map(interval_minutes)
            )

            # Candle close time
            data["candle_close_time"] = (
                data["datetime"] +
                pd.to_timedelta(
                    data["interval_minutes"],
                    unit="m"
                )
            )

        # Metadata
        data["source_timezone"] = source_timezone
        data["target_timezone"] = target_timezone

        return data
