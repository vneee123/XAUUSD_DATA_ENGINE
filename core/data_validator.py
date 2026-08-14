import pandas as pd


class DataValidator:

    @staticmethod
    def validate(df: pd.DataFrame) -> dict:

        report = {}

        report["rows"] = len(df)

        report["null_values"] = int(
            df.isnull().sum().sum()
        )

        report["duplicate_timestamps"] = int(
            df["datetime"].duplicated().sum()
        )

        invalid_high = (
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["high"] < df["low"])
        )

        invalid_low = (
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["low"] > df["high"])
        )

        report["invalid_high"] = int(
            invalid_high.sum()
        )

        report["invalid_low"] = int(
            invalid_low.sum()
        )

        report["non_positive_prices"] = int(
            (
                df[
                    ["open", "high", "low", "close"]
                ] <= 0
            ).sum().sum()
        )

        report["timestamp_not_sorted"] = not (
            df["datetime"].is_monotonic_increasing
        )

        report["valid"] = all([
            report["null_values"] == 0,
            report["duplicate_timestamps"] == 0,
            report["invalid_high"] == 0,
            report["invalid_low"] == 0,
            report["non_positive_prices"] == 0,
            report["timestamp_not_sorted"] is False
        ])

        return report
