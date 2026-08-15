import numpy as np
import pandas as pd
from models.predictor import Predictor

class SignalEngine:
    @staticmethod
    def generate_signal(features_df, model, threshold=0.6):
        """
        Generate trading signal from the latest row of features_df.
        Returns dict with signal, entry, stop_loss, take_profit levels.
        """
        if features_df is None or features_df.empty:
            return None

        # Ambil baris terakhir
        latest = features_df.iloc[-1:].copy()
        close_price = latest["close"].values[0]
        atr = latest["atr_14"].values[0] if "atr_14" in latest.columns else 0.5

        # Kolom fitur yang digunakan untuk prediksi (semua kecuali base/metadata)
        base_cols = {
            "datetime", "datetime_local", "open", "high", "low", "close",
            "symbol", "interval", "candle_close_time", "candle_status",
            "candle_closed", "interval_minutes", "is_current_candle",
            "is_future_data", "source_timezone", "target_timezone"
        }
        feature_cols = [col for col in latest.columns if col not in base_cols]
        X = latest[feature_cols].values.astype(np.float32)

        # Prediksi
        pred, proba = Predictor.predict(model, X)
        prob = proba[0] if proba is not None else [0.5, 0.5]
        confidence = max(prob)
        signal_class = int(pred[0])

        # Hitung level order
        # Support/resistance dari rolling high/low dan Bollinger
        bb_upper = latest["bb_upper_20"].values[0] if "bb_upper_20" in latest.columns else close_price + atr
        bb_lower = latest["bb_lower_20"].values[0] if "bb_lower_20" in latest.columns else close_price - atr
        rolling_high = latest["rolling_high_5"].values[0] if "rolling_high_5" in latest.columns else close_price + atr
        rolling_low = latest["rolling_low_5"].values[0] if "rolling_low_5" in latest.columns else close_price - atr

        if signal_class == 1 and confidence >= threshold:
            # BUY signal
            entry = min(bb_lower, rolling_low)  # entry di support
            stop_loss = entry - atr * 1.5
            take_profit_1 = entry + atr * 2.5
            take_profit_2 = entry + atr * 4.0
            signal = "BUY"
        elif signal_class == 0 and confidence >= threshold:
            # SELL signal
            entry = max(bb_upper, rolling_high)  # entry di resistance
            stop_loss = entry + atr * 1.5
            take_profit_1 = entry - atr * 2.5
            take_profit_2 = entry - atr * 4.0
            signal = "SELL"
        else:
            signal = "HOLD"
            entry = stop_loss = take_profit_1 = take_profit_2 = None

        return {
            "signal": signal,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "close_price": close_price,
            "atr": atr,
            "confidence": confidence,
            "predicted_class": signal_class
        }
