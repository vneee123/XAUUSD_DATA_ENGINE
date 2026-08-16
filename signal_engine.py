import numpy as np
import pandas as pd
from models.predictor import Predictor

# Konfigurasi: 1 pip = 0.1 points (XAUUSD)
PIP_SIZE = 0.1
SL_PIPS = 50       # 50 pips = 5.0 points
TP1_PIPS = 50      # 50 pips = 5.0 points
TP2_PIPS = 100     # 100 pips = 10.0 points

class SignalEngine:
    @staticmethod
    def generate_signal_with_X(X, model, feature_names, latest_df, threshold=0.6):
        pred, proba = Predictor.predict(model, X)
        prob = proba[0] if proba is not None else [0.5, 0.5]
        confidence = max(prob)
        signal_class = int(pred[0])

        close_price = latest_df["close"].values[0]
        atr = latest_df["atr_14"].values[0] if "atr_14" in latest_df.columns else 1.0

        sl_points = SL_PIPS * PIP_SIZE
        tp1_points = TP1_PIPS * PIP_SIZE
        tp2_points = TP2_PIPS * PIP_SIZE

        if signal_class == 1 and confidence >= threshold:
            entry = close_price
            stop_loss = entry - sl_points
            take_profit_1 = entry + tp1_points
            take_profit_2 = entry + tp2_points
            signal = "BUY"
        elif signal_class == 0 and confidence >= threshold:
            entry = close_price
            stop_loss = entry + sl_points
            take_profit_1 = entry - tp1_points
            take_profit_2 = entry - tp2_points
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
            "predicted_class": signal_class,
            "sl_pips": SL_PIPS,
            "tp1_pips": TP1_PIPS,
            "tp2_pips": TP2_PIPS
        }
