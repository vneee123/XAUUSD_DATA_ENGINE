import os
import sys
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from pathlib import Path
from io import StringIO
from dotenv import load_dotenv
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine
from features.multi_timeframe import MultiTimeframeAligner
from dataset.builder import DatasetBuilder
from models.trainer import ModelTrainer
from models.predictor import Predictor
from signal_engine import SignalEngine
from backtest_engine import BacktestEngine

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN must be set")

chat_ids_env = os.getenv("TELEGRAM_CHAT_IDS")
if chat_ids_env:
    CHAT_IDS = [cid.strip() for cid in chat_ids_env.split(",") if cid.strip()]
else:
    single = os.getenv("TELEGRAM_CHAT_ID")
    if single:
        CHAT_IDS = [single.strip()]
    else:
        raise RuntimeError("TELEGRAM_CHAT_ID or TELEGRAM_CHAT_IDS must be set")

BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

SESSION_TIMES = {
    "Tokyo":   {"start": 7,   "end": 16},
    "London":  {"start": 15,  "end": 24},
    "NewYork": {"start": 20,  "end": 5}
}

MAX_ROWS = 500

cache = {"raw": {}, "features": {}, "dataset": {}}
last_update = None

def debug(msg):
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} - {msg}")

# ------------------------------------------------------------------
# TELEGRAM SENDER - clean & professional
# ------------------------------------------------------------------
def send_message(text):
    # Hilangkan simbol berlebihan, rapikan teks
    text = text.replace("*", "")
    text = text.replace("✅", "[OK]")
    text = text.replace("❌", "[ERR]")
    text = text.replace("⚠️", "[WARN]")
    text = text.replace("📊", "")
    text = text.replace("📈", "")
    text = text.replace("📉", "")
    text = text.replace("🤖", "")
    text = text.replace("🧠", "")
    text = text.replace("🚦", "")
    text = text.replace("🆔", "ID:")
    # Hapus baris kosong berlebihan
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    for chat_id in CHAT_IDS:
        url = BASE_URL + "/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": ""}
        try:
            requests.post(url, json=payload, timeout=30).raise_for_status()
        except Exception as e:
            debug(f"Send message error: {e}")
    return True

def send_file(df, filename="data.csv", caption="", format="csv", max_rows=MAX_ROWS):
    if df is None or df.empty:
        send_message("DataFrame kosong, tidak bisa dikirim.")
        return
    total_rows = len(df)
    if total_rows > max_rows:
        df_to_send = df.tail(max_rows).copy()
        caption += f" (last {max_rows} rows of {total_rows})"
    else:
        df_to_send = df.copy()
    if format == "json":
        json_str = df_to_send.to_json(orient="records", date_format="iso", indent=None)
        filename = filename.replace(".csv", ".json")
        files = {"document": (filename, json_str, "application/json")}
    else:
        csv_buffer = StringIO()
        df_to_send.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"document": (filename, csv_buffer.getvalue(), "text/csv")}
    data = {"chat_id": CHAT_IDS[0], "caption": caption}
    url = BASE_URL + "/sendDocument"
    for chat_id in CHAT_IDS:
        data["chat_id"] = chat_id
        try:
            requests.post(url, data=data, files=files, timeout=120).raise_for_status()
        except Exception as e:
            debug(f"Send file error: {e}")
            send_message(f"File send failed: {e}")
    return True

# ------------------------------------------------------------------
# CACHE UPDATE
# ------------------------------------------------------------------
def update_cache():
    global cache, last_update
    try:
        debug("Updating cache...")
        with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
            settings = json.load(f)
        symbol = settings["symbol"]
        timeframes = settings["timeframes"]
        collector = MarketDataCollector()
        raw_dfs = collector.collect(symbol, timeframes)
        processed = {}
        for tf, raw_df in raw_dfs.items():
            cache["raw"][tf] = raw_df.copy()
            normalized = DataNormalizer.normalize(raw_df)
            with_status = CandleStatus.add_status(normalized)
            features = FeatureEngine.calculate(with_status, closed_only=True)
            cache["features"][tf] = features.copy()
            processed[tf] = features
        if "M5" in processed:
            higher_timeframes = ["H1", "M15"]
            higher_dfs = {tf: processed[tf] for tf in higher_timeframes if tf in processed}
            X, y, feature_names = DatasetBuilder.build(
                target_df=processed["M5"],
                higher_dfs=higher_dfs,
                higher_timeframes=higher_timeframes,
                label_horizon=5,
                label_type="binary",
                label_threshold=0.0,
                dropna=True
            )
            cache["dataset"] = {
                "X": X,
                "y": y,
                "feature_names": feature_names
            }
        last_update = datetime.now(ZoneInfo("Asia/Jakarta"))
        debug(f"Cache updated at {last_update}")
        return True
    except Exception as e:
        debug(f"Cache update error: {e}")
        traceback.print_exc()
        send_message(f"Cache update error: {e}")
        return False

def get_aligned_features():
    m5 = cache["features"].get("M5")
    h1 = cache["features"].get("H1")
    m15 = cache["features"].get("M15")
    if any(df is None or df.empty for df in [m5, h1, m15]):
        debug("Missing features for alignment")
        return None
    higher_dfs = {"H1": h1, "M15": m15}
    aligned = MultiTimeframeAligner.align(m5, higher_dfs, ["H1", "M15"])
    debug(f"Aligned features shape: {aligned.shape}")
    return aligned

def combine_all_data():
    dfs = []
    for tf, df in cache["raw"].items():
        if df is not None and not df.empty:
            temp = df.copy()
            temp["timeframe"] = tf
            temp["type"] = "raw"
            dfs.append(temp)
    for tf, df in cache["features"].items():
        if df is not None and not df.empty:
            temp = df.copy()
            temp["timeframe"] = tf
            temp["type"] = "features"
            dfs.append(temp)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def train_and_get_model(model_type="random_forest"):
    ds = cache["dataset"]
    if not ds or ds.get("X") is None or ds.get("y") is None:
        send_message("Dataset tidak tersedia untuk training.")
        return None
    X = ds["X"]
    y = ds["y"]
    feature_names = ds["feature_names"]
    debug(f"Training {model_type} on X shape {X.shape}")
    model, acc, report = ModelTrainer.train(X, y, model_type=model_type)
    Predictor.save_model(model, feature_names, model_type)
    send_message(f"Model {model_type} trained. Out-of-sample accuracy: {acc:.3f}")
    return model

def get_signal_from_cache():
    try:
        model, feature_names = Predictor.load_model("random_forest")
        debug(f"Loaded model with {len(feature_names)} features")
    except FileNotFoundError:
        send_message("Model belum ada. Melatih model default...")
        model = train_and_get_model("random_forest")
        if model is None:
            return None
        _, feature_names = Predictor.load_model("random_forest")

    aligned_df = get_aligned_features()
    if aligned_df is None or aligned_df.empty:
        send_message("Aligned features tidak tersedia.")
        return None

    latest = aligned_df.iloc[-1:].copy()
    missing = set(feature_names) - set(latest.columns)
    if missing:
        debug(f"Missing {len(missing)} features, filling with 0")
        for col in missing:
            latest[col] = 0.0
    X = latest[feature_names].values.astype(np.float32)
    debug(f"Prediction X shape: {X.shape}")

    signal = SignalEngine.generate_signal_with_X(X, model, feature_names, latest, threshold=0.6)
    return signal

def run_backtest(model_type="random_forest"):
    ds = cache["dataset"]
    if not ds or ds.get("X") is None or ds.get("y") is None:
        send_message("Dataset tidak tersedia.")
        return None
    X = ds["X"]
    y = ds["y"]
    result = BacktestEngine.run(X, y, model_type=model_type)
    return result

def send_scheduled_data(format="json"):
    send_message("Running scheduled data send...")
    update_cache()
    combined = combine_all_data()
    if combined is not None and not combined.empty:
        send_file(combined, filename=f"all_data.{format}", caption="All data (raw+features)", format=format)
    ds = cache["dataset"]
    if ds and ds.get("X") is not None and ds.get("y") is not None:
        X = ds["X"]
        y = ds["y"]
        feature_names = ds["feature_names"]
        if X is not None and len(X) > 0:
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{format}", caption="ML Dataset", format=format)
    send_message("Scheduled data send completed.")

# ------------------------------------------------------------------
# COMMAND HANDLER - clean and professional
# ------------------------------------------------------------------
def handle_command(text, chat_id):
    parts = text.strip().split()
    if not parts:
        return
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ["/start", "/help"]:
        help_text = (
            "XAUUSD_DATA_ENGINE Bot v1.8 - Commands:\n"
            "/signal              - Get trading signal with SL/TP\n"
            "/predict             - Get prediction probability only\n"
            "/signal_detail       - Full signal with detailed analysis\n"
            "/backtest [model]    - Run backtest (random_forest, logistic, xgboost)\n"
            "/train [model]       - Train model\n"
            "/optimize            - Find optimal threshold\n"
            "/risk                - Show current risk/reward\n"
            "/raw [tf] [format]   - Send raw data\n"
            "/features [tf] [format] - Send feature data\n"
            "/dataset [format]    - Send ML dataset\n"
            "/all [format]        - Send all data in one file\n"
            "/send_now [format]   - Trigger scheduled send\n"
            "/status              - Cache status\n"
            "/info                - System info\n"
            "/get_chat_id         - Your chat ID\n"
            "/clear_cache         - Clear cache and model\n"
            "/help                - This help\n\n"
            "Auto schedule: Tokyo (07-16), London (15-24), NewYork (20-05) WIB"
        )
        send_message(help_text)
        return

    fmt = "json"
    if args and args[-1].lower() in ["csv", "json"]:
        fmt = args[-1].lower()
        args = args[:-1]

    model_type = "random_forest"
    if args and args[0].lower() in ["random_forest", "logistic", "xgboost"]:
        model_type = args[0].lower()
        args = args[1:]

    # --- NEW / UPDATED COMMANDS ---
    if cmd == "/clear_cache":
        global cache
        cache = {"raw": {}, "features": {}, "dataset": {}}
        for p in Path("models/saved").glob("*.pkl"):
            p.unlink()
        send_message("Cache and model cleared. Bot will rebuild on next request.")
        return

    if cmd == "/predict":
        signal = get_signal_from_cache()
        if signal:
            msg = f"Prediction: {signal['signal']}\nConfidence: {signal['confidence']:.3f}"
            send_message(msg)
        return

    if cmd == "/signal_detail":
        signal = get_signal_from_cache()
        if signal is None:
            return
        if signal["signal"] == "HOLD":
            msg = f"HOLD\nConfidence: {signal['confidence']:.3f}\nNo strong signal."
        else:
            risk = abs(signal['entry'] - signal['stop_loss'])
            reward1 = abs(signal['take_profit_1'] - signal['entry'])
            reward2 = abs(signal['take_profit_2'] - signal['entry'])
            msg = (
                f"Signal: {signal['signal']}\n"
                f"Entry: {signal['entry']:.3f}\n"
                f"Stop Loss: {signal['stop_loss']:.3f} (Risk: {risk:.2f} points)\n"
                f"Take Profit 1: {signal['take_profit_1']:.3f} (R:R {reward1/risk:.2f})\n"
                f"Take Profit 2: {signal['take_profit_2']:.3f} (R:R {reward2/risk:.2f})\n"
                f"Close: {signal['close_price']:.3f}\n"
                f"ATR: {signal['atr']:.3f}\n"
                f"Confidence: {signal['confidence']:.3f}"
            )
        send_message(msg)
        return

    if cmd == "/optimize":
        send_message("Optimization not yet implemented. Use /train and adjust threshold manually.")
        return

    if cmd == "/risk":
        signal = get_signal_from_cache()
        if signal is None or signal["signal"] == "HOLD":
            send_message("No active trade to assess risk.")
            return
        risk = abs(signal['entry'] - signal['stop_loss'])
        reward1 = abs(signal['take_profit_1'] - signal['entry'])
        reward2 = abs(signal['take_profit_2'] - signal['entry'])
        msg = (
            f"Risk/Reward Analysis:\n"
            f"Risk: {risk:.2f} points\n"
            f"Reward TP1: {reward1:.2f} points (R:R {reward1/risk:.2f})\n"
            f"Reward TP2: {reward2:.2f} points (R:R {reward2/risk:.2f})"
        )
        send_message(msg)
        return

    # --- EXISTING COMMANDS ---
    if cmd == "/get_chat_id":
        send_message(f"Your Chat ID: {chat_id}")
        return

    if cmd == "/train":
        send_message(f"Training model {model_type}...")
        model = train_and_get_model(model_type)
        if model:
            send_message(f"Model {model_type} saved successfully.")
        else:
            send_message("Training failed.")
        return

    if cmd == "/backtest":
        send_message(f"Running backtest for {model_type}...")
        result = run_backtest(model_type)
        if result:
            msg = f"Backtest Result ({model_type})\nWinrate: {result['winrate']:.3f}\nTotal Signals: {result['total_signals']}\nCorrect: {result['correct_signals']}"
            send_message(msg)
        else:
            send_message("Backtest failed.")
        return

    if cmd == "/signal":
        send_message("Generating signal...")
        signal = get_signal_from_cache()
        if signal is None:
            return
        if signal["signal"] == "HOLD":
            msg = f"HOLD\nConfidence: {signal['confidence']:.3f}"
        else:
            msg = (
                f"Signal: {signal['signal']}\n"
                f"Entry: {signal['entry']:.3f}\n"
                f"SL: {signal['stop_loss']:.3f}\n"
                f"TP1: {signal['take_profit_1']:.3f}\n"
                f"TP2: {signal['take_profit_2']:.3f}\n"
                f"Close: {signal['close_price']:.3f}\n"
                f"Confidence: {signal['confidence']:.3f}"
            )
        send_message(msg)
        return

    if cmd == "/raw":
        tf = args[0] if args else "M5"
        df = cache["raw"].get(tf)
        if df is None or df.empty:
            send_message(f"Raw data for {tf} not available.")
        else:
            send_file(df, filename=f"raw_{tf}.{fmt}", caption=f"Raw {tf}", format=fmt)
        return

    if cmd == "/features":
        tf = args[0] if args else "M5"
        df = cache["features"].get(tf)
        if df is None or df.empty:
            send_message(f"Features for {tf} not available.")
        else:
            send_file(df, filename=f"features_{tf}.{fmt}", caption=f"Features {tf}", format=fmt)
        return

    if cmd == "/dataset":
        ds = cache["dataset"]
        if not ds or ds.get("X") is None or ds.get("y") is None:
            send_message("Dataset not available.")
        else:
            X = ds["X"]
            y = ds["y"]
            feature_names = ds["feature_names"]
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{fmt}", caption="ML Dataset", format=fmt)
        return

    if cmd == "/all":
        send_message(f"Sending all data in one file ({fmt})...")
        combined = combine_all_data()
        if combined is not None and not combined.empty:
            send_file(combined, filename=f"all_data.{fmt}", caption="All data (raw+features)", format=fmt)
        ds = cache["dataset"]
        if ds and ds.get("X") is not None and ds.get("y") is not None:
            X = ds["X"]
            y = ds["y"]
            feature_names = ds["feature_names"]
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{fmt}", caption="ML Dataset", format=fmt)
        send_message("All data sent.")
        return

    if cmd == "/send_now":
        send_scheduled_data(format=fmt)
        return

    if cmd == "/status":
        status = f"Cache Status\nLast update: {last_update}\nRaw TFs: {list(cache['raw'].keys())}\nFeatures TFs: {list(cache['features'].keys())}\n"
        ds = cache["dataset"]
        if ds and ds.get("X") is not None:
            status += f"Dataset: X shape {ds['X'].shape}"
        else:
            status += "Dataset: not available"
        send_message(status)
        return

    if cmd == "/info":
        info = (
            f"XAUUSD_DATA_ENGINE Bot v1.8\n"
            f"Server time: {datetime.now(ZoneInfo('Asia/Jakarta'))}\n"
            f"Pipeline: Normalizer -> CandleStatus -> FeatureEngine -> MultiTimeframe -> Dataset -> ML\n"
            f"Chat IDs: {CHAT_IDS}\n"
            f"Max rows per file: {MAX_ROWS}"
        )
        send_message(info)
        return

    else:
        send_message(f"Unknown command. Type /help for list.")
        return

# ------------------------------------------------------------------
# POLLING & SCHEDULER
# ------------------------------------------------------------------
def run_bot():
    offset = None
    debug("Bot polling started.")
    while True:
        try:
            url = BASE_URL + "/getUpdates"
            params = {"offset": offset, "timeout": 60, "allowed_updates": ["message"]}
            resp = requests.get(url, params=params, timeout=65)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                updates = data.get("result", [])
                for update in updates:
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if message and "text" in message:
                        chat_id = message["chat"]["id"]
                        text = message["text"]
                        debug(f"Command from {chat_id}: {text}")
                        handle_command(text, chat_id)
        except Exception as e:
            debug(f"Polling error: {e}")
            traceback.print_exc()
            time.sleep(5)
        time.sleep(1)

last_sent = {"Tokyo": None, "London": None, "NewYork": None}

def is_in_session(now, session):
    start = SESSION_TIMES[session]["start"]
    end = SESSION_TIMES[session]["end"]
    current_hour = now.hour
    if start < end:
        return start <= current_hour < end
    else:
        return current_hour >= start or current_hour < end

def run_scheduler():
    debug("Scheduler started.")
    while True:
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
        for session in SESSION_TIMES:
            if is_in_session(now, session):
                if last_sent[session] != now.date():
                    debug(f"Triggering scheduled send for {session} at {now}")
                    try:
                        send_scheduled_data(format="json")
                        last_sent[session] = now.date()
                    except Exception as e:
                        debug(f"Scheduled send error: {e}")
                        traceback.print_exc()
        time.sleep(60)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("Starting Telegram Bot + Scheduler + ML Signal (v1.8)")
    print(f"Chat IDs: {CHAT_IDS}")
    print(f"Max rows per file: {MAX_ROWS}")

    update_cache()
    send_message("XAUUSD_DATA_ENGINE Bot v1.8\nSystem ready. Type /help for commands.")

    try:
        Predictor.load_model("random_forest")
        print("Model already exists.")
    except FileNotFoundError:
        print("Model not found, training...")
        train_and_get_model("random_forest")

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    bot_thread.start()
    scheduler_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
