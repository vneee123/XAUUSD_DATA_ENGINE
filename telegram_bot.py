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

# ============================================================
# PROJECT ROOT & SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

# ============================================================
# IMPORTS DARI PROYEK
# ============================================================

from collectors.market_data import MarketDataCollector
from core.data_normalizer import DataNormalizer
from core.candle_status import CandleStatus
from features.engine import FeatureEngine
from dataset.builder import DatasetBuilder
from models.trainer import ModelTrainer
from models.predictor import Predictor
from signal_engine import SignalEngine

# ============================================================
# KONFIGURASI MULTI CHAT ID
# ============================================================

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

# ============================================================
# CACHE GLOBAL
# ============================================================

cache = {
    "raw": {},
    "features": {},
    "dataset": {}
}
last_update = None

# ============================================================
# TELEGRAM SENDER
# ============================================================

def send_message(text):
    for chat_id in CHAT_IDS:
        url = BASE_URL + "/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    return True

def send_file(df, filename="data.csv", caption="", format="csv", max_rows=MAX_ROWS):
    if df is None or df.empty:
        send_message("⚠️ DataFrame kosong, tidak bisa dikirim.")
        return
    total_rows = len(df)
    if total_rows > max_rows:
        df_to_send = df.tail(max_rows).copy()
        caption += f" (hanya {max_rows} baris terakhir dari total {total_rows} baris)"
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
        resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
    return True

# ============================================================
# UPDATE CACHE
# ============================================================

def update_cache():
    global cache, last_update
    try:
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
        print(f"✅ Cache updated at {last_update}")
        return True
    except Exception as e:
        print(f"❌ Error updating cache: {e}")
        send_message(f"❌ Error updating cache: {e}")
        return False

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

# ============================================================
# TRAIN MODEL & GET SIGNAL
# ============================================================

def train_and_get_model():
    ds = cache["dataset"]
    if not ds or ds.get("X") is None or ds.get("y") is None:
        send_message("❌ Dataset tidak tersedia untuk training.")
        return None
    X = ds["X"]
    y = ds["y"]
    model, acc, report = ModelTrainer.train(X, y, model_type="random_forest")
    # Simpan model
    Predictor.save_model(model, "random_forest")
    send_message(f"🧠 Model trained with accuracy: {acc:.3f}")
    return model

def get_signal_from_cache():
    # Coba load model, jika tidak ada, train dulu
    try:
        model = Predictor.load_model("random_forest")
    except FileNotFoundError:
        send_message("🧠 Model belum ada. Melatih model dari dataset...")
        model = train_and_get_model()
        if model is None:
            send_message("❌ Gagal melatih model.")
            return None

    # Ambil features M5
    features_df = cache["features"].get("M5")
    if features_df is None or features_df.empty:
        send_message("❌ Features M5 tidak tersedia.")
        return None

    signal = SignalEngine.generate_signal(features_df, model, threshold=0.6)
    return signal

# ============================================================
# SEND SCHEDULED DATA
# ============================================================

def send_scheduled_data(format="json"):
    send_message(f"📊 **Menjalankan pengiriman data terjadwal (format {format})...**")
    success = update_cache()
    if not success:
        return
    combined = combine_all_data()
    if combined is not None and not combined.empty:
        send_file(combined, filename=f"all_data.{format}", caption="📊 Semua data (raw+features) semua timeframe", format=format)
    ds = cache["dataset"]
    if ds and ds.get("X") is not None and ds.get("y") is not None:
        X = ds["X"]
        y = ds["y"]
        feature_names = ds["feature_names"]
        if X is not None and len(X) > 0:
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{format}", caption="🤖 Dataset ML", format=format)
    send_message("✅ **Semua data terjadwal telah terkirim.**")

# ============================================================
# HANDLE COMMAND
# ============================================================

def handle_command(text, chat_id):
    parts = text.strip().split()
    if not parts:
        return
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ["/start", "/help"]:
        help_text = (
            "🤖 **XAUUSD_DATA_ENGINE Bot**\n\n"
            "📌 **Command:**\n"
            "/signal              - Kirim sinyal trading + limit order\n"
            "/raw [tf] [format]   - Kirim data mentah (contoh: /raw M5 json)\n"
            "/features [tf] [format] - Kirim feature engineering\n"
            "/dataset [format]    - Kirim dataset ML\n"
            "/all [format]        - Kirim SATU FILE gabungan semua data\n"
            "/send_now [format]   - Kirim data terjadwal\n"
            "/train               - Latih ulang model\n"
            "/status              - Status cache\n"
            "/info                - Informasi sistem\n"
            "/get_chat_id         - Tampilkan chat ID Anda\n"
            "/help                - Bantuan\n\n"
            "🕒 Jadwal otomatis: Tokyo (07-16), London (15-24), NewYork (20-05) WIB"
        )
        send_message(help_text)

    fmt = "json"
    if args and args[-1].lower() in ["csv", "json"]:
        fmt = args[-1].lower()
        args = args[:-1]

    if cmd == "/get_chat_id":
        send_message(f"🆔 Chat ID Anda: {chat_id}")

    elif cmd == "/train":
        send_message("🧠 Melatih ulang model...")
        model = train_and_get_model()
        if model:
            send_message("✅ Model berhasil dilatih dan disimpan.")
        else:
            send_message("❌ Gagal melatih model.")

    elif cmd == "/signal":
        send_message("📈 **Menghasilkan sinyal trading...**")
        signal = get_signal_from_cache()
        if signal is None:
            return
        if signal["signal"] == "HOLD":
            msg = f"⚠️ **HOLD**\nConfidence: {signal['confidence']:.2f}\nTidak ada sinyal kuat."
        else:
            msg = (
                f"🚦 **{signal['signal']}**\n"
                f"📊 Entry: {signal['entry']:.3f}\n"
                f"🛑 Stop Loss: {signal['stop_loss']:.3f}\n"
                f"🎯 Take Profit 1: {signal['take_profit_1']:.3f}\n"
                f"🎯 Take Profit 2: {signal['take_profit_2']:.3f}\n"
                f"📈 Close: {signal['close_price']:.3f}\n"
                f"📉 ATR: {signal['atr']:.3f}\n"
                f"🧠 Confidence: {signal['confidence']:.2f}"
            )
        send_message(msg)

    elif cmd == "/raw":
        tf = args[0] if args else "M5"
        df = cache["raw"].get(tf)
        if df is None or df.empty:
            send_message(f"❌ Data raw untuk {tf} tidak tersedia.")
        else:
            send_file(df, filename=f"raw_{tf}.{fmt}", caption=f"📈 Data mentah {tf}", format=fmt)

    elif cmd == "/features":
        tf = args[0] if args else "M5"
        df = cache["features"].get(tf)
        if df is None or df.empty:
            send_message(f"❌ Data features untuk {tf} tidak tersedia.")
        else:
            send_file(df, filename=f"features_{tf}.{fmt}", caption=f"🧮 Feature Engineering {tf}", format=fmt)

    elif cmd == "/dataset":
        ds = cache["dataset"]
        if not ds or ds.get("X") is None or ds.get("y") is None:
            send_message("❌ Dataset tidak tersedia.")
        else:
            X = ds["X"]
            y = ds["y"]
            feature_names = ds["feature_names"]
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{fmt}", caption="🤖 Dataset ML", format=fmt)

    elif cmd == "/all":
        send_message(f"📦 **Mengirim semua data dalam SATU FILE (format {fmt})...**")
        combined = combine_all_data()
        if combined is not None and not combined.empty:
            send_file(combined, filename=f"all_data.{fmt}", caption="📊 Semua data (raw+features) semua timeframe", format=fmt)
        ds = cache["dataset"]
        if ds and ds.get("X") is not None and ds.get("y") is not None:
            X = ds["X"]
            y = ds["y"]
            feature_names = ds["feature_names"]
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            send_file(df_dataset, filename=f"dataset.{fmt}", caption="🤖 Dataset ML", format=fmt)
        send_message("✅ **Semua data terkirim.**")

    elif cmd == "/send_now":
        send_scheduled_data(format=fmt)

    elif cmd == "/status":
        status = f"📊 **Status Cache**\n"
        status += f"🕒 Terakhir update: {last_update}\n" if last_update else "🕒 Cache belum diupdate.\n"
        status += f"📁 Raw timeframes: {list(cache['raw'].keys())}\n"
        status += f"🧮 Features timeframes: {list(cache['features'].keys())}\n"
        ds = cache["dataset"]
        if ds and ds.get("X") is not None:
            status += f"🤖 Dataset: X shape {ds['X'].shape}"
        else:
            status += "🤖 Dataset: tidak tersedia"
        send_message(status)

    elif cmd == "/info":
        info = "📌 **XAUUSD_DATA_ENGINE Bot**\n"
        info += f"🕒 Waktu server: {datetime.now(ZoneInfo('Asia/Jakarta'))}\n"
        info += "⚙️ Pipeline: Normalizer → CandleStatus → FeatureEngine → MultiTimeframe → Dataset → ML Model\n"
        info += f"📨 Chat IDs aktif: {CHAT_IDS}\n"
        info += f"📦 Maks baris per file: {MAX_ROWS}\n"
        info += "📦 Versi: 1.3"
        send_message(info)

    else:
        send_message(f"❌ Command tidak dikenal. Ketik /help untuk daftar command.")

# ============================================================
# POLLING BOT
# ============================================================

def run_bot():
    offset = None
    print("🤖 Bot started polling...")
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
                        handle_command(text, chat_id)
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(2)
        time.sleep(1)

# ============================================================
# SCHEDULER
# ============================================================

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
    print("🕒 Scheduler started. Checking every 60 seconds...")
    while True:
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
        for session in SESSION_TIMES:
            if is_in_session(now, session):
                if last_sent[session] != now.date():
                    print(f"⏰ Triggering scheduled send for {session} at {now}")
                    try:
                        send_scheduled_data(format="json")
                        last_sent[session] = now.date()
                    except Exception as e:
                        print(f"❌ Error sending scheduled data: {e}")
        time.sleep(60)

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Starting Telegram Bot + Scheduler + ML Signal...")
    print(f"📨 Target chat IDs: {CHAT_IDS}")
    print(f"📦 Maks baris per file: {MAX_ROWS}")
    update_cache()
    # Pre-train model di awal (opsional)
    try:
        Predictor.load_model("random_forest")
        print("✅ Model already exists.")
    except FileNotFoundError:
        print("🧠 Model not found, training...")
        train_and_get_model()

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    bot_thread.start()
    scheduler_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
