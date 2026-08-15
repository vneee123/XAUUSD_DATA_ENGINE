import os
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram.sender import TelegramSender
from telegram.cache import DataCache
import pandas as pd

class TelegramBot:
    def __init__(self, token=None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be set")
        self.sender = TelegramSender(token=self.token)
        self.cache = DataCache()
        self.offset = None
        self.running = True

    def handle_command(self, text, chat_id):
        parts = text.strip().split()
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/start" or cmd == "/help":
            help_text = (
                "🤖 Telegram Bot untuk XAUUSD_DATA_ENGINE\n\n"
                "Command yang tersedia:\n"
                "/raw [timeframe] - Kirim data mentah (contoh: /raw M5)\n"
                "/features [timeframe] - Kirim data feature engineering\n"
                "/dataset - Kirim dataset siap pakai (M5 dengan fitur multi-timeframe)\n"
                "/all - Kirim semua data (raw, features, dataset)\n"
                "/send_now - Kirim data otomatis (sesuai sesi) segera\n"
                "/status - Lihat status cache terakhir\n"
                "/info - Informasi bot\n"
                "/help - Tampilkan bantuan ini"
            )
            self.sender.send_message(help_text)

        elif cmd == "/raw":
            tf = args[0] if args else "M5"
            df = self.cache.get_raw(tf)
            if df is None or df.empty:
                self.sender.send_message(f"Data raw untuk {tf} tidak tersedia.")
            else:
                self.sender.send_file(df, filename=f"raw_{tf}.csv", caption=f"Data mentah {tf}")

        elif cmd == "/features":
            tf = args[0] if args else "M5"
            df = self.cache.get_features(tf)
            if df is None or df.empty:
                self.sender.send_message(f"Data features untuk {tf} tidak tersedia.")
            else:
                self.sender.send_file(df, filename=f"features_{tf}.csv", caption=f"Feature Engineering {tf}")

        elif cmd == "/dataset":
            ds = self.cache.get_dataset()
            if not ds:
                self.sender.send_message("Dataset tidak tersedia.")
            else:
                X = ds.get("X")
                y = ds.get("y")
                feature_names = ds.get("feature_names")
                if X is None or y is None:
                    self.sender.send_message("Dataset tidak lengkap.")
                else:
                    df_dataset = pd.DataFrame(X, columns=feature_names)
                    df_dataset["label"] = y
                    self.sender.send_file(df_dataset, filename="dataset.csv", caption="Dataset ML (features + label)")

        elif cmd == "/all":
            raw_m5 = self.cache.get_raw("M5")
            if raw_m5 is not None and not raw_m5.empty:
                self.sender.send_file(raw_m5, filename="raw_M5.csv", caption="Data mentah M5")
            features_m5 = self.cache.get_features("M5")
            if features_m5 is not None and not features_m5.empty:
                self.sender.send_file(features_m5, filename="features_M5.csv", caption="Feature Engineering M5")
            ds = self.cache.get_dataset()
            if ds:
                X = ds.get("X")
                y = ds.get("y")
                feature_names = ds.get("feature_names")
                if X is not None and y is not None:
                    df_dataset = pd.DataFrame(X, columns=feature_names)
                    df_dataset["label"] = y
                    self.sender.send_file(df_dataset, filename="dataset.csv", caption="Dataset ML")
            self.sender.send_message("Semua data terkirim.")

        elif cmd == "/send_now":
            from telegram.scheduler import send_scheduled_data
            send_scheduled_data()
            self.sender.send_message("Pengiriman data terjadwal telah dijalankan.")

        elif cmd == "/status":
            status = f"Cache terakhir diperbarui: {self.cache.last_update}\n"
            status += f"Raw timeframes: {list(self.cache.cache['raw'].keys())}\n"
            status += f"Features timeframes: {list(self.cache.cache['features'].keys())}\n"
            ds = self.cache.get_dataset()
            if ds:
                status += f"Dataset tersedia: X shape {ds['X'].shape if ds.get('X') is not None else 'N/A'}"
            self.sender.send_message(status)

        elif cmd == "/info":
            info = "XAUUSD_DATA_ENGINE Telegram Bot\n"
            info += f"Versi: 1.0\n"
            info += f"Waktu server: {datetime.now(ZoneInfo('Asia/Jakarta'))}\n"
            info += "Pipeline: Normalizer -> CandleStatus -> FeatureEngine -> MultiTimeframe -> Dataset"
            self.sender.send_message(info)

        else:
            self.sender.send_message(f"Command tidak dikenal. Ketik /help untuk daftar command.")

    def process_updates(self):
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {
            "offset": self.offset,
            "timeout": 60,
            "allowed_updates": ["message"]
        }
        try:
            resp = requests.get(url, params=params, timeout=65)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                updates = data.get("result", [])
                for update in updates:
                    self.offset = update["update_id"] + 1
                    message = update.get("message")
                    if message and "text" in message:
                        chat_id = message["chat"]["id"]
                        text = message["text"]
                        self.handle_command(text, chat_id)
        except Exception as e:
            print(f"Error in getUpdates: {e}")
            time.sleep(2)

    def run(self):
        print("Bot started polling...")
        while self.running:
            self.process_updates()
            time.sleep(1)

    def stop(self):
        self.running = False
