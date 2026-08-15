import os
import requests
import pandas as pd
from io import StringIO

class TelegramSender:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    def send_message(self, text):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def send_file(self, df, filename="data.csv", caption=None, format="csv"):
        if format == "csv":
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            files = {
                "document": (filename, csv_buffer.getvalue(), "text/csv")
            }
        elif format == "json":
            json_str = df.to_json(orient="records", date_format="iso")
            files = {
                "document": (filename, json_str, "application/json")
            }
        else:
            raise ValueError("format must be 'csv' or 'json'")

        url = f"https://api.telegram.org/bot{self.token}/sendDocument"
        data = {
            "chat_id": self.chat_id,
            "caption": caption or "",
        }
        resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
