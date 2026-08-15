import time
import pandas as pd
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from telegram.sender import TelegramSender
from telegram.cache import DataCache
from telegram.updater import update_all_cache

SESSION_TIMES = {
    "Tokyo": (dt_time(7, 0), dt_time(16, 0)),
    "London": (dt_time(15, 0), dt_time(0, 0)),
    "NewYork": (dt_time(20, 0), dt_time(5, 0)),
}

last_sent = {
    "Tokyo": None,
    "London": None,
    "NewYork": None
}

def send_scheduled_data():
    print("Updating data cache...")
    try:
        update_all_cache()
    except Exception as e:
        print(f"Error updating cache: {e}")
        sender = TelegramSender()
        sender.send_message(f"Error updating data cache: {e}")
        return

    sender = TelegramSender()
    cache = DataCache()

    raw_dict = cache.get_raw()
    for tf, df in raw_dict.items():
        if df is not None and not df.empty:
            sender.send_file(df, filename=f"raw_{tf}.csv", caption=f"Data mentah {tf} (scheduled)")

    features_dict = cache.get_features()
    for tf, df in features_dict.items():
        if df is not None and not df.empty:
            sender.send_file(df, filename=f"features_{tf}.csv", caption=f"Feature Engineering {tf} (scheduled)")

    ds = cache.get_dataset()
    if ds:
        X = ds.get("X")
        y = ds.get("y")
        feature_names = ds.get("feature_names")
        if X is not None and y is not None:
            df_dataset = pd.DataFrame(X, columns=feature_names)
            df_dataset["label"] = y
            sender.send_file(df_dataset, filename="dataset.csv", caption="Dataset ML (scheduled)")

    sender.send_message("Data terjadwal telah dikirim.")

def is_between_interval(current, start, end):
    if start < end:
        return start <= current < end
    else:
        return current >= start or current < end

def should_send(session_name):
    now = datetime.now(ZoneInfo('Asia/Jakarta'))
    current_time = now.time()
    start, end = SESSION_TIMES[session_name]
    if not is_between_interval(current_time, start, end):
        return False
    if last_sent[session_name] == now.date():
        return False
    return True

def run_scheduler():
    print("Scheduler started. Checking every 60 seconds...")
    while True:
        now = datetime.now(ZoneInfo('Asia/Jakarta'))
        for session in SESSION_TIMES:
            if should_send(session):
                print(f"Triggering scheduled send for {session} at {now}")
                try:
                    send_scheduled_data()
                    last_sent[session] = now.date()
                except Exception as e:
                    print(f"Error sending scheduled data: {e}")
        time.sleep(60)
