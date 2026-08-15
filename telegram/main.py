import threading
import time
from telegram.bot import TelegramBot
from telegram.scheduler import run_scheduler

def run_bot():
    bot = TelegramBot()
    bot.run()

def run_scheduler_thread():
    run_scheduler()

def main():
    print("Starting Telegram Bot and Scheduler...")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    scheduler_thread = threading.Thread(target=run_scheduler_thread, daemon=True)
    bot_thread.start()
    scheduler_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
