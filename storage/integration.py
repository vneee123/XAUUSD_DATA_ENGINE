from storage.database import Database

def save_cache_to_db(cache):
    db = Database()
    try:
        for tf, df in cache["raw"].items():
            if df is not None and not df.empty:
                db.insert_raw(df, tf)
        for tf, df in cache["features"].items():
            if df is not None and not df.empty:
                db.insert_features(df, tf)
        print("Data saved to database successfully")
    except Exception as e:
        print(f"Database save error: {e}")
