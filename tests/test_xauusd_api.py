import os

import pytest
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

URL = "https://api.twelvedata.com/time_series"


@pytest.mark.integration
def test_xauusd_api():

    if not API_KEY:
        pytest.skip(
            "TWELVE_DATA_API_KEY tidak ditemukan di environment"
        )

    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": 5,
        "apikey": API_KEY,
        "format": "JSON",
    }

    try:
        response = requests.get(
            URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        pytest.skip(
            f"XAUUSD API tidak dapat diakses: {exc}"
        )

    try:
        data = response.json()

    except ValueError as exc:

        pytest.fail(
            f"Response API bukan JSON valid: {exc}"
        )

    assert isinstance(
        data,
        dict
    ), "Response API harus berupa object JSON"

    if data.get("status") == "error":

        pytest.fail(
            f"API ERROR: {data}"
        )

    meta = data.get("meta", {})
    values = data.get("values", [])

    assert meta, "API tidak mengembalikan meta"
    assert values, "API tidak mengembalikan candle"

    assert meta.get("symbol") == "XAU/USD", (
        f"Symbol tidak sesuai: {meta.get('symbol')}"
    )

    assert meta.get("interval") == "1h", (
        f"Interval tidak sesuai: {meta.get('interval')}"
    )

    assert len(values) > 0, (
        "Tidak ada candle XAU/USD"
    )

    print()
    print("HTTP STATUS:", response.status_code)
    print("API STATUS: OK")
    print("SYMBOL:", meta.get("symbol"))
    print("INTERVAL:", meta.get("interval"))
    print("CANDLE COUNT:", len(values))

    print()
    print("LATEST CANDLES:")

    for candle in values:
        print(candle)
