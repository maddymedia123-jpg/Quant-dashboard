"""15m BTC candles from other venues for the same months as Gate's sample (research only).

Bybit and Bitget USDT perps, Binance USDT-M perp (live REST, to include October), Coinbase BTC-USD spot.
OKX and Kraken did not answer from this machine. From 1 Jun 2026 (a month of warm-up before Gate's
11 Jul start) to now. -> research/data/<venue>_15m.csv
"""
import datetime as dt
import time
from pathlib import Path

import httpx
import pandas as pd

OUT = Path(__file__).parent / "data"
START = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
BAR = 900_000
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def page(c, venue, end_ms):
    if venue == "bybit":
        r = c.get("https://api.bybit.com/v5/market/kline", params={"category": "linear", "symbol": "BTCUSDT",
                  "interval": "15", "end": end_ms, "limit": 1000}).json()["result"]["list"]
        return [[int(x[0]), *map(float, x[1:6])] for x in r]
    if venue == "bitget":
        r = c.get("https://api.bitget.com/api/v2/mix/market/history-candles", params={"symbol": "BTCUSDT",
                  "productType": "usdt-futures", "granularity": "15m", "endTime": end_ms, "limit": 200}).json()["data"]
        return [[int(x[0]), *map(float, x[1:6])] for x in r]
    if venue == "binance":
        r = c.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": "BTCUSDT", "interval": "15m",
                  "endTime": end_ms, "limit": 1500}).json()
        return [[int(x[0]), *map(float, x[1:6])] for x in r]
    if venue == "coinbase":                        # [time s, low, high, open, close, volume], 300 per call
        s = end_ms // 1000
        r = c.get("https://api.exchange.coinbase.com/products/BTC-USD/candles", params={"granularity": 900,
                  "start": dt.datetime.fromtimestamp(s - 299 * 900, dt.timezone.utc).isoformat(),
                  "end": dt.datetime.fromtimestamp(s, dt.timezone.utc).isoformat()}).json()
        return [[int(x[0]) * 1000, float(x[3]), float(x[2]), float(x[1]), float(x[4]), float(x[5])] for x in r]
    raise KeyError(venue)


def fetch(venue):
    rows, end = [], int(time.time() * 1000) // BAR * BAR
    with httpx.Client(timeout=30, headers={"User-Agent": "trap-intel-research/1.0"}) as c:
        while end > START:
            got = page(c, venue, end)
            if not got:
                break
            rows += got
            end = min(r[0] for r in got) - BAR
            time.sleep(0.25)
    df = pd.DataFrame(rows, columns=COLS).sort_values("timestamp").drop_duplicates("timestamp")
    df = df[df.timestamp >= START].reset_index(drop=True)
    df.to_csv(OUT / f"{venue}_15m.csv", index=False)
    gaps = int((df.timestamp.diff().dropna() != BAR).sum())
    print(f"{venue}: {len(df)} bars, {gaps} gaps")


if __name__ == "__main__":
    for v in ("bybit", "bitget", "binance", "coinbase"):
        try:
            fetch(v)
        except Exception as e:  # noqa: BLE001 - research: report and move on
            print(v, "failed:", type(e).__name__, e)
