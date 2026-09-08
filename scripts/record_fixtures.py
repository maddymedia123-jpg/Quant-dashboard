"""Record live JSON responses into tests/fixtures so tests never touch the network.
Run once: .venv/Scripts/python scripts/record_fixtures.py
"""
from __future__ import annotations

import json
import pathlib

import httpx

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINTS = {
    "kraken_ohlc_15m.json": "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=15",
    "kraken_ohlc_1h.json": "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60",
    "kraken_ticker.json": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "binance_funding.json": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=21",
    "binance_oi.json": "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
    "binance_oi_hist.json": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=25",
    "binance_ls.json": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=1",
    "binance_taker.json": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=1h&limit=1",
    "bybit_tickers.json": "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
    "bybit_funding.json": "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=21",
    "bybit_oi.json": "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h&limit=25",
    "bybit_ratio.json": "https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=BTCUSDT&period=1h&limit=1",
    "deribit_book_summary.json": "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
    "fng.json": "https://api.alternative.me/fng/?limit=2",
}

with httpx.Client(timeout=20, headers={"User-Agent": "trap-intel-fixtures/1.0"}) as client:
    for name, url in ENDPOINTS.items():
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
        # trim very large payloads so the repo stays small
        if name.startswith("kraken_ohlc"):
            key = next(k for k in data["result"] if k != "last")
            data["result"][key] = data["result"][key][-400:]
        (OUT / name).write_text(json.dumps(data), encoding="utf-8")
        print(f"wrote {name} ({(OUT / name).stat().st_size} B)")
