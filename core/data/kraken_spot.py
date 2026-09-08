"""Kraken public REST: OHLC for all category timeframes + last price."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from core.config import SYMBOLS, TF_MINUTES, TIMEFRAMES
from core.data.types import SpotSnapshot

log = logging.getLogger(__name__)
BASE = "https://api.kraken.com/0/public"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _result_key(payload: dict) -> str:
    return next(k for k in payload["result"] if k != "last")


def parse_ohlc(payload: dict) -> pd.DataFrame:
    if payload.get("error"):
        raise ValueError(f"kraken error: {payload['error']}")
    rows = payload["result"][_result_key(payload)]
    # kraken row: [time(s), open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        [[int(r[0]) * 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows],
        columns=COLS,
    )
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").reset_index(drop=True)


def parse_ticker(payload: dict) -> float:
    if payload.get("error"):
        raise ValueError(f"kraken error: {payload['error']}")
    return float(payload["result"][_result_key(payload)]["c"][0])


async def _get(client: httpx.AsyncClient, path: str, **params) -> dict:
    r = await client.get(f"{BASE}/{path}", params=params)
    r.raise_for_status()
    return r.json()


async def fetch_spot(client: httpx.AsyncClient) -> SpotSnapshot:
    pair = SYMBOLS["kraken_pair"]
    try:
        tasks = [_get(client, "OHLC", pair=pair, interval=TF_MINUTES[tf]) for tf in TIMEFRAMES]
        tasks.append(_get(client, "Ticker", pair=pair))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        frames: dict[str, pd.DataFrame] = {}
        for tf, res in zip(TIMEFRAMES, results[:-1]):
            if isinstance(res, Exception):
                log.warning("kraken %s failed: %s", tf, res)
                continue
            frames[tf] = parse_ohlc(res)
        if not frames:
            raise RuntimeError("no OHLC frames returned")
        ticker = results[-1]
        last = parse_ticker(ticker) if not isinstance(ticker, Exception) else float(frames[next(iter(frames))]["close"].iloc[-1])
        return SpotSnapshot(source="kraken", last=last, frames=frames)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("kraken spot unavailable: %s", e)
        return SpotSnapshot.unavailable("kraken", str(e))
