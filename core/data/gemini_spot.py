"""Gemini exchange (US) public candles - spot fallback when Kraken is unreachable."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from core.data.types import SpotSnapshot

log = logging.getLogger(__name__)
BASE = "https://api.gemini.com"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
GEMINI_TF = {"15m": "15m", "1h": "1hr", "1d": "1day"}  # 4h and 1w are resampled from 1h / 1d
DAY_MS = 86_400_000


def parse_candles(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows], columns=COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    ms = minutes * 60_000
    ts = df["timestamp"].astype("int64")
    if minutes == 10080:  # epoch day 0 was a Thursday; +3 aligns buckets to Monday 00:00 UTC
        day = ts // DAY_MS
        bucket = ((day + 3) // 7 * 7 - 3) * DAY_MS
    else:
        bucket = (ts // ms) * ms
    g = df.assign(bucket=bucket.astype("int64")).groupby("bucket", sort=True)
    out = pd.DataFrame({
        "timestamp": g["open"].first().index.to_numpy(dtype="int64"),
        "open": g["open"].first().to_numpy(),
        "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(),
        "close": g["close"].last().to_numpy(),
        "volume": g["volume"].sum().to_numpy(),
    })
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.reset_index(drop=True)


async def _get(client: httpx.AsyncClient, path: str):
    r = await client.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()


async def fetch_gemini_spot(client: httpx.AsyncClient) -> SpotSnapshot:
    try:
        res = await asyncio.gather(*[_get(client, f"/v2/candles/btcusd/{g}") for g in GEMINI_TF.values()],
                                   _get(client, "/v1/pubticker/btcusd"), return_exceptions=True)
        frames: dict[str, pd.DataFrame] = {}
        for tf, data in zip(GEMINI_TF, res[:-1]):
            if isinstance(data, Exception):
                log.warning("gemini %s failed: %s", tf, data)
                continue
            frames[tf] = parse_candles(data)
        if "1h" in frames:
            frames["4h"] = resample(frames["1h"], 240)
        if "1d" in frames:
            frames["1w"] = resample(frames["1d"], 10080)
        if not frames:
            raise RuntimeError("no candles returned")
        tick = res[-1]
        last = float(tick["last"]) if not isinstance(tick, Exception) else float(frames[next(iter(frames))]["close"].iloc[-1])
        return SpotSnapshot(source="gemini", last=last, frames=frames)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("gemini spot unavailable: %s", e)
        return SpotSnapshot.unavailable("gemini", str(e))
