"""Spot OHLCV chain: Kraken first, Gemini exchange second."""
from __future__ import annotations

import httpx

from datetime import datetime, timezone

import pandas as pd

from core.data.gemini_spot import fetch_gemini_spot
from core.data.kraken_spot import fetch_spot as fetch_kraken
from core.data.types import SpotSnapshot

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
# Monday 1970-01-05 00:00 UTC: weeks are counted from here, so two-week candles pair the same weeks
# on every run instead of shifting with whatever the first candle in the response happened to be.
EPOCH_MONDAY_MS = 4 * DAY_MS


def _aggregate(df: pd.DataFrame, bucket: pd.Series) -> pd.DataFrame:
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


def two_week(weekly: pd.DataFrame) -> pd.DataFrame:
    """2W candles: consecutive pairs of Monday-aligned weeks, paired from a fixed epoch Monday."""
    ts = weekly["timestamp"].astype("int64")
    week_no = (ts - EPOCH_MONDAY_MS) // WEEK_MS
    return _aggregate(weekly, (week_no // 2) * 2 * WEEK_MS + EPOCH_MONDAY_MS)


def calendar_month(daily: pd.DataFrame) -> pd.DataFrame:
    """1M candles: every daily candle grouped by the UTC calendar month it opened in."""
    # plain datetime arithmetic, not pd.to_datetime: the pandas build this runs on crashed the process
    # (an access violation, not an exception) converting epoch milliseconds
    def month_start(ms: int) -> int:
        d = datetime.fromtimestamp(ms / 1000, timezone.utc)
        return int(datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp() * 1000)

    starts = [month_start(int(t)) for t in daily["timestamp"]]
    return _aggregate(daily, pd.Series(starts, index=daily.index, dtype="int64"))


def with_derived_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add the 2W and 1M candles the Weekly and Monthly desks read. A frame that cannot be built is
    simply absent, which every consumer already treats as "not enough history"."""
    out = dict(frames)
    weekly, daily = frames.get("1w"), frames.get("1d")
    if weekly is not None and len(weekly) and "2w" not in out:
        out["2w"] = two_week(weekly)
    if daily is not None and len(daily) and "1M" not in out:
        out["1M"] = calendar_month(daily)
    return out


async def fetch_spot_chain(client: httpx.AsyncClient) -> SpotSnapshot:
    errors = []
    for fn in (fetch_kraken, fetch_gemini_spot):
        s = await fn(client)
        if s.available:
            return s.model_copy(update={"frames": with_derived_frames(s.frames)})
        errors.append(f"{s.source}: {s.error}")
    return SpotSnapshot.unavailable("kraken,gemini", " | ".join(errors))
