"""Price OHLCV chain: the Gate.io BTC perpetual first (futures price action), then Kraken and Gemini spot.

The snapshot is still called "spot" in code; its source says which market answered."""
from __future__ import annotations

import httpx

from datetime import datetime, timezone

import pandas as pd

from core.data.gate_candles import fetch_gate_candles
from core.data.gemini_spot import fetch_gemini_spot
from core.data.kraken_spot import fetch_spot as fetch_kraken
from core.data.types import SpotSnapshot

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
HALF_WEEK_MS = WEEK_MS // 2        # 84 hours: Monday 00:00 to Thursday 12:00 UTC, then to Monday
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


def twelve_hour(hourly: pd.DataFrame) -> pd.DataFrame:
    """12H candles: hourly candles grouped into the UTC half-days opening at 00:00 and 12:00."""
    ts = hourly["timestamp"].astype("int64")
    return _aggregate(hourly, (ts // (12 * HOUR_MS)) * (12 * HOUR_MS))


def three_day(daily: pd.DataFrame) -> pd.DataFrame:
    """3D candles: daily candles in runs of three counted from the fixed epoch Monday, so the same days
    group together on every run whatever the first candle in the response."""
    ts = daily["timestamp"].astype("int64")
    day_no = (ts - EPOCH_MONDAY_MS) // DAY_MS
    return _aggregate(daily, (day_no // 3) * 3 * DAY_MS + EPOCH_MONDAY_MS)


def half_week(four_hour: pd.DataFrame) -> pd.DataFrame:
    """0.5W candles: 4h candles grouped into half-weeks, Monday 00:00 to Thursday 12:00 UTC and Thursday
    12:00 to Monday 00:00, so two of them make exactly the weekly candle."""
    ts = four_hour["timestamp"].astype("int64")
    return _aggregate(four_hour, ((ts - EPOCH_MONDAY_MS) // HALF_WEEK_MS) * HALF_WEEK_MS + EPOCH_MONDAY_MS)


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


def calendar_quarter(daily: pd.DataFrame) -> pd.DataFrame:
    """3M candles: daily candles grouped by UTC calendar quarter (Jan, Apr, Jul, Oct)."""
    def quarter_start(ms: int) -> int:
        d = datetime.fromtimestamp(ms / 1000, timezone.utc)
        return int(datetime(d.year, (d.month - 1) // 3 * 3 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    starts = [quarter_start(int(t)) for t in daily["timestamp"]]
    return _aggregate(daily, pd.Series(starts, index=daily.index, dtype="int64"))


def with_derived_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add the derived candles (12h, 3d, 0.5w, 2w, 1M, 3M) the desks and his cascade read. A frame that
    cannot be built is simply absent, which every consumer already treats as "not enough history"."""
    out = dict(frames)
    hourly, four_hour = frames.get("1h"), frames.get("4h")
    weekly, daily = frames.get("1w"), frames.get("1d")
    if hourly is not None and len(hourly) and "12h" not in out:
        out["12h"] = twelve_hour(hourly)
    if daily is not None and len(daily) and "3d" not in out:
        out["3d"] = three_day(daily)
    if four_hour is not None and len(four_hour) and "0.5w" not in out:
        out["0.5w"] = half_week(four_hour)
    if weekly is not None and len(weekly) and "2w" not in out:
        out["2w"] = two_week(weekly)
    if daily is not None and len(daily) and "1M" not in out:
        out["1M"] = calendar_month(daily)
    if daily is not None and len(daily) and "3M" not in out:
        out["3M"] = calendar_quarter(daily)
    return out


async def fetch_spot_chain(client: httpx.AsyncClient) -> SpotSnapshot:
    errors = []
    for fn in (fetch_gate_candles, fetch_kraken, fetch_gemini_spot):
        s = await fn(client)
        if s.available:
            return s.model_copy(update={"frames": with_derived_frames(s.frames)})
        errors.append(f"{s.source}: {s.error}")
    return SpotSnapshot.unavailable("gate,kraken,gemini", " | ".join(errors))
