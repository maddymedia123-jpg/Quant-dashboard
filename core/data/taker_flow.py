"""Real taker order flow: who paid the spread, per candle, on the Gate.io BTC_USDT perpetual (keyless).

Gate.io's futures contract stats carry `long_taker_size` and `short_taker_size` for every period - the
volume of market orders that lifted the offer and that hit the bid. Their difference is the real taker
delta, and its running sum is cumulative volume delta (CVD). This replaces the candle close-position
proxy wherever it is available: the proxy guessed delta from where a candle closed; this measures it.

Why Gate.io: Binance's futures candles carry the same split, but Binance refuses US addresses and the
dashboard is hosted on US servers. Gate.io already serves this dashboard's funding and open interest from
there. Found by calling it:

* Sizes are in CONTRACTS. A BTC_USDT contract is `quanto_multiplier` bitcoin (0.0001), read from the
  contract itself, so a size is never shown as bitcoin without that conversion.
* 1,000 rows per interval are served: ten days of 15m, six weeks of 1h, five months of 4h, almost three
  years of 1d. Weekly and monthly flow are summed from the daily rows.
* The newest row is the period still in progress, exactly as with price candles."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

from core.data.types import TakerFlowSnapshot

log = logging.getLogger(__name__)

BASE = "https://api.gateio.ws/api/v4/futures/usdt"
CONTRACT = "BTC_USDT"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
FETCHED = ("15m", "1h", "4h", "1d")
ROWS = 300                    # per interval: enough for a 20-bar read and its history, small enough to poll
DEFAULT_MULTIPLIER = 0.0001   # used only if the contract endpoint fails; the value Gate.io publishes
MAX_BYTES = 2 * 1024 * 1024
COLUMNS = ["timestamp", "buy", "sell", "delta", "long_liq_usd", "short_liq_usd"]
DAY_MS, WEEK_MS = 86_400_000, 7 * 86_400_000
EPOCH_MONDAY_MS = 4 * DAY_MS


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_stats(rows, multiplier: float) -> pd.DataFrame:
    """One row per period, in BTC: taker buy, taker sell, their difference, and liquidations in USD.

    A row missing either taker size is dropped rather than read as zero - a zero would be a fake
    balanced period in the middle of the series."""
    if not isinstance(rows, list):
        raise ValueError(f"unexpected payload {type(rows).__name__}")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t, buy, sell = _num(r.get("time")), _num(r.get("long_taker_size")), _num(r.get("short_taker_size"))
        if t is None or buy is None or sell is None:
            continue
        out.append([int(t) * 1000, buy * multiplier, sell * multiplier, (buy - sell) * multiplier,
                    _num(r.get("long_liq_usd")) or 0.0, _num(r.get("short_liq_usd")) or 0.0])
    df = pd.DataFrame(out, columns=COLUMNS)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _sum_by(daily: pd.DataFrame, bucket: list[int]) -> pd.DataFrame:
    g = daily.assign(bucket=bucket).groupby("bucket", sort=True)
    out = g[["buy", "sell", "delta", "long_liq_usd", "short_liq_usd"]].sum().reset_index()
    out = out.rename(columns={"bucket": "timestamp"})
    out["timestamp"] = out["timestamp"].astype("int64")
    return out[COLUMNS]


def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monday-aligned weeks, the same weeks the price candles use."""
    return _sum_by(daily, [int(t) - (int(t) - EPOCH_MONDAY_MS) % WEEK_MS for t in daily["timestamp"]])


def monthly(daily: pd.DataFrame) -> pd.DataFrame:
    def start(ms: int) -> int:
        d = datetime.fromtimestamp(ms / 1000, timezone.utc)
        return int(datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return _sum_by(daily, [start(int(t)) for t in daily["timestamp"]])


def build(raw: dict, multiplier: float) -> TakerFlowSnapshot:
    """Assemble the snapshot from whatever arrived; one interval failing is a degraded feed, named."""
    frames, problems = {}, []
    for tf in FETCHED:
        rows = raw.get(tf)
        if isinstance(rows, BaseException) or rows is None:
            problems.append(f"{tf}: {type(rows).__name__}: {rows}" if rows is not None else f"{tf}: not fetched")
            continue
        try:
            df = parse_stats(rows, multiplier)
        except ValueError as e:
            problems.append(f"{tf}: {e}")
            continue
        if len(df):
            frames[tf] = df
    if "1d" in frames:
        frames["1w"], frames["1M"] = weekly(frames["1d"]), monthly(frames["1d"])
    if not frames:
        return TakerFlowSnapshot.unavailable("gate.io", "; ".join(problems) or "no taker rows")
    return TakerFlowSnapshot(source="gate.io", contract=CONTRACT, btc_per_contract=multiplier, frames=frames,
                             error="; ".join(problems) or None)


async def _get(client: httpx.AsyncClient, path: str, **params):
    r = await client.get(f"{BASE}{path}", params=params, headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    if len(r.content) > MAX_BYTES:
        raise ValueError(f"{len(r.content) / 1024:.0f} KB from {path}")
    return r.json()


async def fetch_taker_flow(client: httpx.AsyncClient) -> TakerFlowSnapshot:
    try:
        calls = [_get(client, "/contract_stats", contract=CONTRACT, interval=tf, limit=ROWS) for tf in FETCHED]
        calls.append(_get(client, f"/contracts/{CONTRACT}"))
        results = await asyncio.gather(*calls, return_exceptions=True)
        contract = results[-1]
        mult = _num(contract.get("quanto_multiplier")) if isinstance(contract, dict) else None
        if not mult:
            log.warning("Gate.io contract size unavailable (%s); using %s", contract, DEFAULT_MULTIPLIER)
            mult = DEFAULT_MULTIPLIER
        return build(dict(zip(FETCHED, results[:-1])), mult)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("Gate.io taker flow unavailable: %s", e)
        return TakerFlowSnapshot.unavailable("gate.io", f"{type(e).__name__}: {e}".strip(": "))
