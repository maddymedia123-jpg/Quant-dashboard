"""Qualified value-area breaks on the 1H, 4H and 1D, and the first pullback after one.

His rule, as relayed: a break only counts when three things line up on a COMPLETED candle -

1. Close outside value: the candle closes above the value area high (or below the value area low), and
   the close before it was still inside. The value area is the 70% volume profile of the candles BEFORE
   the break, so the breaking candle never defines the value it is breaking out of.
2. Volume expansion: the breaking candle trades at least 1.5x the median volume of the 20 candles before it.
3. VWAP acceptance: the close is on the break side of VWAP anchored at the open of the next timeframe up
   (1H against the day, 4H against the week, 1D against the month).

All three: a qualified break. Outside value without the other two: an unqualified poke - the kind that
traps. After a qualified break:

* a completed close back inside value fails it (the break was the trap);
* the FIRST candle to trade back to the broken edge (within a quarter ATR) and close on the break side is
  the first pullback - the entry the rule is built around. Later retests are not the first.

These rules apply to the 1H, 4H and 1D only. Nothing below an hour is read: there are no sub-hour
signals in this framework."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.indicators.cascade import anchored_vwap
from core.indicators.liquidity import volume_profile
from core.indicators.volatility import atr

TIMEFRAMES = ("1h", "4h", "1d")
VWAP_ANCHOR = {"1h": "1d", "4h": "1w", "1d": "1M"}
VOLUME_EXPANSION = 1.5
VOLUME_LOOKBACK = 20
PROFILE_LOOKBACK = 200
SEARCH_BARS = 12          # how many completed candles back a break is looked for
RETEST_ATR = 0.25         # how near the broken edge a pullback must trade
MIN_BARS = 40
DAY_MS = 86_400_000
EPOCH_MONDAY_MS = 4 * DAY_MS


@dataclass(frozen=True)
class BreakRead:
    timeframe: str
    available: bool
    state: str = "none"          # none, qualified, unqualified, failed, pullback, too_few
    direction: str = ""          # up, down
    edge: float | None = None    # the value-area edge that was broken
    vwap: float | None = None
    volume_ratio: float | None = None
    bars_ago: int | None = None  # of the break candle, 0 = the latest completed candle
    pullback_bars_ago: int | None = None
    checks: tuple[tuple[str, bool], ...] = ()
    note: str = ""

    @property
    def headline(self) -> str:
        side = "above value" if self.direction == "up" else "below value"
        return {"qualified": f"Qualified break {side}, awaiting the first pullback",
                "pullback": f"Qualified break {side}, first pullback held: the entry",
                "unqualified": f"Close {side} without the volume or VWAP behind it: trap risk",
                "failed": f"Break {side} failed back inside value: a {'bull' if self.direction == 'up' else 'bear'} trap",
                "none": "No break: price is auctioning inside value",
                "too_few": "Not enough candles"}.get(self.state, self.state)

    def to_state(self) -> dict:
        r = lambda v: None if v is None else round(v, 2)  # noqa: E731
        return {"available": self.available, "state": self.state, "read": self.headline,
                "direction": self.direction or None, "broken_edge": r(self.edge), "anchored_vwap": r(self.vwap),
                "volume_vs_median": r(self.volume_ratio), "break_bars_ago": self.bars_ago,
                "first_pullback_bars_ago": self.pullback_bars_ago,
                "checks": {k: v for k, v in self.checks}, "note": self.note}


def period_start(ms: int, tf: str) -> int:
    """Opening time of the `tf` candle (1d, 1w Monday, 1M) that contains `ms`, in UTC."""
    if tf == "1d":
        return ms - ms % DAY_MS
    if tf == "1w":
        return ms - (ms - EPOCH_MONDAY_MS) % (7 * DAY_MS)
    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return int(datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def read(df: pd.DataFrame | None, tf: str) -> BreakRead:
    """The latest break on this timeframe's completed candles. The last row is treated as forming."""
    if tf not in TIMEFRAMES:
        return BreakRead(tf, False, note=f"qualified breaks are read on {', '.join(TIMEFRAMES)} only")
    if df is None or len(df) < MIN_BARS + 1:
        return BreakRead(tf, False, state="too_few", note=f"fewer than {MIN_BARS} completed {tf} candles")
    closed = df.iloc[:-1].reset_index(drop=True)
    n = len(closed)
    close = closed["close"].astype(float).to_numpy()
    vol = closed["volume"].astype(float).to_numpy()
    a = atr(closed)
    atr_v = float(a.iloc[-1]) if len(a) and not np.isnan(a.iloc[-1]) else 0.0

    # the most recent candle that closed outside the value built before it, with the close before inside
    for i in range(n - 1, max(MIN_BARS, n - SEARCH_BARS) - 1, -1):
        vp = volume_profile(closed.iloc[:i], lookback=PROFILE_LOOKBACK)
        if not vp.available:
            continue
        hi, lo = vp.value_area_high, vp.value_area_low
        prev, c = close[i - 1], close[i]
        if lo <= prev <= hi and (c > hi or c < lo):
            break
    else:
        return BreakRead(tf, True, state="none")

    up = c > hi
    edge = hi if up else lo
    ts = int(closed["timestamp"].iloc[i])
    vwap = anchored_vwap(closed.iloc[: i + 1], period_start(ts, VWAP_ANCHOR[tf]))
    base = vol[max(0, i - VOLUME_LOOKBACK):i]
    median = float(np.median(base)) if len(base) else 0.0
    ratio = float(vol[i] / median) if median > 0 else None
    checks = (("close_outside_value", True),
              ("volume_expansion", ratio is not None and ratio >= VOLUME_EXPANSION),
              ("vwap_acceptance", vwap is not None and (c > vwap if up else c < vwap)))
    common = dict(direction="up" if up else "down", edge=edge, vwap=vwap, volume_ratio=ratio,
                  bars_ago=n - 1 - i, checks=checks)
    if not all(ok for _, ok in checks):
        return BreakRead(tf, True, state="unqualified", **common)

    # after the break: back inside fails it; the first candle to retest the edge and hold is the pullback
    reach = RETEST_ATR * atr_v
    for j in range(i + 1, n):
        cj = close[j]
        if (up and cj <= edge) or (not up and cj >= edge):
            return BreakRead(tf, True, state="failed", **common)
        touched = (float(closed["low"].iloc[j]) <= edge + reach) if up else (float(closed["high"].iloc[j]) >= edge - reach)
        if touched:
            return BreakRead(tf, True, state="pullback", pullback_bars_ago=n - 1 - j, **common)
    return BreakRead(tf, True, state="qualified", **common)


def read_all(frames: dict[str, pd.DataFrame], timeframes) -> dict[str, BreakRead]:
    return {tf: read(frames.get(tf), tf) for tf in timeframes if tf in TIMEFRAMES}
