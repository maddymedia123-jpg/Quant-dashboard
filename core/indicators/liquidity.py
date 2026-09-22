"""Liquidity and order flow from candles.

The rubric asks whether a pool of equal highs or lows was swept and rejected, whether cumulative delta
confirms price or shows absorption, where the volume profile's point of control and value area sit,
and how open interest moved on each timeframe. All four are arithmetic, so they are computed here and
handed to the judges as facts.

One honesty note carried through to the UI: without tick data there is no true CVD. What is computed
here is the standard candle proxy - volume split by where the candle closed in its range - and it is
labelled as a proxy everywhere it is shown, so nobody reads it as tape."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

MIN_TOUCHES = 2            # a pool needs at least two attempts at the same level
CONFIRM_BARS = 2           # and price must have left it: the last bars are where price is, not a pool
DEFAULT_TOLERANCE_PCT = 0.1
VALUE_AREA_SHARE = 0.70
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}
DEFAULT_WINDOWS = ("15m", "1h", "4h")


@dataclass
class Pool:
    side: Literal["buyside", "sellside"]     # buyside sits above price, sellside below
    price: float
    touches: int
    last_index: int
    swept: bool = False
    sweep_index: int | None = None
    reclaimed: bool = False                  # traded through, then closed back inside: a raid
    broken: bool = False                     # closed beyond it: a break, the opposite conclusion
    bars: int = 0

    @property
    def sweep_bars_ago(self) -> int | None:
        return None if self.sweep_index is None else self.bars - 1 - self.sweep_index


@dataclass
class DeltaRead:
    available: bool = False
    last: float = 0.0
    delta_change: float = 0.0
    price_change: float = 0.0
    state: str = "unavailable"               # confirming, bearish absorption, bullish absorption, flat
    method: str = "candle close-position proxy (no tick data)"
    series: list[float] = field(default_factory=list)


@dataclass
class VolumeProfile:
    available: bool = False
    poc: float | None = None
    value_area_high: float | None = None
    value_area_low: float | None = None
    value_area_share: float = 0.0
    position: str = "unknown"                # above value, inside value, below value


def liquidity_pools(df: pd.DataFrame, tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
                    lookback: int = 200, confirm_bars: int = CONFIRM_BARS) -> list[Pool]:
    """Clusters of equal highs and lows, and whether price raided one and came back.

    A sweep is a wick through the level that closes back inside. A close beyond it is a break, which
    is the opposite conclusion, so the two are kept apart."""
    window = df.tail(lookback).reset_index(drop=True)
    n = len(window)
    if n < 3:
        return []
    highs, lows = window["high"].to_numpy(), window["low"].to_numpy()
    closes = window["close"].to_numpy()
    price = float(closes[-1])
    pools: list[Pool] = []

    for side, values in (("buyside", highs), ("sellside", lows)):
        order = sorted(range(n), key=lambda i: values[i], reverse=(side == "buyside"))
        used: set[int] = set()
        for i in order:
            if i in used:
                continue
            level = float(values[i])
            tol = abs(level) * tolerance_pct / 100.0
            members = [j for j in range(n) if j not in used and abs(values[j] - level) <= tol]
            if len(members) < MIN_TOUCHES:
                continue
            used.update(members)
            last = max(members)
            level_price = float(np.mean([values[j] for j in members]))
            # Buyside liquidity sits above price and sellside below it, judged when the cluster last
            # formed - so a pool that has since been taken stays in the list and is marked broken,
            # rather than vanishing and taking the information with it.
            if n - 1 - last < confirm_bars:
                continue       # still trading there; nothing has been left behind to hunt
            ref = float(closes[last])
            if (level_price <= ref) if side == "buyside" else (level_price >= ref):
                continue
            pool = Pool(side=side, price=level_price, touches=len(members), last_index=last, bars=n)

            for j in range(last + 1, n):
                through = highs[j] > level_price + tol if side == "buyside" else lows[j] < level_price - tol
                if not through:
                    continue
                closed_back = closes[j] < level_price if side == "buyside" else closes[j] > level_price
                if closed_back:
                    pool.swept, pool.sweep_index, pool.reclaimed = True, j, True
                else:
                    pool.broken = True
                break      # the first interaction decides: raid or break
            pools.append(pool)
    return pools


def cumulative_delta(df: pd.DataFrame, lookback: int = 20) -> DeltaRead:
    """Volume split by where each candle closed in its range, accumulated.

    Price up while delta falls is sellers absorbing the move, and the mirror for a fall."""
    if df.empty or "volume" not in df:
        return DeltaRead()
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    vol = df["volume"].to_numpy(dtype=float)
    span = np.where(high - low == 0, np.nan, high - low)
    ratio = np.divide(close - low, span, out=np.full_like(vol, 0.5, dtype=float), where=~np.isnan(span))
    delta = vol * (2.0 * ratio - 1.0)
    cvd = np.cumsum(delta)

    look = min(lookback, len(df) - 1)
    if look <= 0:
        return DeltaRead(available=True, last=float(cvd[-1]), series=[float(x) for x in cvd], state="flat")
    d_change = float(cvd[-1] - cvd[-1 - look])
    p_change = float(close[-1] - close[-1 - look])
    scale = float(np.mean(np.abs(delta[-look:]))) or 1.0
    if abs(d_change) < scale * 0.5 or p_change == 0:
        state = "flat"
    elif (p_change > 0) == (d_change > 0):
        state = "confirming"
    else:
        state = "bearish absorption" if p_change > 0 else "bullish absorption"
    return DeltaRead(available=True, last=float(cvd[-1]), delta_change=d_change, price_change=p_change,
                     state=state, series=[float(x) for x in cvd])


def volume_profile(df: pd.DataFrame, bins: int = 48, lookback: int = 200) -> VolumeProfile:
    """Point of control and the value area holding 70% of traded volume."""
    window = df.tail(lookback)
    if window.empty or float(window["volume"].sum()) <= 0:
        return VolumeProfile()
    lo, hi = float(window["low"].min()), float(window["high"].max())
    if hi <= lo:
        return VolumeProfile()

    edges = np.linspace(lo, hi, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    totals = np.zeros(bins)
    for h, l, v in zip(window["high"].to_numpy(), window["low"].to_numpy(), window["volume"].to_numpy()):
        first, last = np.searchsorted(edges, l, "right") - 1, np.searchsorted(edges, h, "left")
        first, last = max(0, min(bins - 1, first)), max(0, min(bins - 1, last - 1 if last > 0 else 0))
        share = v / (last - first + 1)
        totals[first:last + 1] += share

    total = totals.sum()
    poc_i = int(np.argmax(totals))
    lo_i = hi_i = poc_i
    covered = totals[poc_i]
    while covered < VALUE_AREA_SHARE * total and (lo_i > 0 or hi_i < bins - 1):
        below = totals[lo_i - 1] if lo_i > 0 else -1.0
        above = totals[hi_i + 1] if hi_i < bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            covered += above
        else:
            lo_i -= 1
            covered += below

    va_low, va_high = float(edges[lo_i]), float(edges[hi_i + 1])
    price = float(window["close"].iloc[-1])
    position = "inside value" if va_low <= price <= va_high else "above value" if price > va_high else "below value"
    return VolumeProfile(True, float(centres[poc_i]), va_high, va_low, float(covered / total), position)


def oi_changes(history: list[tuple[int, float]], now_ms: int,
               windows: tuple[str, ...] = DEFAULT_WINDOWS) -> dict[str, float | None]:
    """Percent change in open interest over each Live Recon window.

    A window with no sample near its start returns None: unknown is not the same as unchanged, and a
    reading from the wrong window is not an approximation of the right one."""
    out: dict[str, float | None] = {}
    latest = history[-1][1] if history else None
    for name in windows:
        span = TF_MS[name]
        if not history or not latest:
            out[name] = None
            continue
        # The sample used must actually sit near the start of the window. Sampling is sparse, and
        # taking the newest reading older than the cutoff would report a three-hour move as a
        # fifteen-minute one - a real number against the wrong label, which is worse than no number.
        lo, hi = span / 2, span * 2
        candidates = [(ts, v) for ts, v in history[:-1] if v and lo <= now_ms - ts <= hi]
        base = min(candidates, key=lambda tv: abs((now_ms - tv[0]) - span))[1] if candidates else None
        out[name] = None if not base else (latest / base - 1.0) * 100.0
    return out


def summarise(df: pd.DataFrame, oi_history: list[tuple[int, float]], now_ms: int,
              tolerance_pct: float = DEFAULT_TOLERANCE_PCT, windows: tuple[str, ...] = DEFAULT_WINDOWS) -> dict:
    """The compact, JSON-safe reading handed to the rubric judges."""
    price = float(df["close"].iloc[-1]) if len(df) else 0.0
    pools = liquidity_pools(df, tolerance_pct)
    cvd = cumulative_delta(df)
    vp = volume_profile(df)

    def pool_out(side: str) -> dict | None:
        side_pools = [p for p in pools if p.side == side and not p.broken] or                      [p for p in pools if p.side == side]
        if not side_pools:
            return None
        p = min(side_pools, key=lambda p: abs(p.price - price))
        return {"price": round(p.price, 2), "touches": p.touches,
                "distance_pct": round((p.price - price) / price * 100.0, 3) if price else None,
                "swept": p.swept, "reclaimed": p.reclaimed, "broken": p.broken,
                "sweep_bars_ago": p.sweep_bars_ago}

    return {
        "pools": {"buyside": pool_out("buyside"), "sellside": pool_out("sellside"),
                  "recent_sweep": next(({"side": p.side, "price": round(p.price, 2),
                                         "bars_ago": p.sweep_bars_ago}
                                        for p in sorted(pools, key=lambda p: -(p.sweep_index or -1))
                                        if p.swept), None)},
        "cumulative_delta": {"available": cvd.available, "state": cvd.state, "method": cvd.method,
                             "change_over_20_bars": round(cvd.delta_change, 2),
                             "price_change_over_20_bars": round(cvd.price_change, 2)},
        "volume_profile": ({"available": False} if not vp.available else
                           {"available": True, "point_of_control": round(vp.poc, 2),
                            "value_area_high": round(vp.value_area_high, 2),
                            "value_area_low": round(vp.value_area_low, 2),
                            "price_position": vp.position}),
        "open_interest": {k: (None if v is None else round(v, 3))
                          for k, v in oi_changes(oi_history, now_ms, windows).items()},
    }
