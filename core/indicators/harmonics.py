"""Harmonic patterns - Gartley, Bat, Butterfly, Crab, Cypher - and his rule for when one counts.

Swings come from an ATR zigzag on COMPLETED candles: a pivot is confirmed only once price has moved
ZIGZAG_ATR x ATR away from it, so noise does not make pivots. Five alternating pivots X-A-B-C-D are
checked against each pattern's Fibonacci ratios (bullish when X is a low, bearish when X is a high):

    pattern     AB of XA        BC of AB      CD of BC       D (of XA, or of XC for the Cypher)
    Gartley     0.618           0.382-0.886   1.13-1.618     0.786 retracement of XA
    Bat         0.382-0.50      0.382-0.886   1.618-2.618    0.886 retracement of XA
    Butterfly   0.786           0.382-0.886   1.618-2.24     1.27-1.618 extension of XA
    Crab        0.382-0.618     0.382-0.886   2.24-3.618     1.618 extension of XA
    Cypher      0.382-0.618     C at 1.272-1.414 of XA, beyond A   0.786 retracement of XC

Single values get a +-RATIO_TOL band. The potential reversal zone (PRZ) is where the D-leg rule and the
CD rule overlap; a pattern whose two rules do not overlap cannot complete and is not reported.

Stages:
* forming   - X, A, B, C confirmed; the PRZ is projected and price has not reached it.
* in_zone   - price has traded into the PRZ on the current leg. This is where a harmonic is acted on.
* completed - D is a confirmed pivot inside the PRZ: the reversal has started.

His rule: a harmonic is only valid next to a VWAP, value-area or gamma level. The PRZ is checked against
the anchored VWAP, the point of control and value-area edges, and the dealer gamma levels (zero gamma,
walls, max gamma). With one inside the PRZ, or within half an ATR of it, the pattern is "valid"; without,
it is reported as unconfirmed - a shape in the price, not a trade.

Targets are the standard 0.382 and 0.618 retracements of the A-D leg. A close beyond the far edge of the
PRZ by half an ATR invalidates the pattern."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.indicators.volatility import atr as atr_series

TIMEFRAMES = ("1h", "4h", "1d")
TRACK_RECORD = ("Replayed on Binance BTC (1h 2025-26, 4h 2024-26, 1d 2018-26): 47 setups reached their zone. "
                "Of those next to a VWAP or value-area level, 43% hit the first target before invalidation; of "
                "those without, 65%. Too few to trust either way, and gamma levels have no history to test.")
ZIGZAG_ATR = 2.0
RATIO_TOL = 0.03
NEAR_ATR = 0.5
MIN_BARS = 60
LOOKBACK = 300

# name: (AB of XA, BC of AB, CD of BC, D rule) with ranges as (lo, hi)
SPECS = {
    "Gartley": ((0.618, 0.618), (0.382, 0.886), (1.13, 1.618), (0.786, 0.786)),
    "Bat": ((0.382, 0.50), (0.382, 0.886), (1.618, 2.618), (0.886, 0.886)),
    "Butterfly": ((0.786, 0.786), (0.382, 0.886), (1.618, 2.24), (1.27, 1.618)),
    "Crab": ((0.382, 0.618), (0.382, 0.886), (2.24, 3.618), (1.618, 1.618)),
}
CYPHER = {"AB": (0.382, 0.618), "XC": (1.272, 1.414), "D_of_XC": (0.786, 0.786)}


def _band(lo: float, hi: float) -> tuple[float, float]:
    return lo - RATIO_TOL, hi + RATIO_TOL


def _within(v: float, rng: tuple[float, float]) -> bool:
    lo, hi = _band(*rng)
    return lo <= v <= hi


@dataclass(frozen=True)
class Pivot:
    index: int
    price: float
    kind: str           # high | low


@dataclass(frozen=True)
class Harmonic:
    name: str
    side: str                       # bullish | bearish
    stage: str                      # forming | in_zone | completed
    points: tuple[Pivot, ...]       # X A B C (D when completed)
    prz: tuple[float, float]
    targets: tuple[float, float]
    invalidation: float
    confluence: tuple[str, ...] = ()
    distance_pct: float | None = None   # price to the nearest PRZ edge, for a forming pattern

    @property
    def valid(self) -> bool:
        return bool(self.confluence)

    @property
    def headline(self) -> str:
        where = {"forming": "forming", "in_zone": "price is in its reversal zone",
                 "completed": "completed, reversal under way"}[self.stage]
        verdict = ("valid: " + ", ".join(self.confluence) + " in the zone" if self.valid else
                   "unconfirmed: no VWAP, value-area or gamma level at the zone")
        return f"{self.side.title()} {self.name}, {where}. {verdict[0].upper() + verdict[1:]}."

    def to_state(self) -> dict:
        r = lambda v: round(float(v), 2)  # noqa: E731
        return {"pattern": self.name, "side": self.side, "stage": self.stage, "valid": self.valid,
                "read": self.headline, "prz": [r(self.prz[0]), r(self.prz[1])],
                "targets": [r(t) for t in self.targets], "invalidation": r(self.invalidation),
                "confluence": list(self.confluence),
                "points": {k: r(p.price) for k, p in zip("XABCD", self.points)},
                "distance_pct": None if self.distance_pct is None else round(self.distance_pct, 2)}


def zigzag(df: pd.DataFrame, atr_value: float, mult: float = ZIGZAG_ATR) -> tuple[list[Pivot], Pivot | None]:
    """Confirmed alternating pivots, and the current leg's running extreme (not yet confirmed)."""
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    th = mult * atr_value
    pivots: list[Pivot] = []
    trend = None                     # "up" while hunting a high, "down" while hunting a low
    ext_i, ext_hi, ext_lo = 0, hi[0], lo[0]
    lo_i, hi_i = 0, 0
    for i in range(1, len(df)):
        if trend is None:
            if hi[i] > ext_hi:
                ext_hi, hi_i = hi[i], i
            if lo[i] < ext_lo:
                ext_lo, lo_i = lo[i], i
            if ext_hi - lo[i] >= th and hi_i < i:
                pivots.append(Pivot(hi_i, ext_hi, "high"))
                trend, ext_i, ext_lo = "down", i, lo[i]
            elif hi[i] - ext_lo >= th and lo_i < i:
                pivots.append(Pivot(lo_i, ext_lo, "low"))
                trend, ext_i, ext_hi = "up", i, hi[i]
            continue
        if trend == "up":
            if hi[i] >= ext_hi:
                ext_hi, ext_i = hi[i], i
            elif ext_hi - lo[i] >= th:
                pivots.append(Pivot(ext_i, ext_hi, "high"))
                trend, ext_i, ext_lo = "down", i, lo[i]
        else:
            if lo[i] <= ext_lo:
                ext_lo, ext_i = lo[i], i
            elif hi[i] - ext_lo >= th:
                pivots.append(Pivot(ext_i, ext_lo, "low"))
                trend, ext_i, ext_hi = "up", i, hi[i]
    current = (None if trend is None else
               Pivot(ext_i, ext_hi, "high") if trend == "up" else Pivot(ext_i, ext_lo, "low"))
    return pivots, current


def _prz(name: str, x: float, a: float, b: float, c: float, bull: bool) -> tuple[float, float] | None:
    """The zone where D must land, or None when X-A-B-C does not fit the pattern or its rules disagree."""
    s = 1.0 if bull else -1.0                          # bullish: X low, A high
    xa = (a - x) * s
    ab = (a - b) * s
    if xa <= 0 or ab <= 0:
        return None
    if name == "Cypher":
        if not _within(ab / xa, CYPHER["AB"]):
            return None
        xc = (c - x) * s
        if not _within(xc / xa, CYPHER["XC"]):
            return None
        lo, hi = _band(*CYPHER["D_of_XC"])
        zone = sorted((c - s * lo * xc, c - s * hi * xc))
        return zone[0], zone[1]
    ab_r, bc_r, cd_r, d_r = SPECS[name]
    bc = (c - b) * s
    if not (_within(ab / xa, ab_r) and bc > 0 and _within(bc / ab, bc_r)):
        return None
    d_lo, d_hi = _band(*d_r)
    by_xa = sorted((a - s * d_lo * xa, a - s * d_hi * xa))
    c_lo, c_hi = cd_r                                    # CD ranges are already wide; no extra band
    by_cd = sorted((c - s * c_lo * bc, c - s * c_hi * bc))
    lo, hi = max(by_xa[0], by_cd[0]), min(by_xa[1], by_cd[1])
    return (lo, hi) if lo <= hi else None


def levels_for(df: pd.DataFrame, tf: str, gamma: dict | None) -> list[tuple[str, float]]:
    """The levels a harmonic must sit next to: anchored VWAP, value area, dealer gamma."""
    from core.indicators.cascade import anchored_vwap
    from core.indicators.liquidity import volume_profile
    from core.indicators.vwap_break import VWAP_ANCHOR, period_start
    out: list[tuple[str, float]] = []
    closed = df.iloc[:-1]
    if len(closed):
        last = int(closed["timestamp"].iloc[-1])
        v = anchored_vwap(closed, period_start(last, VWAP_ANCHOR.get(tf, "1d")))
        if v:
            out.append((f"{VWAP_ANCHOR.get(tf, '1d')} VWAP", v))
    vp = volume_profile(closed)
    if vp.available:
        out += [("point of control", vp.poc), ("value area high", vp.value_area_high),
                ("value area low", vp.value_area_low)]
    g = gamma or {}
    if g.get("available"):
        for key, name in (("zero_gamma_level", "zero gamma"), ("put_wall", "put wall"),
                          ("call_wall", "call wall"), ("max_gamma_strike", "max gamma")):
            if g.get(key):
                out.append((name, float(g[key])))
    return out


def _confluence(prz: tuple[float, float], levels: list[tuple[str, float]], atr_value: float) -> tuple[str, ...]:
    pad = NEAR_ATR * atr_value
    return tuple(dict.fromkeys(n for n, p in levels if prz[0] - pad <= p <= prz[1] + pad))


def find(df: pd.DataFrame | None, tf: str, gamma: dict | None = None) -> list[Harmonic]:
    """Harmonics on this timeframe's completed candles: the latest completed one, and those forming or
    in their zone on the current leg. Nothing below an hour is read."""
    if tf not in TIMEFRAMES or df is None or len(df) < MIN_BARS + 1:
        return []
    closed = df.iloc[:-1].tail(LOOKBACK).reset_index(drop=True)
    a = atr_series(closed)
    atr_value = float(a.iloc[-1]) if len(a) and not np.isnan(a.iloc[-1]) else 0.0
    if atr_value <= 0:
        return []
    pivots, current = zigzag(closed, atr_value)
    levels = levels_for(df, tf, gamma)
    price = float(closed["close"].iloc[-1])
    pad = NEAR_ATR * atr_value
    out: list[Harmonic] = []

    def build(name, pts, stage, prz, bull, dist=None):
        x, a_, d_ref = pts[0].price, pts[1].price, (pts[4].price if len(pts) == 5 else (prz[0] if bull else prz[1]))
        ad = a_ - d_ref
        targets = (d_ref + 0.382 * ad, d_ref + 0.618 * ad)
        invalid = prz[0] - pad if bull else prz[1] + pad
        return Harmonic(name, "bullish" if bull else "bearish", stage, tuple(pts), prz, targets, invalid,
                        _confluence(prz, levels, atr_value), dist)

    # completed: the last five confirmed pivots, D the newest
    if len(pivots) >= 5:
        pts = pivots[-5:]
        bull = pts[0].kind == "low"
        for name in (*SPECS, "Cypher"):
            prz = _prz(name, *(p.price for p in pts[:4]), bull)
            if prz and prz[0] - pad <= pts[4].price <= prz[1] + pad:
                out.append(build(name, pts, "completed", prz, bull))
    # forming / in zone: X A B C are the last four confirmed pivots, the current leg runs toward D
    if len(pivots) >= 4 and current is not None:
        pts = pivots[-4:]
        bull = pts[0].kind == "low"
        if (current.kind == "low") == bull:                   # the leg is heading the way D lies
            for name in (*SPECS, "Cypher"):
                prz = _prz(name, *(p.price for p in pts), bull)
                if not prz:
                    continue
                reached = current.price <= prz[1] if bull else current.price >= prz[0]
                broken = price < prz[0] - pad if bull else price > prz[1] + pad
                if broken:
                    continue
                if reached:
                    out.append(build(name, pts, "in_zone", prz, bull))
                else:
                    edge = prz[1] if bull else prz[0]
                    out.append(build(name, pts, "forming", prz, bull, dist=(edge / price - 1) * 100))
    return out


def find_all(frames: dict, timeframes, gamma: dict | None = None) -> dict[str, list[Harmonic]]:
    return {tf: find(frames.get(tf), tf, gamma) for tf in timeframes if tf in TIMEFRAMES}
