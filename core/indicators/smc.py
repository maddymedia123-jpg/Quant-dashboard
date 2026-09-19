"""Smart Money Concepts, computed from candles.

The Live Recon rubric asks five questions the rest of the deterministic layer could not answer: is
there a break of structure or change of character, is price at an unmitigated order block, is there an
unfilled fair value gap, is the lower timeframe making higher highs, and is price in premium or
discount. Those are all arithmetic on OHLC, so they belong here rather than in a prompt - a model
asked to eyeball them will invent them.

Every reading carries an availability flag. Thin data returns "unclear", never a guess."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from core.indicators.pivots import find_pivots

MIN_BARS = 20          # below this there is no structure worth naming
SWING_LEFT = 2
SWING_RIGHT = 2


@dataclass(frozen=True)
class Swing:
    index: int
    price: float
    kind: Literal["high", "low"]


@dataclass(frozen=True)
class StructureEvent:
    kind: Literal["BOS", "CHOCH"]      # continuation, or the first break the other way
    direction: Literal["bullish", "bearish"]
    index: int
    price: float                        # the swing level that broke

    @property
    def label(self) -> str:
        return f"{self.direction} {self.kind}"


@dataclass
class Structure:
    available: bool = False
    bias: str = "unclear"                       # bullish, bearish or unclear
    events: list[StructureEvent] = field(default_factory=list)
    swing_high: float | None = None             # most recent confirmed swing high
    swing_low: float | None = None
    note: str = ""

    @property
    def last_event(self) -> StructureEvent | None:
        return self.events[-1] if self.events else None


@dataclass(frozen=True)
class Gap:
    direction: Literal["bullish", "bearish"]
    index: int                                  # the middle candle of the three
    low: float
    high: float
    filled: bool

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class OrderBlock:
    kind: Literal["demand", "supply"]
    index: int
    low: float
    high: float
    mitigated: bool


@dataclass(frozen=True)
class PremiumDiscount:
    high: float
    low: float
    equilibrium: float
    position: float                             # 0 at the range low, 1 at the range high
    zone: Literal["premium", "discount", "equilibrium"]


def swings(df: pd.DataFrame, left: int = SWING_LEFT, right: int = SWING_RIGHT) -> list[Swing]:
    """Confirmed swing points in time order, alternating high and low.

    Where several same-kind pivots run together the most extreme one wins, which is what a trader
    reads off the chart and what keeps a break of structure meaningful."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    found = ([Swing(i, float(highs[i]), "high") for i in find_pivots(highs, left, right, "high")]
             + [Swing(i, float(lows[i]), "low") for i in find_pivots(lows, left, right, "low")])
    found.sort(key=lambda s: s.index)

    out: list[Swing] = []
    for s in found:
        if out and out[-1].kind == s.kind:
            better = (s.price > out[-1].price) if s.kind == "high" else (s.price < out[-1].price)
            if better:
                out[-1] = s
            continue
        out.append(s)
    return out


def structure(df: pd.DataFrame, left: int = SWING_LEFT, right: int = SWING_RIGHT) -> Structure:
    """Break of structure and change of character from the swing sequence.

    A close through the last swing high continues an uptrend (BOS) or reverses a downtrend (CHOCH),
    and the mirror for lows. The distinction is exactly the prior bias, so it is tracked as we walk."""
    if len(df) < MIN_BARS:
        return Structure(note=f"needs {MIN_BARS} bars, has {len(df)}")

    sw = swings(df, left, right)
    if len(sw) < 2:
        return Structure(note="no confirmed swings yet")

    closes = df["close"].to_numpy()
    events: list[StructureEvent] = []
    bias = "unclear"
    last_high = last_low = None
    broken: set[tuple[int, str]] = set()

    for i in range(len(df)):
        for s in sw:
            if s.index + right > i:          # a swing is only usable once confirmed
                continue
            if s.kind == "high":
                if last_high is None or s.index > last_high.index:
                    last_high = s
            elif last_low is None or s.index > last_low.index:
                last_low = s

        c = float(closes[i])
        if last_high is not None and c > last_high.price and (last_high.index, "high") not in broken:
            broken.add((last_high.index, "high"))
            kind = "CHOCH" if bias == "bearish" else "BOS"
            events.append(StructureEvent(kind, "bullish", i, last_high.price))
            bias = "bullish"
        elif last_low is not None and c < last_low.price and (last_low.index, "low") not in broken:
            broken.add((last_low.index, "low"))
            kind = "CHOCH" if bias == "bullish" else "BOS"
            events.append(StructureEvent(kind, "bearish", i, last_low.price))
            bias = "bearish"

    highs = [s.price for s in sw if s.kind == "high"]
    lows = [s.price for s in sw if s.kind == "low"]
    return Structure(available=True, bias=bias, events=events,
                     swing_high=highs[-1] if highs else None,
                     swing_low=lows[-1] if lows else None,
                     note="" if events else "no break of structure in range")


def fair_value_gaps(df: pd.DataFrame, max_age: int = 200) -> list[Gap]:
    """Three-candle imbalances: candle 1 and candle 3 do not overlap.

    A gap is filled once any later candle trades back through it."""
    if len(df) < 3:
        return []
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    out: list[Gap] = []
    start = max(1, len(df) - max_age)
    for i in range(start, len(df) - 1):
        prev_high, prev_low = float(highs[i - 1]), float(lows[i - 1])
        next_high, next_low = float(highs[i + 1]), float(lows[i + 1])
        if next_low > prev_high:
            lo, hi, direction = prev_high, next_low, "bullish"
        elif next_high < prev_low:
            lo, hi, direction = next_high, prev_low, "bearish"
        else:
            continue
        after = df.iloc[i + 2:]
        filled = bool(len(after) and ((after["low"] <= lo).any() if direction == "bullish"
                                      else (after["high"] >= hi).any()))
        out.append(Gap(direction, i, lo, hi, filled))
    return out


def nearest_unfilled(gaps: list[Gap], price: float, direction: str) -> Gap | None:
    """The closest unfilled gap on the side that matters: bullish below price, bearish above."""
    side = [g for g in gaps if g.direction == direction and not g.filled
            and (g.high <= price if direction == "bullish" else g.low >= price)]
    if not side:
        return None
    return min(side, key=lambda g: abs(price - g.mid))


def order_blocks(df: pd.DataFrame, left: int = SWING_LEFT, right: int = SWING_RIGHT) -> list[OrderBlock]:
    """The last opposite candle before an impulse that breaks structure.

    Only breaks count, so this does not mark every pullback candle as institutional interest. A block
    is mitigated once price trades back into it after the break."""
    st = structure(df, left, right)
    if not st.available:
        return []
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    out: list[OrderBlock] = []

    for ev in st.events:
        want_down = ev.direction == "bullish"        # a demand block is the last down candle
        j = ev.index
        found = None
        while j >= 0 and ev.index - j <= 20:
            is_down = closes[j] < opens[j]
            if is_down == want_down:
                found = j
                break
            j -= 1
        if found is None:
            continue
        lo, hi = float(lows[found]), float(highs[found])
        after = df.iloc[ev.index + 1:]
        mitigated = bool(len(after) and ((after["low"] <= hi).any() if want_down
                                         else (after["high"] >= lo).any()))
        out.append(OrderBlock("demand" if want_down else "supply", found, lo, hi, mitigated))
    return out


def premium_discount(df: pd.DataFrame, lookback: int = 100) -> PremiumDiscount:
    """Where price sits in the dealing range: above equilibrium is premium, below is discount."""
    window = df.tail(lookback)
    hi, lo = float(window["high"].max()), float(window["low"].min())
    price = float(window["close"].iloc[-1])
    eq = (hi + lo) / 2.0
    span = hi - lo
    pos = 0.5 if span <= 0 else (price - lo) / span
    zone = "equilibrium" if 0.45 <= pos <= 0.55 else "premium" if pos > 0.55 else "discount"
    return PremiumDiscount(hi, lo, eq, pos, zone)


def summarise(df: pd.DataFrame) -> dict:
    """The compact, JSON-safe reading handed to the rubric judges."""
    st = structure(df)
    if not st.available:
        return {"structure": {"available": False, "note": st.note},
                "fair_value_gaps": {"available": False}, "order_blocks": {"available": False},
                "premium_discount": {"available": False}}

    price = float(df["close"].iloc[-1])
    gaps = fair_value_gaps(df)
    blocks = order_blocks(df)
    pdz = premium_discount(df)
    last = st.last_event

    def gap_out(direction: str) -> dict | None:
        g = nearest_unfilled(gaps, price, direction)
        return None if g is None else {"low": round(g.low, 2), "high": round(g.high, 2),
                                       "distance_pct": round((g.mid - price) / price * 100.0, 3)}

    def block_out(kind: str) -> dict | None:
        live = [b for b in blocks if b.kind == kind and not b.mitigated]
        if not live:
            return None
        b = min(live, key=lambda b: abs(price - (b.low + b.high) / 2.0))
        return {"low": round(b.low, 2), "high": round(b.high, 2),
                "price_inside": bool(b.low <= price <= b.high),
                "distance_pct": round(((b.low + b.high) / 2.0 - price) / price * 100.0, 3)}

    return {
        "structure": {
            "available": True, "bias": st.bias,
            "last_event": None if last is None else {
                "type": last.kind, "direction": last.direction, "level": round(last.price, 2),
                "bars_ago": len(df) - 1 - last.index},
            "swing_high": None if st.swing_high is None else round(st.swing_high, 2),
            "swing_low": None if st.swing_low is None else round(st.swing_low, 2),
        },
        "fair_value_gaps": {"available": True, "unfilled_below": gap_out("bullish"),
                            "unfilled_above": gap_out("bearish"),
                            "unfilled_count": sum(1 for g in gaps if not g.filled)},
        "order_blocks": {"available": True, "nearest_demand": block_out("demand"),
                         "nearest_supply": block_out("supply"),
                         "unmitigated_count": sum(1 for b in blocks if not b.mitigated)},
        "premium_discount": {"available": True, "zone": pdz.zone, "position": round(pdz.position, 3),
                             "equilibrium": round(pdz.equilibrium, 2),
                             "range_high": round(pdz.high, 2), "range_low": round(pdz.low, 2)},
    }
