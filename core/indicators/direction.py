"""Composite per-category direction call: EMA stack, RSI, CVD slope, S/R position."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.indicators.ema import EmaStack
from core.indicators.levels import Level


@dataclass
class DirectionCall:
    direction: str      # BULLISH | BEARISH | NEUTRAL
    confidence: float   # 0..1
    score: float        # -1..1
    anchor: float | None
    invalidation: float | None
    drivers: list[str]
    flips_if: str


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def cvd_slope(df: pd.DataFrame, bars: int = 5) -> float:
    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    buyer_ratio = (df["close"] - df["low"]) / rng
    delta = (buyer_ratio - 0.5) * 2.0 * df["volume"]
    cvd = delta.cumsum()
    if len(cvd) <= bars:
        return 0.0
    return float(cvd.iloc[-1] - cvd.iloc[-1 - bars])


def direction_call(df: pd.DataFrame, stack: EmaStack, rsi_series: pd.Series, levels: list[Level], atr_value: float | None) -> DirectionCall:
    price = float(df["close"].iloc[-1])
    atr_v = atr_value or price * 0.005
    drivers: list[str] = []

    if stack.state == "BULL":
        ema_vote = 1.0
        drivers.append("EMA stack bullish (price > 21 > 50 > 100 > 200)")
    elif stack.state == "BEAR":
        ema_vote = -1.0
        drivers.append("EMA stack bearish (price < 21 < 50 < 100 < 200)")
    else:
        e50 = stack.values.get(50)
        ema_vote = 0.0 if e50 is None else (0.3 if price > e50 else -0.3)
        if e50 is not None:
            drivers.append(f"Price {'above' if price > e50 else 'below'} EMA50, stack mixed")

    r = rsi_series.dropna()
    if len(r) >= 6:
        last, prev = float(r.iloc[-1]), float(r.iloc[-6])
        rsi_vote = _clamp((last - 50.0) / 25.0) * 0.7 + (0.3 if last > prev else -0.3 if last < prev else 0.0)
        drivers.append(f"RSI {last:.0f} and {'rising' if last > prev else 'falling'}")
    else:
        rsi_vote = 0.0

    slope = cvd_slope(df)
    vol_ref = float(df["volume"].tail(20).mean()) or 1.0
    cvd_vote = _clamp(slope / (vol_ref * 2.0))
    if abs(cvd_vote) >= 0.3:
        drivers.append(f"CVD {'rising' if cvd_vote > 0 else 'falling'} over last 5 bars")

    sr_vote = 0.0
    near_sup = [l for l in levels if l.kind == "support" and l.touches >= 2 and 0 <= price - l.price <= 0.5 * atr_v]
    near_res = [l for l in levels if l.kind == "resistance" and l.touches >= 2 and 0 <= l.price - price <= 0.5 * atr_v]
    if near_sup and not near_res:
        sr_vote = 0.5
        drivers.append(f"Holding support {near_sup[0].price:,.0f} ({near_sup[0].touches} touches)")
    elif near_res and not near_sup:
        sr_vote = -0.5
        drivers.append(f"Capped under resistance {near_res[0].price:,.0f} ({near_res[0].touches} touches)")

    score = 0.35 * ema_vote + 0.2 * rsi_vote + 0.2 * cvd_vote + 0.25 * sr_vote
    direction = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
    confidence = min(1.0, abs(score))

    supports = sorted([l for l in levels if l.kind == "support"], key=lambda l: price - l.price)
    resistances = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: l.price - price)
    anchor = invalidation = None
    if direction == "BULLISH" and supports:
        anchor = supports[0].price
        invalidation = anchor - 0.5 * atr_v
        flips = f"Close below {invalidation:,.0f} or EMA stack turns bearish"
    elif direction == "BEARISH" and resistances:
        anchor = resistances[0].price
        invalidation = anchor + 0.5 * atr_v
        flips = f"Close above {invalidation:,.0f} or EMA stack turns bullish"
    elif direction == "NEUTRAL":
        flips = "Break of nearest support/resistance with EMA confirmation"
    else:
        flips = "EMA stack reversal"
    return DirectionCall(direction, confidence, float(score), anchor, invalidation, drivers, flips)
