"""Preemptive compression, leverage and liquidity-gravity assessment.

The assessment is deliberately advisory: missing feeds remain ``None`` and cannot be treated as
evidence. The DEFCON-1 flag requires measured compression, leverage/absorption stress and a
directional liquidity cue; it does not wait for a candle-close breakout confirmation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LONG, SHORT, NONE = "LONG", "SHORT", "NONE"
MIN_BARS = 24
COMPRESSION_TRIGGER = 0.75
FAILURE_TRIGGER = 0.78
NEARBY_PCT = 2.5


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class PreemptiveSignal:
    available: bool = False
    direction: str = NONE
    defcon_1: bool = False
    compression_score: float | None = None
    oi_buildup_score: float | None = None
    funding_skew_score: float | None = None
    absorption_drop: float | None = None
    nearby_liquidity_score: float | None = None
    pull_velocity: float | None = None          # estimated USD per hour
    pull_velocity_atr: float | None = None       # estimated ATR per hour
    structural_failure_probability: float | None = None
    entry_price: float | None = None
    target_low: float | None = None
    target_high: float | None = None
    safety_line: float | None = None
    plain_english_reason: str = "Insufficient live evidence for a preemptive signal."
    evidence: tuple[str, ...] = ()


def _compression(frame: pd.DataFrame, atr_value: float | None) -> tuple[float | None, float | None]:
    if frame is None or len(frame) < MIN_BARS:
        return None, None
    required = {"high", "low", "close"}
    if not required.issubset(frame.columns):
        return None, None

    highs = frame["high"].astype(float).to_numpy()
    lows = frame["low"].astype(float).to_numpy()
    closes = frame["close"].astype(float).to_numpy()
    if not (np.isfinite(highs).all() and np.isfinite(lows).all() and np.isfinite(closes).all()):
        return None, None
    previous = np.concatenate(([closes[0]], closes[:-1]))
    true_ranges = np.maximum(highs - lows, np.maximum(np.abs(highs - previous), np.abs(lows - previous)))
    recent_atr = float(np.mean(true_ranges[-6:]))
    baseline_atr = float(np.mean(true_ranges[-24:-6]))
    current_atr = _finite(atr_value) or float(np.mean(true_ranges[-14:]))
    if current_atr <= 0 or baseline_atr <= 0:
        return None, None

    contraction = _clamp(1.0 - recent_atr / baseline_atr)
    normalized_span = float(np.ptp(np.concatenate((highs[-12:], lows[-12:])))) / (current_atr * math.sqrt(12))
    tight_range = _clamp(1.0 - normalized_span)
    return 0.55 * contraction + 0.45 * tight_range, current_atr


def _liquidity_pull(price: float | None, liq_levels: dict | None,
                    fvgs: tuple | list | None) -> tuple[dict[str, float], float, float | None, float | None]:
    votes = {LONG: 0.0, SHORT: 0.0}
    targets: list[float] = []
    if price is None or price <= 0:
        return votes, 0.0, None, None

    if liq_levels and liq_levels.get("available"):
        for key, direction in (("longs", SHORT), ("shorts", LONG)):
            for cluster in liq_levels.get(key) or ():
                cluster_price = _finite(cluster.get("price"))
                notional = _finite(cluster.get("notional_usd"))
                if cluster_price is None or notional is None or notional <= 0:
                    continue
                distance = abs(cluster_price / price - 1.0) * 100
                if distance <= NEARBY_PCT:
                    proximity = 1.0 - distance / NEARBY_PCT
                    votes[direction] += math.log1p(notional / 1_000_000) * proximity
                    targets.append(cluster_price)

    for gap in fvgs or ():
        if isinstance(gap, dict):
            low, high, filled = _finite(gap.get("low")), _finite(gap.get("high")), gap.get("filled", False)
        else:
            low, high, filled = _finite(getattr(gap, "low", None)), _finite(getattr(gap, "high", None)), getattr(gap, "filled", False)
        if low is None or high is None or filled:
            continue
        midpoint = (low + high) / 2
        distance = abs(midpoint / price - 1.0) * 100
        if distance <= NEARBY_PCT:
            direction = LONG if midpoint > price else SHORT
            votes[direction] += max(0.1, 1.0 - distance / NEARBY_PCT)
            targets.append(midpoint)

    total = votes[LONG] + votes[SHORT]
    score = 0.0 if total <= 0 else abs(votes[LONG] - votes[SHORT]) / total
    target_low = min((target for target in targets if target < price), default=None)
    target_high = min((target for target in targets if target > price), default=None)
    return votes, score, target_low, target_high


def evaluate(
    frame: pd.DataFrame | None,
    *,
    price: float | None = None,
    atr: float | None = None,
    oi_change_pct: float | None = None,
    funding_rate: float | None = None,
    absorption_drop: float | None = None,
    absorption_side: str | None = None,
    liq_levels: dict | None = None,
    fvgs: tuple | list | None = None,
    bar_minutes: int = 60,
) -> PreemptiveSignal:
    """Assess imminent directional pull from candles and optional live derivatives/book feeds.

    ``absorption_drop`` is a fraction from 0 to 1 (for example 0.6 means 60% of a tested wall
    disappeared). ``absorption_side`` is ``bids`` or ``asks``. OI and funding are the latest
    available desk-window observations; callers should pass ``None`` for unavailable or stale feeds.
    """
    current_price = _finite(price)
    if current_price is None and frame is not None and not frame.empty and "close" in frame:
        current_price = _finite(frame["close"].iloc[-1])

    compression, atr_value = _compression(frame, atr)
    oi = _finite(oi_change_pct)
    oi_score = None if oi is None else _clamp((oi - 0.25) / 2.75)
    funding = _finite(funding_rate)
    funding_score = None if funding is None else _clamp(abs(funding) / 0.0005)
    drop = _finite(absorption_drop)
    drop = None if drop is None else _clamp(drop)

    pulls = {LONG: 0.0, SHORT: 0.0}
    if funding is not None and funding_score is not None and funding_score > 0.15:
        pulls[SHORT if funding > 0 else LONG] += funding_score
    if frame is not None and len(frame) >= 7 and oi_score is not None and oi_score > 0:
        closes = frame["close"].astype(float).to_numpy()
        move = closes[-1] / closes[-7] - 1 if closes[-7] else 0
        if abs(move) > 0.001:
            pulls[SHORT if move > 0 else LONG] += oi_score
    if drop is not None and absorption_side in ("bids", "asks"):
        pulls[SHORT if absorption_side == "bids" else LONG] += drop

    liq_pulls, nearby_score, target_low, target_high = _liquidity_pull(current_price, liq_levels, fvgs)
    pulls[LONG] += liq_pulls[LONG]
    pulls[SHORT] += liq_pulls[SHORT]
    direction = LONG if pulls[LONG] > pulls[SHORT] else SHORT if pulls[SHORT] > pulls[LONG] else NONE

    measured = [x for x in (compression, oi_score, funding_score, drop) if x is not None]
    if not measured:
        return PreemptiveSignal(entry_price=current_price)

    # Missing inputs are omitted from the denominator, never converted into neutral readings.
    nearby_available = bool((liq_levels is not None and liq_levels.get("available"))
                            or fvgs is not None)
    weights = ((compression, 0.30), (oi_score, 0.23), (funding_score, 0.17), (drop, 0.20),
               (nearby_score if nearby_available else None, 0.10))
    present_weight = sum(weight for value, weight in weights if value is not None)
    stress = sum(float(value) * weight for value, weight in weights if value is not None) / present_weight
    probability = _clamp(0.04 + 0.94 * stress)

    stress_trigger = ((oi_score is not None and oi_score >= 0.35)
                      or (funding_score is not None and funding_score >= 0.6)
                      or (drop is not None and drop >= 0.5))
    directional_cue = direction != NONE and (funding_score is not None or drop is not None
                                                or nearby_score > 0.2 or (oi_score is not None and oi_score > 0.3))
    defcon = bool(compression is not None and compression >= COMPRESSION_TRIGGER
                  and stress_trigger and directional_cue and probability >= FAILURE_TRIGGER)

    total_pull = pulls[LONG] + pulls[SHORT]
    signed_pull = (pulls[LONG] - pulls[SHORT]) / total_pull if total_pull else 0.0
    pull_atr = None if compression is None else signed_pull * compression * min(0.5, 0.1 + stress * 0.4)
    hours_per_bar = max(float(bar_minutes), 1.0) / 60.0
    velocity = None if pull_atr is None or atr_value is None else round(pull_atr * atr_value / hours_per_bar, 2)

    safety = None
    if current_price and atr_value and direction != NONE:
        safety = round(current_price - atr_value * 1.5 if direction == LONG else current_price + atr_value * 1.5, 2)
    evidence = []
    if compression is not None:
        evidence.append(f"range/ATR compression {compression:.0%}")
    if oi is not None:
        evidence.append(f"open interest {oi:+.2f}%")
    if funding is not None:
        evidence.append(f"funding {funding * 100:.4f}%")
    if drop is not None:
        evidence.append(f"tested book absorption fell {drop:.0%}")
    if nearby_score > 0:
        evidence.append("nearby liquidation cluster or unfilled FVG")

    if defcon:
        verb = "upside squeeze" if direction == LONG else "downside flush"
        reason = (f"DEFCON-1 preemptive {verb}: compression and leverage/absorption stress are pulling "
                  f"toward nearby liquidity; estimated structural failure probability {probability:.0%} "
                  "before candle-close confirmation.")
    elif direction != NONE:
        reason = f"Directional pull points {direction.lower()}, but the preemptive threshold is not met."
    else:
        reason = "No balanced, sufficiently supported preemptive direction; missing feeds are not votes."

    return PreemptiveSignal(
        available=True, direction=direction, defcon_1=defcon,
        compression_score=None if compression is None else round(compression, 3),
        oi_buildup_score=None if oi_score is None else round(oi_score, 3),
        funding_skew_score=None if funding_score is None else round(funding_score, 3),
        absorption_drop=None if drop is None else round(drop, 3),
        nearby_liquidity_score=round(nearby_score, 3),
        pull_velocity=velocity,
        pull_velocity_atr=None if pull_atr is None else round(pull_atr / hours_per_bar, 4),
        structural_failure_probability=round(probability, 3),
        entry_price=current_price, target_low=target_low, target_high=target_high,
        safety_line=safety, plain_english_reason=reason, evidence=tuple(evidence),
    )


assess = evaluate
