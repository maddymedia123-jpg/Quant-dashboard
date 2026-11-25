"""Preemptive compression squeeze and liquidation gravity calculator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LONG, SHORT, NONE = "LONG", "SHORT", "NONE"


@dataclass(frozen=True)
class PreemptiveSignal:
    available: bool = True
    defcon_1: bool = False
    direction: str = NONE          # "LONG" | "SHORT" | "NONE"
    pull_velocity: float = 0.0     # 0.0 to 100.0
    structural_failure_probability: float | None = None  # 0.0 to 1.0
    compression_score: float = 0.0
    entry_price: float | None = None
    target_low: float | None = None
    target_high: float | None = None
    safety_line: float | None = None
    plain_english_reason: str = ""


@dataclass(frozen=True)
class GravityRead:
    direction: str          # "LONG" | "SHORT" | "NEUTRAL"
    pull_velocity: float    # 0.0 to 100.0
    failure_prob: float     # 0.0 to 100.0
    compression_score: float  # 0.0 to 100.0
    defcon_1: bool
    target_zone: tuple[float, float] | None
    invalidation: float | None
    reason: str


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
        return out if out == out and abs(out) != float("inf") else None
    except (TypeError, ValueError):
        return None


def evaluate(
    frame: Any = None,
    *,
    price: float | None = None,
    atr: float | None = None,
    oi_change_pct: float | None = None,
    funding_rate: float | None = None,
    absorption_drop: float | None = None,
    absorption_side: str | None = None,
    liq_levels: dict[str, Any] | None = None,
    fvgs: Any = None,
    bar_minutes: int = 60,
) -> GravityRead:
    px = _safe_float(price)
    if px is None and frame is not None and not getattr(frame, "empty", True) and "close" in frame:
        px = _safe_float(frame["close"].iloc[-1])

    atr_val = _safe_float(atr)
    oi_pct = _safe_float(oi_change_pct) or 0.0
    fund = _safe_float(funding_rate) or 0.0
    abs_drop = max(0.0, min(1.0, _safe_float(absorption_drop) or 0.0))

    compression = 0.0
    if frame is not None and not getattr(frame, "empty", True) and len(frame) >= 6:
        recent = frame.tail(min(len(frame), 12))
        highs = recent["high"].astype(float) if "high" in recent else None
        lows = recent["low"].astype(float) if "low" in recent else None
        if highs is not None and lows is not None and px and px > 0:
            span_pct = ((float(highs.max()) - float(lows.min())) / px) * 100.0
            compression = max(0.0, min(100.0, (2.5 - span_pct) * 45.0))
            if atr_val and atr_val > 0 and (atr_val / px) * 100.0 < 0.45:
                compression = min(100.0, compression + 15.0)

    leverage_heat = min(100.0, abs(oi_pct) * 8.0 + abs(fund) * 50_000.0)
    side = "NEUTRAL"
    if fund > 0.0002 or (absorption_side or "").upper() in ("BID", "BUY", "LONG_TRAP"):
        side = "SHORT"
    elif fund < -0.0001 or (absorption_side or "").upper() in ("ASK", "SELL", "SHORT_TRAP"):
        side = "LONG"

    target_zone: tuple[float, float] | None = None
    invalidation: float | None = None
    liq_dict = liq_levels or {}
    below_px = _safe_float(liq_dict.get("below") or liq_dict.get("downside"))
    above_px = _safe_float(liq_dict.get("above") or liq_dict.get("upside"))

    if side == "SHORT" and px is not None:
        low_t = below_px if below_px is not None else px * 0.97
        high_t = px * 0.992
        target_zone = (min(low_t, high_t), max(low_t, high_t))
        invalidation = px * 1.012
    elif side == "LONG" and px is not None:
        high_t = above_px if above_px is not None else px * 1.03
        low_t = px * 1.008
        target_zone = (min(low_t, high_t), max(low_t, high_t))
        invalidation = px * 0.988

    pull_velocity = round(
        max(0.0, min(100.0, compression * 0.40 + leverage_heat * 0.30 + abs_drop * 100.0 * 0.30)),
        1,
    )
    failure_prob = round(
        max(0.0, min(100.0, compression * 0.45 + abs_drop * 100.0 * 0.35 + min(100.0, abs(oi_pct) * 10.0) * 0.20)),
        1,
    )
    defcon_1 = bool(
        side in ("LONG", "SHORT")
        and compression >= 55.0
        and failure_prob >= 62.0
        and pull_velocity >= 55.0
    )

    if side == "NEUTRAL":
        reason = "Balanced positioning; no asymmetric compression/liquidation magnet."
    elif defcon_1:
        reason = (
            f"DEFCON-1 preemptive {side} pressure ({failure_prob:.0f}% failure risk, "
            f"{pull_velocity:.0f} velocity) ahead of candle close."
        )
    else:
        reason = (
            f"{side} gravity {pull_velocity:.0f}/100 with {failure_prob:.0f}% structural "
            f"failure risk on {bar_minutes}m cadence."
        )

    return GravityRead(
        direction=side,
        pull_velocity=pull_velocity,
        failure_prob=failure_prob,
        compression_score=round(compression, 1),
        defcon_1=defcon_1,
        target_zone=target_zone,
        invalidation=invalidation,
        reason=reason,
    )


def to_preemptive_signal(
    read: GravityRead,
    *,
    entry_price: float | None = None,
) -> PreemptiveSignal:
    low_t = read.target_zone[0] if read.target_zone else None
    high_t = read.target_zone[1] if read.target_zone else None
    return PreemptiveSignal(
        available=True,
        defcon_1=read.defcon_1,
        direction=read.direction if read.direction in (LONG, SHORT) else NONE,
        pull_velocity=read.pull_velocity,
        structural_failure_probability=round(read.failure_prob / 100.0, 4),
        compression_score=read.compression_score,
        entry_price=_safe_float(entry_price),
        target_low=low_t,
        target_high=high_t,
        safety_line=read.invalidation,
        plain_english_reason=read.reason,
    )
