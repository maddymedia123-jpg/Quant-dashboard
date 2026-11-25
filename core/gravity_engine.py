"""Preemptive compression squeeze and liquidation gravity calculator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LONG, SHORT, NONE = "LONG", "SHORT", "NONE"


@dataclass(frozen=True)
class PreemptiveSignal:
    available: bool = True
    defcon_1: bool = False
    watch_alert: bool = False
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
    watch_alert: bool
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


def _fvg_magnet(fvgs: Any, px: float | None, side: str) -> tuple[float, float] | None:
    """Extract nearest FVG zone in the anticipated sweep/flush direction."""
    if not fvgs or px is None:
        return None
    items = fvgs.values() if isinstance(fvgs, dict) else fvgs
    candidates: list[tuple[float, float, float]] = []
    for item in items or ():
        low = _safe_float(item.get("low") if isinstance(item, dict) else getattr(item, "low", None))
        high = _safe_float(item.get("high") if isinstance(item, dict) else getattr(item, "high", None))
        if low is None or high is None:
            continue
        lo, hi = min(low, high), max(low, high)
        mid = (lo + hi) / 2.0
        if side == SHORT and hi < px:
            candidates.append((px - mid, lo, hi))
        elif side == LONG and lo > px:
            candidates.append((mid - px, lo, hi))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, lo, hi = candidates[0]
    return (lo, hi)


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
            compression = max(0.0, min(100.0, (2.6 - span_pct) * 46.0))
            if atr_val and atr_val > 0 and (atr_val / px) * 100.0 < 0.50:
                compression = min(100.0, compression + 15.0)

    leverage_heat = min(100.0, abs(oi_pct) * 9.0 + abs(fund) * 55_000.0)
    side = "NEUTRAL"
    if fund > 0.00015 or (absorption_side or "").upper() in ("BID", "BUY", "LONG_TRAP"):
        side = "SHORT"
    elif fund < -0.0001 or (absorption_side or "").upper() in ("ASK", "SELL", "SHORT_TRAP"):
        side = "LONG"

    target_zone: tuple[float, float] | None = None
    invalidation: float | None = None
    liq_dict = liq_levels or {}
    below_px = _safe_float(liq_dict.get("below") or liq_dict.get("downside"))
    above_px = _safe_float(liq_dict.get("above") or liq_dict.get("upside"))
    fvg_zone = _fvg_magnet(fvgs, px, side)

    if side == "SHORT" and px is not None:
        if below_px is not None:
            target_zone = (min(below_px, px * 0.992), max(below_px, px * 0.992))
        elif fvg_zone is not None:
            target_zone = fvg_zone
        else:
            target_zone = (px * 0.968, px * 0.992)
        invalidation = px * 1.011
    elif side == "LONG" and px is not None:
        if above_px is not None:
            target_zone = (min(px * 1.008, above_px), max(px * 1.008, above_px))
        elif fvg_zone is not None:
            target_zone = fvg_zone
        else:
            target_zone = (px * 1.008, px * 1.032)
        invalidation = px * 0.989

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
        and compression >= 50.0
        and failure_prob >= 58.0
        and pull_velocity >= 52.0
    )
    watch_alert = bool(
        side in ("LONG", "SHORT")
        and not defcon_1
        and (compression >= 40.0 or pull_velocity >= 42.0 or failure_prob >= 45.0)
    )

    if side == "NEUTRAL":
        reason = "No asymmetric compression or leverage imbalance detected yet."
    elif defcon_1:
        reason = (
            f"PREEMPTIVE DEFCON-1 ({side}): {compression:.0f}/100 coil with {failure_prob:.0f}% "
            f"failure risk and {pull_velocity:.0f} magnet velocity ahead of {bar_minutes}m close."
        )
    elif watch_alert:
        reason = (
            f"PREEMPTIVE WATCH ({side}): building {compression:.0f}/100 compression, "
            f"{pull_velocity:.0f} magnet velocity, and {failure_prob:.0f}% sweep risk on {bar_minutes}m window."
        )
    else:
        reason = (
            f"{side} magnet pull {pull_velocity:.0f}/100 ({failure_prob:.0f}% structural failure risk, "
            f"{compression:.0f}/100 compression) on {bar_minutes}m window."
        )

    return GravityRead(
        direction=side,
        pull_velocity=pull_velocity,
        failure_prob=failure_prob,
        compression_score=round(compression, 1),
        defcon_1=defcon_1,
        watch_alert=watch_alert,
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
        watch_alert=read.watch_alert,
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
