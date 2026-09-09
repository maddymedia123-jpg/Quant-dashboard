"""Long/short squeeze risk from OI build, price stagnation, funding deviation, L/S skew."""
from __future__ import annotations

from dataclasses import dataclass

from core.data.types import FuturesSnapshot


@dataclass
class SqueezeRisk:
    score: int | None
    direction: str  # LONG_SQUEEZE | SHORT_SQUEEZE | NONE | UNKNOWN
    drivers: list[str]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def squeeze_risk(fut: FuturesSnapshot, price_change_24h_pct: float | None) -> SqueezeRisk:
    if not fut.available or (fut.oi_change_24h_pct is None and fut.funding_rate is None and fut.long_short_ratio is None):
        return SqueezeRisk(None, "UNKNOWN", ["futures data unavailable"])
    missing: list[str] = []
    if fut.oi_change_24h_pct is None:
        missing.append("OI 24h change unavailable")
    oi_up = _clamp(fut.oi_change_24h_pct / 5.0, 0.0, 1.0) if fut.oi_change_24h_pct is not None else 0.0
    px = price_change_24h_pct or 0.0
    flat = 1.0 - _clamp(abs(px) / 2.0, 0.0, 1.0)
    fr = fut.funding_rate or 0.0
    mean = fut.funding_7d_mean or 0.0
    if fut.funding_7d_mean is None and fut.funding_rate is not None:
        missing.append("funding 7d mean unavailable (compared to 0)")
    funding_hot = _clamp((fr - mean) / 0.0002, -1.0, 1.0)
    ls = fut.long_short_ratio
    if ls is None:
        missing.append("long/short ratio unavailable")
    ls_skew = 0.0 if ls is None else _clamp((ls - 1.0) / 0.5, -1.0, 1.0)

    long_crowd = max(0.0, funding_hot) * 0.4 + max(0.0, ls_skew) * 0.3 + oi_up * 0.2 + flat * 0.1
    short_crowd = max(0.0, -funding_hot) * 0.4 + max(0.0, -ls_skew) * 0.3 + oi_up * 0.2 + flat * 0.1

    drivers: list[str] = list(missing)
    if oi_up >= 0.5 and fut.oi_change_24h_pct is not None:
        drivers.append(f"Open interest +{fut.oi_change_24h_pct:.1f}% in 24h")
    if flat >= 0.5:
        drivers.append(f"Price {px:+.1f}% while positioning builds")
    if funding_hot >= 0.5:
        drivers.append(f"Funding {fr:+.4%} above 7d mean {mean:+.4%}")
    if funding_hot <= -0.5:
        drivers.append(f"Funding {fr:+.4%} below 7d mean {mean:+.4%}")
    if ls is not None and ls_skew >= 0.5:
        drivers.append(f"Long/short ratio {ls:.2f} long-crowded")
    if ls is not None and ls_skew <= -0.5:
        drivers.append(f"Long/short ratio {ls:.2f} short-crowded")

    if long_crowd >= short_crowd:
        score = round(long_crowd * 100)
        return SqueezeRisk(score, "LONG_SQUEEZE" if score >= 40 else "NONE", drivers)
    score = round(short_crowd * 100)
    return SqueezeRisk(score, "SHORT_SQUEEZE" if score >= 40 else "NONE", drivers)
