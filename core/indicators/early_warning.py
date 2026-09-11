"""Deterministic early-warning setups: traps and squeezes as they form, before the Director confirms them."""
from __future__ import annotations

from dataclasses import dataclass

from core.data.types import FuturesSnapshot
from core.indicators.category import CategoryAnalysis

SQUEEZE_FORMING_MIN = 40
SQUEEZE_WARNING = 60


@dataclass
class EarlyWarning:
    kind: str    # BULL_TRAP_FORMING | BEAR_TRAP_FORMING | SQUEEZE_FORMING | SQUEEZE_WARNING
    level: str   # forming | warning
    drivers: list[str]


def early_warnings(a: CategoryAnalysis, futures: FuturesSnapshot, last_squeeze: int | None) -> list[EarlyWarning]:
    out: list[EarlyWarning] = []
    atr = a.vol.atr or a.price * 0.005
    price = a.price
    res = [l for l in a.levels if l.kind == "resistance" and l.touches >= 2 and 0 <= l.price - price <= 0.5 * atr]
    sup = [l for l in a.levels if l.kind == "support" and l.touches >= 2 and 0 <= price - l.price <= 0.5 * atr]
    bear_div = [d for d in a.divergences if d.kind == "regular" and d.direction == "bearish"]
    bull_div = [d for d in a.divergences if d.kind == "regular" and d.direction == "bullish"]

    f = futures if futures.available else None
    funding_hot = f is not None and f.funding_rate is not None and f.funding_7d_mean is not None and f.funding_rate > f.funding_7d_mean
    funding_cold = f is not None and f.funding_rate is not None and f.funding_7d_mean is not None and f.funding_rate < f.funding_7d_mean
    longs_crowded = f is not None and f.long_short_ratio is not None and f.long_short_ratio > 1.2
    shorts_crowded = f is not None and f.long_short_ratio is not None and f.long_short_ratio < 0.8

    if res and bear_div and (funding_hot or longs_crowded):
        drivers = [f"Price {price:,.0f} pressing resistance {res[0].price:,.0f} ({res[0].touches} touches)", bear_div[0].label + " RSI divergence"]
        if funding_hot:
            drivers.append(f"Funding {f.funding_rate:+.4%} above 7d mean")
        if longs_crowded:
            drivers.append(f"Long/short {f.long_short_ratio:.2f} long-crowded")
        out.append(EarlyWarning("BULL_TRAP_FORMING", "forming", drivers))
    if sup and bull_div and (funding_cold or shorts_crowded):
        drivers = [f"Price {price:,.0f} testing support {sup[0].price:,.0f} ({sup[0].touches} touches)", bull_div[0].label + " RSI divergence"]
        if funding_cold:
            drivers.append(f"Funding {f.funding_rate:+.4%} below 7d mean")
        if shorts_crowded:
            drivers.append(f"Long/short {f.long_short_ratio:.2f} short-crowded")
        out.append(EarlyWarning("BEAR_TRAP_FORMING", "forming", drivers))

    rw = getattr(a, "reversal", None)
    if rw is not None and rw.status == "ACTIVE":
        out.append(EarlyWarning("REVERSAL_WINDOW", "warning", [f"Fibonacci zone F{rw.zone_k} with {rw.bias} confluence"] + rw.confluence))

    sc = a.squeeze.score
    if sc is not None:
        if sc >= SQUEEZE_WARNING:
            out.append(EarlyWarning("SQUEEZE_WARNING", "warning", [f"{a.squeeze.direction.replace('_', ' ')} risk {sc}/100"] + a.squeeze.drivers))
        elif sc >= SQUEEZE_FORMING_MIN and last_squeeze is not None and sc > last_squeeze:
            out.append(EarlyWarning("SQUEEZE_FORMING", "forming", [f"Squeeze score rising {last_squeeze} → {sc}", a.squeeze.direction.replace("_", " ")] + a.squeeze.drivers))
    return out
