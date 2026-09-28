"""The TRAP department's deterministic eye: where a fakeout may be forming.

The spec's watchdog hunts fakeouts, swing failure patterns, stop runs and engineered liquidity. Those
have definitions, so candidates are found in code and only then judged by the desk agents: a trap is
never asserted without the evidence that produced it, and every level on a trap card comes from the data.

A candidate needs a trigger and at least one supporting factor. The triggers are the two shapes the spec
names: liquidity taken and immediately reclaimed (the swing failure), or price holding beyond value while
delta drains the other way. One signal on its own is a data point, not a trap.

Nothing here decides whether a trap is real - that is the TRAP Head's call, after it has asked the
category desk whether the move is an authentic higher-timeframe break."""
from __future__ import annotations

from dataclasses import dataclass, field

BULL_TRAP, BEAR_TRAP = "bull_trap", "bear_trap"
MIN_SUPPORT = 1                  # trigger plus this many supporting factors
# How far beyond the raid price must go before the break counts as authentic. A quarter ATR is a
# hair trigger: ordinary noise would invalidate a live trap within minutes of it being declared.
INVALIDATION_ATR = 0.5
DEFAULT_ATR_PCT = 0.004          # fallback when the volatility matrix has no ATR for this timeframe


@dataclass(frozen=True)
class TrapCandidate:
    side: str                    # bull_trap (a fake break up) | bear_trap (a fake break down)
    timeframe: str
    level: float                 # the liquidity or value edge being faked
    invalidation: float          # beyond this the break was authentic, not a raid
    plays_out: float             # where the trap pays off: back inside value
    strength: int                # trigger plus supporting factors
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return "bull trap" if self.side == BULL_TRAP else "bear trap"

    def resolved_by(self, price: float) -> str | None:
        """Has price settled the question? Invalidated means the break was real after all."""
        if self.side == BULL_TRAP:
            if price >= self.invalidation:
                return "invalidated"
            if price <= self.plays_out:
                return "played_out"
        else:
            if price <= self.invalidation:
                return "invalidated"
            if price >= self.plays_out:
                return "played_out"
        return None


def _get(d, *path, default=None):
    cur = d if isinstance(d, dict) else {}
    for step in path:
        if not isinstance(cur, dict) or step not in cur or cur[step] is None:
            return default
        cur = cur[step]
    return cur


def _atr_for(state: dict, timeframe: str, price: float) -> float:
    atr = _get(state, "volatility", timeframe, "atr")
    return float(atr) if atr else price * DEFAULT_ATR_PCT


def _candidate(side: str, timeframe: str, level: float, price: float, atr: float, poc: float | None,
               evidence: list[str]) -> TrapCandidate:
    buffer = atr * INVALIDATION_ATR
    if side == BULL_TRAP:
        invalidation = level + buffer
        plays_out = poc if poc is not None and poc < level else level - 2 * buffer
    else:
        invalidation = level - buffer
        plays_out = poc if poc is not None and poc > level else level + 2 * buffer
    return TrapCandidate(side=side, timeframe=timeframe, level=round(level, 2),
                         invalidation=round(invalidation, 2), plays_out=round(plays_out, 2),
                         strength=len(evidence), evidence=tuple(evidence))


def _for_timeframe(state: dict, tf: str) -> list[TrapCandidate]:
    price = float(_get(state, "spot", "price", default=0.0) or 0.0)
    if not price:
        return []
    liq, smc = _get(state, "liquidity", tf, default={}), _get(state, "smc", tf, default={})
    fut = _get(state, "derivatives", "futures", default={})
    atr = _atr_for(state, tf, price)
    vp = _get(liq, "volume_profile", default={})
    poc = _get(vp, "point_of_control")
    delta_state = str(_get(liq, "cumulative_delta", "state", default="")).lower()
    bands = _get(state, "volatility", tf, "bands", default={})
    oi = _get(fut, "oi_change_24h_pct")
    funding, funding_mean = _get(fut, "funding_rate"), _get(fut, "funding_7d_mean")
    zone = _get(smc, "premium_discount", "zone")
    structure = _get(smc, "structure", "bias")
    out: list[TrapCandidate] = []

    for side, pool_key, other in ((BULL_TRAP, "buyside", "sellside"), (BEAR_TRAP, "sellside", "buyside")):
        up = side == BULL_TRAP
        absorption = "bearish absorption" if up else "bullish absorption"
        pool = _get(liq, "pools", pool_key, default={})
        trigger, level = None, None

        if pool.get("swept") and pool.get("reclaimed") and not pool.get("broken"):
            level = float(pool.get("price") or 0.0) or None
            if level:
                bars = pool.get("sweep_bars_ago")
                trigger = (f"{pool_key} liquidity at {level:,.0f} was swept and reclaimed"
                           + (f" {bars} bars ago" if bars is not None else "")
                           + f" ({pool.get('touches', '?')} touches)")
        if trigger is None and delta_state == absorption:
            edge = _get(vp, "value_area_high") if up else _get(vp, "value_area_low")
            position = str(_get(vp, "price_position", default="")).lower()
            wanted = "above value" if up else "below value"
            if edge and position == wanted:
                level = float(edge)
                trigger = f"price is holding {wanted} at {level:,.0f} while delta drains the other way"
        if trigger is None or not level:
            continue

        evidence = [trigger]
        if delta_state == absorption:
            evidence.append(f"cumulative delta shows {absorption}")
        if oi is not None and float(oi) > 0:
            evidence.append(f"open interest is expanding ({float(oi):+.1f}% over 24h), so the move is crowded")
        if funding is not None and funding_mean is not None:
            hot = float(funding) > float(funding_mean) if up else float(funding) < float(funding_mean)
            if hot:
                evidence.append(f"funding at {float(funding) * 100:+.3f}% is beyond its 7-day mean "
                                f"({float(funding_mean) * 100:+.3f}%)")
        stretch = _get(bands, "sigma2_up") if up else _get(bands, "sigma2_dn")
        if stretch and ((price > float(stretch)) if up else (price < float(stretch))):
            evidence.append(f"price is beyond the 2 sigma channel at {float(stretch):,.0f}")
        if zone == ("premium" if up else "discount"):
            evidence.append(f"price is in {zone} of the dealing range")
        if structure in ("bearish", "bullish") and structure != ("bullish" if up else "bearish"):
            evidence.append(f"the break runs against {structure} structure on {tf}")

        if len(evidence) - 1 >= MIN_SUPPORT:
            out.append(_candidate(side, tf, level, price, atr, poc, evidence))
    return out


def find_candidates(state: dict, timeframes) -> list[TrapCandidate]:
    """Every trap shape the evidence supports, strongest first."""
    found: list[TrapCandidate] = []
    for tf in timeframes:
        found.extend(_for_timeframe(state or {}, tf))
    return sorted(found, key=lambda c: (-c.strength, c.timeframe, c.side))
