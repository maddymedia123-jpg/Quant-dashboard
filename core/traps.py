"""The TRAP department's deterministic eye: where a fakeout may be forming.

The spec's watchdog hunts fakeouts, swing failure patterns, stop runs and engineered liquidity. Those
have definitions, so candidates are found in code and only then judged by the desk agents: a trap is
never asserted without the evidence that produced it, and every level on a trap card comes from the data.

A candidate needs a trigger and at least one *independent* supporting factor. The triggers are the two
shapes the spec names: liquidity taken and immediately reclaimed (the swing failure), or price holding
beyond value while delta drains the other way. One signal on its own is a data point, not a trap, and a
factor that merely re-states the reading that fired the trigger is not a second signal.

A candidate must also still be live, which is what the gates below enforce:

* The raid has to be recent. "Immediately reclaimed" is a handful of bars, not a memory - without a
  bound, a level swept 189 bars ago still produced a card.
* Price has to have come back through the swept level. A fake break up is only a fake once price is
  below the high it took; while price is still above it, the break is simply in progress.
* Price must not already have answered the question. If `resolved_by` fires at the price that produced
  the candidate, the trap was born settled and `settle()` would close it on the next page load.

Nothing here decides whether a trap is real - that is the TRAP Head's call, after it has asked the
category desk whether the move is an authentic higher-timeframe break."""
from __future__ import annotations

from dataclasses import dataclass, field

BULL_TRAP, BEAR_TRAP = "bull_trap", "bear_trap"
MIN_SUPPORT = 1                  # trigger plus this many supporting factors
# How far beyond the raid price must go before the break counts as authentic. A quarter ATR is a
# hair trigger: ordinary noise would invalidate a live trap within minutes of it being declared.
INVALIDATION_ATR = 0.5
# A swing failure is a two-or-three-bar event, with room for the reclaim to form. Measured over the
# fixture candles, surviving candidates sit at a median of 4 bars but stretch to 189, so an explicit
# bound is what keeps a card from describing a raid that is hours or days dead.
MAX_SWEEP_BARS = 5
# The payoff has to be worth at least what the trap risks, or the card advertises a trade that loses
# money when it is right half the time. Measured: 38 of 676 candidates had the point of control nearer
# than the invalidation buffer, one paying 68 points against 154 of risk.
MIN_REWARD_RISK = 1.0
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
    # The point of control is where the trap pays off, but only once it is far enough away to be worth
    # the risk; a POC sitting just inside the level gets pushed out to the flat fallback.
    travel = buffer * MIN_REWARD_RISK
    if side == BULL_TRAP:
        invalidation = level + buffer
        target = poc if poc is not None and poc < level else level - 2 * buffer
        plays_out = min(target, level - travel)
    else:
        invalidation = level - buffer
        target = poc if poc is not None and poc > level else level + 2 * buffer
        plays_out = max(target, level + travel)
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
        trigger, level, delta_is_trigger = None, None, False

        if pool.get("swept") and pool.get("reclaimed") and not pool.get("broken"):
            level = float(pool.get("price") or 0.0) or None
            bars = pool.get("sweep_bars_ago")
            fresh = bars is None or int(bars) <= MAX_SWEEP_BARS
            # the reclaim has to have happened: a fake break up means price is back below the high it took
            reclaimed = level is not None and ((price < level) if up else (price > level))
            if level and fresh and reclaimed:
                trigger = (f"{pool_key} liquidity at {level:,.0f} was swept and reclaimed"
                           + (f" {bars} bars ago" if bars is not None else "")
                           + f" ({pool.get('touches', '?')} touches)")
            else:
                level = None
        if trigger is None and delta_state == absorption:
            edge = _get(vp, "value_area_high") if up else _get(vp, "value_area_low")
            position = str(_get(vp, "price_position", default="")).lower()
            wanted = "above value" if up else "below value"
            if edge and position == wanted:
                level = float(edge)
                delta_is_trigger = True
                trigger = f"price is holding {wanted} at {level:,.0f} while delta drains the other way"
        if trigger is None or not level:
            continue

        evidence = [trigger]
        # not when the same delta read is what fired the trigger: restating it would let one signal
        # satisfy MIN_SUPPORT on its own, and would inflate the strength the card prints
        if delta_state == absorption and not delta_is_trigger:
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

        if len(evidence) - 1 < MIN_SUPPORT:
            continue
        candidate = _candidate(side, tf, level, price, atr, poc, evidence)
        # born settled: settle() would close this on the next page load, so it is not a trap to declare
        if candidate.resolved_by(price) is None:
            out.append(candidate)
    return out


def find_candidates(state: dict, timeframes) -> list[TrapCandidate]:
    """Every trap shape the evidence supports, strongest first."""
    found: list[TrapCandidate] = []
    for tf in timeframes:
        found.extend(_for_timeframe(state or {}, tf))
    return sorted(found, key=lambda c: (-c.strength, c.timeframe, c.side))
