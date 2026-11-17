"""Hammad's Section 6: ONE desk-owned Master Swing Trade card.

The trader's own positions stay theirs (the Active Trade screen, up to ten cards, KILL). Beside them the
worker holds at most one card that no one typed in: opened when the monthly or the weekly desk's
confluence gate says TRADE while price is inside the zone, from the same deterministic plan builder the
screen uses, and closed when its last target is reached, its hard invalidation is hit, or the desk that
opened it now trades the other way. The monthly desk outranks the weekly. No agents are consulted: the
gate's three-of-seven families are the conviction, and the plan stands on its levels.

The desks are read through a reader object (DeskReader for the real market; tests hand in fakes), so the
rules here are testable without candles."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core import trades
from core.trade_plan import deterministic_plan
from core.trades import LONG, SHORT

log = logging.getLogger(__name__)

SWING_DESKS = ("monthly", "weekly")     # in rank order: the higher desk opens the card when both would
ASSET = "BTC/USDT"
EVENT = "master"
# Bumped whenever what opens or closes the card changes: a card opened under older rules is closed as
# superseded on the next pass, and the channel is told. Rules 1 (17 Nov 00:03 UTC) opened at the zone's
# middle with reach of one desk range, so the first card sat five thousand dollars from price with the
# verifier's stop; rules 2 open at the live price inside the zone with the desk's invalidation as the stop.
RULES = 2


@dataclass
class MasterResult:
    opened: int = 0
    closed: list = field(default_factory=list)
    problems: list = field(default_factory=list)


class DeskReader:
    """The real desks, for the worker: the price, and per desk its state (the very one its judges read),
    its chart, its gamma walls and its confluence gate. Each state is built once per pass."""

    def __init__(self, m, analyses: dict, now_ms: int):
        self.m, self.analyses, self.now_ms = m, analyses, now_ms
        self.price = getattr(m.spot, "last", None)
        self._state: dict = {}

    def state(self, category: str) -> dict:
        if category not in self._state:
            from core.agents.live_recon import build_state
            from core.agents.recon_profiles import PROFILES
            self._state[category] = build_state(self.m, self.analyses, self.now_ms, PROFILES[category])
        return self._state[category]

    def frame(self, category: str):
        from core.agents.recon_profiles import PROFILES
        return self.m.spot.frames.get(PROFILES[category].timeframes[1])

    def gamma(self, category: str) -> dict | None:
        from core.indicators import desk_derivs
        return desk_derivs.gamma_for(category, self.m)

    def gate(self, category: str):
        """(framework, gate) for the desk, as the summary computes them."""
        from core import confluence
        from core.layman_summary import framework
        a = self.analyses.get(category)
        if a is None:
            raise ValueError(f"no analysis for the {category} desk")
        f = framework(a)
        ev = evidence_for(category, self.m, self.analyses, self.state(category), self.now_ms)
        return f, confluence.read(f, a, ev, self.now_ms)


def evidence_for(category: str, m, analyses: dict, state: dict, now_ms: int):
    """The gate's evidence for a desk, built from core alone the way the page builds it (app._desk_evidence):
    the desk's harmonics, its break reads, its liquidity, this week's prints, the dollar and yields, the
    crowd and the resting liquidation clusters."""
    from core.agents.context import _rates
    from core.agents.recon_profiles import PROFILES
    from core.confluence import HIGHER, Evidence
    from core.indicators import desk_derivs
    from core.indicators.event_trap import read_all as event_reads
    from core.indicators.harmonics import find_all
    from core.indicators.vwap_break import read_all as break_reads
    tfs = PROFILES[category].timeframes
    # the higher-timeframe bar reads the levels of the desks above this one, never its own
    htf = tuple((f"{k} level", float(lv.price)) for k in HIGHER.get(category, ()) if analyses.get(k) is not None
                for lv in analyses[k].levels)
    try:
        own = desk_derivs.read(category, m, now_ms)
    except Exception:  # noqa: BLE001 - the gate then reads no derivatives, a family that says nothing
        own = None
    desk = analyses.get(category)
    return Evidence(
        harmonics=tuple(h for found in find_all(m.spot.frames, tfs, desk_derivs.gamma_for(category, m)).values()
                        for h in found),
        htf_levels=htf, derivs=own,
        fear_greed=m.sentiment.fear_greed if m.sentiment.available else None,
        price_change_24h_pct=desk.price_change_24h_pct if desk is not None else None,
        oi_history=tuple(desk_derivs.oi_history(m)),
        breaks=tuple(break_reads(m.spot.frames, tfs).values()),
        liquidity=(state or {}).get("liquidity"),
        prints=tuple(event_reads(m.spot.frames.get("1h"), m.calendar.events if m.calendar.available else [], now_ms)),
        rates=_rates(m),
        futures=m.futures if m.futures.available else None,
        liq_levels=m.liq_levels.to_state() if m.liq_levels.available else None)


def current_master(store) -> dict | None:
    for t in store.trades(status="open", limit=50):
        if (t.get("plan") or {}).get("master"):
            return t
    return None


def _in_zone(price: float, zone: tuple[float, float]) -> bool:
    """Inside the zone, or within one zone-width beyond either edge. The zone is a quarter range either side
    of the level (core.layman_summary.ZONE_ATR), so the reach is the gate's own "at the zone" pad and never
    the desk's whole range."""
    half = (zone[1] - zone[0]) / 2
    return zone[0] - half <= price <= zone[1] + half


def _tradeable(side: str, price: float, invalidation: float | None, targets: list[float]) -> bool:
    """A card to open at market: the invalidation on the far side of the entry and every target ahead of it."""
    if invalidation is None or not targets:
        return False
    if side == LONG:
        return invalidation < price and min(targets) > price
    return invalidation > price and max(targets) < price


def open_master(store, desks, now_ms: int) -> int | None:
    """Open the master card if a swing desk's gate says TRADE with price in its zone and none is open: entry
    the live price, the desk's invalidation the hard stop, the desk's own targets."""
    from core.agents.recon_profiles import PROFILES
    if current_master(store) is not None or not desks.price:
        return None
    price = float(desks.price)
    for cat in SWING_DESKS:
        f, g = desks.gate(cat)
        if g.verdict != "TRADE" or f.zone is None or f.side not in (LONG, SHORT) or not _in_zone(price, f.zone):
            continue
        tps = [float(t) for t in (f.targets[0], f.targets[1], f.target3) if t is not None]
        if not _tradeable(f.side, price, f.invalidation, tps):
            continue
        entry, inv = round(price, 2), float(f.invalidation)
        st = desks.state(cat)
        plan, verdict, _ = deterministic_plan(store, st, desks.frame(cat), f.side, entry, inv, cat, now_ms,
                                              gamma=desks.gamma(cat), asset=ASSET)
        risk = abs(entry - inv)
        targets = [trades.TimedTarget(f"TP{i}", t, f"the {cat} desk's target", None,
                                      round(abs(t - entry) / risk, 2) if risk else 0.0)
                   for i, t in enumerate(tps, start=1)]
        schedule = trades.monitoring_schedule(now_ms, PROFILES[cat].timeframes, targets)
        plan.update({
            "master": True, "desk": cat, "master_rules": RULES,
            "gate": {"verdict": g.verdict, "reason": g.reason, "zone": [f.zone[0], f.zone[1]], "invalidation": inv},
            "targets": [{"name": t.name, "price": t.price, "basis": t.basis, "due_ms": t.due_ms,
                         "due_from": t.due_from, "reward_r": t.reward_r} for t in targets],
            "schedule": [{"timeframe": c.timeframe, "at_ms": c.at_ms, "reason": c.reason} for c in schedule],
            "stop_verdict": "hard invalidation",
            "stop_reasons": [f"The {cat} desk's invalidation at ${inv:,.0f} is the hard stop, as his Section 6 asks. "
                             f"The verifier read it as '{verdict.verdict}'"
                             + (f": {'; '.join(str(r) for r in verdict.reasons)}" if verdict.reasons else ".")],
        })
        plan["notes"] = [f"Master swing card: opened by the {cat} desk's confluence gate ({g.reason}) with price "
                         "inside the zone. Held until its targets, its hard invalidation, or the desk turning. "
                         "No agents were consulted."] + list(plan.get("notes") or [])
        tid = store.open_trade(opened_ms=now_ms, asset=ASSET, side=f.side, entry=entry, stop=inv, given_stop=inv,
                               stop_verdict="hard invalidation", category=cat, plan=plan)
        tp = ", ".join(f"{t.name} ${t.price:,.0f}" for t in targets)
        store.add_trade_event(tid, now_ms, EVENT, f"master:opened:{tid}",
                              f"🧭 Master swing {f.side} opened by the {cat} desk · entry ${entry:,.0f} · "
                              f"hard stop ${inv:,.0f} · {tp}. {g.reason}", price)
        log.info("master card #%s opened: %s %s by the %s desk", tid, f.side, entry, cat)
        return tid
    return None


def review_masters(store, desks, now_ms: int) -> list[int]:
    """Close the master card(s) whose hard invalidation or last target has been reached, or whose desk now
    trades the other way. The trader's own cards are never touched."""
    closed: list[int] = []
    price = desks.price
    for row in store.trades(status="open", limit=50):
        plan = row.get("plan") or {}
        if not plan.get("master"):
            continue
        reason = None
        if plan.get("master_rules") != RULES:
            reason = ("superseded: opened under the previous master-card rules; the desk reopens it on the next "
                      "pass if its gate still calls for it")
        elif price:
            targets = [trades.TimedTarget(t.get("name", ""), float(t.get("price") or 0.0), str(t.get("basis") or ""),
                                          t.get("due_ms"), float(t.get("reward_r") or 0.0), str(t.get("due_from") or ""))
                       for t in plan.get("targets") or []]
            prog = trades.progress(row["side"], float(row["entry"]), float(row["stop"]), targets, float(price))
            if prog.stopped:
                reason = f"hard invalidation: price reached the stop at ${float(row['stop']):,.0f}"
            elif targets and len(prog.hit) == len(targets):
                reason = f"all targets reached ({', '.join(prog.hit)})"
        if reason is None:
            cat = str(plan.get("desk") or row.get("category") or "")
            f, g = desks.gate(cat)
            if g.verdict == "TRADE" and f.side in (LONG, SHORT) and f.side != row["side"]:
                reason = f"the {cat} desk now trades the other way ({f.side}): {g.reason}"
        if reason:
            store.close_trade(row["id"], reason, price, now_ms)
            store.add_trade_event(row["id"], now_ms, EVENT, f"master:closed:{row['id']}",
                                  f"🧭 Master swing {row['side']} closed · {reason}", price)
            log.info("master card #%s closed: %s", row["id"], reason)
            closed.append(row["id"])
    return closed


def master_pass(store, desks, now_ms: int, result: MasterResult | None = None) -> MasterResult:
    """One worker pass: review what is open, then open if a desk calls for it. A desk that cannot be read is
    a problem on the pass, never a crash; a card closed this pass leaves the next pass to open anew."""
    r = result or MasterResult()
    try:
        r.closed = review_masters(store, desks, now_ms)
    except Exception as e:  # noqa: BLE001
        r.problems.append(f"master review: {type(e).__name__}: {e}")
    if not r.closed:
        try:
            if open_master(store, desks, now_ms) is not None:
                r.opened += 1
        except Exception as e:  # noqa: BLE001
            r.problems.append(f"master open: {type(e).__name__}: {e}")
    return r
