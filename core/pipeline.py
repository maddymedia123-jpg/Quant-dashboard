"""The work every pass does whoever runs it: the dashboard on a page load, or the 24/7 worker each minute.

Alerting, scoring and trap settlement used to live inside app.py, so they only ran while somebody had the
page open - a 4h close, a stop or a TRAP call at night went unseen. They are here now, called by both,
so the page and the worker cannot drift apart. Two senders are safe because every send goes through the
store's one-statement claim (UPDATE ... RETURNING): whichever claims an alert first sends it, the other
finds nothing.

Each step fails on its own and is reported, never raised: a pass that stops halfway would skip the alerts
after the failing step."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.alerts import MAX_PER_PASS, Dispatcher, Event, detect_events

log = logging.getLogger(__name__)


@dataclass
class PassResult:
    verdicts: dict = field(default_factory=dict)
    settled: list = field(default_factory=list)
    sent: int = 0
    problems: list[str] = field(default_factory=list)


def ledger(store, m, analyses: dict, now_ms: int, classification: str | None = None,
           result: PassResult | None = None) -> PassResult:
    """Anchor each desk's call, score what is due, settle the traps price has answered, and score every
    war-room window whose candle has closed."""
    from core.accuracy import score_due_calls, score_plans, score_reports
    from core.agents.recon_profiles import PROFILES
    from core.agents.trap_desk import settle
    from core.anchors import anchor_step
    from core import live_accuracy

    r = result or PassResult()
    px = m.spot.last or (next(iter(analyses.values())).price if analyses else None)
    try:
        for key, a in analyses.items():
            r.verdicts[key] = anchor_step(store, key, a, m, classification, now_ms)
        if analyses and px:
            score_due_calls(store, now_ms, px)
            score_reports(store, now_ms, px)
            score_plans(store, m.spot.frames, now_ms)
    except Exception as e:  # noqa: BLE001 - one step's failure must not skip the rest of the pass
        r.problems.append(f"ledger: {type(e).__name__}: {e}")
    try:
        if px:
            r.settled = settle(store, float(px), now_ms)
    except Exception as e:  # noqa: BLE001
        r.problems.append(f"traps: {type(e).__name__}: {e}")
    try:
        for profile in PROFILES.values():
            live_accuracy.score_due(store, m.spot.frames, now_ms, profile)
    except Exception as e:  # noqa: BLE001
        r.problems.append(f"accuracy: {type(e).__name__}: {e}")
    return r


def market_events(store, m, analyses: dict, now_ms: int) -> list[Event]:
    """Plan alerts for the newest 1-agent plan, a desk's cascade only into A or C once it has held, TRAP
    calls, and the Market Pulse (once per report, once per 4h close)."""
    from core import market_alerts as ma
    from core import plan_alerts as pa
    from core import pulse
    from core.agents.recon_profiles import PROFILES
    from core.config import CATEGORIES
    from core.indicators.cascade import classify

    context, events = [], []
    for key in CATEGORIES:
        bias_tf, anchor_tf, engine_tfs = PROFILES[key].cascade
        c = classify(m.spot.frames, bias_tf, anchor_tf, engine_tfs)
        track = store.track_cascade(key, c.state, now_ms)
        e = ma.important_cascade_event(key, PROFILES[key].label, track, c, now_ms, m.spot.last,
                                       recent=store.market_alerts_keyed(f"cascade:{key}:", limit=20))
        if e is not None:
            events.append(e)
        if key in ("live", "intraday") and c.state not in ma.SILENT_STATES:
            trend = c.bias.direction if c.bias is not None else None
            context.append(f"{PROFILES[key].label} {ma.state_name(c.state, trend).lower()}")
    plan_row = pa.latest_plan_report(store)
    events += pa.events(plan_row, m.spot.frames,
                        m.taker_flow.frames if m.taker_flow.available else None, now_ms, m.spot.last,
                        context="; ".join(context))
    events += ma.trap_events(store.traps(limit=200), now_ms)
    latest = store.reports(limit=1)
    rep_pulse = pulse.report_event(latest[0] if latest else None, analyses.get("live"), m, now_ms)
    # a fresh report's pulse stands in for the close's: one dispatch, not two back to back; a close while
    # the plan is live speaks from the plan, so both messages name the same target and safety line
    p = rep_pulse or pulse.close_event(analyses.get("live"), m, now_ms, plan_row=plan_row)
    if p is not None:
        events.append(p)
    return events


def market_alerts(store, m, analyses: dict, now_ms: int, secrets: dict, may_send: bool,
                  result: PassResult | None = None) -> PassResult:
    """Detect, log once (the ledger's unique key), then claim and send. Detection runs even when sending
    is off, so nothing is lost; the claim decides who sends."""
    r = result or PassResult()
    try:
        for e in market_events(store, m, analyses, now_ms):
            store.add_market_alert(now_ms, e.kind, e.dedup_key, e.headline, e.price)
    except Exception as e:  # noqa: BLE001 - the other detectors and the send step still run
        r.problems.append(f"market alerts: {type(e).__name__}: {e}")
    try:
        # the Quant Forecast's day plan: made, audited, replaced and scored here, its alerts logged with the rest
        from core import qf_alerts
        from core.indicators.desk_derivs import gamma_for
        for e in qf_alerts.run(store, m, now_ms, gamma_for("intraday", m)):
            store.add_market_alert(now_ms, e.kind, e.dedup_key, e.headline, e.price)
    except Exception as e:  # noqa: BLE001 - the forecast failing must not hold back the other alerts
        r.problems.append(f"quant forecast: {type(e).__name__}: {e}")
    if not may_send:
        return r
    pending = store.claim_market_alerts(now_ms, limit=MAX_PER_PASS)
    if not pending:
        return r
    delivered, problems = Dispatcher(secrets).send(
        [Event(p["kind"], p["dedup_key"], p["headline"], p.get("price")) for p in pending])
    keys = {e.dedup_key for e in delivered}
    store.mark_market_notified([p["id"] for p in pending if p["dedup_key"] in keys])
    store.release_market_alerts([p["id"] for p in pending if p["dedup_key"] not in keys])
    r.sent += len(keys)
    r.problems += problems[:1]
    return r


def trade_progress(row: dict, price: float | None):
    """A pinned trade's live numbers at this price, or None."""
    from core import trades
    if not price:
        return None
    plan = row.get("plan") or {}
    targets = [trades.TimedTarget(t.get("name", ""), float(t.get("price") or 0.0), str(t.get("basis") or ""),
                                  t.get("due_ms"), float(t.get("reward_r") or 0.0), str(t.get("due_from") or ""))
               for t in plan.get("targets") or []]
    return trades.progress(row["side"], float(row["entry"]), float(row["stop"]), targets, float(price))


def trade_alerts(store, row: dict, prog, now_ms: int, secrets: dict, may_send: bool,
                 result: PassResult | None = None) -> PassResult:
    """What has happened to one pinned trade: detect, log once, then claim and send."""
    from core import trades
    r = result or PassResult()
    plan = row.get("plan") or {}
    checkpoints = [trades.Checkpoint(c.get("timeframe", ""), int(c.get("at_ms") or 0), str(c.get("reason") or ""))
                   for c in plan.get("schedule") or []]
    traps = [type("T", (), {"trap_id": t.get("trap_id"), "side": t.get("side", ""),
                            "timeframe": t.get("timeframe", ""), "near": t.get("near", ""),
                            "distance": t.get("distance", 0.0)})() for t in plan.get("traps") or []]
    for e in detect_events(row, prog.price if prog else None, prog, traps, checkpoints, now_ms):
        # the unique index decides what is new: the same target reached is one event, not one per refresh
        store.add_trade_event(row["id"], now_ms, e.kind, e.dedup_key, e.headline, e.price)
    if not may_send:
        return r
    pending = store.claim_trade_events(row["id"], now_ms)
    if not pending:
        return r
    delivered, problems = Dispatcher(secrets).send(
        [Event(p["kind"], p["dedup_key"], p["headline"], p.get("price")) for p in pending])
    keys = {e.dedup_key for e in delivered}
    store.mark_notified([p["id"] for p in pending if p["dedup_key"] in keys])
    store.release_trade_events([p["id"] for p in pending if p["dedup_key"] not in keys])
    r.sent += len(keys)
    r.problems += problems[:1]
    return r


def all_trade_alerts(store, price: float | None, now_ms: int, secrets: dict, may_send: bool,
                     result: PassResult | None = None) -> PassResult:
    """Every open trade, as the Active Trade screen does for each pinned card."""
    r = result or PassResult()
    for row in store.trades(status="open", limit=50):
        try:
            prog = trade_progress(row, price)
            if prog is not None:
                trade_alerts(store, row, prog, now_ms, secrets, may_send, r)
        except Exception as e:  # noqa: BLE001 - one trade's failure must not silence the others
            r.problems.append(f"trade #{row.get('id')}: {type(e).__name__}: {e}")
    return r
