"""What the MCP server (mcp_server.py) lets an AI assistant read: the shared store, as the dashboard shows it.

Every function here takes the store and returns a JSON-safe dict built from rows the 24/7 worker already
wrote - the day's Quant Forecast plan, its status and alerts, the record, the desks' anchors, the week's
Monday plan. Nothing is fetched from an exchange and no forecast is built or scored here: an assistant
reads what the code produced, it never computes a number of its own (the house rule since message 5). The
store methods these tools call are listed in READS; the test holds them to that list, so a tool that
started writing would fail the suite before it reached the server."""
from __future__ import annotations

import datetime as dt
import json
import time

from core import accuracy, qf_alerts, qf_payload, qf_selfaudit, qf_weekly
from core.anchors import horizon_ms_for
from core.config import CATEGORIES
from core.quant_forecast import DAY_MS, day_start

READS = frozenset({"qf_plans", "qf_week", "market_alerts_keyed", "market_alerts_recent", "book_report", "get_anchor",
                   "anchor_log", "calls", "reports", "report_body", "traps", "trades", "trade_events", "live_anchors",
                   "get_live_anchor", "side_notes", "recon_scores"})
RECORD_DAYS = 30
ALERT_LIMIT = 20


def _now(now_ms: int | None) -> int:
    return int(now_ms) if now_ms is not None else int(time.time() * 1000)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return dt.datetime.fromtimestamp(int(ms) / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(obj):
    """Plain JSON: dataclasses, numpy scalars and timestamps become strings or numbers."""
    return json.loads(json.dumps(obj, default=_default))


def _default(o):
    if hasattr(o, "__dataclass_fields__"):
        return {k: getattr(o, k) for k in o.__dataclass_fields__}
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def parse_day(day_utc: str | None, now_ms: int) -> int:
    """A 'YYYY-MM-DD' date to its 00:00 UTC day start; today's when empty."""
    if not day_utc:
        return day_start(now_ms)
    d = dt.datetime.strptime(day_utc.strip(), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def _alerts_for(store, day_ms: int) -> list[dict]:
    rows = store.market_alerts_keyed(f"qf:{day_ms}:", limit=ALERT_LIMIT)
    return sorted(rows, key=lambda r: (int(r["ts_ms"]), int(r.get("id") or 0)))


def _alert_kind(row: dict) -> str:
    parts = str(row.get("dedup_key") or "").split(":")
    return parts[3] if len(parts) > 3 else str(row.get("kind") or "")


def tier_status(row: dict, sent: list[dict]) -> str:
    """The plan's tier word as the wire panel shows it, read from the row and the alerts the bot sent for
    it (the page reads the candles for the same word; the server reads no candles)."""
    if row.get("status") == "invalidated":
        return "invalidated"
    if (row.get("payload") or {}).get("regime") == "locked":
        return "locked"
    if row.get("outcome"):
        return "invalidated" if row["outcome"].get("invalidated") else "on track"
    mine = [a for a in sent if str(a.get("dedup_key") or "").startswith(f"qf:{row['day_ms']}:{row['version']}:")]
    last = _alert_kind(mine[-1]) if mine else ""
    return {"caution": "caution", "held": "model held"}.get(last, "on track")


def _row_summary(r: dict) -> dict:
    p = r.get("payload") or {}
    return {"id": r.get("id"), "version": r["version"], "status": r["status"], "regime": p.get("regime"),
            "created_utc": _iso(r.get("created_ms")), "closed_utc": _iso(r.get("closed_ms")),
            "closed_reason": r.get("closed_reason"), "outcome": r.get("outcome")}


def todays_plan(store, day_utc: str | None = None, now_ms: int | None = None) -> dict:
    """The day's Quant Forecast plan in the schema-v7 JSON the dashboard offers for download, plus a small
    `store` block (the row it came from and the day's other plans). The active plan when the day has one,
    else the day's last plan (a past day: the one that finished it)."""
    now = _now(now_ms)
    day = parse_day(day_utc, now)
    rows = store.qf_plans(day)
    if not rows:
        return {"available": False, "day_utc": _iso(day), "reason": "no plan for this day (a plan is made on the "
                "first pass after 00:00 UTC that has a live price and three weeks of hourly candles)"}
    row = next((r for r in reversed(rows) if r["status"] == "active"), rows[-1])
    sent = _alerts_for(store, day)
    status = tier_status(row, sent)
    week = store.qf_week(qf_weekly.week_start(day))
    out = qf_payload.payload(row["payload"], status, rows, sent[-1]["headline"] if sent else "",
                             qf_alerts.live_record(store), now, None, store.book_report(),
                             week=week["payload"] if week else None, self_audit=qf_selfaudit.latest(store, now))
    out["store"] = {"day_utc": _iso(day), "served_from": _row_summary(row),
                    "plans_this_day": [_row_summary(r) for r in rows]}
    return _clean(out)


def plan_status(store, now_ms: int | None = None) -> dict:
    """Where today's plan stands: each plan row of the day (version 0 the morning plan, 1 Scenario B), the
    active plan's tier word, its compact timeline, which window is live, its directive and one-line wire,
    and every alert the bot sent for the day, oldest first."""
    now = _now(now_ms)
    day = day_start(now)
    rows = store.qf_plans(day)
    sent = _alerts_for(store, day)
    out = {"day_utc": _iso(day), "now_utc": _iso(now), "plans": [_row_summary(r) for r in rows],
           "alerts_today": [{"time_utc": _iso(a["ts_ms"]), "kind": _alert_kind(a), "headline": a["headline"]} for a in sent]}
    active = next((r for r in reversed(rows) if r["status"] == "active"), None)
    if active is None:
        out["active"] = None
        out["note"] = ("no plan yet today" if not rows else "the day's last plan is closed; the next plan comes at 00:00 UTC")
        return _clean(out)
    p = active["payload"]
    out["active"] = {"version": active["version"], "regime": p.get("regime"), "tier_status": tier_status(active, sent),
                     "timeline": qf_alerts.timeline_line(p, now), "window": qf_alerts.window_state(p, now),
                     "wire": qf_alerts.wire_line(p) if hasattr(qf_alerts, "wire_line") else qf_payload.qf.wire_line(p),
                     "directive": qf_alerts.directive_line(p),
                     "projected_high": p.get("projected_high"), "projected_low": p.get("projected_low"),
                     "invalidation": p.get("invalidation"), "magnitude_scaler": p.get("magnitude"),
                     "calibration": p.get("calibration"), "gravity_shadow": (p.get("gravity") or {}).get("shadow")}
    return _clean(out)


def _ledger(store) -> dict:
    s = accuracy.summary(store)
    desks = {}
    for key, h in s["by_category"].items():
        cat = CATEGORIES.get(key)
        desks[key] = {"desk": cat.label if cat else key, "horizon": cat.horizon_label if cat else None,
                      "scored_calls": h.n, "hits": h.hits,
                      "hit_rate": None if h.rate is None else round(h.rate, 3),
                      "chance_rate": None if h.chance is None else round(h.chance, 3)}
    return {"desks": desks, "open_calls": s["open_calls"], "total_scored": s["total_scored"],
            "reports_24h": {k: {"n": h.n, "hits": h.hits} for k, h in s["by_classification_24h"].items()},
            "reports_7d": {k: {"n": h.n, "hits": h.hits} for k, h in s["by_classification_7d"].items()},
            "method": "a desk call is scored once per horizon unless its direction changes; the chance rate is "
                      "what a coin with the same rule would score, so compare hit_rate with chance_rate, not with 50%"}


def forecast_record(store, now_ms: int | None = None) -> dict:
    """How the Quant Forecast has done: the 60-day backtest counts carried by the newest morning plan, the
    morning plans' live record over the last 10 and 30 scored days, each of those days in brief, and the
    desks' accuracy ledger with its chance baseline."""
    now = _now(now_ms)
    rows = store.qf_plans(limit=200)
    mornings = [r for r in rows if r["version"] == 0]
    backtest = next((r["payload"].get("backtest") for r in mornings if (r.get("payload") or {}).get("backtest")), None)
    scored = [r for r in mornings if r.get("outcome")]
    days = []
    for r in scored[:RECORD_DAYS]:
        o, p = r["outcome"], r["payload"]
        days.append({"day_utc": _iso(r["day_ms"]), "regime": p.get("regime"), "status": r["status"],
                     "projected_high": (p.get("projected_high") or {}).get("price"),
                     "projected_low": (p.get("projected_low") or {}).get("price"),
                     "actual_high": o.get("high"), "actual_low": o.get("low"), "close": o.get("close"),
                     **{k: o.get(k) for k in ("scored", "direction_hit", "scenario_a_right", "high_hit", "low_hit",
                                              "high_time_hit", "low_time_hit", "band_held", "invalidated",
                                              "shadow_high_hit", "shadow_low_hit") if k in o}})
    return _clean({"as_of_utc": _iso(now), "backtest_60d": backtest,
                   "live_last_10_days": qf_alerts.live_record(store, 10), "live_last_30_days": qf_alerts.live_record(store, 30),
                   "scored_days": days, "desk_ledger": _ledger(store),
                   "how_to_read": "direction days and range days are counted apart (a range day is right whenever its "
                                  "box holds); shadow_* counts are liquidation gravity's projection judged beside the "
                                  "live plan on the same days - it does not steer the live plan"})


def desk_anchors(store, now_ms: int | None = None) -> dict:
    """The four desks' anchored verdicts as the store holds them (the direction a desk is anchored to, what
    it would say live, since when) and the last re-anchors with their triggers."""
    now = _now(now_ms)
    desks = {}
    for key, cat in CATEGORIES.items():
        a = store.get_anchor(key)
        desks[key] = {"desk": cat.label, "horizon": cat.horizon_label, "horizon_hours": horizon_ms_for(key) / 3_600_000,
                      "anchored_since_utc": _iso(a["since_ms"]) if a else None, "anchor": a["payload"] if a else None}
    changes = [{"time_utc": _iso(r["ts_ms"]), "desk": r["category"], "from": r.get("from_dir"), "to": r["to_dir"],
                "price": r.get("price"), "triggers": r.get("triggers")} for r in store.anchor_log(limit=20)]
    return _clean({"as_of_utc": _iso(now), "desks": desks, "recent_reanchors": changes})


def recent_alerts(store, limit: int = ALERT_LIMIT, now_ms: int | None = None) -> dict:
    """The last alerts the worker sent to the channel, newest first: Quant Forecast tiers, cascade and TRAP
    alerts, the Market Pulse, trade alerts."""
    now = _now(now_ms)
    rows = store.market_alerts_recent(limit=max(1, min(int(limit), 100)))
    return _clean({"as_of_utc": _iso(now),
                   "alerts": [{"time_utc": _iso(r["ts_ms"]), "kind": r["kind"], "key": r["dedup_key"], "headline": r["headline"],
                               "price": r.get("price"), "sent": bool(r.get("notified"))} for r in rows]})


def monday_plan(store, now_ms: int | None = None) -> dict:
    """This week's Sticky Monday Master Plan: the weekly bias, the dealers' put and call walls, the baseline
    floor and ceiling, written once on the first pass after Monday 00:00 UTC."""
    now = _now(now_ms)
    week = qf_weekly.week_start(now)
    row = store.qf_week(week)
    if not row:
        return {"available": False, "week_utc": _iso(week), "reason": "no plan written for this week yet"}
    return _clean({"available": True, "week_utc": _iso(week), "written_utc": _iso(row.get("created_ms")), **row["payload"]})


def _limit(n, cap: int) -> int:
    try:
        return max(1, min(int(n), cap))
    except (TypeError, ValueError):
        return cap


def _desk(key: str | None) -> dict:
    cat = CATEGORIES.get(key or "")
    return {"desk_label": cat.label if cat else key, "horizon": cat.horizon_label if cat else None}


def _bool(v):
    return None if v is None else bool(v)


def _loads(v, default):
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v) if v else default
    except (TypeError, ValueError):
        return default


def _report_row(r: dict) -> dict:
    return {"report_id": r["report_id"], "time_utc": _iso(r["ts_ms"]), "mode": r.get("mode"),
            "classification": r.get("classification"), "price": r.get("price"), "sigma24_pct": r.get("sigma24_pct"),
            "majority_direction": r.get("majority_dir"), "plan": r.get("plan"), "plan_outcome": r.get("plan_outcome"),
            "plan_scored_utc": _iso(r.get("plan_scored_ms")),
            "scored": {"move_24h_pct": r.get("move_24h_pct"), "hit_24h": _bool(r.get("hit_24h")),
                       "move_7d_pct": r.get("move_7d_pct"), "hit_7d": _bool(r.get("hit_7d"))}}


def trap_reports(store, limit: int = 20, now_ms: int | None = None) -> dict:
    """The Trap Intelligence reports, newest first, without their bodies: id, time, pipeline (1-agent or
    15-agent), classification, price, 24h sigma, majority direction, the plan line and how the report
    scored at 24 hours and 7 days. `trap_report` reads one body."""
    now = _now(now_ms)
    rows = store.reports(limit=_limit(limit, 100))
    return _clean({"as_of_utc": _iso(now), "reports": [_report_row(r) for r in rows],
                   "how_to_read": "hit_24h / hit_7d: did price move the way the report's majority said over that "
                                  "horizon; plan_outcome: what became of the report's trade plan (target, stop, "
                                  "cancelled = the setup broke before any confirming close, not_triggered, no_trade, "
                                  "open); the Pulse speaks from a plan only while it is open"})


def trap_report(store, report_id: str | None = None, mode: str | None = None, now_ms: int | None = None) -> dict:
    """One Trap Intelligence report in full: every desk brief with its raw metrics, interpretation and
    traps detected, the bullish and bearish desks' summaries, the director's classification, scenarios,
    contradictions, blind spots and trade plan, the category summaries, the raw metric table and the
    report's own plan. `report_id` names one; empty reads the newest (of `mode` when given)."""
    now = _now(now_ms)
    r = store.report_body(report_id or None, mode or None)
    if not r:
        want = f"report {report_id}" if report_id else f"a {mode} report" if mode else "a report"
        return {"available": False, "as_of_utc": _iso(now), "reason": f"no body saved for {want}"}
    body = dict(r["payload"] or {})
    body.pop("runs", None)                      # model latencies and token counts: not part of the analysis
    return _clean({"available": True, "as_of_utc": _iso(now), **_report_row(r), "report": body})


def desk_calls(store, desk: str | None = None, limit: int = 50, now_ms: int | None = None) -> dict:
    """The desks' direction calls, newest first, each with its score once its horizon has passed (the
    realised move and whether the call was right), plus counts per desk over the rows returned.
    `desk` is live, intraday, weekly or monthly; empty is all four."""
    now = _now(now_ms)
    lim = _limit(limit, 500)
    rows = store.calls(limit=2000 if desk else lim)
    if desk:
        rows = [r for r in rows if r.get("category") == desk][:lim]
    calls, per = [], {}
    for r in rows:
        scored = r.get("scored_ms") is not None
        calls.append({"time_utc": _iso(r["ts_ms"]), "desk": r["category"], **_desk(r["category"]),
                      "direction": r["direction"], "confidence": r.get("confidence"), "price": r.get("price"),
                      "sigma_pct": r.get("sigma_pct"), "horizon_hours": int(r["horizon_ms"]) / 3_600_000,
                      "scored_utc": _iso(r.get("scored_ms")), "realised_pct": r.get("realised_pct"),
                      "hit": _bool(r.get("hit")) if scored else None})
        p = per.setdefault(r["category"], {"calls": 0, "scored": 0, "hits": 0})
        p["calls"] += 1
        if scored:
            p["scored"] += 1
            p["hits"] += 1 if r.get("hit") else 0
    return _clean({"as_of_utc": _iso(now), "desk": desk, "calls": calls, "per_desk": per,
                   "how_to_read": "a call is scored once at its horizon; for hit rates against chance use "
                                  "forecast_record.desk_ledger, which applies the same rule to every desk"})


def _horizon(timeframe) -> str:
    from core.traps import horizon
    return horizon(timeframe)


def traps(store, status: str | None = None, limit: int = 50, now_ms: int | None = None) -> dict:
    """The TRAP desk's declared traps, newest first: side (bull_trap / bear_trap), timeframe, the level,
    where it is invalidated and where it plays out, the score and head call, the evidence lines, and how
    it was settled. `status` is active, played_out, invalidated or expired; empty is all."""
    now = _now(now_ms)
    rows = store.traps(status=status or None, limit=_limit(limit, 200))
    out = [{"id": r["id"], "declared_utc": _iso(r["declared_ms"]), "declared_by": r["category"], "side": r["side"],
            "timeframe": r["timeframe"], "horizon": _horizon(r["timeframe"]), "level": r["level"], "invalidation": r["invalidation"], "plays_out": r["plays_out"],
            "score": r.get("score"), "head_call": r.get("head_call"), "evidence": _loads(r.get("evidence"), []),
            "notes": _loads(r.get("notes"), []), "status": r["status"], "resolved_utc": _iso(r.get("resolved_ms")),
            "resolved_price": r.get("resolved_price"), "detail": _loads(r.get("detail"), None)} for r in rows]
    return _clean({"as_of_utc": _iso(now), "status": status, "traps": out,
                   "how_to_read": "a trap is a market event, not a desk's opinion: the same raid is one trap whichever "
                                  "desk declared it; invalidation sits half an ATR past the level; horizon short = a 15m or "
                                  "1h trap, shown and traded on the Live desk only, never in a desk summary or the Pulse; "
                                  "horizon swing = 4h and longer, read by every desk on that candle"})


def trade_journal(store, status: str | None = None, limit: int = 20, now_ms: int | None = None) -> dict:
    """The pinned trades: side, entry, the stop the desk set and the one the trader gave with the desk's
    verdict on it, the plan written at open (never rewritten), status and close, and every event logged
    on the trade. `status` is open or closed; empty is all."""
    now = _now(now_ms)
    rows = store.trades(status=status or None, limit=_limit(limit, 100))
    trades = []
    for r in rows:
        events = [{"time_utc": _iso(e["ts_ms"]), "kind": e["kind"], "headline": e["headline"], "price": e.get("price"),
                   "sent": bool(e.get("notified"))} for e in store.trade_events(trade_id=r["id"], limit=100)]
        trades.append({"id": r["id"], "opened_utc": _iso(r["opened_ms"]), "asset": r["asset"], "side": r["side"],
                       "entry": r["entry"], "stop": r["stop"], "given_stop": r.get("given_stop"),
                       "stop_verdict": r.get("stop_verdict"), "desk": r["category"], **_desk(r["category"]),
                       "status": r["status"], "closed_utc": _iso(r.get("closed_ms")), "closed_reason": r.get("closed_reason"),
                       "closed_price": r.get("closed_price"), "last_refresh_utc": _iso(r.get("refreshed_ms")),
                       "plan": r.get("plan") or {}, "events": events})
    return _clean({"as_of_utc": _iso(now), "status": status, "trades": trades,
                   "how_to_read": "the plan is the thesis at open and does not change; read the events for what has "
                                  "happened since; an empty list means no trade is pinned"})


def order_book(store, now_ms: int | None = None) -> dict:
    """The worker's latest order-book report: depth around the mid (best bid and ask, spread, resting size
    within 0.10 / 0.25 / 0.50 / 1.00 percent, the imbalance), the sweeps and scoops seen in the window,
    and the method that defines them. This is what the Market Pulse reads."""
    now = _now(now_ms)
    rep = store.book_report()
    if not rep:
        return {"available": False, "as_of_utc": _iso(now), "reason": "the worker has not written an order-book report yet"}
    rest = {k: v for k, v in rep.items() if k not in ("ts_ms", "as_of_ms")}
    return _clean({"available": True, "as_of_utc": _iso(now), "report_utc": _iso(rep.get("ts_ms")),
                   "book_as_of_utc": _iso(rep.get("as_of_ms")), **rest})


def recon_windows(store, desk: str = "live", limit: int = 5, now_ms: int | None = None) -> dict:
    """A desk's last anchored windows with the full reasoning behind each: the bias and margin, the bull
    and bear points by domain (quant, auction, delta, ICT, derivatives) and by item, the trap
    probabilities, the handshake with the other desks, the side notes added during the window, and the
    score once the window closed. `desk` is live, intraday, weekly or monthly."""
    now = _now(now_ms)
    desk = desk or "live"
    heads = store.live_anchors(limit=_limit(limit, 50), category=desk)
    scores = {int(s["window_open_ms"]): s for s in store.recon_scores(category=desk, limit=200)}
    windows = []
    for h in heads:
        w = int(h["window_open_ms"])
        full = store.get_live_anchor(w, category=desk)
        sc = scores.get(w)
        windows.append({"window_open_utc": _iso(w), "window_close_utc": _iso(h.get("window_close_ms")),
                        "published_utc": _iso(h.get("published_ms")), "bias": h.get("bias"), "margin": h.get("margin"),
                        "confidence": h.get("confidence"), "trap": h.get("trap"), "price": h.get("price"),
                        "reasoning": (full or {}).get("payload"),
                        "side_notes": [{"time_utc": _iso(n["ts_ms"]), "kind": n["kind"], "headline": n["headline"],
                                        "detail": n.get("detail"), "price": n.get("price")}
                                       for n in store.side_notes(w, category=desk)],
                        "score": None if not sc else {
                            "scored_utc": _iso(sc["scored_ms"]), "open_price": sc["open_price"], "close_price": sc["close_price"],
                            "move_pct": sc["move_pct"], "realised": sc["realised"], "direction_hit": _bool(sc["direction_hit"]),
                            "trap_hit": _bool(sc.get("trap_hit")), "bull_brier": sc.get("bull_brier"), "bear_brier": sc.get("bear_brier")}})
    return _clean({"as_of_utc": _iso(now), "desk": desk, **_desk(desk), "windows": windows,
                   "how_to_read": "margin is bull points minus bear points on the rubric; the anchor is pinned at the "
                                  "window open and scored once at its close; side notes are what changed meanwhile"})


TOOLS = {"todays_plan": todays_plan, "plan_status": plan_status, "forecast_record": forecast_record,
         "desk_anchors": desk_anchors, "recent_alerts": recent_alerts, "monday_plan": monday_plan,
         "trap_reports": trap_reports, "trap_report": trap_report, "desk_calls": desk_calls, "traps": traps,
         "trade_journal": trade_journal, "order_book": order_book, "recon_windows": recon_windows}
__all__ = ["TOOLS", "READS", "DAY_MS", *TOOLS]
