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
                   "anchor_log", "calls", "reports"})
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


TOOLS = {"todays_plan": todays_plan, "plan_status": plan_status, "forecast_record": forecast_record,
         "desk_anchors": desk_anchors, "recent_alerts": recent_alerts, "monday_plan": monday_plan}
__all__ = ["TOOLS", "READS", "DAY_MS", *TOOLS]
