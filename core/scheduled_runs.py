"""Scheduled agent runs (election-day fix 6, owner's GO ALL, 5 November).

Until now the head agents ran only when a person clicked Run Analysis: on 4 November nothing with judgment
looked at the market for 14 hours while BTC ran 78,336 to 81,194. The 24/7 worker now runs the 1-agent
Director and the Live war room on its own, twice over:

* after a High-impact USD print has settled (PRINT_SETTLE_MS after the release, so the first 15-minute
  candle of the reaction has closed), and
* after a Quant Forecast day plan is invalidated (the whole-day line broken by a 4-hour close or two
  hourly closes).

Each trigger runs once (its key is a row in the market_alerts ledger, written before the run, so two
processes cannot both run it), never within RUN_GAP_MS of the last run, and never more than MAX_RUNS_PER_DAY
in a day: a handful of model runs, not a stream. A trigger that is throttled is recorded as skipped and not
retried. The run's ledger row is its record, not its announcement (wording batch 5, 6 Nov): a run that
produced a report sends nothing of its own, because the report's own Market Pulse and plan alerts follow
from the report ledger as they do for a report run by hand; a run the model refused sends NO NEW REPORT,
with the plan still in force and when the next automatic run can come. No model names or error codes
reach the channel; those go to the pass's problems and the log.

Nothing here runs on fixture data or without a model key, and no failure here can stop the worker's pass.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

PRINT_SETTLE_MS = 15 * 60_000          # the first 15m candle after the print has closed
PRINT_STALE_MS = 2 * 3_600_000         # a print older than this (a reboot, an outage) is not re-run
INVALID_STALE_MS = 2 * 3_600_000       # the same for an invalidation
RUN_GAP_MS = 30 * 60_000               # never two runs within half an hour
MAX_RUNS_PER_DAY = 6
DAY_MS = 86_400_000
KEY = "run:"
KIND = "run"


def _utc(ms: int) -> str:
    d = datetime.fromtimestamp(ms / 1000, timezone.utc)
    return f"{d:%H:%M} UTC ({datetime.fromtimestamp(ms / 1000 + 5 * 3600, timezone.utc):%H:%M} PKT)"


CAP_MARK = "automatic runs already ran today"     # in the cap row's text; due() must not count that row as a run


def _head(what: str, price) -> str:
    return f"🤖 {what}" + (f" · BTC ${float(price):,.0f}" if price else "")


def _plan_word(side) -> str:
    return {"SHORT": "sell", "LONG": "buy"}.get(str(side or "").upper(), "none")


def _is_skip(row: dict) -> bool:
    h = str(row.get("headline", ""))
    return h.startswith("skipped") or CAP_MARK in h    # "skipped …" is the shape the rows had before 6 Nov


def _next_after_cap(recent, now_ms: int) -> str:
    """When the oldest of today's runs falls out of the 24-hour window."""
    from core.trades import fmt_when
    today = [int(r.get("ts_ms") or 0) for r in recent or []
             if not _is_skip(r) and now_ms - int(r.get("ts_ms") or 0) < DAY_MS]
    return fmt_when(min(today) + DAY_MS) if today else fmt_when(now_ms)


def _in_force(store) -> str:
    """What the trader should still act on when no new report came."""
    try:
        from core.plan_alerts import latest_plan_report
        r = latest_plan_report(store)
    except Exception:  # noqa: BLE001 - an unreadable ledger must not lose the message
        r = None
    if not r:
        return "No plan is in force."
    plan = r.get("plan") or {}
    if isinstance(plan, str):                 # store.reports() hands the column back as JSON text
        try:
            plan = json.loads(plan)
        except ValueError:
            plan = {}
    side = str(plan.get("side") or "").upper()
    if side in ("SHORT", "LONG") and plan.get("zone_low") is not None and plan.get("zone_high") is not None:
        return (f"The last plan (report {r.get('report_id')}, {_plan_word(side)} ${float(plan['zone_low']):,.0f}–"
                f"${float(plan['zone_high']):,.0f}) stays in force.")
    return f"The last report ({r.get('report_id')}) said stand aside; that still holds."


def triggers(cal_events, qf_rows, now_ms: int) -> list[dict]:
    """Every reason to run that stands now: [{"key", "reason", "plain", "at_ms"}], oldest first. `reason` is
    for the log and the pass's problems; `plain` is the channel's words ("the US CPI m/m release at …")."""
    out: list[dict] = []
    by_minute: dict[int, list[str]] = {}
    for e in cal_events or []:
        t = e.get("time_ms")
        if t is None or e.get("impact") != "High" or e.get("country") != "USD":
            continue
        t = int(t)
        if now_ms - PRINT_STALE_MS < t <= now_ms - PRINT_SETTLE_MS:
            by_minute.setdefault(t, []).append(str(e.get("title") or "release").strip())
    for t in sorted(by_minute):
        out.append({"key": f"{KEY}print:{t}", "at_ms": t,
                    "reason": f"after US {' + '.join(by_minute[t])} (High) at {_utc(t)}",
                    "plain": f"the US {' + '.join(by_minute[t])} release at {_utc(t)}"})
    for r in qf_rows or []:
        closed = r.get("closed_ms")
        if r.get("status") != "invalidated" or not closed:
            continue
        closed = int(closed)
        if now_ms - INVALID_STALE_MS < closed <= now_ms:
            out.append({"key": f"{KEY}qf_invalid:{r.get('id')}", "at_ms": closed,
                        "reason": f"after the day plan (forecast #{r.get('version')}) was invalidated at {_utc(closed)}",
                        "plain": f"the day plan being invalidated at {_utc(closed)}"})
    return sorted(out, key=lambda d: d["at_ms"])


def due(recent: list[dict], cal_events, qf_rows, now_ms: int) -> tuple[dict | None, str | None]:
    """(the trigger to run now, None) or (None, why not) or (None, None) when nothing is due. `recent`: the
    ledger's 'run:' rows (ts_ms, dedup_key, headline)."""
    done = {str(r.get("dedup_key")) for r in recent or []}
    fresh = [t for t in triggers(cal_events, qf_rows, now_ms) if t["key"] not in done]
    if not fresh:
        return None, None
    runs = [r for r in recent or [] if not _is_skip(r)]
    last = max((int(r.get("ts_ms") or 0) for r in runs), default=0)
    if last and now_ms - last < RUN_GAP_MS:
        return None, "waiting"                    # the gap: the trigger stands and is read again next pass
    today = sum(1 for r in runs if now_ms - int(r.get("ts_ms") or 0) < DAY_MS)
    if today >= MAX_RUNS_PER_DAY:
        return fresh[0], "daily cap"
    return fresh[0], None


def _director(store, m, analyses: dict, now_ms: int, settings) -> dict:
    """Runs the 1-agent Director and records its report. Returns {report_id, classification, side}; raises
    when the Director produced nothing, so the caller's line says the model was unavailable."""
    from core.accuracy import majority_direction, record_report
    from core.agents.one_agent import client_for, run_one_agent_sync
    snapshot_id = f"S-{m.generated_at:%Y%m%d-%H%M%S}" if getattr(m, "generated_at", None) else None
    rep = run_one_agent_sync(client_for(settings), m, analyses, settings.director_model, None, snapshot_id)
    if rep.director is None:
        raise RuntimeError(f"Director failed ({(rep.runs[0].error if rep.runs else 'no answer') or 'no answer'})")
    intraday = analyses.get("intraday")
    price = m.spot.last or next(iter(analyses.values())).price
    record_report(store, rep, price, intraday.vol.sigma_pct if intraday is not None else None,
                  majority_direction(analyses), now_ms)
    side = getattr(rep.plan, "side", None) if rep.plan is not None else None
    return {"report_id": rep.report_id,
            "classification": rep.director.trap_classification.replace("_", " ").lower(),
            "side": side}


def _war_room(store, m, analyses: dict, now_ms: int, settings) -> str:
    import asyncio
    from dataclasses import replace
    from core import live_anchor
    from core.agents import handshake
    from core.agents.judge import Judge
    from core.agents.live_recon import run_live_recon
    from core.agents.llm import LLMClient
    from core.agents.recon_profiles import LIVE
    judge = Judge(settings, typesafe_key=settings.typesafe_key, llm=LLMClient(settings))
    recon = asyncio.run(run_live_recon(judge, m, analyses, profile=LIVE))
    try:
        recon = handshake.apply(recon, handshake.consult(store, LIVE.category, recon.bias, now_ms))
    except Exception as e:  # noqa: BLE001 - an unreadable ledger leaves the war room's own conviction
        recon = replace(recon, notes=tuple(recon.notes) + (f"Cross-desk handshake unavailable: {e}",))
    cal = getattr(m, "calendar", None)
    macro = None
    if getattr(cal, "available", False):
        soon = cal.upcoming(now_ms, LIVE.window_close(now_ms) - now_ms)
        macro = soon[0].get("title") if soon else None
    published = live_anchor.publish(store, recon, now_ms, price=m.spot.last, macro_event=macro, profile=LIVE)
    if published.anchored:
        try:
            handshake.notify(store, LIVE.category, recon.bias, now_ms, price=m.spot.last)
        except Exception as e:  # noqa: BLE001
            log.warning("scheduled run: could not notify the other desks: %s", e)
    what = "anchored the window's summary" if published.anchored else f"{len(published.appended)} side note(s)"
    return f"Live war room: {recon.bias} ({what})"


def run(store, m, analyses: dict, now_ms: int, settings, result=None, may_run: bool = True) -> dict | None:
    """One pass's check: run the agents if a trigger is due and allowed. Returns the ledger row's key and
    headline when something was written, else None. Never raises."""
    problems = getattr(result, "problems", None)
    try:
        if not may_run or not analyses:
            return None
        cal = getattr(m, "calendar", None)
        events = (getattr(cal, "events", None) or []) if getattr(cal, "available", False) else []
        day = now_ms - now_ms % DAY_MS
        rows = store.qf_plans(day) + (store.qf_plans(day - DAY_MS) if now_ms - day < INVALID_STALE_MS else [])
        recent = store.market_alerts_keyed(KEY, limit=50)
        trigger, why = due(recent, events, rows, now_ms)
        if trigger is None:
            return None
        price = getattr(getattr(m, "spot", None), "last", None)
        if why == "daily cap":
            head = (f"{_head('NO NEW REPORT', price)}\nTriggered by {trigger['plain']}, but {MAX_RUNS_PER_DAY} {CAP_MARK}. "
                    f"The next automatic run is allowed after {_next_after_cap(recent, now_ms)}.")
            store.add_market_alert(now_ms, KIND, trigger["key"], head, price)
            return {"key": trigger["key"], "headline": head}
        if settings is None:
            if problems is not None:
                problems.append(f"scheduled run {trigger['reason']} skipped: no model key on this machine")
            return None
        # the claim: the key is written before the run, so a second process finds it and does not run. The
        # row is marked sent at once: nothing about a run in progress goes to the channel.
        row_id = store.add_market_alert(now_ms, KIND, trigger["key"],
                                        f"{_head('NEW REPORT COMING', price)}\nTriggered by {trigger['plain']}: "
                                        "the agents are running now.", price)
        if row_id is None:
            return None
        store.mark_market_notified([row_id])
        report = None
        for name, fn in (("Director", _director), ("Live war room", _war_room)):
            try:
                got = fn(store, m, analyses, now_ms, settings)
                if name == "Director":
                    report = got
                else:
                    log.info("scheduled run: %s", got)
            except Exception as e:  # noqa: BLE001 - the other half still runs; the pass's problems say what failed
                log.exception("scheduled run: %s failed", name)
                if problems is not None:
                    problems.append(f"scheduled run: {name} failed: {type(e).__name__}: {e}")
        if report:
            # the report's own Pulse carries the news: this line is the ledger's record and is not sent
            head = (f"{_head('NEW REPORT', price)}\nTriggered by {trigger['plain']}. Report {report['report_id']}: "
                    f"{report['classification']}, plan {_plan_word(report.get('side'))}. Details follow in the Pulse.")
            store.set_market_alert_headline(row_id, head)
        else:
            head = (f"{_head('NO NEW REPORT', price)}\nTriggered by {trigger['plain']}, but the model was unavailable. "
                    f"{_in_force(store)} The next automatic run comes with the next release or plan change, "
                    f"not before {_utc(now_ms + RUN_GAP_MS)}.")
            store.set_market_alert_headline(row_id, head, notified=0)
        log.info("scheduled run: %s", head.replace("\n", " "))
        return {"key": trigger["key"], "headline": head}
    except Exception as e:  # noqa: BLE001 - never the worker's pass
        log.exception("scheduled run failed")
        if problems is not None:
            problems.append(f"scheduled run: {type(e).__name__}: {e}")
        return None
