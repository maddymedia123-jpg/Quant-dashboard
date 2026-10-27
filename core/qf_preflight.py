"""Quant Forecast pre-flight: the two checks that run before a day's plan is built (Hammad, 28 Oct,
"multi-agent pre-flight consensus").

1. Trap Head. If an engineered trap is active on the 4h, the plan's directional bias is overridden and
   the day is forced NEUTRAL / RANGE-BOUND. "Active on the 4h" is read from the trap ledger, not re-judged:
   a row in `traps` with status 'active' (declared by the TRAP desk and not yet settled by price - the
   desk settles them as 'played_out' or 'invalidated' in core.agents.trap_desk.settle; nothing expires on
   a timer) and timeframe '4h'. Every stored trap was *declared*, which core.agents.trap_desk.score only
   does when the TRAP Head's call was ENGINEERED_TRAP, so head_call is checked only to exclude a row that
   somehow carries a different call; a NULL/empty head_call (rows older than the column) still counts.
   The trap's type is its side: bull_trap (a fake break up) or bear_trap (a fake break down).

2. Economic Data. Any High-impact USD event on the Forex Factory calendar (the snapshot already on the
   market object, `m.calendar`; no fetch of its own) whose release time falls inside the plan's window
   [now_ms, end_ms] becomes a forced high-volatility chop window from MACRO_BEFORE_MS before it to
   MACRO_AFTER_MS after it. Two events at the same minute (CPI and core CPI) make one window with both
   titles. core.quant_forecast.build_plan marks the hours it touches as chop and widens their sigma.

Neither check may block the daily plan: `preflight` never raises. A store or calendar that fails leaves
that check clear and records why in "errors", so the plan still goes out and the drawer says what was
not checked."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

TRAP_TIMEFRAME = "4h"
ENGINEERED = "ENGINEERED_TRAP"
MACRO_BEFORE_MS = 30 * 60_000        # the positioning into a print
MACRO_AFTER_MS = 90 * 60_000         # the print and the whipsaw after it
MACRO_IMPACT = "High"
MACRO_COUNTRIES = ("USD",)
CLEAR = "Trap Agent: Clear"
TRAP_WORD = {"bull_trap": "BULL TRAP", "bear_trap": "BEAR TRAP"}


def _engineered(row: dict) -> bool:
    call = row.get("head_call")
    return not call or str(call).upper() == ENGINEERED


def trap_check(store) -> tuple[dict, list[str]]:
    """Active engineered 4h traps, newest first, and the label for the top line."""
    errors: list[str] = []
    rows: list[dict] = []
    try:
        for r in store.traps(status="active", timeframes=(TRAP_TIMEFRAME,)):
            if str(r.get("status", "active")) != "active" or str(r.get("timeframe")) != TRAP_TIMEFRAME:
                continue
            if not _engineered(r):
                continue
            rows.append({"id": r.get("id"), "type": r.get("side"), "side": r.get("side"),
                         "timeframe": r.get("timeframe"), "level": r.get("level"),
                         "invalidation": r.get("invalidation"), "plays_out": r.get("plays_out"),
                         "since_ms": r.get("declared_ms"), "score": r.get("score"),
                         "category": r.get("category")})
    except Exception as e:  # noqa: BLE001 - the plan must not wait on the trap ledger
        log.warning("quant forecast pre-flight: trap ledger unreadable: %s", e)
        errors.append(f"trap ledger unreadable ({type(e).__name__}: {str(e)[:80]})")
        rows = []
    if rows:
        kinds = sorted({TRAP_WORD.get(str(r["type"]), str(r["type"]).upper()) for r in rows})
        label = f"Trap Agent: {' + '.join(kinds)} active (4H)"
    else:
        label = CLEAR
    return {"active": bool(rows), "label": label, "traps": rows}, errors


def macro_check(m, now_ms: int, end_ms: int) -> tuple[list[dict], list[str]]:
    """High-impact USD releases inside [now_ms, end_ms], each with its forced chop window."""
    cal = getattr(m, "calendar", None)
    if cal is None:
        return [], ["calendar not on the snapshot"]
    try:
        if not getattr(cal, "available", False):
            return [], [f"calendar unavailable ({getattr(cal, 'error', None) or 'no events'})"]
        by_minute: dict[int, dict] = {}
        for e in getattr(cal, "events", None) or []:
            t = e.get("time_ms")
            if t is None or not (int(now_ms) <= int(t) <= int(end_ms)):
                continue
            if e.get("impact") != MACRO_IMPACT or e.get("country") not in MACRO_COUNTRIES:
                continue
            t = int(t)
            title = str(e.get("title") or "macro release").strip()
            if t in by_minute:
                if title not in by_minute[t]["titles"]:
                    by_minute[t]["titles"].append(title)
            else:
                by_minute[t] = {"titles": [title], "t_ms": t, "impact": str(e.get("impact"))}
        out = []
        for t in sorted(by_minute):
            w = by_minute[t]
            out.append({"title": " + ".join(w["titles"]), "t_ms": t,
                        "start_ms": max(int(now_ms), t - MACRO_BEFORE_MS),
                        "end_ms": min(int(end_ms), t + MACRO_AFTER_MS), "impact": w["impact"]})
        return out, []
    except Exception as e:  # noqa: BLE001 - the plan must not wait on the calendar
        log.warning("quant forecast pre-flight: calendar unreadable: %s", e)
        return [], [f"calendar unreadable ({type(e).__name__}: {str(e)[:80]})"]


def preflight(store, m, now_ms: int, end_ms: int) -> dict:
    """Both checks, never raising:
    {"trap": {"active", "label", "traps": [...]}, "macro": [{"title", "t_ms", "start_ms", "end_ms",
    "impact"}], "forced_neutral": bool, "errors": [str]}."""
    try:
        trap, e1 = trap_check(store)
    except Exception as e:  # noqa: BLE001 - belt and braces: the check above already catches
        trap, e1 = {"active": False, "label": CLEAR, "traps": []}, [f"trap check failed ({type(e).__name__})"]
    try:
        macro, e2 = macro_check(m, now_ms, end_ms)
    except Exception as e:  # noqa: BLE001
        macro, e2 = [], [f"macro check failed ({type(e).__name__})"]
    return {"trap": trap, "macro": macro, "forced_neutral": bool(trap["active"]), "errors": e1 + e2}


def plan_kwargs(pre: dict) -> dict:
    """The extra keyword arguments for core.quant_forecast.build_plan."""
    pre = pre or {}
    return {"force_neutral": bool(pre.get("forced_neutral")),
            "forced_windows": [(w["start_ms"], w["end_ms"], w["title"]) for w in pre.get("macro") or []]}
