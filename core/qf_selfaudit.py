"""Self-audit card and persistent trader memory / learning helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GRAVITY_LABELS = ("est. liquidation zone", "air pocket", "call wall", "put wall", "gamma flip")
WIDE_X, NARROW_X = 1.25, 0.75
LOOKBACK_DAYS = 3
NO_CALIBRATION = "not changed (too little history to calibrate)"

DEFAULT_MEMORY_RULES: list[dict[str, str]] = [
    {"id": "MEM-01", "lesson": "Sideways ranges can yield trades (range edges, exiting shorts near lows, bounce/reversal reads) while frustrating late holders; do not force one template onto every range."},
    {"id": "MEM-02", "lesson": "Price often sweeps below trendlines, support, or range edges to grab stops/liquidations or tap demand before reversing/expanding; avoid treating a single poke as confirmation."},
    {"id": "MEM-03", "lesson": "Pair Price (Y) levels with Time (X) windows and explicit time/price invalidation so stale calls do not linger."},
    {"id": "MEM-04", "lesson": "Connect 2-4+ fitting puzzle clues (structure, liquidity/gamma, order-flow absorption/exhaustion, crowding/funding/L-S, divergences, volatility regime, macro/session) rather than isolated metrics."},
    {"id": "MEM-05", "lesson": "Check positioning and liquidity around macro prints and let older headlines decay naturally."},
    {"id": "MEM-06", "lesson": "Account for session handovers and weekend lower-spot-volume sweep-and-reject dynamics."},
    {"id": "MEM-07", "lesson": "Focus on multi-day swing structure rather than minor 500-600 point noise or foggy 1H flips."},
    {"id": "MEM-08", "lesson": "Update/close the single Master Swing Trade immediately on TP/SL/invalidation and record wins/misses to memory."},
]


def _default_memory_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "trader_memory.json"


def load_memory(path: Path | None = None) -> dict[str, Any]:
    target = path or _default_memory_path()
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("rules", list(DEFAULT_MEMORY_RULES))
                data.setdefault("audit_entries", [])
                data.setdefault("clue_scores", {})
                return data
        except Exception:  # noqa: BLE001
            pass
    return {
        "rules": list(DEFAULT_MEMORY_RULES),
        "audit_entries": [],
        "clue_scores": {},
    }


def save_memory(state: dict[str, Any], path: Path | None = None) -> Path:
    target = path or _default_memory_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return target


def record_audit_lesson(
    *,
    summary: str,
    root_cause: str,
    hit: bool = False,
    helpful_clues: list[str] | None = None,
    misleading_clues: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Persist a miss/hit audit lesson and update clue reliability scores in memory."""
    state = load_memory(path)
    scores: dict[str, int] = state.setdefault("clue_scores", {})
    for clue in helpful_clues or []:
        scores[clue] = int(scores.get(clue, 0)) + 1
    for clue in misleading_clues or []:
        scores[clue] = int(scores.get(clue, 0)) - 1

    entry = {
        "hit": bool(hit),
        "root_cause": root_cause,
        "summary": summary,
        "helpful_clues": list(helpful_clues or []),
        "misleading_clues": list(misleading_clues or []),
    }
    entries: list[dict[str, Any]] = state.setdefault("audit_entries", [])
    entries.append(entry)
    state["audit_entries"] = entries[-100:]
    save_memory(state, path)
    return entry


def calibration_text(plan: dict) -> str:
    """What the plan did to its multipliers, for the card: the sigma scale it was built with."""
    c = plan.get("calibration") or {}
    if not c.get("days") or c.get("ratio") is None:
        return NO_CALIBRATION
    return (f"auto-calibrated (sigma x{c['scale']:.2f}: the last {c['days']} days traded "
            f"{c['ratio']:.2f}x the range the model expected, bounded {c['bounds'][0]}-{c['bounds'][1]})")


def self_audit(row: dict, persist: bool = False, memory_path: Path | None = None) -> dict | None:
    """The audit of one scored morning plan, or None when both levels landed inside tolerance or the day
    was not scored."""
    plan, o = row.get("payload") or {}, row.get("outcome") or {}
    if not o.get("scored") or plan.get("regime") == "locked" or not plan.get("projected_high"):
        return None
    ph, pl = float(plan["projected_high"]["price"]), float(plan["projected_low"]["price"])
    misses = []
    if not o.get("high_hit"):
        misses.append(("high", ph, float(o["high"]), (float(o["high"]) - ph) / ph * 100.0, o["high"] > ph,
                       plan["projected_high"]["t_ms"], o.get("high_ms")))
    if not o.get("low_hit"):
        misses.append(("low", pl, float(o["low"]), (float(o["low"]) - pl) / pl * 100.0, o["low"] < pl,
                       plan["projected_low"]["t_ms"], o.get("low_ms")))
    if not misses:
        return None
    side, projected, actual, pct, beyond, due_ms, at_ms = max(misses, key=lambda m: abs(m[3]))
    exp_range, act_range = ph - pl, float(o["high"]) - float(o["low"])
    ratio = (act_range / exp_range) if exp_range > 0 else None
    root = None
    misleading: list[str] = []
    if beyond:
        lo, hi = sorted((projected, actual))
        ran_to = [lv for lv in plan.get("levels") or [] if lv["label"] in GRAVITY_LABELS and lo <= lv["price"] <= hi]
        if ran_to:
            lv = max(ran_to, key=lambda x: abs(x["price"] - projected))
            noun = "dealer wall" if lv["label"] in ("call wall", "put wall", "gamma flip") else lv["label"]
            root = f"liquidation gravity underestimated: price ran to the {noun} at ${lv['price']:,.0f} past the projected {side}"
            misleading.append("liquidity_magnet_cap")
    if root is None and ratio is not None and beyond and ratio >= WIDE_X:
        root = f"volatility underestimated: the day traded {ratio:.1f}x the expected range"
        misleading.append("volatility_magnitude")
    if root is None and ratio is not None and not beyond and ratio <= NARROW_X:
        root = f"volatility overestimated: the day traded {ratio:.1f}x the expected range"
        misleading.append("volatility_range")
    if root is None and o.get("invalidated"):
        root = "the lean was on the wrong side: the whole-day invalidation was breached"
        misleading.append("directional_lean")
    if root is None and due_ms and at_ms:
        hours = round((int(at_ms) - int(due_ms)) / 3_600_000)
        if abs(hours) >= 2:
            root = f"timing: the {side} printed {abs(hours)} hours {'later' if hours > 0 else 'earlier'} than projected"
            misleading.append("time_window")
    if root is None:
        root = f"the {side} landed just outside tolerance; no single driver stands out"

    note = (f"🔍 SELF-AUDIT REPORT: {side.capitalize()} projection missed by {abs(pct):.1f}%. "
            f"Root cause: {root}. System multipliers: {calibration_text(plan)}.")
    if persist:
        record_audit_lesson(
            summary=note,
            root_cause=root,
            hit=False,
            misleading_clues=misleading,
            path=memory_path,
        )
    return {"day_ms": plan.get("day_ms"), "side": side, "missed_pct": round(abs(pct), 2), "projected": projected,
            "actual": actual, "beyond": beyond, "range_ratio": None if ratio is None else round(ratio, 2),
            "root_cause": root, "calibration": calibration_text(plan),
            "note": note}


def latest(store, now_ms: int) -> dict | None:
    """The audit of the most recent scored morning plan within LOOKBACK_DAYS, or None."""
    since = int(now_ms) - LOOKBACK_DAYS * 86_400_000
    for row in store.qf_plans(limit=20):
        if row.get("version") == 0 and row.get("outcome") and int(row.get("day_ms") or 0) >= since:
            return self_audit(row)
    return None
