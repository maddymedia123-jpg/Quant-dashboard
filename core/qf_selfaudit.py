"""The self-audit card (Hammad, message 9, panel D): after a day is scored, how far the projected high
or low missed the tape, and why, in one line.

The miss is arithmetic on the scored outcome. The root cause is chosen by rule, in this order, from the
plan's own numbers: a level of the plan's that price ran to beyond the projection (liquidation gravity),
a day that traded far wider or narrower than the expected range (volatility), a breached invalidation
(the lean), or the move landing at a different hour (timing). No model names a cause.

"System multipliers auto-calibrated": the card reports the sigma scale the plan was actually built with
(core.quant_forecast.calibration, approved 30 Oct: the last ten days' realised over expected range,
bounded 0.8-1.3). A plan built with too little history to calibrate says so instead."""
from __future__ import annotations

GRAVITY_LABELS = ("est. liquidation zone", "air pocket", "call wall", "put wall", "gamma flip")
WIDE_X, NARROW_X = 1.25, 0.75
LOOKBACK_DAYS = 3
NO_CALIBRATION = "not changed (too little history to calibrate)"


def calibration_text(plan: dict) -> str:
    """What the plan did to its multipliers, for the card: the sigma scale it was built with."""
    c = plan.get("calibration") or {}
    if not c.get("days") or c.get("ratio") is None:
        return NO_CALIBRATION
    return (f"auto-calibrated (sigma x{c['scale']:.2f}: the last {c['days']} days traded "
            f"{c['ratio']:.2f}x the range the model expected, bounded {c['bounds'][0]}-{c['bounds'][1]})")


def self_audit(row: dict) -> dict | None:
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
    if beyond:
        lo, hi = sorted((projected, actual))
        ran_to = [lv for lv in plan.get("levels") or [] if lv["label"] in GRAVITY_LABELS and lo <= lv["price"] <= hi]
        if ran_to:
            lv = max(ran_to, key=lambda x: abs(x["price"] - projected))
            noun = "dealer wall" if lv["label"] in ("call wall", "put wall", "gamma flip") else lv["label"]
            root = f"liquidation gravity underestimated: price ran to the {noun} at ${lv['price']:,.0f} past the projected {side}"
    if root is None and ratio is not None and beyond and ratio >= WIDE_X:
        root = f"volatility underestimated: the day traded {ratio:.1f}x the expected range"
    if root is None and ratio is not None and not beyond and ratio <= NARROW_X:
        root = f"volatility overestimated: the day traded {ratio:.1f}x the expected range"
    if root is None and o.get("invalidated"):
        root = "the lean was on the wrong side: the whole-day invalidation was breached"
    if root is None and due_ms and at_ms:
        hours = round((int(at_ms) - int(due_ms)) / 3_600_000)
        if abs(hours) >= 2:
            root = f"timing: the {side} printed {abs(hours)} hours {'later' if hours > 0 else 'earlier'} than projected"
    if root is None:
        root = f"the {side} landed just outside tolerance; no single driver stands out"
    return {"day_ms": plan.get("day_ms"), "side": side, "missed_pct": round(abs(pct), 2), "projected": projected,
            "actual": actual, "beyond": beyond, "range_ratio": None if ratio is None else round(ratio, 2),
            "root_cause": root, "calibration": calibration_text(plan),
            "note": (f"🔍 SELF-AUDIT REPORT: {side.capitalize()} projection missed by {abs(pct):.1f}%. "
                     f"Root cause: {root}. System multipliers: {calibration_text(plan)}.")}


def latest(store, now_ms: int) -> dict | None:
    """The audit of the most recent scored morning plan within LOOKBACK_DAYS, or None."""
    since = int(now_ms) - LOOKBACK_DAYS * 86_400_000
    for row in store.qf_plans(limit=20):
        if row.get("version") == 0 and row.get("outcome") and int(row.get("day_ms") or 0) >= since:
            return self_audit(row)
    return None
