"""Accuracy ledger: score anchored direction calls and Director reports against realised price."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

# imported rather than recomputed: two definitions of a category's horizon would drift, which is exactly
# what the missing DEFAULT_ATR_PCT cost earlier. core.anchors does not import this module, so no cycle.
from core.anchors import horizon_ms_for
from core.config import CATEGORIES

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
MIN_SAMPLE = 10


@dataclass
class HitRate:
    n: int = 0
    hits: int = 0

    @property
    def rate(self) -> float | None:
        return (self.hits / self.n) if self.n else None

    @property
    def small(self) -> bool:
        return self.n < MIN_SAMPLE


def majority_direction(analyses: dict) -> str:
    if not analyses:
        return "NEUTRAL"
    c = Counter(a.direction.direction for a in analyses.values())
    return c.most_common(1)[0][0]


def _move_pct(price_now: float, price_then: float) -> float:
    return (price_now / price_then - 1.0) * 100.0


def call_hit(direction: str, move_pct: float, sigma_pct: float | None) -> bool:
    s = sigma_pct if sigma_pct and sigma_pct > 0 else 1.0
    if direction == "BULLISH":
        return move_pct > 0.25 * s
    if direction == "BEARISH":
        return move_pct < -0.25 * s
    return abs(move_pct) <= 0.5 * s


def score_due_calls(store, now_ms: int, price: float) -> int:
    n = 0
    for c in store.due_calls(now_ms):
        move = _move_pct(price, float(c["price"]))
        store.score_call(int(c["id"]), now_ms, move, call_hit(c["direction"], move, c["sigma_pct"]))
        n += 1
    return n


def record_report(store, report, price: float, sigma24_pct: float | None, majority_dir: str, now_ms: int) -> bool:
    if report is None or report.director is None:
        return False
    plan = report.plan.model_dump_json() if getattr(report, "plan", None) is not None else None
    return store.add_report(report.report_id, now_ms, report.director.trap_classification, price, sigma24_pct,
                            majority_dir, mode=getattr(report, "mode", "15-agent"),
                            snapshot_id=getattr(report, "snapshot_id", None), plan=plan)


# ---------- plan outcomes ----------
# A plan is scored the way it would have been traded: price reaches the zone, then a CLOSED candle on the
# plan's confirmation timeframe closes beyond confirm_close (the taker-ratio rule is not replayed - there is
# no per-candle history of it here), and from that close the first of stop or first target to be touched
# decides it. A candle that spans both counts as the stop: the cautious reading.
PLAN_OUTCOMES = ("target", "stop", "not_triggered", "no_trade", "open")


def plan_outcome(plan: dict, ts_ms: int, frames: dict, now_ms: int) -> str:
    side = plan.get("side")
    if side not in ("LONG", "SHORT"):
        return "no_trade"
    low, high, stop = plan["zone_low"], plan["zone_high"], plan["stop"]
    # the nearest target, whatever order the plan listed them in
    t1 = max(plan["targets"]) if side == "SHORT" else min(plan["targets"])
    confirm = plan.get("confirm_close")
    expires = ts_ms + int(float(plan.get("expires_hours") or 24) * 3_600_000)
    fine, conf = frames.get("15m"), frames.get(plan.get("confirm_timeframe") or "1h")
    if fine is None or conf is None or not len(fine):
        return "open"
    bar_ms = TF_MS.get(plan.get("confirm_timeframe") or "1h", 3_600_000)
    rows = [r for r in fine.itertuples() if int(r.timestamp) >= ts_ms]
    touched = next((int(r.timestamp) for r in rows if r.high >= low and r.low <= high and int(r.timestamp) < expires), None)
    if touched is None:
        return "not_triggered" if now_ms >= expires else "open"
    entry_ms = None
    for r in conf.itertuples():
        t = int(r.timestamp)
        if t + bar_ms <= touched or t + bar_ms > min(now_ms, expires + bar_ms):
            continue                               # before the touch, still forming, or past expiry
        level = confirm if confirm is not None else (low if side == "SHORT" else high)
        if (side == "SHORT" and r.close < level) or (side == "LONG" and r.close > level):
            entry_ms = t + bar_ms
            break
    if entry_ms is None:
        return "not_triggered" if now_ms >= expires + bar_ms else "open"
    for r in rows:
        if int(r.timestamp) < entry_ms:
            continue
        hit_stop = r.high >= stop if side == "SHORT" else r.low <= stop
        hit_t1 = r.low <= t1 if side == "SHORT" else r.high >= t1
        if hit_stop:
            return "stop"
        if hit_t1:
            return "target"
    return "open"


TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def score_plans(store, frames: dict, now_ms: int) -> int:
    """Settle every plan still open; returns how many reached a final outcome this pass."""
    import json
    n = 0
    for r in store.reports(limit=500):
        if not r.get("plan") or r.get("plan_scored_ms"):
            continue
        out = plan_outcome(json.loads(r["plan"]), int(r["ts_ms"]), frames, now_ms)
        if out != (r.get("plan_outcome") or "open"):
            store.set_plan_outcome(int(r["id"]), out, None if out == "open" else now_ms)
            n += out != "open"
    return n


def mode_scorecard(store) -> dict:
    """1-agent against 15-agent: classification hit rates at 24h and 7d, and how the plans ended."""
    out: dict[str, dict] = {}
    for r in store.reports(limit=1000):
        mode = r.get("mode") or "15-agent"
        card = out.setdefault(mode, {"reports": 0, "h24": HitRate(), "d7": HitRate(),
                                     "plans": Counter()})
        card["reports"] += 1
        if r["hit_24h"] is not None:
            card["h24"].n += 1
            card["h24"].hits += int(r["hit_24h"])
        if r["hit_7d"] is not None:
            card["d7"].n += 1
            card["d7"].hits += int(r["hit_7d"])
        if r.get("plan"):
            card["plans"][r.get("plan_outcome") or "open"] += 1
    return out


def report_hit(classification: str, move_pct: float, sigma24_pct: float | None, majority_dir: str | None, horizon: str) -> bool:
    s = (sigma24_pct if sigma24_pct and sigma24_pct > 0 else 1.0) * (math.sqrt(7) if horizon == "7d" else 1.0)
    if classification == "BULL_TRAP":
        return move_pct < 0
    if classification == "BEAR_TRAP":
        return move_pct > 0
    if classification == "RANGE_TRAP":
        return abs(move_pct) < s
    # NO_TRAP: genuine move in the direction the deterministic layer had at report time
    if majority_dir == "BULLISH":
        return move_pct > 0
    if majority_dir == "BEARISH":
        return move_pct < 0
    return abs(move_pct) < s


def score_reports(store, now_ms: int, price: float) -> int:
    n = 0
    for r in store.reports(limit=500):
        move = _move_pct(price, float(r["price"]))
        if r["hit_24h"] is None and r["ts_ms"] + DAY_MS <= now_ms:
            store.score_report(int(r["id"]), "24h", move, report_hit(r["classification"], move, r["sigma24_pct"], r["majority_dir"], "24h"))
            n += 1
        if r["hit_7d"] is None and r["ts_ms"] + WEEK_MS <= now_ms:
            store.score_report(int(r["id"]), "7d", move, report_hit(r["classification"], move, r["sigma24_pct"], r["majority_dir"], "7d"))
            n += 1
    return n


def summary(store) -> dict:
    # Only calls made over the horizon the category uses NOW. A direction call is a claim about a
    # specific window, so blending one-hour calls with four-hour ones would average two different
    # questions into a single hit rate - the same trap RUBRIC_VERSION guards for the rubric. The
    # horizon is stored per call, so a definition change retires the old history instead of corrupting
    # the number.
    current = {key: horizon_ms_for(key) for key in CATEGORIES}
    by_cat: dict[str, HitRate] = {}
    stale_horizon = 0
    for c in store.calls(limit=2000, scored_only=True):
        want = current.get(c["category"])
        if want is not None and c["horizon_ms"] != want:
            stale_horizon += 1
            continue
        h = by_cat.setdefault(c["category"], HitRate())
        h.n += 1
        h.hits += int(c["hit"] or 0)
    by_cls: dict[str, HitRate] = {}
    by_cls_7d: dict[str, HitRate] = {}
    for r in store.reports(limit=500):
        if r["hit_24h"] is not None:
            h = by_cls.setdefault(r["classification"], HitRate())
            h.n += 1
            h.hits += int(r["hit_24h"])
        if r["hit_7d"] is not None:
            h = by_cls_7d.setdefault(r["classification"], HitRate())
            h.n += 1
            h.hits += int(r["hit_7d"])
    recent_calls = store.calls(limit=20, scored_only=True)
    recent_reports = [r for r in store.reports(limit=20) if r["hit_24h"] is not None or r["hit_7d"] is not None]
    return {"by_category": by_cat, "by_classification_24h": by_cls, "by_classification_7d": by_cls_7d,
            "recent_calls": recent_calls, "recent_reports": recent_reports,
            "stale_horizon_calls": stale_horizon,
            "total_scored": sum(h.n for h in by_cat.values()) + sum(h.n for h in by_cls.values()),
            "open_calls": len([c for c in store.calls(limit=2000) if c["scored_ms"] is None])}
