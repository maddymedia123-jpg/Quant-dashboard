"""Accuracy ledger: score anchored direction calls and Director reports against realised price."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

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
    return store.add_report(report.report_id, now_ms, report.director.trap_classification, price, sigma24_pct, majority_dir)


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
    by_cat: dict[str, HitRate] = {}
    for c in store.calls(limit=2000, scored_only=True):
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
            "total_scored": sum(h.n for h in by_cat.values()) + sum(h.n for h in by_cls.values()),
            "open_calls": len([c for c in store.calls(limit=2000) if c["scored_ms"] is None])}
