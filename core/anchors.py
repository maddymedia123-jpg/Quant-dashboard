"""Anchored verdicts: the displayed direction per category stays fixed until a defined trigger fires."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from core.config import CATEGORIES, MACRO_EVENTS, TF_MINUTES
from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis

TRIGGER_OI_PCT = 5.0
# 29 Oct, 16:01-16:34 UTC: funding sat at +0.002% and wobbled through zero, so "funding_flip" fired nine times
# in 33 minutes, every desk re-anchored each time with no change of direction, and each re-anchor added a
# scored call - one wrong episode counted ten times. A sign counts only beyond the dead band, and a flip
# is a strictly opposite sign to the anchored one; the long/short ratio likewise must clear 1.0 by a margin.
FUNDING_DEADBAND = 0.00003        # 0.003% per 8h; ordinary positive funding is 0.01%
LS_DEADBAND = 0.03
SQUEEZE_LEVEL = 60
STALE_MULT = 4
MACRO_WINDOW_MS = 86_400_000


@dataclass
class AnchorContext:
    direction: str
    confidence: float
    anchor: float | None
    invalidation: float | None
    price: float
    ts_ms: int
    funding_sign: int | None
    oi_change_24h_pct: float | None
    long_short_ratio: float | None
    squeeze_score: int | None
    classification: str | None
    macro_within_24h: bool
    sigma_pct: float | None
    drivers: list[str] = field(default_factory=list)
    flips_if: str = ""


@dataclass
class AnchoredVerdict:
    category: str
    anchored: AnchorContext
    live: AnchorContext
    since_ms: int
    triggers_fired: list[str]
    changed: bool

    @property
    def unconfirmed(self) -> bool:
        return self.live.direction != self.anchored.direction


def horizon_ms_for(key: str) -> int:
    cat = CATEGORIES[key]
    return cat.horizon_bars * TF_MINUTES[cat.chart_tf] * 60_000


def macro_within(now_ms: int, events=MACRO_EVENTS, window_ms: int = MACRO_WINDOW_MS) -> bool:
    for e in events:
        try:
            day = datetime.strptime(e["date_utc"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        start, end = int(day.timestamp() * 1000), int(day.timestamp() * 1000) + 86_400_000
        if start - window_ms <= now_ms < end:
            return True
    return False


def _sign(v: float | None, deadband: float = 0.0) -> int | None:
    if v is None:
        return None
    if abs(v) < deadband:
        return 0
    return 1 if v > 0 else -1 if v < 0 else 0


def macro_soon(m: MarketSnapshot, now_ms: int) -> bool:
    if m.calendar.available and m.calendar.events:
        return bool(m.calendar.upcoming(now_ms, MACRO_WINDOW_MS))
    return macro_within(now_ms)


def build_context(a: CategoryAnalysis, m: MarketSnapshot, classification: str | None, now_ms: int) -> AnchorContext:
    f = m.futures
    return AnchorContext(
        direction=a.direction.direction, confidence=float(a.direction.confidence),
        anchor=a.direction.anchor, invalidation=a.direction.invalidation, price=float(a.price), ts_ms=int(now_ms),
        funding_sign=_sign(f.funding_rate, FUNDING_DEADBAND) if f.available else None,
        oi_change_24h_pct=f.oi_change_24h_pct if f.available else None,
        long_short_ratio=f.long_short_ratio if f.available else None,
        squeeze_score=a.squeeze.score, classification=classification, macro_within_24h=macro_soon(m, now_ms),
        sigma_pct=a.vol.sigma_pct, drivers=list(a.direction.drivers), flips_if=a.direction.flips_if,
    )


def evaluate_triggers(prev: AnchorContext, cur: AnchorContext, horizon_ms: int) -> list[str]:
    fired: list[str] = []
    if prev.funding_sign is not None and cur.funding_sign is not None and prev.funding_sign * cur.funding_sign < 0:
        fired.append("funding_flip")
    if prev.oi_change_24h_pct is not None and cur.oi_change_24h_pct is not None and abs(cur.oi_change_24h_pct - prev.oi_change_24h_pct) >= TRIGGER_OI_PCT:
        fired.append("oi_shift")
    if (prev.long_short_ratio is not None and cur.long_short_ratio is not None
            and (prev.long_short_ratio - 1.0) * (cur.long_short_ratio - 1.0) < 0
            and abs(cur.long_short_ratio - 1.0) >= LS_DEADBAND):
        fired.append("long_short_cross")
    if cur.squeeze_score is not None and cur.squeeze_score >= SQUEEZE_LEVEL and (prev.squeeze_score is None or prev.squeeze_score < SQUEEZE_LEVEL):
        fired.append("squeeze")
    if prev.invalidation is not None:
        if prev.direction == "BULLISH" and cur.price < prev.invalidation:
            fired.append("invalidation")
        elif prev.direction == "BEARISH" and cur.price > prev.invalidation:
            fired.append("invalidation")
    if cur.classification is not None and prev.classification is not None and cur.classification != prev.classification:
        fired.append("trap_change")
    if cur.macro_within_24h and not prev.macro_within_24h:
        fired.append("macro_window")
    if cur.ts_ms - prev.ts_ms > STALE_MULT * horizon_ms:
        fired.append("stale")
    return fired


def call_due(store, key: str, direction: str, prev_direction: str | None, now_ms: int, horizon_ms: int) -> bool:
    """Whether a re-anchor is a new claim worth scoring: the direction changed, or the desk's last call is
    older than its horizon (a held view is re-scored once per window). A re-anchor on a data trigger that
    keeps the same direction minutes after the last call would score the same claim again."""
    if direction != prev_direction:
        return True
    last = store.last_call(key)
    return last is None or now_ms - int(last["ts_ms"]) >= horizon_ms


def anchor_step(store, key: str, a: CategoryAnalysis, m: MarketSnapshot, classification: str | None, now_ms: int) -> AnchoredVerdict:
    cur = build_context(a, m, classification, now_ms)
    horizon = horizon_ms_for(key)
    store.add_signal(now_ms, key, cur.squeeze_score, cur.price)
    prev_row = store.get_anchor(key)
    if prev_row is None:
        store.set_anchor(key, asdict(cur), now_ms)
        store.log_anchor_change(now_ms, key, None, cur.direction, cur.price, ["initial"])
        store.add_call(now_ms, key, cur.direction, cur.confidence, cur.price, cur.sigma_pct, horizon)
        return AnchoredVerdict(key, cur, cur, now_ms, ["initial"], True)
    prev = AnchorContext(**prev_row["payload"])
    fired = evaluate_triggers(prev, cur, horizon)
    if fired:
        store.set_anchor(key, asdict(cur), now_ms)
        store.log_anchor_change(now_ms, key, prev.direction, cur.direction, cur.price, fired)
        if call_due(store, key, cur.direction, prev.direction, now_ms, horizon):
            store.add_call(now_ms, key, cur.direction, cur.confidence, cur.price, cur.sigma_pct, horizon)
        return AnchoredVerdict(key, cur, cur, now_ms, fired, True)
    return AnchoredVerdict(key, prev, cur, int(prev_row["since_ms"]), [], False)
