"""Anchored verdicts: the displayed direction per category stays fixed until a defined trigger fires."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from core.config import CATEGORIES, MACRO_EVENTS, TF_MINUTES
from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis

TRIGGER_OI_PCT = 5.0
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


def _sign(v: float | None) -> int | None:
    if v is None:
        return None
    return 1 if v > 0 else -1 if v < 0 else 0


def build_context(a: CategoryAnalysis, m: MarketSnapshot, classification: str | None, now_ms: int) -> AnchorContext:
    f = m.futures
    return AnchorContext(
        direction=a.direction.direction, confidence=float(a.direction.confidence),
        anchor=a.direction.anchor, invalidation=a.direction.invalidation, price=float(a.price), ts_ms=int(now_ms),
        funding_sign=_sign(f.funding_rate) if f.available else None,
        oi_change_24h_pct=f.oi_change_24h_pct if f.available else None,
        long_short_ratio=f.long_short_ratio if f.available else None,
        squeeze_score=a.squeeze.score, classification=classification, macro_within_24h=macro_within(now_ms),
        sigma_pct=a.vol.sigma_pct, drivers=list(a.direction.drivers), flips_if=a.direction.flips_if,
    )


def evaluate_triggers(prev: AnchorContext, cur: AnchorContext, horizon_ms: int) -> list[str]:
    fired: list[str] = []
    if prev.funding_sign is not None and cur.funding_sign is not None and prev.funding_sign != cur.funding_sign:
        fired.append("funding_flip")
    if prev.oi_change_24h_pct is not None and cur.oi_change_24h_pct is not None and abs(cur.oi_change_24h_pct - prev.oi_change_24h_pct) >= TRIGGER_OI_PCT:
        fired.append("oi_shift")
    if prev.long_short_ratio is not None and cur.long_short_ratio is not None and (prev.long_short_ratio - 1.0) * (cur.long_short_ratio - 1.0) < 0:
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
        store.add_call(now_ms, key, cur.direction, cur.confidence, cur.price, cur.sigma_pct, horizon)
        return AnchoredVerdict(key, cur, cur, now_ms, fired, True)
    return AnchoredVerdict(key, prev, cur, int(prev_row["since_ms"]), [], False)
