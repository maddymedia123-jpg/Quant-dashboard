from dataclasses import replace
from datetime import datetime, timezone

from core.anchors import AnchorContext, anchor_step, evaluate_triggers, horizon_ms_for, macro_within
from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from core.store import Store

H = horizon_ms_for("live")


def _ctx(**kw) -> AnchorContext:
    base = dict(direction="BULLISH", confidence=0.5, anchor=100.0, invalidation=99.0, price=101.0, ts_ms=0,
                funding_sign=1, oi_change_24h_pct=1.0, long_short_ratio=1.2, squeeze_score=30, classification="NO_TRAP",
                macro_within_24h=False, sigma_pct=1.0)
    base.update(kw)
    return AnchorContext(**base)


def test_no_trigger_when_nothing_changed():
    p = _ctx()
    assert evaluate_triggers(p, replace(p, ts_ms=1000), H) == []


def test_each_trigger_fires_alone():
    p = _ctx()
    assert evaluate_triggers(p, replace(p, funding_sign=-1), H) == ["funding_flip"]
    assert evaluate_triggers(p, replace(p, oi_change_24h_pct=6.5), H) == ["oi_shift"]
    assert evaluate_triggers(p, replace(p, long_short_ratio=0.9), H) == ["long_short_cross"]
    assert evaluate_triggers(p, replace(p, squeeze_score=65), H) == ["squeeze"]
    assert evaluate_triggers(p, replace(p, price=98.5), H) == ["invalidation"]
    assert evaluate_triggers(p, replace(p, classification="BULL_TRAP"), H) == ["trap_change"]
    assert evaluate_triggers(p, replace(p, macro_within_24h=True), H) == ["macro_window"]
    assert evaluate_triggers(p, replace(p, ts_ms=4 * H + 1), H) == ["stale"]
    bear = _ctx(direction="BEARISH", invalidation=103.0)
    assert evaluate_triggers(bear, replace(bear, price=104.0), H) == ["invalidation"]
    assert evaluate_triggers(bear, replace(bear, price=102.0), H) == []


def test_missing_data_never_fires_data_triggers():
    p = _ctx(funding_sign=None, oi_change_24h_pct=None, long_short_ratio=None, squeeze_score=None, classification=None)
    assert evaluate_triggers(p, replace(p, ts_ms=10), H) == []


def test_macro_within_window():
    ev = [{"title": "x", "date_utc": "2026-09-17", "impact": "HIGH", "note": ""}]
    d = lambda s: int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
    assert macro_within(d("2026-09-16 12:00"), ev) is True
    assert macro_within(d("2026-09-17 20:00"), ev) is True
    assert macro_within(d("2026-09-15 12:00"), ev) is False
    assert macro_within(d("2026-09-18 00:01"), ev) is False


def test_anchor_step_initial_then_hold_then_reanchor():
    m = fixture_market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    s = Store(":memory:")
    v1 = anchor_step(s, "live", a, m, None, now_ms=1_000)
    assert v1.changed and v1.triggers_fired == ["initial"] and s.anchor_log()[0]["to_dir"] == a.direction.direction
    assert len(s.calls()) == 1
    # same market a moment later: anchored verdict held, nothing logged
    v2 = anchor_step(s, "live", a, m, None, now_ms=2_000)
    assert not v2.changed and v2.since_ms == 1_000 and v2.unconfirmed is False and len(s.anchor_log()) == 1
    # funding flips sign → re-anchor
    m2 = m.model_copy(update={"futures": FuturesSnapshot(source="x", funding_rate=-0.0001, oi_change_24h_pct=m.futures.oi_change_24h_pct,
                                                          long_short_ratio=m.futures.long_short_ratio, funding_7d_mean=0.0)})
    a2 = analyze_category(CATEGORIES["live"], m2.spot.frames, m2.futures)
    v3 = anchor_step(s, "live", a2, m2, None, now_ms=3_000)
    assert v3.changed and "funding_flip" in v3.triggers_fired and v3.since_ms == 3_000
    assert len(s.anchor_log()) == 2 and len(s.calls()) == 2 and s.last_signal("live")["ts_ms"] == 3_000
