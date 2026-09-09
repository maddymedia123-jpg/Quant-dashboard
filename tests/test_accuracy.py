import math
from datetime import datetime, timezone

from core.accuracy import DAY_MS, WEEK_MS, HitRate, call_hit, report_hit, score_due_calls, score_reports, summary
from core.agents.schemas import DirectorReport, Scenario, TradePlan, TrapReport
from core.accuracy import record_report
from core.store import Store


def test_call_hit_rules_around_sigma():
    assert call_hit("BULLISH", 0.3, 1.0) and not call_hit("BULLISH", 0.2, 1.0)
    assert call_hit("BEARISH", -0.3, 1.0) and not call_hit("BEARISH", 0.1, 1.0)
    assert call_hit("NEUTRAL", 0.4, 1.0) and not call_hit("NEUTRAL", 0.6, 1.0)
    assert call_hit("BULLISH", 0.3, None)  # sigma fallback 1.0


def test_report_hit_rules():
    assert report_hit("BULL_TRAP", -0.1, 1.0, None, "24h") and not report_hit("BULL_TRAP", 0.1, 1.0, None, "24h")
    assert report_hit("BEAR_TRAP", 0.1, 1.0, None, "24h")
    assert report_hit("RANGE_TRAP", 0.9, 1.0, None, "24h") and not report_hit("RANGE_TRAP", 1.1, 1.0, None, "24h")
    assert report_hit("RANGE_TRAP", 2.0, 1.0, None, "7d") and not report_hit("RANGE_TRAP", math.sqrt(7) + 0.1, 1.0, None, "7d")
    assert report_hit("NO_TRAP", 0.5, 1.0, "BULLISH", "24h") and not report_hit("NO_TRAP", -0.5, 1.0, "BULLISH", "24h")
    assert report_hit("NO_TRAP", 0.2, 1.0, "NEUTRAL", "24h")


def test_score_due_calls_and_summary():
    s = Store(":memory:")
    s.add_call(0, "live", "BULLISH", 0.5, 100.0, 1.0, 1000)
    s.add_call(0, "live", "BEARISH", 0.5, 100.0, 1.0, 1000)
    s.add_call(0, "weekly", "NEUTRAL", 0.2, 100.0, 1.0, 10_000)  # not due yet
    assert score_due_calls(s, now_ms=1000, price=101.0) == 2
    sm = summary(s)
    assert sm["by_category"]["live"].n == 2 and sm["by_category"]["live"].hits == 1
    assert sm["by_category"]["live"].small is True and sm["open_calls"] == 1
    assert HitRate(0, 0).rate is None and HitRate(4, 3).rate == 0.75


def _report(rid, cls):
    d = DirectorReport(executive_summary="e", trap_classification=cls, classification_rationale="r",
                       scenarios=[Scenario(name="a", probability=50, path="p", invalidation="i"), Scenario(name="b", probability=50, path="p", invalidation="i")],
                       trade_plan=TradePlan(existing_position_management="h", new_entry_conditions="n", position_sizing="s", stop_loss="l", max_leverage="3x"))
    return TrapReport(report_id=rid, generated_at=datetime.now(timezone.utc), window="w", provider="fake", director=d)


def test_record_and_score_reports_at_both_horizons():
    s = Store(":memory:")
    assert record_report(s, _report("TIR-A", "BULL_TRAP"), price=100.0, sigma24_pct=1.0, majority_dir="BEARISH", now_ms=0) is True
    assert record_report(s, _report("TIR-A", "BULL_TRAP"), price=100.0, sigma24_pct=1.0, majority_dir="BEARISH", now_ms=0) is False
    assert score_reports(s, now_ms=DAY_MS - 1, price=99.0) == 0
    assert score_reports(s, now_ms=DAY_MS, price=99.0) == 1
    assert score_reports(s, now_ms=WEEK_MS, price=102.0) == 1
    r = s.reports()[0]
    assert r["hit_24h"] == 1 and r["hit_7d"] == 0
    sm = summary(s)
    assert sm["by_classification_24h"]["BULL_TRAP"].hits == 1 and sm["by_classification_7d"]["BULL_TRAP"].hits == 0
    assert record_report(s, None, 1.0, 1.0, "NEUTRAL", 0) is False
