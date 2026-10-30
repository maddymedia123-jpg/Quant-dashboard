"""Message 8/9: the Sticky Monday Master Plan and the self-audit card."""
from types import SimpleNamespace as NS

from core import qf_selfaudit, qf_weekly
from core.quant_forecast import DAY_MS
from core.store import Store

MONDAY = 1_790_553_600_000          # a Monday 00:00 UTC
H = 3_600_000


def weekly(direction="BULLISH", confidence=0.6):
    return NS(price=80_000.0,
              direction=NS(direction=direction, confidence=confidence, invalidation=76_000.0, drivers=["a", "b", "c", "d"]),
              levels=[NS(kind="support", price=78_000.0), NS(kind="support", price=74_000.0),
                      NS(kind="resistance", price=84_000.0)],
              vol=NS(exp_low=77_000.0, exp_high=83_000.0))


def test_the_week_starts_on_monday_utc():
    assert qf_weekly.week_start(MONDAY) == MONDAY
    assert qf_weekly.week_start(MONDAY + 3 * DAY_MS + 5 * H) == MONDAY
    assert qf_weekly.week_start(MONDAY - 1) == MONDAY - 7 * DAY_MS
    assert qf_weekly.week_start(MONDAY + 7 * DAY_MS) == MONDAY + 7 * DAY_MS


def test_the_monday_plan_reads_bias_walls_and_baseline_in_plain_words():
    p = qf_weekly.build(weekly(), {"put_wall": 76_000.0, "call_wall": 86_000.0}, MONDAY + H)
    assert p["weekly_bias"] == "BULLISH" and p["weekly_gex_walls"] == [76_000.0, 86_000.0]
    assert p["baseline"] == {"floor": 78_000.0, "ceiling": 84_000.0} and p["week_ms"] == MONDAY
    s = p["macro_baseline_summary"]
    assert "leans bullish (a moderate lean)" in s and "floor $78,000 and ceiling $84,000" in s
    assert "$76,000 below and $86,000 above" in s and "weekly close below $76,000" in s
    assert len(p["drivers"]) == 3
    r = qf_weekly.build(weekly("NEUTRAL", 0.1), None, MONDAY)
    assert r["weekly_bias"] == "RANGE" and "range week" in r["macro_baseline_summary"] and r["weekly_gex_walls"] == [None, None]
    assert "dealers" not in r["macro_baseline_summary"] and "flips" not in r["macro_baseline_summary"]


def test_the_monday_plan_is_sticky_for_the_week_and_new_on_the_next():
    s = Store(":memory:")
    assert qf_weekly.ensure(s, None, None, MONDAY) is None, "nothing to build from yet"
    first = qf_weekly.ensure(s, weekly("BULLISH"), None, MONDAY + H)
    assert first["weekly_bias"] == "BULLISH"
    again = qf_weekly.ensure(s, weekly("BEARISH"), None, MONDAY + 4 * DAY_MS)
    assert again["weekly_bias"] == "BULLISH" and again["created_ms"] == MONDAY + H, "the week keeps its first plan"
    nxt = qf_weekly.ensure(s, weekly("BEARISH"), None, MONDAY + 7 * DAY_MS + H)
    assert nxt["weekly_bias"] == "BEARISH" and nxt["week_ms"] == MONDAY + 7 * DAY_MS


def _row(high, low, *, high_hit, low_hit, invalidated=False, levels=(), high_ms=None, low_ms=None, day=MONDAY):
    plan = {"day_ms": day, "regime": "bullish", "projected_high": {"price": 82_000.0, "t_ms": day + 14 * H},
            "projected_low": {"price": 79_000.0, "t_ms": day + 4 * H},
            "levels": [{"label": lb, "price": px, "side": "above", "p_touch": 0.3} for lb, px in levels]}
    out = {"scored": True, "high": high, "low": low, "high_hit": high_hit, "low_hit": low_hit, "invalidated": invalidated,
           "high_ms": high_ms or day + 14 * H, "low_ms": low_ms or day + 4 * H}
    return {"version": 0, "day_ms": day, "payload": plan, "outcome": out}


def test_the_self_audit_names_the_miss_and_a_cause_by_rule():
    assert qf_selfaudit.self_audit(_row(82_100, 79_050, high_hit=True, low_hit=True)) is None, "both landed: no card"
    # ran past the high to an estimated liquidation zone: gravity
    a = qf_selfaudit.self_audit(_row(83_640, 79_100, high_hit=False, low_hit=True,
                                     levels=[("est. liquidation zone", 83_500.0), ("POC", 83_000.0)]))
    assert a["side"] == "high" and a["missed_pct"] == 2.0 and a["beyond"] is True
    assert a["root_cause"].startswith("liquidation gravity underestimated: price ran to the est. liquidation zone at $83,500")
    assert a["note"].startswith("🔍 SELF-AUDIT REPORT: High projection missed by 2.0%. Root cause: liquidation gravity")
    assert a["note"].endswith("System multipliers: not changed (too little history to calibrate)."), \
        "a plan built without a calibration says so"
    # a dealer wall counts as gravity too
    w = qf_selfaudit.self_audit(_row(83_640, 79_100, high_hit=False, low_hit=True, levels=[("call wall", 83_200.0)]))
    assert "dealer wall at $83,200" in w["root_cause"]
    # far wider than expected with no level in the way: volatility underestimated
    v = qf_selfaudit.self_audit(_row(84_000, 78_000, high_hit=False, low_hit=False))
    assert v["side"] == "high" and v["root_cause"].startswith("volatility underestimated: the day traded 2.0x")
    # never got near either level: overestimated
    n = qf_selfaudit.self_audit(_row(80_500, 79_800, high_hit=False, low_hit=False))
    assert n["root_cause"].startswith("volatility overestimated")
    # breached the invalidation, short of the high, range about right: the lean
    i = qf_selfaudit.self_audit(_row(81_200, 78_600, high_hit=False, low_hit=True, invalidated=True))
    assert i["root_cause"].startswith("the lean was on the wrong side")
    # high came six hours late, otherwise fine: timing
    t = qf_selfaudit.self_audit(_row(81_200, 78_600, high_hit=False, low_hit=True, high_ms=MONDAY + 20 * H))
    assert t["root_cause"] == "timing: the high printed 6 hours later than projected"
    assert qf_selfaudit.self_audit({"payload": {"regime": "locked"}, "outcome": {"scored": True}}) is None


def test_the_latest_audit_is_the_most_recent_scored_morning_plan():
    class S:
        def qf_plans(self, limit=20):
            return [{"version": 1, "day_ms": MONDAY + DAY_MS, "payload": {}, "outcome": None},
                    _row(83_640, 79_100, high_hit=False, low_hit=True, levels=[("air pocket", 83_400.0)], day=MONDAY + DAY_MS),
                    _row(84_000, 78_000, high_hit=False, low_hit=False, day=MONDAY)]
    a = qf_selfaudit.latest(S(), MONDAY + 2 * DAY_MS)
    assert a["day_ms"] == MONDAY + DAY_MS and "air pocket at $83,400" in a["root_cause"]
    assert qf_selfaudit.latest(S(), MONDAY + 10 * DAY_MS) is None, "nothing within the lookback"
