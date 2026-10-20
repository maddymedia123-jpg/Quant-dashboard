"""Plan alerts: the 1-agent plan watched against candles - WATCH, CONFIRMED, NO SETUP, TARGET, STOPPED,
EXPIRED - each once, in plain words. And the claim that stops two open pages sending one alert twice."""
import json

import pandas as pd
import pytest

from core import plan_alerts as pa
from core.store import Store

H = 3_600_000
T0 = 1_790_000_000_000 - (1_790_000_000_000 % H)
PLAN = {"side": "SHORT", "zone_low": 83075, "zone_high": 83535, "confirm_timeframe": "1h", "confirm_close": 83077,
        "taker_ratio_limit": 0.90, "confirmations": ["1h close below 83,077", "taker buy/sell below 0.90"],
        "stop": 85238, "targets": [80600, 82237, 78000], "expires_hours": 24}


def fine(path):
    """15m candles from (high, low, close) tuples, starting at T0."""
    return pd.DataFrame([{"timestamp": T0 + i * 900_000, "open": c, "high": h, "low": lo, "close": c}
                         for i, (h, lo, c) in enumerate(path)])


def hourly(f):
    g = f.assign(b=(f["timestamp"] // H) * H).groupby("b")
    return pd.DataFrame({"timestamp": g["timestamp"].first().index, "open": g["open"].first().values,
                         "high": g["high"].max().values, "low": g["low"].min().values,
                         "close": g["close"].last().values})


def flow(ratios):
    return pd.DataFrame({"timestamp": [T0 + i * H for i in range(len(ratios))],
                         "buy": [r * 100 for r in ratios], "sell": [100.0] * len(ratios)})


def steps(path, ratios=None, now=None, plan=PLAN):
    f = fine(path)
    flows = {"1h": flow(ratios)} if ratios is not None else None
    return pa.replay(plan, T0, {"15m": f, "1h": hourly(f)}, flows, now or T0 + 48 * H, 83_000.0,
                     "BULL_TRAP", "Live Recon pause in an uptrend")


UP = [(83600, 83400, 83450)] * 4            # hour 0: into the zone, closing inside it
REJECT = [(83500, 83000, 83050)] * 4        # hour 1: closes below 83,077
FALL = [(83000, 82100, 82150)] * 4          # hour 2: through the first target 82,237


def test_the_whole_short_reads_in_plain_words():
    s = steps(UP + REJECT + FALL, ratios=[1.0, 0.8, 0.8])
    assert [x.name for x in s] == ["watch", "confirmed", "target1"]
    watch = s[0].headline
    assert watch.startswith("🟡 WATCH · BTC $83,000") and "short zone ($83,075–$83,535)" in watch
    assert "from the bull trap report" in watch and "Next check: 1h close at" in watch
    assert "Short only if: 1h close below 83,077; taker buy/sell below 0.90." in watch
    assert "Cancelled if a 1h candle closes above $83,535. Plan wrong past $85,238." in watch
    assert watch.endswith("Context: Live Recon pause in an uptrend")
    assert s[1].headline.startswith("✅ SETUP CONFIRMED · short · BTC $83,050") and "(taker buy/sell 0.80)" in s[1].headline
    assert "targets $82,237 / $80,600 / $78,000" in s[1].headline, "nearest target first"
    assert s[2].headline.startswith("🎯 TARGET 1 HIT · $82,237") and "+0.98%" in s[2].headline
    assert "Next: $80,600." in s[2].headline


def test_a_close_without_the_flow_keeps_waiting():
    """The 20 Oct case: the close was there, but taker buy/sell was 1.14 - no confirmation."""
    s = steps(UP + REJECT + FALL, ratios=[1.0, 1.14, 1.2], now=T0 + 3 * H)
    assert [x.name for x in s] == ["watch"]


def test_a_close_through_the_far_side_cancels():
    s = steps(UP + [(83700, 83500, 83650)] * 4)
    assert [x.name for x in s] == ["watch", "cancelled"]
    assert "⚪ NO SETUP · short cancelled" in s[1].headline and "closed at $83,650, above the zone" in s[1].headline


def test_the_stop_after_entry_reads_as_wrong():
    s = steps(UP + REJECT + [(85300, 83000, 85250)] * 4, ratios=[1.0, 0.8, 1.2])
    assert [x.name for x in s] == ["watch", "confirmed", "stopped"]
    assert s[2].headline.startswith("🔴 STOPPED · short from $83,050") and "-2.63%" in s[2].headline
    assert "bull trap call was wrong" in s[2].headline


def test_never_reaching_the_zone_expires():
    s = steps([(83000, 82900, 82950)] * 8, now=T0 + 30 * H)
    assert [x.name for x in s] == ["expired"] and "never reached the zone" in s[0].headline
    assert steps([(83000, 82900, 82950)] * 8, now=T0 + 3 * H) == [], "still waiting: nothing to say"


def test_a_long_plan_mirrors_it():
    plan = {"side": "LONG", "zone_low": 80000, "zone_high": 80400, "confirm_timeframe": "1h", "confirm_close": 80500,
            "stop": 79500, "targets": [81500, 81000], "expires_hours": 24}
    path = [(80300, 80100, 80200)] * 4 + [(80600, 80200, 80550)] * 4 + [(81100, 80500, 81050)] * 4
    s = steps(path, plan=plan)
    assert [x.name for x in s] == ["watch", "confirmed", "target1"] and "TARGET 1 HIT · $81,000" in s[2].headline


def test_no_trade_plans_are_silent():
    assert steps(UP, plan={"side": "NONE"}) == []


# ---------- events: keys, staleness, lateness ----------
def row(plan=PLAN, rid="TIR1-A"):
    return {"report_id": rid, "ts_ms": T0, "plan": json.dumps(plan), "classification": "BULL_TRAP"}


def test_each_step_has_a_fixed_key_and_late_ones_say_when():
    f = fine(UP + REJECT + FALL)
    frames, flows = {"15m": f, "1h": hourly(f)}, {"1h": flow([1.0, 0.8, 0.8])}
    ev = pa.events(row(), frames, flows, T0 + H + 5 * 60_000, 83_050.0)
    assert [e.dedup_key for e in ev] == ["plan:TIR1-A:watch"] and ev[0].kind == "plan"
    ev = pa.events(row(), frames, flows, T0 + 2 * H + 10 * 60_000, 82_000.0)
    assert [e.dedup_key for e in ev] == ["plan:TIR1-A:confirmed", "plan:TIR1-A:target1"], \
        "the WATCH is over two hours old by now: history, not news"
    assert ev[0].kind == "plan_confirmed" and "(This happened at" not in ev[0].headline
    late = pa.events(row(), frames, flows, T0 + 2 * H + 40 * 60_000, 82_000.0)
    assert "(This happened at" in late[0].headline, "sent 40 minutes after the close, it says when it closed"


def test_an_old_watch_is_not_sent_as_news():
    f = fine(UP + [(83500, 83200, 83300)] * 20)
    ev = pa.events(row(), {"15m": f, "1h": hourly(f)}, None, T0 + 5 * H, 83_300.0)
    assert ev == []


def test_only_the_newest_one_agent_plan_is_watched(tmp_path):
    s = Store(tmp_path / "p.sqlite")
    s.add_report("TIR1-OLD", T0, "BULL_TRAP", 1.0, 1.0, "NEUTRAL", mode="1-agent", plan=json.dumps(PLAN))
    s.add_report("TIR-15", T0 + 1, "NO_TRAP", 1.0, 1.0, "NEUTRAL")
    assert pa.latest_plan_report(s)["report_id"] == "TIR1-OLD", "a 15-agent report has no plan to replace it"
    s.add_report("TIR1-NEW", T0 + 2, "RANGE_TRAP", 1.0, 1.0, "NEUTRAL", mode="1-agent", plan=None)
    assert pa.latest_plan_report(s) is None, "the newest 1-agent report replaces the plan, even with none"
    s.close()


# ---------- two pages, one message ----------
def test_two_pages_cannot_both_send_the_same_alert(tmp_path):
    s = Store(tmp_path / "a.sqlite")
    s.add_market_alert(T0, "plan", "k1", "h1")
    s.add_market_alert(T0 + 1, "plan", "k2", "h2")
    first = s.claim_market_alerts(T0 + 5)
    second = s.claim_market_alerts(T0 + 6)
    assert [r["dedup_key"] for r in first] == ["k1", "k2"] and second == [], "the second page finds them taken"
    s.mark_market_notified([first[0]["id"]])
    s.release_market_alerts([first[1]["id"]])                      # the channel refused k2
    again = s.claim_market_alerts(T0 + 7)
    assert [r["dedup_key"] for r in again] == ["k2"], "a refused alert is tried again, a sent one never"
    s.close()


def test_a_claim_from_a_page_that_died_is_released(tmp_path):
    s = Store(tmp_path / "b.sqlite")
    s.add_market_alert(T0, "plan", "k1", "h1")
    assert len(s.claim_market_alerts(T0)) == 1
    assert s.claim_market_alerts(T0 + 60_000) == [], "still claimed a minute later"
    assert len(s.claim_market_alerts(T0 + Store.CLAIM_TIMEOUT_MS + 1)) == 1, "released after the timeout"
    s.close()


@pytest.mark.parametrize("ratio_limit", [None])
def test_without_a_taker_rule_the_close_alone_confirms(ratio_limit):
    plan = dict(PLAN, taker_ratio_limit=ratio_limit)
    s = steps(UP + REJECT + FALL, ratios=[1.2, 1.2, 1.2], plan=plan)
    assert [x.name for x in s] == ["watch", "confirmed", "target1"]
