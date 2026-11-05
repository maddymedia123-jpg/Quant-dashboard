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
    return pa.replay(plan, T0, {"15m": f, "1h": hourly(f)}, flows, now or T0 + 48 * H, 83_000.0)


UP = [(83600, 83400, 83450)] * 4            # hour 0: into the zone, closing inside it
REJECT = [(83500, 83000, 83050)] * 4        # hour 1: closes below 83,077
FALL = [(83000, 82100, 82150)] * 4          # hour 2: through the first target 82,237


def test_the_whole_short_reads_in_plain_words():
    s = steps(UP + REJECT + FALL, ratios=[1.0, 0.8, 0.8])
    assert [x.name for x in s] == ["watch", "confirmed", "target1"]
    watch = s[0].headline
    lines = watch.splitlines()
    assert lines[0] == "🟡 WATCH · BTC $83,000"
    # 4 Nov, 21:00 UTC: "Price reached the short zone" under a headline price below the zone read as a
    # contradiction; the candle high touched it. Now: touched, the candle high, then the current price.
    assert lines[1].startswith("The sell zone $83,075–$83,535 was touched: the candle high reached $83,600 at ")
    assert lines[1].endswith(" BTC is now $83,000, back below the zone.")
    assert "Price reached" not in watch
    assert lines[2].startswith("Sell only if a 1-hour candle closes below $83,077 with sellers doing more of the trading. "
                               "Next close: "), "written from the plan's numbers, not the model's wording"
    assert "1h close below 83,077" not in watch, "the model's own confirmation text is not sent"
    assert lines[3] == "Forget it if a 1-hour candle closes above $83,535. Stop $85,238 · targets $82,237 / $80,600 / $78,000."
    assert "turning hours" in lines[4]
    for gone in ("Context", "report", "fake breakout", "Cancelled if", "Plan wrong", "Next check"):
        assert gone not in watch, gone
    c = s[1].headline.splitlines()
    assert c[0] == "✅ SELL NOW · BTC $83,050"
    assert c[1] == "A 1-hour candle closed at $83,050, below $83,077, with sellers doing more of the trading."
    assert c[2] == "Stop $85,238 (2.6% away) · targets $82,237 / $80,600 / $78,000 (1.0% / 3.0% / 6.1%)."
    assert c[3] == "Entry was planned at $83,075–$83,535; price is past it, so the risk is the distance to the stop from here."
    assert c[4] == f"Good until {pa._when(T0 + 24 * H)}."
    t = s[2].headline.splitlines()
    assert t[0] == "🎯 TARGET 1 OF 3 HIT · BTC $82,237"
    assert t[1] == "Short from $83,050: +0.98% in your favour."
    assert t[2] == "Next target $80,600."
    assert t[3] == "Suggested: take some profit here and move the stop to $83,050 (break-even)."
    # entry inside the zone: the entry line says so; the second target no longer suggests the break-even move
    inside = dict(PLAN, confirm_close=83_400)
    s2 = steps(UP + [(83500, 83000, 83300)] * 4 + FALL + [(80500, 80300, 80400)] * 4, ratios=[1.0, 0.8, 0.8, 0.8], plan=inside)
    assert [x.name for x in s2] == ["watch", "confirmed", "target1", "target2"]
    assert "Entry was planned at $83,075–$83,535; price is inside it." in s2[1].headline
    assert s2[3].headline.startswith("🎯 TARGET 2 OF 3 HIT · BTC $80,600\n") and "Suggested: take some profit here." in s2[3].headline


def test_a_close_without_the_flow_keeps_waiting():
    """The 20 Oct case: the close was there, but taker buy/sell was 1.14 - no confirmation."""
    s = steps(UP + REJECT + FALL, ratios=[1.0, 1.14, 1.2], now=T0 + 3 * H)
    assert [x.name for x in s] == ["watch"]


def test_a_close_through_the_far_side_cancels():
    s = steps(UP + [(83700, 83500, 83650)] * 4)
    assert [x.name for x in s] == ["watch", "cancelled"]
    n = s[1].headline.splitlines()
    assert n[0] == "⚪ NO TRADE · short off · BTC $83,650"
    assert n[1].startswith("A 1-hour candle closed at $83,650, above the zone $83,075–$83,535, before any confirmation. The sell idea from the ")
    assert n[1].endswith(" UTC report is over.") and n[2] == "Next: the hourly Pulse, or a new report."
    through = steps(UP + [(85300, 83000, 83100)] * 4)
    assert [x.name for x in through] == ["watch", "cancelled"]
    assert through[1].headline.startswith("⚪ NO TRADE · short off · BTC $85,238\nPrice went through the stop $85,238 before any confirmation. The sell idea from the ")


def test_the_stop_after_entry_reads_as_wrong():
    s = steps(UP + REJECT + [(85300, 83000, 85250)] * 4, ratios=[1.0, 0.8, 1.2])
    assert [x.name for x in s] == ["watch", "confirmed", "stopped"]
    x = s[2].headline.splitlines()
    assert x[0] == "🔴 STOPPED OUT · BTC $85,238"
    assert x[1].startswith("Short from $83,050 hit the stop $85,238: 2.63% loss. The sell idea from the ") and x[1].endswith(" UTC report was wrong.")
    assert x[2] == "Next: the hourly Pulse, or a new report. No re-entry from this plan."
    assert "-2.63" not in s[2].headline and "bull trap" not in s[2].headline


def test_never_reaching_the_zone_expires():
    s = steps([(83000, 82900, 82950)] * 8, now=T0 + 30 * H)
    assert [x.name for x in s] == ["expired"]
    assert s[0].headline.startswith("⌛ PLAN EXPIRED · BTC $83,000\nThe sell idea from the ")
    assert f" UTC report ran out at {pa._when(T0 + 24 * H)}: price never reached the zone $83,075–$83,535. No trade.\nNext: the hourly Pulse, or a new report." in s[0].headline
    touched = steps(UP + [(83500, 83200, 83300)] * 4, now=T0 + 30 * H)
    assert [x.name for x in touched] == ["watch", "expired"]
    assert ": the zone was touched but no 1-hour candle confirmed. No trade." in touched[1].headline
    assert steps([(83000, 82900, 82950)] * 8, now=T0 + 3 * H) == [], "still waiting: nothing to say"


def test_a_long_plan_mirrors_it():
    plan = {"side": "LONG", "zone_low": 80000, "zone_high": 80400, "confirm_timeframe": "1h", "confirm_close": 80500,
            "stop": 79500, "targets": [81500, 81000], "expires_hours": 24}
    path = [(80300, 80100, 80200)] * 4 + [(80600, 80200, 80550)] * 4 + [(81100, 80500, 81050)] * 4
    s = steps(path, plan=plan)
    assert [x.name for x in s] == ["watch", "confirmed", "target1"] and "TARGET 1 OF 2 HIT · BTC $81,000" in s[2].headline
    assert s[1].headline.startswith("✅ BUY NOW · BTC $80,550\nA 1-hour candle closed at $80,550, above $80,500.\n")
    assert "Buy only if a 1-hour candle closes above $80,500. Next close: " in s[0].headline
    assert "Forget it if a 1-hour candle closes below $80,000. Stop $79,500 · targets $81,000 / $81,500." in s[0].headline
    assert "the candle low reached $80,100 at " in s[0].headline, "a long is touched from above: the candle low"
    assert "BTC is now $83,000, back above the zone." in s[0].headline


def test_the_watch_says_where_price_stands_against_the_zone_now():
    """The current price is named after the touch, with its side of the zone: inside, back below (a short
    that retreated), or above (through the zone toward the stop)."""
    f = fine(UP)
    frames = {"15m": f, "1h": hourly(f)}
    inside = pa.replay(PLAN, T0, frames, None, T0 + H, 83_300.0)[0].headline
    assert inside.startswith("🟡 WATCH · BTC $83,300") and "BTC is now $83,300, inside the zone." in inside
    above = pa.replay(PLAN, T0, frames, None, T0 + H, 83_900.0)[0].headline
    assert "BTC is now $83,900, above the zone." in above
    unknown = pa.replay(PLAN, T0, frames, None, T0 + H, None)[0].headline
    assert "the candle high reached $83,600 at " in unknown and "BTC is now" not in unknown
    t = pa.zone_touch(PLAN, T0, frames)
    assert t is not None and t.at_ms == T0 and t.extreme == 83_600.0 and t.side == "SHORT"
    assert pa.zone_touch(PLAN, T0, {"15m": fine([(83000, 82900, 82950)] * 4)}) is None
    assert pa.zone_touch(PLAN, T0, {}) is None and pa.zone_touch({"side": "NONE"}, T0, frames) is None


def test_a_plan_made_inside_an_event_regime_says_so_on_the_watch():
    """The Pulse's regime line says 'no range plan, smaller size'; a report's plan sent the same hour
    contradicted it. A plan tagged with the regime now says it was made inside it, with its halved size,
    and that the no-range rule is the day plan's."""
    plan = dict(PLAN, event_regime="Congressional Elections", risk_pct=0.5, max_leverage=2.5)
    watch = steps(UP, plan=plan)[0].headline
    assert ("Event regime (Congressional Elections): the report made this plan inside it, with size halved "
            "(risk 0.5%, max 2.5x). The no-range-plan rule belongs to the day plan, not to this trade.") in watch
    bare = steps(UP, plan=dict(plan, risk_pct=None, max_leverage=None))[0].headline
    assert "inside it, with size halved. The no-range-plan rule" in bare
    assert "Event regime" not in steps(UP)[0].headline


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


def test_the_newest_plan_of_either_pipeline_is_watched(tmp_path):
    """Since 5 November the classic Director writes the same numbers plan, so the newest report that
    carries one is the plan being watched, whichever pipeline wrote it. A report saved without a plan (a
    Director that failed, a report from before plans existed) does not silence the plan before it."""
    s = Store(tmp_path / "p.sqlite")
    s.add_report("TIR1-OLD", T0, "BULL_TRAP", 1.0, 1.0, "NEUTRAL", mode="1-agent", plan=json.dumps(PLAN))
    s.add_report("TIR-NOPLAN", T0 + 1, "NO_TRAP", 1.0, 1.0, "NEUTRAL")
    assert pa.latest_plan_report(s)["report_id"] == "TIR1-OLD", "a report without a plan leaves the watched plan alone"
    s.add_report("TIR-15", T0 + 2, "RANGE_TRAP", 1.0, 1.0, "NEUTRAL", mode="15-agent", plan=json.dumps(PLAN))
    assert pa.latest_plan_report(s)["report_id"] == "TIR-15", "the classic report's plan replaces it"
    s.add_report("TIR1-NONE", T0 + 3, "RANGE_TRAP", 1.0, 1.0, "NEUTRAL", mode="1-agent", plan=json.dumps({"side": "NONE"}))
    assert pa.latest_plan_report(s)["report_id"] == "TIR1-NONE", "a stand-aside is a plan: it replaces the trade"
    empty = Store(tmp_path / "e.sqlite")
    empty.add_report("TIR-OLDEST", T0, "NO_TRAP", 1.0, 1.0, "NEUTRAL")
    assert pa.latest_plan_report(empty) is None
    s.close()
    empty.close()


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


# ---------- no jargon reaches the bot ----------
def test_no_bot_message_carries_a_banned_word():
    """His Section 8: the reader is not a trader. Every plan step, cascade alert and TRAP message is scanned."""
    from core import market_alerts as ma
    from core.indicators.cascade import Cascade, TimeframeRead
    msgs = []
    for path, ratios in ((UP + REJECT + FALL, [1.0, 0.8, 0.8]), (UP + [(83700, 83500, 83650)] * 4, None),
                         (UP + REJECT + [(85300, 83000, 85250)] * 4, [1.0, 0.8, 1.2]),
                         ([(83000, 82900, 82950)] * 8, None)):
        msgs += [x.headline for x in steps(path, ratios=ratios, now=T0 + 30 * H)]
    plan = dict(PLAN, confirmations=["4h CVD remains net negative", "RSI below 70", "VWAP rejection"])
    msgs += [x.headline for x in steps(UP + REJECT + FALL, ratios=[1.0, 0.8, 0.8], plan=plan)]
    eng = tuple(TimeframeRead(t, "engine", "neutral", "MIXED", "unclear") for t in ("15m", "1h"))
    for prev, state, bias in (("B", "A", "bullish"), ("A", "C", "bearish"), ("A", "B", "bullish"), ("B", "N", "neutral")):
        c = Cascade(state, "", TimeframeRead("1d", "bias", bias, "BULL", "bullish"),
                    TimeframeRead("4h", "anchor", "neutral", "MIXED", "unclear"), eng, bias, None, None, 0, "x")
        msgs.append(ma.cascade_event("live", "Live Recon", prev, c, T0, 82_000.0).headline)
    trap = {"id": 1, "side": "BEAR_TRAP", "timeframe": "1h", "level": 81963, "plays_out": 82378, "invalidation": 81756,
            "declared_ms": T0, "category": "live", "status": "invalidated", "resolved_ms": T0 + 1, "resolved_price": 81700}
    msgs += [e.headline for e in ma.trap_events([trap], T0 + 2)]
    assert len(msgs) >= 12
    for msg in msgs:
        low = msg.lower()
        for word in pa.BANNED_WORDS:
            assert word not in low, f"{word!r} in: {msg}"


def test_the_turning_window_is_the_measured_us_session():
    from core.indicators import turn_window as tw
    morning = T0 - (T0 % (24 * H)) + 6 * H
    w = tw.next_window(morning)
    assert not w.active and w.label.startswith("12:00–16:00 UTC (17:00–21:00 PKT")
    assert tw.next_window(morning + 7 * H).active, "13:00 UTC is inside"
    late = tw.next_window(morning + 11 * H)                       # 17:00 UTC: today's has ended
    assert late.start_ms == w.start_ms + 24 * H
    state = tw.to_state(morning)
    assert state["share_of_turns"] == 0.35 and "Fibonacci" in state["note"] and "no edge" in state["note"]
