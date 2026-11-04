"""Market Pulse Dispatch: his template, once per report and once per fresh hour boundary (the 1H, 2H and 4H
closes, named), plain words only."""
import json
from types import SimpleNamespace as NS

from core import alerts, pulse
from core.alerts import Event
from core.plan_alerts import BANNED_WORDS

H = 3_600_000
CLOSE = 1_792_713_600_000 - 1_792_713_600_000 % (4 * H)       # a 4h boundary


def lvl(kind, price):
    return NS(kind=kind, price=price)


def live(direction="BEARISH", confidence=0.7, invalidation=81_300.0):
    return NS(price=81_100.0, direction=NS(direction=direction, confidence=confidence, invalidation=invalidation),
              levels=[lvl("support", 80_800.0), lvl("support", 80_200.0), lvl("resistance", 81_600.0)],
              vol=NS(exp_low=80_500.0, exp_high=81_700.0))


RANK = {"Low": 1, "Medium": 2, "High": 3}


def market(events=(), heads=()):
    """`events`: calendar rows (impact High unless given). upcoming() filters like the real snapshot."""
    events = [{"impact": "High", "country": "USD", **e} for e in events]
    upcoming = lambda now, within, min_impact="High", countries=("USD",): [  # noqa: E731
        e for e in events if RANK[e["impact"]] >= RANK[min_impact] and now - H <= e["time_ms"] <= now + within]
    cal = NS(available=True, upcoming=upcoming, events=events)
    news = NS(available=True, recent=lambda now, within, impact: list(heads))
    return NS(calendar=cal, news=news, spot=NS(last=81_100.0))


PLAN = {"side": "SHORT", "zone_low": 81_800, "zone_high": 82_200, "stop": 82_650, "targets": [80_400, 80_900],
        "expires_hours": 12}


def test_the_message_follows_his_template_in_order():
    t = pulse.close_event(live(), market(), CLOSE + 60_000).headline
    order = ["MARKET PULSE DISPATCH", "⏰ Time:", "🎯 MARKET DIRECTION: BEARISH", "💡 WHAT IS HAPPENING:",
             "🕒 EXACT TIME TO WATCH:", "Next Turning Window:", "📍 IMPORTANT PRICE LEVELS:",
             "Target Level: $80,500", "Invalidation (Safety Line): $81,300", "⚠️ IMPORTANT NEWS / EVENT WARNINGS:"]
    at = [t.index(s) for s in order]
    assert at == sorted(at)
    assert "PKT" in t.split("Next Turning Window:")[1].splitlines()[0]
    assert len(t) < alerts.MAX_CHARS


def test_the_turning_window_says_which_day_it_is_on():
    """29 Oct: the 16:00 UTC (21:00 PKT) dispatch read 'Between 17:00 PKT and 21:00 PKT' - tomorrow's window,
    read as one that had just ended."""
    day = CLOSE - CLOSE % (24 * H)
    line = lambda ms: pulse.close_event(live(), market(), ms).headline.split("Next Turning Window:")[1].splitlines()[0]  # noqa: E731
    assert line(day + 16 * H).strip() == "Tomorrow, between 17:00 PKT and 21:00 PKT"
    assert line(day + 8 * H).strip() == "Later today, between 17:00 PKT and 21:00 PKT"
    assert line(day + 12 * H).strip().startswith("NOW, until 21:00 PKT")


def test_no_jargon_reaches_the_reader():
    for d in ("BULLISH", "BEARISH", "NEUTRAL"):
        t = pulse.close_event(live(d), market(), CLOSE).headline.lower()
        assert not [w for w in BANNED_WORDS if w in t], d
    t = pulse.report_event({"report_id": "R1", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}, live(), market(),
                           CLOSE).headline.lower()
    assert not [w for w in BANNED_WORDS if w in t]


def test_sideways_names_both_edges():
    t = pulse.close_event(live("NEUTRAL", 0.1, None), market(), CLOSE).headline
    assert "RANGE-BOUND" in t and "$80,800 or $81,600, whichever breaks first" in t
    assert "a 1-hour close below $80,800 or above $81,600 picks a side" in t


def test_one_per_close_and_only_while_fresh():
    a = pulse.close_event(live(), market(), CLOSE + 5 * 60_000)
    b = pulse.close_event(live(), market(), CLOSE + 25 * 60_000)
    assert a.dedup_key == b.dedup_key == f"pulse:4h:{CLOSE}", "the ledger's unique key makes it one message"
    assert pulse.close_event(live(), market(), CLOSE + 31 * 60_000) is None, "a late pulse is skipped, not sent"
    assert pulse.close_event(None, market(), CLOSE) is None


def test_every_hour_boundary_sends_one_pulse_naming_the_closes_it_covers():
    """Owner's decision 4 Nov (his Section 10): the 1H, 2H and 4H closes, one message per boundary."""
    four = pulse.close_event(live(), market(), CLOSE + 60_000)
    assert four.dedup_key == f"pulse:4h:{CLOSE}" and "DISPATCH · 1h + 2h + 4h candle closes · Live desk" in four.headline
    one = pulse.close_event(live(), market(), CLOSE + H + 60_000)
    assert one.dedup_key == f"pulse:1h:{CLOSE + H}" and "DISPATCH · 1h candle close · Live desk" in one.headline
    two = pulse.close_event(live(), market(), CLOSE + 2 * H + 60_000)
    assert two.dedup_key == f"pulse:2h:{CLOSE + 2 * H}" and "DISPATCH · 1h + 2h candle closes · Live desk" in two.headline
    three = pulse.close_event(live(), market(), CLOSE + 3 * H + 60_000)
    assert three.dedup_key == f"pulse:1h:{CLOSE + 3 * H}" and "1h candle close ·" in three.headline
    assert pulse.close_event(live(), market(), CLOSE + H + 31 * 60_000) is None, "an hourly pulse is as perishable as a 4h one"
    assert [pulse.closes_at(CLOSE + k * H) for k in range(5)] == [["1h", "2h", "4h"], ["1h"], ["1h", "2h"], ["1h"], ["1h", "2h", "4h"]]
    # with a live plan the hourly pulse speaks from the plan too, and still names the closes
    row = {"report_id": "TIR1-9", "ts_ms": CLOSE - 2 * H, "plan": json.dumps(PLAN), "plan_outcome": None}
    e = pulse.close_event(live("BULLISH"), market(), CLOSE + H + 60_000, plan_row=row)
    assert e.dedup_key == f"pulse:1h:{CLOSE + H}" and "1h candle close · plan from report TIR1-9" in e.headline


def test_a_report_with_a_plan_speaks_from_the_plan():
    row = {"report_id": "TIR1-1", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}
    e = pulse.report_event(row, live("BULLISH"), market(), CLOSE + 60_000)
    assert e.dedup_key == "pulse:report:TIR1-1"
    assert "MARKET DIRECTION: BEARISH" in e.headline, "the plan's side, not the desk's lean"
    assert "Target Level: $80,900" in e.headline and "Safety Line): $82,650" in e.headline
    assert "below the sell zone ($81,800–$82,200)" in e.headline
    assert "Do not sell here: wait for a rise into the zone, then sell only after a 1-hour candle closes below $81,800." in e.headline


def test_the_confirmation_names_the_candle_and_the_level_and_where_price_stands():
    plan = dict(PLAN, confirm_timeframe="4h", confirm_close=81_750, taker_ratio_limit=0.9)
    row = {"report_id": "R", "ts_ms": CLOSE, "plan": json.dumps(plan)}
    inside = pulse.report_event(row, live(), NS(**{**vars(market()), "spot": NS(last=82_000.0)}), CLOSE).headline
    assert "inside the sell zone" in inside
    assert "Sell only after a 4-hour candle closes below $81,750, with sellers doing more of the trading." in inside
    long_plan = {"side": "LONG", "zone_low": 80_000, "zone_high": 80_300, "stop": 79_600, "targets": [81_000]}
    above = pulse.report_event({"report_id": "L", "ts_ms": CLOSE, "plan": json.dumps(long_plan)}, live(),
                               market(), CLOSE).headline
    assert "above the buy zone ($80,000–$80,300)" in above
    assert "Do not buy here: wait for a dip into the zone, then buy only after a 1-hour candle closes above $80,300." in above


def test_the_pulse_names_the_touch_with_the_candle_high_then_the_current_price():
    """4 Nov 21:26: the Pulse said 'BTC is $80,088, below the sell zone' an hour after the candle high had
    touched the zone. With the candles to hand it says touched, the candle high, then where price is now."""
    import pandas as pd
    f = pd.DataFrame([{"timestamp": CLOSE + i * 900_000, "open": 81_700.0, "high": h, "low": 81_500.0, "close": 81_700.0}
                      for i, h in enumerate((81_750.0, 81_900.0, 81_820.0, 81_760.0))])
    m = NS(**{**vars(market()), "spot": NS(last=81_100.0, frames={"15m": f})})
    row = {"report_id": "TIR1-T", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}
    h = pulse.close_event(live(), m, CLOSE + 4 * H, plan_row=row).headline
    assert "The sell zone ($81,800–$82,200) was touched at " in h and ": the candle high reached $81,900." in h
    assert "BTC is now $81,100, back below it." in h
    assert "Do not sell here: wait for a rise into the zone, then sell only after a 1-hour candle closes below $81,800." in h
    assert "BTC is $81,100, below the sell zone" not in h
    # inside the zone after the touch: the step is the confirming close
    inside = NS(**{**vars(market()), "spot": NS(last=82_000.0, frames={"15m": f})})
    hi = pulse.close_event(live(), inside, CLOSE + 4 * H, plan_row=row).headline
    assert "BTC is now $82,000, inside it." in hi and "Sell only after a 1-hour candle closes below $81,800." in hi
    # no candles, or no touch yet: the old sentence
    none = pulse.close_event(live(), market(), CLOSE + 4 * H, plan_row=row).headline
    assert "BTC is $81,100, below the sell zone ($81,800–$82,200)." in none and "touched" not in none
    assert pulse.plan_read(row, 81_100.0)["happening"].startswith("BTC is $81,100, below the sell zone")


def test_a_plan_tagged_with_the_regime_says_so_in_the_pulse():
    now = CLOSE + 60_000
    tagged = dict(PLAN, event_regime="Congressional Elections", risk_pct=0.5, max_leverage=2.5)
    row = {"report_id": "TIR1-R", "ts_ms": CLOSE, "plan": json.dumps(tagged)}
    ev = [{"title": "Congressional Elections", "time_ms": now - 23 * H}]
    h = pulse.report_event(row, live(), market(ev), now).headline
    assert ("This plan was made inside the Congressional Elections event regime: size halved (risk 0.5%, max 2.5x). "
            "The no-range-plan rule below is the day plan's, not this trade's.") in h
    plain = pulse.report_event({"report_id": "P", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}, live(), market(ev), now).headline
    assert "made inside the" not in plain


def test_a_report_without_a_plan_sends_nothing_and_a_stale_one_is_skipped():
    """A 15-agent report, or a 1-agent 'stand aside', used to send the Live desk's view again: a compare
    run was two dispatches, and every Run was a message whether or not it said anything new."""
    row = {"report_id": "TIR1-2", "ts_ms": CLOSE, "plan": json.dumps({"side": "NONE"})}
    assert pulse.report_event(row, live(), market(), CLOSE) is None
    assert pulse.report_event({"report_id": "TIR-15", "ts_ms": CLOSE, "plan": None}, live(), market(), CLOSE) is None
    with_plan = {"report_id": "TIR1-3", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}
    assert pulse.report_event(with_plan, live(), market(), CLOSE) is not None
    assert pulse.report_event(with_plan, live(), market(), CLOSE + 31 * 60_000) is None
    assert pulse.report_event(None, live(), market(), CLOSE) is None


def test_the_desk_target_is_the_farther_of_the_first_level_and_the_range_edge():
    """29 Oct: 'Target Level: $80,801' sat 0.4% from price - the first floor, not a target. The range
    edge is the target unless a level sits beyond it."""
    assert pulse.desk_read(live()).get("target") == "$80,500", "exp_low 80,500 is beyond the first floor 80,800"
    near_edge = live()
    near_edge.vol = NS(exp_low=80_900.0, exp_high=81_700.0)
    assert pulse.desk_read(near_edge)["target"] == "$80,800", "the first floor sits beyond a narrow range"
    bull = live("BULLISH")
    assert pulse.desk_read(bull)["target"] == "$81,700"


def test_a_4h_close_while_a_plan_is_live_speaks_from_the_plan():
    """The close pulse named the desk's nearest level while the report pulse named the plan's target, so
    two messages an hour apart gave two targets for one trade."""
    row = {"report_id": "TIR1-9", "ts_ms": CLOSE - 2 * H, "plan": json.dumps(PLAN), "plan_outcome": None}
    e = pulse.close_event(live("BULLISH"), market(), CLOSE + 60_000, plan_row=row)
    assert e.dedup_key == f"pulse:4h:{CLOSE}" and "plan from report TIR1-9" in e.headline
    assert "MARKET DIRECTION: BEARISH" in e.headline and "Target Level: $80,900" in e.headline
    assert "Safety Line): $82,650" in e.headline
    # closed by the ledger, or out of hours: back to the desk
    done = dict(row, plan_outcome="stop")
    cancelled = dict(row, plan_outcome="cancelled")
    assert pulse.plan_live(cancelled, CLOSE) is False
    assert "Live desk" in pulse.close_event(live(), market(), CLOSE + 60_000, plan_row=cancelled).headline, "a cancelled plan hands the pulse back to the Live desk"
    assert "Live desk" in pulse.close_event(live(), market(), CLOSE + 60_000, plan_row=done).headline
    old = dict(row, ts_ms=CLOSE - 13 * H)                      # PLAN expires after 12 hours
    assert "Live desk" in pulse.close_event(live(), market(), CLOSE + 60_000, plan_row=old).headline
    assert pulse.plan_live(row, CLOSE) and not pulse.plan_live(None, CLOSE)
    assert pulse.plan_live({"report_id": "x", "ts_ms": CLOSE, "plan": json.dumps({"side": "NONE"})}, CLOSE) is False
    assert pulse.close_event(None, market(), CLOSE, plan_row=row) is not None, "a plan needs no desk"
    assert pulse.close_event(None, market(), CLOSE) is None


def test_news_lists_us_prints_and_fresh_headlines_or_says_there_are_none():
    ev = [{"title": "CPI m/m", "time_ms": CLOSE + 9 * H}]
    t = pulse.close_event(live(), market(ev, [{"title": "SEC delays ETF", "published_ms": 1, "topic": "etf"}]),
                          CLOSE).headline
    assert "US CPI m/m at" in t and "Headline: SEC delays ETF" in t
    assert "No US data releases or Bitcoin-moving headlines" in pulse.close_event(live(), market(), CLOSE).headline


def test_only_bitcoin_or_market_wide_headlines_and_bitcoin_ones_first():
    heads = [{"title": "Small chain drained in exploit", "published_ms": 9, "topic": "security", "assets": ["ETH"]},
             {"title": "Exchange abroad fights regulator in court", "published_ms": 8, "topic": "regulation"},
             {"title": "Leader eyes blockchain to dodge sanctions", "published_ms": 7, "topic": "geopolitics"},
             {"title": "Bitcoin holds $81K as crude hits a high", "published_ms": 3, "topic": "geopolitics",
              "assets": ["BTC"]},
             {"title": "SEC sues bitcoin miner", "published_ms": 2, "topic": "regulation", "assets": ["BTC"]}]
    t = pulse.close_event(live(), market(heads=heads), CLOSE).headline
    news = t.split("EVENT WARNINGS:")[1]
    assert "Small chain" not in news and "Exchange abroad" not in news
    assert news.index("Bitcoin holds $81K") < news.index("SEC sues bitcoin miner")
    assert "Leader eyes" not in news, "two Bitcoin headlines fill the two places"
    off_topic = pulse._news(market(heads=heads[:2]), CLOSE)
    assert off_topic.startswith("No US data releases or Bitcoin-moving headlines"), off_topic


def test_fed_odds_appear_only_with_a_fed_meeting_this_week():
    fomc = [{"title": "FOMC Statement", "time_ms": CLOSE + 2 * 86_400_000}]
    odds = NS(available=True, tradeable=lambda a: [{"question": "Will the Fed hold rates in October?",
                                                   "probability": 0.92, "volume_24h": 5}])
    near = NS(**{**vars(market()), "predictions": odds,
                 "calendar": NS(available=True, upcoming=lambda now, within, **kw: fomc, events=[])})
    assert 'markets give 92% to "Will the Fed hold rates in October?"' in pulse._news(near, CLOSE)
    far = NS(**{**vars(market()), "predictions": odds})
    assert "Fed meeting" not in pulse._news(far, CLOSE)


def test_long_batches_go_out_as_several_messages_under_the_limit():
    big = [Event("pulse", f"k{i}", "x" * 800) for i in range(5)]
    batches = alerts._batches(big)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert all(sum(len(e.headline) + 4 for e in b) <= alerts.MAX_CHARS for b in batches)
    huge = alerts._batches([Event("pulse", "k", "y" * 5000)])
    assert len(huge) == 1 and len(huge[0][0].headline) < alerts.MAX_CHARS


def test_only_the_messages_a_channel_accepted_count_as_delivered(monkeypatch):
    d = alerts.Dispatcher({"DISCORD_WEBHOOK_URL": "https://example.invalid/hook"})
    sent = []

    def fake_post(url, payload):
        sent.append(payload["content"])
        return len(sent) != 2                        # the second message is refused
    monkeypatch.setattr(d, "_post", fake_post)
    events = [Event("pulse", f"k{i}", "z" * 800) for i in range(5)]
    delivered, problems = d.send(events)
    assert len(sent) == 3 and [e.dedup_key for e in delivered] == ["k0", "k1", "k4"]
    assert problems == ["Discord did not accept the message"]


# ---------- election-day fixes 3 and 4: the event section names the regime and the scheduled prints ----------

def _event_section(t: str) -> list[str]:
    return [ln for ln in t.split("⚠️ IMPORTANT NEWS / EVENT WARNINGS:")[1].splitlines() if ln and ln != "=" * 40]


def test_the_event_section_leads_with_the_regime_then_printed_and_scheduled_prints():
    """4 Nov, 14:00 UTC: the election had printed 23 hours earlier (gone from the list), ISM Services at 15:00
    was Medium (never listed), and the section read 'No US data releases'. Now the regime leads, the print of
    the last day is named with its read, and a Medium print is named and labelled."""
    now = CLOSE + 60_000
    ev = [{"title": "Congressional Elections", "time_ms": now - 23 * H},
          {"title": "ISM Services PMI", "time_ms": now + H, "impact": "Medium"},
          {"title": "Unemployment Claims", "time_ms": now - 5 * H},
          {"title": "Building Permits", "time_ms": now + 2 * H, "impact": "Low"},
          {"title": "CPI m/m", "time_ms": now + 30 * H}]
    sec = _event_section(pulse.close_event(live(), market(ev), now).headline)
    assert sec[0].startswith("🚨 EVENT REGIME: Congressional Elections printed 23 hours ago (")
    assert "the day plan's rule is no range plan, wider safety lines, smaller size" in sec[0]
    assert "That rule is the Quant Forecast's; a report's trade plan is sized by its own regime note." in sec[0]
    assert sec[1].startswith("US Unemployment Claims printed at ") and sec[1].endswith("the reaction is still being traded.")
    assert sec[2].startswith("US ISM Services PMI (medium impact) at ") and "expect a reaction around it" in sec[2]
    assert not any("Building Permits" in ln or "CPI" in ln for ln in sec), "Low impact and beyond the day stay out"
    assert sum("Congressional Elections" in ln for ln in sec) == 1, "the regime line is the election's one mention"
    # no major print: no regime line, and a High print ahead still reads 'sharp moves'
    quiet = _event_section(pulse.close_event(live(), market([{"title": "CPI m/m", "time_ms": now + 9 * H}]), now).headline)
    assert quiet == ["US CPI m/m at " + quiet[0].split(" at ")[1]] and "expect sharp moves around it" in quiet[0]
    assert not any("EVENT REGIME" in ln for ln in quiet)
