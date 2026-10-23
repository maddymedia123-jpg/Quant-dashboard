"""Market Pulse Dispatch: his template, once per report and once per fresh 4h close, plain words only."""
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


def market(events=(), heads=()):
    cal = NS(available=True, upcoming=lambda now, within: list(events))
    news = NS(available=True, recent=lambda now, within, impact: list(heads))
    return NS(calendar=cal, news=news, spot=NS(last=81_100.0))


PLAN = {"side": "SHORT", "zone_low": 81_800, "zone_high": 82_200, "stop": 82_650, "targets": [80_400, 80_900],
        "expires_hours": 12}


def test_the_message_follows_his_template_in_order():
    t = pulse.close_event(live(), market(), CLOSE + 60_000).headline
    order = ["MARKET PULSE DISPATCH", "⏰ Time:", "🎯 MARKET DIRECTION: BEARISH", "💡 WHAT IS HAPPENING:",
             "🕒 EXACT TIME TO WATCH:", "Next Turning Window:", "📍 IMPORTANT PRICE LEVELS:",
             "Target Level: $80,800", "Invalidation (Safety Line): $81,300", "⚠️ IMPORTANT NEWS / EVENT WARNINGS:"]
    at = [t.index(s) for s in order]
    assert at == sorted(at)
    assert "PKT" in t.split("Next Turning Window:")[1].splitlines()[0]
    assert len(t) < alerts.MAX_CHARS


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


def test_a_report_with_a_plan_speaks_from_the_plan():
    row = {"report_id": "TIR1-1", "ts_ms": CLOSE, "plan": json.dumps(PLAN)}
    e = pulse.report_event(row, live("BULLISH"), market(), CLOSE + 60_000)
    assert e.dedup_key == "pulse:report:TIR1-1"
    assert "MARKET DIRECTION: BEARISH" in e.headline, "the plan's side, not the desk's lean"
    assert "Target Level: $80,900" in e.headline and "Safety Line): $82,650" in e.headline
    assert "$81,800–$82,200 zone" in e.headline


def test_a_report_without_a_plan_falls_back_to_the_live_desk_and_a_stale_one_is_skipped():
    row = {"report_id": "TIR1-2", "ts_ms": CLOSE, "plan": json.dumps({"side": "NONE"})}
    e = pulse.report_event(row, live(), market(), CLOSE)
    assert "Live desk" in e.headline and "Target Level: $80,800" in e.headline
    assert pulse.report_event(row, live(), market(), CLOSE + 31 * 60_000) is None
    assert pulse.report_event(None, live(), market(), CLOSE) is None


def test_news_lists_us_prints_and_fresh_headlines_or_says_there_are_none():
    ev = [{"title": "CPI m/m", "time_ms": CLOSE + 9 * H}]
    t = pulse.close_event(live(), market(ev, [{"title": "SEC delays ETF", "published_ms": 1}]), CLOSE).headline
    assert "US CPI m/m at" in t and "Headline: SEC delays ETF" in t
    assert "No major news" in pulse.close_event(live(), market(), CLOSE).headline


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
