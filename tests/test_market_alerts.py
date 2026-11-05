"""Market alerts: cascade state changes and TRAP calls, logged once and pushed through the dispatcher."""
import json

import httpx
import pytest

from core import market_alerts as ma
from core.alerts import URGENT_MARK, Dispatcher, Event
from dataclasses import replace

import pandas as pd

from core.indicators.cascade import Cascade, TimeframeRead
from core.store import Store

NOW = 1_790_000_000_000
H = 3_600_000


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def cascade(state, bias="bullish"):
    """Live Recon's cascade: daily bias, 4h anchor, 15m and 1h engine."""
    engine = tuple(TimeframeRead(tf, "engine", "neutral", "MIXED", "unclear") for tf in ("15m", "1h"))
    return Cascade(state, "", TimeframeRead("1d", "bias", bias, "BULL", "bullish"),
                   TimeframeRead("4h", "anchor", "neutral", "MIXED", "unclear"), engine, bias, None, None, 0,
                   "Ride the expansion toward the 1d targets.")


def trap(i, declared=NOW - H, status="active", resolved=None, side="BULL_TRAP"):
    return {"id": i, "side": side, "timeframe": "4h", "level": 80_000.0, "plays_out": 78_000.0,
            "invalidation": 81_000.0, "declared_ms": declared, "category": "intraday", "status": status,
            "resolved_ms": resolved, "resolved_price": 78_000.0 if resolved else None}


# ---------- cascade changes ----------
def test_the_first_reading_is_a_baseline_not_a_change():
    assert ma.cascade_event("live", "Live Recon", None, cascade("A"), NOW) is None


def test_an_unchanged_state_is_silent():
    assert ma.cascade_event("live", "Live Recon", "A", cascade("A"), NOW) is None


def test_a_change_reads_as_what_happened_the_horizon_the_price_then_what_it_means():
    """Wording batch 3 (6 Nov): emoji, what happened, the desk's horizon in plain words, the price; then
    the sentence. No desk name, no state letters, no 'old state -> new state'."""
    e = ma.cascade_event("live", "Live Recon", "A", cascade("B"), NOW, price=78_494.0)
    assert e.kind == "cascade" and not e.urgent
    assert e.headline.split("\n") == [
        "⏸️ UPTREND PAUSING · next 4 hours · BTC $78,494",
        "The daily trend is still up, but the shorter candles (1h and 15m) lost their push. A breather, not a reversal.",
    ]
    assert e.dedup_key == f"cascade:live:A->B:{NOW}"


def test_every_state_has_a_plain_name_and_the_letters_never_show():
    for prev, new, head in (("B", "A", "📈 UP SETUP · next 4 hours"), ("A", "B", "⏸️ UPTREND PAUSING · next 4 hours"),
                            ("B", "C", "⚠️ UPTREND WEAKENING · next 4 hours")):
        h = ma.cascade_event("live", "Live Recon", prev, cascade(new), NOW).headline
        assert h.split("\n")[0] == head, h
        for gone in ("cascade", "->", "vector", "engine", "Live Recon", "Strong setup", "losing control", "1h/15m"):
            assert gone not in h, gone
    down = ma.cascade_event("weekly", "Weekly", "B", cascade("A", "bearish"), NOW, price=80_180.0).headline
    assert down.split("\n") == [
        "📉 DOWN SETUP · this week · BTC $80,180",
        "The daily trend is down and the shorter candles (1h and 15m) now agree: the best conditions for a move lower this week.",
        "No trade from this alone: wait for the report's plan or the Pulse.",
    ]
    with_level = ma.cascade_event("live", "Live Recon", "B", replace(cascade("A", "bearish"), vwap=80_650.0), NOW, price=80_180.0).headline
    assert with_level.endswith("wait for the report's plan or the Pulse. The trend's average price is $80,650; a 4-hour candle closing above it ends this.")


def test_the_explanation_follows_where_the_desk_came_from():
    said = {prev: ma.cascade_event("live", "Live Recon", prev, cascade("B"), NOW).headline.split("\n")[1]
            for prev in ("A", "C", "N")}
    assert "lost their push" in said["A"]
    assert "stopped fighting it" in said["C"]
    assert "picked a direction again" in said["N"]
    c = ma.cascade_event("live", "Live Recon", "B", cascade("C"), NOW).headline
    assert c.split("\n")[1:] == [
        "The shorter candles (1h and 15m) have started to turn against the daily uptrend but have not yet closed below its average price.",
        "Not a trade yet. It becomes a reversal if a 4-hour candle closes below the daily average price; a REVERSAL CONFIRMED alert will say so.",
    ]
    assert "vwap" not in c.lower(), "jargon the bot may not say"
    assert "Watch for a reversal" not in c, "fix 5: the level and the confirmation rule, not a vague watch"
    lvl = ma.cascade_event("live", "Live Recon", "B", replace(cascade("C", "bearish"), vwap=77_512.0, closes_beyond_vwap=2), NOW, price=80_317.0).headline
    assert lvl.split("\n") == [
        "⚠️ DOWNTREND WEAKENING · next 4 hours · BTC $80,317",
        "The shorter candles (1h and 15m) have closed above the daily trend's average price $77,512 twice.",
        "Not a trade yet. It becomes a reversal if a 4-hour candle closes above $77,512; a REVERSAL CONFIRMED alert will say so.",
    ]
    three = ma.cascade_event("live", "Live Recon", "B", replace(cascade("C", "bearish"), vwap=77_512.0, closes_beyond_vwap=3), NOW).headline
    assert "$77,512 three times." in three
    assert "$77,512 5 times." in ma.cascade_event("live", "Live Recon", "B", replace(cascade("C", "bearish"), vwap=77_512.0, closes_beyond_vwap=5), NOW).headline
    flat = ma.cascade_event("live", "Live Recon", "B", cascade("N", "neutral"), NOW).headline
    assert flat.startswith("↔️ NO TREND · next 4 hours\n") and "not breakouts" in flat


def test_a_control_shift_is_urgent():
    assert ma.cascade_event("weekly", "Weekly", "B", cascade("C"), NOW).urgent


def test_a_data_gap_is_not_a_market_event():
    assert ma.cascade_event("monthly", "Monthly", "A", cascade("?"), NOW) is None
    assert ma.cascade_event("monthly", "Monthly", "?", cascade("A"), NOW) is None


def test_the_same_change_on_a_new_candle_is_a_new_alert_but_not_on_the_same_one():
    a = ma.cascade_event("live", "Live Recon", "A", cascade("B"), NOW)
    b = ma.cascade_event("live", "Live Recon", "A", cascade("B"), NOW + H)
    assert a.dedup_key != b.dedup_key


# ---------- TRAP calls ----------
def test_a_trap_declared_today_is_announced_and_an_old_one_is_not():
    events = ma.trap_events([trap(1), trap(2, declared=NOW - 2 * 86_400_000)], NOW)
    assert [e.dedup_key for e in events] == ["trap:1:declared"]
    assert events[0].urgent
    assert events[0].headline.split("\n") == ["🪤 TRAP CALL · today (4h) · BTC $80,000",
                                              "The pop above $80,000 should fail and drop.",
                                              "Target $78,000 · wrong if price goes past $81,000."]


def test_a_settled_trap_is_announced_with_its_outcome():
    e = ma.trap_events([trap(3, declared=NOW - 3 * 86_400_000, status="invalidated", resolved=NOW - H,
                             side="BEAR_TRAP")], NOW)
    assert [x.dedup_key for x in e] == ["trap:3:invalidated"]
    assert e[0].headline == ("❌ TRAP FAILED · today (4h) · BTC $78,000\nThe dip below $80,000 was real: "
                             "price went past $81,000. The call was wrong.")
    worked = ma.trap_events([trap(4, declared=NOW - 3 * 86_400_000, status="played_out", resolved=NOW - H)], NOW)
    assert worked[0].headline == "✅ TRAP WORKED · today (4h) · BTC $78,000\nThe pop above $80,000 failed as called: price reached the target $78,000."


# ---------- the ledger ----------
def test_the_ledger_returns_the_state_it_replaces(store):
    assert store.swap_cascade_state("live", "A", NOW) is None
    assert store.swap_cascade_state("live", "B", NOW + 1) == "A"
    assert store.swap_cascade_state("weekly", "C", NOW) is None, "each desk has its own state"


def test_an_alert_is_logged_once_and_sent_oldest_first(store):
    assert store.add_market_alert(NOW + 5, "trap", "k2", "second") is not None
    assert store.add_market_alert(NOW, "cascade", "k1", "first") is not None
    assert store.add_market_alert(NOW + 9, "cascade", "k1", "again") is None, "the same key is one alert"
    pending = store.market_alerts(unnotified_only=True)
    assert [p["headline"] for p in pending] == ["first", "second"]
    store.mark_market_notified([pending[0]["id"]])
    assert [p["headline"] for p in store.market_alerts(unnotified_only=True)] == ["second"]


def test_a_reopened_ledger_keeps_the_tables(tmp_path):
    path = tmp_path / "t.sqlite"
    s = Store(path)
    s.add_market_alert(NOW, "trap", "k", "h")
    s.swap_cascade_state("live", "A", NOW)
    s.close()
    s = Store(path)
    assert s.swap_cascade_state("live", "B", NOW) == "A" and len(s.market_alerts()) == 1
    s.close()


# ---------- sending ----------
def _discord(sent, status=204):
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(status, request=request)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_alerts_reach_discord_with_urgent_ones_marked():
    sent = []
    d = Dispatcher({"DISCORD_WEBHOOK_URL": "https://discord.test/hook"}, client=_discord(sent))
    shift = ma.cascade_event("weekly", "Weekly", "B", cascade("C"), NOW)
    calm = ma.cascade_event("live", "Live Recon", "A", cascade("B"), NOW)
    delivered, problems = d.send([shift, calm])
    assert len(delivered) == 2 and not problems
    blocks = sent[0]["content"].split("\n\n")
    assert len(blocks) == 2, "a blank line between alerts, so two three-line alerts do not run together"
    assert blocks[0].startswith("⚠️ UPTREND WEAKENING · this week\n") and blocks[1].startswith("⏸️ UPTREND PAUSING · next 4 hours\n")
    assert "🚨" not in sent[0]["content"], "wording batch 3: the headline's own emoji is the mark; no siren in front of it"


def test_a_refused_send_is_not_marked_delivered():
    d = Dispatcher({"DISCORD_WEBHOOK_URL": "https://discord.test/hook"}, client=_discord([], status=429))
    delivered, problems = d.send([Event("cascade", "k", "h")])
    assert delivered == [] and "Discord did not accept" in problems[0]


def test_the_test_alert_says_what_will_arrive():
    sent = []
    ok, _ = Dispatcher({"DISCORD_WEBHOOK_URL": "https://discord.test/hook"}, client=_discord(sent)).send_test()
    assert ok and "Cascade changes and TRAP calls will arrive here" in sent[0]["content"]


def test_no_test_can_reach_a_real_channel(_no_real_alerts, monkeypatch):
    """The guard in conftest: without an injected client a send is blocked before any request is made."""
    requests = []
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: requests.append(url))
    delivered, problems = Dispatcher({"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/x/y"}).send(
        [Event("trap", "k", "h")])
    assert requests == [], "no HTTP request left the test"
    assert delivered == [] and _no_real_alerts == ["https://discord.com/api/webhooks/x/y"]


# ---------- the page ----------
def test_the_page_logs_alerts_but_never_sends_from_fixtures(monkeypatch, tmp_path, _no_real_alerts):
    """Offline fixtures are not the market: the page detects and logs, and does not dispatch."""
    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    import streamlit as st
    st.cache_resource.clear()                                  # the app caches its Store across runs
    st.cache_data.clear()
    import time as _t
    seed = Store()
    for desk in ("live", "intraday", "weekly", "monthly"):
        seed.swap_cascade_state(desk, "Z", NOW)               # a stale state, so this load is a change
    seed.put_trap("live", "BEAR_TRAP", "1h", 80_000.0, 79_500.0, 81_000.0, int(_t.time() * 1000) - H, 70.0,
                  "BEAR_TRAP", ["test"])
    seed.close()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"),
                           default_timeout=180)
    at.secrets["DISCORD_WEBHOOK_URL"] = "https://discord.com/api/webhooks/x/y"
    at.run()
    assert not at.exception
    s = Store()
    logged = s.market_alerts()
    states = [s.swap_cascade_state(d, "Z", NOW) for d in ("live", "intraday", "weekly", "monthly")]
    s.close()
    assert "trap:1:declared" in {r["dedup_key"] for r in logged}, logged
    assert not any(r["kind"] in ("cascade", "shift") for r in logged), "cascade changes are context, not alerts"
    assert all(st_ != "Z" for st_ in states), "the cascade states are still recorded"
    assert all(r["notified"] == 0 for r in logged) and _no_real_alerts == [], "logged, never sent"


# ---------- the important cascade changes, held ----------
MIN = 60_000


def test_track_keeps_when_a_state_began_and_what_it_replaced(store):
    assert store.track_cascade("live", "B", NOW) == {"state": "B", "since_ms": NOW, "prior_state": None, "first": True}
    assert store.track_cascade("live", "A", NOW + MIN)["since_ms"] == NOW + MIN
    held = store.track_cascade("live", "A", NOW + 9 * MIN)
    assert held == {"state": "A", "since_ms": NOW + MIN, "prior_state": "B", "first": False}, "a refresh keeps the start"


def test_a_strong_setup_waits_until_it_has_held(store):
    store.track_cascade("intraday", "B", NOW)
    t = store.track_cascade("intraday", "A", NOW + MIN)
    assert ma.important_cascade_event("intraday", "Intraday", t, cascade("A"), NOW + 5 * MIN) is None, "too soon"
    t = store.track_cascade("intraday", "A", NOW + 17 * MIN)
    e = ma.important_cascade_event("intraday", "Intraday", t, cascade("A"), NOW + 17 * MIN, 82_909.0)
    assert e is not None and e.dedup_key == f"cascade:intraday:B->A:{NOW + MIN}"
    assert e.headline.startswith("📈 UP SETUP · today · BTC $82,909\n")


def test_the_0213_flicker_sends_nothing(store):
    """20 Oct: Intraday went A at 02:13 and back to B at 02:17 - both were sent. Neither is now."""
    store.track_cascade("intraday", "B", NOW)
    t = store.track_cascade("intraday", "A", NOW + 13 * MIN)
    assert ma.important_cascade_event("intraday", "Intraday", t, cascade("A"), NOW + 13 * MIN) is None
    t = store.track_cascade("intraday", "B", NOW + 17 * MIN)
    assert ma.important_cascade_event("intraday", "Intraday", t, cascade("B"), NOW + 40 * MIN) is None


def test_pauses_and_no_trend_stay_quiet_and_a_control_shift_does_not(store):
    store.track_cascade("live", "A", NOW)
    t = store.track_cascade("live", "B", NOW + MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("B"), NOW + 60 * MIN) is None
    t = store.track_cascade("live", "N", NOW + 61 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("N", "neutral"), NOW + 120 * MIN) is None
    store.track_cascade("weekly", "B", NOW)
    t = store.track_cascade("weekly", "C", NOW + MIN)
    assert ma.important_cascade_event("weekly", "Weekly", t, cascade("C"), NOW + 30 * MIN) is None, "weekly holds an hour"
    e = ma.important_cascade_event("weekly", "Weekly", t, cascade("C"), NOW + 62 * MIN)
    assert e is not None and e.urgent and e.headline.startswith("⚠️ UPTREND WEAKENING · this week\n")


def test_a_return_to_the_same_state_within_hours_is_the_same_episode(store):
    """29 Oct: Live went C at 16:00, 16:17 and 18:55, flapping through B and N, and the identical message
    went out three times. The second and third are now the first's episode."""
    store.track_cascade("live", "B", NOW)
    t = store.track_cascade("live", "C", NOW + MIN)
    e = ma.important_cascade_event("live", "Live Recon", t, cascade("C"), NOW + 6 * MIN, recent=[])
    assert e is not None
    sent = [{"ts_ms": NOW + 6 * MIN, "dedup_key": e.dedup_key}]
    store.track_cascade("live", "N", NOW + 10 * MIN)
    t = store.track_cascade("live", "C", NOW + 17 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("C"), NOW + 23 * MIN, recent=sent) is None
    store.track_cascade("live", "B", NOW + 60 * MIN)
    t = store.track_cascade("live", "C", NOW + 175 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("C"), NOW + 181 * MIN, recent=sent) is None, "under 8h"
    t = store.track_cascade("live", "C", NOW + 190 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("C"), NOW + 190 * MIN, recent=sent) is None, "4 Nov: 10:29 was 3h01 after 07:28"
    t = store.track_cascade("live", "C", NOW + 8 * 60 * MIN + 7 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("C"), NOW + 8 * 60 * MIN + 7 * MIN, recent=sent) is not None, "a new episode after 8h"
    # a different state is not the same episode, nor is another desk's alert
    t = store.track_cascade("live", "A", NOW + 30 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("A"), NOW + 36 * MIN, recent=sent) is not None
    assert not ma.same_episode(sent, "intraday", "C", NOW + 10 * MIN)
    assert ma.same_episode(sent, "live", "C", NOW + 10 * MIN) and not ma.same_episode(None, "live", "C", NOW + 10 * MIN)


def test_the_first_reading_after_a_reboot_is_not_news(store):
    t = store.track_cascade("live", "A", NOW)
    assert ma.important_cascade_event("live", "Live Recon", t, cascade("A"), NOW + 99 * MIN) is None


# ---------- election-day fix 5: REVERSAL CONFIRMED on the anchor candle's close ----------
DAY = 1_793_750_400_000                       # 4 Nov 2026 00:00 UTC
FOUR_H = [(DAY, 77_454, 76_496), (DAY + 4 * H, 76_496, 78_140), (DAY + 8 * H, 78_140, 78_477),
          (DAY + 12 * H, 78_477, 80_296)]      # the real 4h candles: (open time, open, close)


def four_hour(rows):
    return pd.DataFrame({"timestamp": [t for t, _, _ in rows], "open": [o for _, o, _ in rows],
                         "high": [max(o, c) + 50 for _, o, c in rows], "low": [min(o, c) - 50 for _, o, c in rows],
                         "close": [c for _, _, c in rows], "volume": [100.0] * len(rows)})


def test_the_fourth_of_november_reversal_is_confirmed_at_0800_six_hours_before_the_spike(store):
    """The Live desk went C at 07:23 UTC (the 1h had closed twice above the day's VWAP against the daily
    downtrend). The 04:00-08:00 4h candle closed 78,140 above that VWAP: REVERSAL CONFIRMED goes out at
    08:00, once; the warning at 07:28 stays a warning, and the 10:29 / 14:37 / 19:12 repeats never go out."""
    c = replace(cascade("C", "bearish"), vwap=77_512.0, closes_beyond_vwap=2)
    since = DAY + 7 * H + 23 * MIN
    store.track_cascade("live", "B", DAY + 7 * H)
    t = store.track_cascade("live", "C", since)
    frames = {"4h": four_hour(FOUR_H[:2]), "1h": pd.DataFrame()}
    warn = ma.important_cascade_event("live", "Live Recon", t, c, since + 5 * MIN, 78_271.0, recent=[])
    assert warn is not None and "Not a trade yet." in warn.headline
    # 07:59: the 4h candle is still forming, so nothing confirms yet
    assert ma.reversal_event("live", "Live Recon", t, c, frames, DAY + 7 * H + 59 * MIN, 78_100.0, recent=[]) is None
    r = ma.reversal_event("live", "Live Recon", t, c, frames, DAY + 8 * H + 2 * MIN, 78_140.0, recent=[])
    assert r is not None and r.kind == "reversal" and r.urgent
    assert r.dedup_key == f"cascade:live:C->confirmed:{DAY + 4 * H}"
    assert r.headline.split("\n") == [
        "🔄 REVERSAL CONFIRMED · next 4 hours · BTC $78,140",
        "A 4-hour candle closed at $78,140 (08:00 UTC, 13:00 PKT), above the daily trend's average price $77,512: the daily downtrend has lost control.",
        "If you are short from the trend, it is over. Expect a move up. Wrong only if a 4-hour candle closes back below $77,512.",
    ]
    sent = [{"ts_ms": DAY + 8 * H + 2 * MIN, "dedup_key": r.dedup_key}]
    # the next 4h close (12:00, 78,477) does not confirm it again, and the C warning is silent for 8 hours after
    later = {"4h": four_hour(FOUR_H[:3])}
    assert ma.reversal_event("live", "Live Recon", t, c, later, DAY + 12 * H + 2 * MIN, 78_477.0, recent=sent) is None
    store.track_cascade("live", "B", DAY + 10 * H)
    t2 = store.track_cascade("live", "C", DAY + 10 * H + 29 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t2, c, DAY + 10 * H + 35 * MIN, recent=sent) is None, "10:29"
    store.track_cascade("live", "B", DAY + 14 * H)
    t3 = store.track_cascade("live", "C", DAY + 14 * H + 37 * MIN)
    assert ma.important_cascade_event("live", "Live Recon", t3, c, DAY + 14 * H + 43 * MIN, recent=sent) is None, "14:37"


def test_a_reversal_needs_a_completed_anchor_close_against_the_trend_and_a_vwap(store):
    c = replace(cascade("C", "bearish"), vwap=77_512.0, closes_beyond_vwap=2)
    since = DAY + 7 * H + 23 * MIN
    store.track_cascade("live", "B", DAY + 7 * H)
    t = store.track_cascade("live", "C", since)
    # a 4h candle that closed below the VWAP (with the trend) confirms nothing
    low = {"4h": four_hour([(DAY, 77_454, 76_496), (DAY + 4 * H, 76_496, 77_300)])}
    assert ma.reversal_event("live", "Live Recon", t, c, low, DAY + 8 * H + 2 * MIN, recent=[]) is None
    # a candle that closed before the shift began does not count, even above the VWAP
    early = {"4h": four_hour([(DAY, 77_454, 77_900)])}
    assert ma.reversal_event("live", "Live Recon", t, c, early, DAY + 8 * H + 2 * MIN, recent=[]) is None
    # no VWAP, no anchor frame, or a desk no longer in C: nothing
    assert ma.reversal_event("live", "Live Recon", t, cascade("C", "bearish"), {"4h": four_hour(FOUR_H[:2])}, DAY + 8 * H + 2 * MIN, recent=[]) is None
    assert ma.reversal_event("live", "Live Recon", t, c, {}, DAY + 8 * H + 2 * MIN, recent=[]) is None
    tb = store.track_cascade("live", "B", DAY + 8 * H + MIN)
    assert ma.reversal_event("live", "Live Recon", tb, replace(c, state="B"), {"4h": four_hour(FOUR_H[:2])}, DAY + 8 * H + 2 * MIN, recent=[]) is None
    # the mirror: a bullish bias is confirmed by a close BELOW the VWAP
    up = replace(cascade("C", "bullish"), vwap=80_000.0, closes_beyond_vwap=2,
                 anchor=TimeframeRead("1d", "anchor", "neutral", "MIXED", "unclear"))      # the Intraday desk anchors on the daily
    store.track_cascade("intraday", "B", DAY + 7 * H)
    tu = store.track_cascade("intraday", "C", since)
    with_day = {"1d": pd.DataFrame({"timestamp": [DAY - 24 * H], "open": [80_500.0], "high": [80_600.0], "low": [79_300.0],
                                    "close": [79_400.0], "volume": [100.0]})}
    assert ma.reversal_event("intraday", "Intraday", tu, up, with_day, DAY + 8 * H, recent=[]) is None, "closed 00:00, before 07:23"
    nxt = {"1d": pd.DataFrame({"timestamp": [DAY], "open": [77_454.0], "high": [81_200.0], "low": [75_700.0],
                               "close": [79_400.0], "volume": [100.0]})}
    r = ma.reversal_event("intraday", "Intraday", tu, up, nxt, DAY + 24 * H + MIN, 79_400.0, recent=[])
    assert r is not None and "A daily candle closed at $79,400 (00:00 UTC, 05:00 PKT), below the daily trend's average price $80,000" in r.headline
    assert "If you are long from the trend, it is over. Expect a move down." in r.headline and "vwap" not in r.headline.lower()
    assert "Wrong only if a daily candle closes back above $80,000." in r.headline and r.dedup_key == f"cascade:intraday:C->confirmed:{DAY}"
    assert r.headline.startswith("🔄 REVERSAL CONFIRMED · today · BTC $79,400\n")
