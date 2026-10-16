"""Market alerts: cascade state changes and TRAP calls, logged once and pushed through the dispatcher."""
import json

import httpx
import pytest

from core import market_alerts as ma
from core.alerts import Dispatcher, Event
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
    return Cascade(state, "", TimeframeRead("1d", "bias", bias, "BULL", "bullish"), None, (), bias, None, None, 0,
                   "Ride the expansion toward the 1d targets. Second sentence is not sent.")


def trap(i, declared=NOW - H, status="active", resolved=None, side="BULL_TRAP"):
    return {"id": i, "side": side, "timeframe": "4h", "level": 80_000.0, "plays_out": 78_000.0,
            "invalidation": 81_000.0, "declared_ms": declared, "category": "intraday", "status": status,
            "resolved_ms": resolved, "resolved_price": 78_000.0 if resolved else None}


# ---------- cascade changes ----------
def test_the_first_reading_is_a_baseline_not_a_change():
    assert ma.cascade_event("live", "Live Recon", None, cascade("A"), NOW) is None


def test_an_unchanged_state_is_silent():
    assert ma.cascade_event("live", "Live Recon", "A", cascade("A"), NOW) is None


def test_a_change_names_both_states_the_bias_and_the_first_line_of_the_playbook():
    e = ma.cascade_event("live", "Live Recon", "A", cascade("B"), NOW, price=78_494.0)
    assert e.kind == "cascade" and not e.urgent
    assert e.headline.startswith("Live Recon: cascade A · full confluence -> B · pullback (1d bias bullish).")
    assert "Ride the expansion toward the 1d targets." in e.headline and "Second sentence" not in e.headline
    assert "$78,494" in e.headline and e.dedup_key == f"cascade:live:A->B:{NOW}"


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
    assert events[0].urgent and "TRAP declared: 4h bull trap at $80,000 (intraday desk)" in events[0].headline
    assert "Plays out at $78,000; invalid past $81,000." in events[0].headline


def test_a_settled_trap_is_announced_with_its_outcome():
    e = ma.trap_events([trap(3, declared=NOW - 3 * 86_400_000, status="invalidated", resolved=NOW - H,
                             side="BEAR_TRAP")], NOW)
    assert [x.dedup_key for x in e] == ["trap:3:invalidated"]
    assert "TRAP invalidated: the break was real: 4h bear trap" in e[0].headline


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
    lines = sent[0]["content"].split("\n")
    assert lines[0].startswith("[!] Weekly: cascade B") and lines[1].startswith("Live Recon: cascade A")


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
    seed = Store()
    for desk in ("live", "intraday", "weekly", "monthly"):
        seed.swap_cascade_state(desk, "Z", NOW)               # a stale state, so this load is a change
    seed.close()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"),
                           default_timeout=180)
    at.secrets["DISCORD_WEBHOOK_URL"] = "https://discord.com/api/webhooks/x/y"
    at.run()
    assert not at.exception
    s = Store()
    logged = s.market_alerts()
    s.close()
    assert len(logged) == 4 and all(r["kind"] in ("cascade", "shift") for r in logged), logged
    assert all(r["notified"] == 0 for r in logged) and _no_real_alerts == [], "logged, never sent"
