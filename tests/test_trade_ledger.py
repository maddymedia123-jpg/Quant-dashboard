"""Pinned trades in the database: what survives a refresh, a reboot, and a double click.

The spec asks for cards that do not vanish on navigation or refresh, several positions tracked at once,
and a KILL that halts alerts for that position. Session state gives the first and loses everything on a
redeploy, which on Streamlit Cloud happens unannounced - so the plan lives here and the card is rendered
from it. The plan is written once: Refresh must not be able to rewrite the thesis a trader is holding
against.
"""
import pytest

from core.store import Store

NOW = 1_760_000_000_000


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def open_one(s: Store, *, asset="BTC/USDT", side="LONG", entry=80_000.0, stop=79_000.0,
             given_stop=None, verdict="supplied", category="live", plan=None, opened_ms=NOW) -> int:
    return s.open_trade(opened_ms=opened_ms, asset=asset, side=side, entry=entry, stop=stop,
                        given_stop=given_stop, stop_verdict=verdict, category=category,
                        plan=plan if plan is not None else {"targets": [{"name": "TP1", "price": 80_600.0}],
                                                            "thesis": "the original thesis"})


# ---------- pinned and separate ----------
def test_a_pinned_trade_round_trips_with_its_whole_plan(store):
    tid = open_one(store, given_stop=79_200.0, verdict="would be hunted")
    row = store.trade(tid)
    assert row["asset"] == "BTC/USDT" and row["side"] == "LONG" and row["status"] == "open"
    assert row["entry"] == 80_000.0 and row["stop"] == 79_000.0 and row["given_stop"] == 79_200.0
    assert row["stop_verdict"] == "would be hunted" and row["category"] == "live"
    assert row["plan"]["targets"][0]["price"] == 80_600.0
    assert row["plan"]["thesis"] == "the original thesis"


def test_several_positions_are_tracked_at_once_newest_first(store):
    first = open_one(store, opened_ms=NOW)
    second = open_one(store, asset="ETH/USDT", opened_ms=NOW + 1_000)
    rows = store.trades()
    assert [r["id"] for r in rows] == [second, first], "each in its own card, newest at the top"
    assert {r["asset"] for r in rows} == {"BTC/USDT", "ETH/USDT"}


def test_a_closed_trade_is_not_in_the_open_list(store):
    live = open_one(store)
    dead = open_one(store, asset="SOL/USDT", opened_ms=NOW + 1)
    store.close_trade(dead, "killed by the trader", 80_100.0, NOW + 2)
    assert [r["id"] for r in store.trades()] == [live], "the killed card is unpinned"
    assert [r["id"] for r in store.trades(status="closed")] == [dead]
    assert len(store.trades(status=None)) == 2, "both are still on the record"


def test_an_unknown_trade_id_is_none_rather_than_an_error(store):
    assert store.trade(999) is None


# ---------- refresh touches the numbers, not the thesis ----------
def test_a_refresh_records_itself_without_changing_the_plan(store):
    tid = open_one(store)
    before = store.trade(tid)
    store.touch_trade(tid, NOW + 60_000)
    after = store.trade(tid)
    assert after["refreshed_ms"] == NOW + 60_000
    assert after["plan"] == before["plan"], "Refresh updates live numbers, never the thesis"
    assert (after["entry"], after["stop"], after["stop_verdict"]) == \
           (before["entry"], before["stop"], before["stop_verdict"])


# ---------- the kill button ----------
def test_the_kill_button_works_once_and_keeps_why_it_ended(store):
    """A rerun must not be able to re-close a trade and overwrite the reason it was closed."""
    tid = open_one(store)
    assert store.close_trade(tid, "killed by the trader", 80_100.0, NOW + 10) is True
    assert store.close_trade(tid, "stop reached", 79_000.0, NOW + 20) is False
    row = store.trade(tid)
    assert row["closed_reason"] == "killed by the trader"
    assert row["closed_price"] == 80_100.0 and row["closed_ms"] == NOW + 10


def test_killing_a_trade_that_does_not_exist_is_false(store):
    assert store.close_trade(4242, "killed", 1.0, NOW) is False


# ---------- the event log ----------
def test_the_same_event_cannot_be_logged_twice(store):
    tid = open_one(store)
    first = store.add_trade_event(tid, NOW, "target", "target:TP1", "TP1 reached", 80_600.0)
    again = store.add_trade_event(tid, NOW + 30_000, "target", "target:TP1", "TP1 reached", 80_650.0)
    assert first is not None and again is None, "one alert per event, not one per refresh"
    assert len(store.trade_events(tid)) == 1


def test_two_trades_can_each_hit_their_own_tp1(store):
    a, b = open_one(store), open_one(store, asset="ETH/USDT", opened_ms=NOW + 1)
    assert store.add_trade_event(a, NOW, "target", "target:TP1", "TP1", 1.0) is not None
    assert store.add_trade_event(b, NOW, "target", "target:TP1", "TP1", 2.0) is not None
    assert len(store.trade_events(a)) == 1 and len(store.trade_events(b)) == 1


def test_events_are_listed_per_trade_newest_first(store):
    tid = open_one(store)
    store.add_trade_event(tid, NOW, "target", "target:TP1", "TP1 reached", 80_600.0)
    store.add_trade_event(tid, NOW + 60_000, "stop", "stop", "stopped out", 79_000.0)
    assert [e["kind"] for e in store.trade_events(tid)] == ["stop", "target"]


def test_only_events_that_were_actually_sent_are_marked(store):
    """mark_notified takes exactly what a channel accepted. Marking rows that were never delivered would
    lose them, because the unnotified query is what the next pass retries."""
    tid = open_one(store)
    sent = store.add_trade_event(tid, NOW, "target", "target:TP1", "TP1 reached", 80_600.0)
    unsent = store.add_trade_event(tid, NOW + 1, "stop", "stop", "stopped out", 79_000.0)
    assert store.mark_notified([sent]) == 1
    assert [e["id"] for e in store.trade_events(tid, unnotified_only=True)] == [unsent]

    # marking again must not re-count a row that was already notified
    assert store.mark_notified([sent, unsent]) == 1, "only the one still waiting"
    assert store.trade_events(tid, unnotified_only=True) == []


def test_marking_nothing_is_not_an_error(store):
    assert store.mark_notified([]) == 0


def test_events_survive_their_trade_being_killed(store):
    """The card is unpinned, but what happened to the position stays on the record."""
    tid = open_one(store)
    store.add_trade_event(tid, NOW, "target", "target:TP1", "TP1 reached", 80_600.0)
    store.close_trade(tid, "killed by the trader", 80_600.0, NOW + 1)
    assert len(store.trade_events(tid)) == 1
