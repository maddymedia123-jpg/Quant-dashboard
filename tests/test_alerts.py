"""Alerts: what counts as news about a pinned trade, and what happens when it is sent.

The important property is that an event fires once. The page auto-refreshes every thirty seconds, so a
target that is "hit" is hit on every pass; without a dedup key and the store's unique index, one target
would send a hundred and twenty alerts an hour. Detection is therefore pure and repeatable, and the store
decides what is new.

Nothing here touches the network: the dispatcher takes an injected client.
"""
import httpx
import pytest

from core import alerts
from core.alerts import Dispatcher, Event, detect_events
from core.store import Store
from core.trades import LONG, SHORT, Checkpoint, TimedTarget, progress

NOW = 1_760_000_000_000
HOUR = 3_600_000
ENTRY = 80_000.0


def trade(**over) -> dict:
    base = {"id": 1, "asset": "BTC/USDT", "side": LONG, "entry": ENTRY, "stop": 79_000.0,
            "status": "open",
            "plan": {"targets": [{"name": "TP1", "price": 80_600.0}, {"name": "TP2", "price": 81_200.0},
                                 {"name": "TP3", "price": 81_800.0}]}}
    base.update(over)
    return base


def targets() -> list[TimedTarget]:
    return [TimedTarget("TP1", 80_600.0, "b", NOW + 3_600_000, 1.5),
            TimedTarget("TP2", 81_200.0, "b", NOW + 7_200_000, 3.0),
            TimedTarget("TP3", 81_800.0, "b", NOW + 10_800_000, 4.5)]


class Trap:
    def __init__(self, trap_id=7, side="bull_trap", timeframe="1h", near="TP1", distance=50.0):
        self.trap_id, self.side, self.timeframe = trap_id, side, timeframe
        self.near, self.distance = near, distance


# ---------- detection ----------
def test_a_target_reached_is_an_event_that_names_the_level_and_the_r():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 80_650.0)
    events = detect_events(trade(), 80_650.0, p, now_ms=NOW)
    assert [e.kind for e in events] == ["target"]
    assert events[0].dedup_key == "target:TP1"
    assert "TP1 reached at 80,600" in events[0].headline and "+0.7R" in events[0].headline


def test_the_stop_being_reached_is_urgent():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 78_900.0)
    events = detect_events(trade(), 78_900.0, p, now_ms=NOW)
    stop_event = next(e for e in events if e.kind == "stop")
    assert stop_event.urgent and "the stop at 79,000 was reached" in stop_event.headline
    assert "-1.1R" in stop_event.headline


def test_several_targets_reached_are_separate_events():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 81_300.0)
    keys = [e.dedup_key for e in detect_events(trade(), 81_300.0, p, now_ms=NOW)]
    assert keys == ["target:TP1", "target:TP2"]


def test_detection_is_repeatable_so_the_store_decides_what_is_new():
    """The same state has to produce the same keys, or an auto-refresh would alert on every pass."""
    p = progress(LONG, ENTRY, 79_000.0, targets(), 80_650.0)
    first = detect_events(trade(), 80_650.0, p, now_ms=NOW)
    again = detect_events(trade(), 80_650.0, p, now_ms=NOW + 30_000)
    assert [e.dedup_key for e in first] == [e.dedup_key for e in again]


def test_a_trap_near_the_position_is_urgent_and_identifies_the_trap():
    events = detect_events(trade(), ENTRY, None, traps=[Trap()], now_ms=NOW)
    assert [e.kind for e in events] == ["trap"]
    assert events[0].urgent and events[0].dedup_key == "trap:7:TP1"
    assert "anchored bull trap on the 1h" in events[0].headline


def test_a_checkpoint_only_fires_once_it_has_passed():
    future = Checkpoint("4h", NOW + 3_600_000, "the 4h close")
    assert detect_events(trade(), ENTRY, None, checkpoints=[future], now_ms=NOW) == []
    passed = Checkpoint("4h", NOW - 60_000, "the 4h close")
    events = detect_events(trade(), ENTRY, None, checkpoints=[passed], now_ms=NOW)
    assert [e.kind for e in events] == ["checkpoint"]
    assert "review the position" in events[0].headline


def test_a_quiet_position_produces_nothing():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 80_100.0)
    assert detect_events(trade(), 80_100.0, p, now_ms=NOW) == []


def test_a_short_reaching_its_target_is_not_read_as_a_loss():
    t = trade(side=SHORT, stop=81_000.0,
              plan={"targets": [{"name": "TP1", "price": 79_400.0}]})
    p = progress(SHORT, ENTRY, 81_000.0,
                 [TimedTarget("TP1", 79_400.0, "b", NOW, 0.6)], 79_300.0)
    events = detect_events(t, 79_300.0, p, now_ms=NOW)
    assert [e.kind for e in events] == ["target"] and "+0.7R" in events[0].headline


def test_a_burst_larger_than_a_pass_is_capped():
    checkpoints = [Checkpoint("1h", NOW - i * 60_000, f"the {i}th close") for i in range(1, 30)]
    events = detect_events(trade(), ENTRY, None, checkpoints=checkpoints, now_ms=NOW)
    assert len(events) == alerts.MAX_PER_PASS


# ---------- the store makes it fire once ----------
@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_the_same_event_is_logged_once_however_often_it_is_detected(store):
    tid = store.open_trade(opened_ms=NOW, asset="BTC/USDT", side=LONG, entry=ENTRY, stop=79_000.0,
                           given_stop=None, stop_verdict="supplied", category="live", plan={})
    p = progress(LONG, ENTRY, 79_000.0, targets(), 80_650.0)
    logged = []
    for pass_no in range(5):                     # five auto-refreshes, one target
        for e in detect_events(store.trade(tid) | {"plan": {"targets": [{"name": "TP1", "price": 80_600.0}]}},
                               80_650.0, p, now_ms=NOW + pass_no * 30_000):
            row = store.add_trade_event(tid, NOW, e.kind, e.dedup_key, e.headline, e.price)
            if row is not None:
                logged.append(row)
    assert len(logged) == 1, "one alert, not five"
    assert len(store.trade_events(tid)) == 1


def test_only_what_a_channel_accepted_is_marked_as_notified(store):
    tid = store.open_trade(opened_ms=NOW, asset="BTC/USDT", side=LONG, entry=ENTRY, stop=79_000.0,
                           given_stop=None, stop_verdict="supplied", category="live", plan={})
    store.add_trade_event(tid, NOW, "target", "target:TP1", "TP1 reached", 80_600.0)
    assert len(store.trade_events(tid, unnotified_only=True)) == 1

    refuses = Dispatcher({"DISCORD_WEBHOOK_URL": "https://example.invalid/hook"},
                         client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    delivered, problems = refuses.send([Event("target", "target:TP1", "TP1 reached", 80_600.0)])
    assert delivered == [] and problems
    assert len(store.trade_events(tid, unnotified_only=True)) == 1, "a refused send is retried, not lost"

    accepts = Dispatcher({"DISCORD_WEBHOOK_URL": "https://example.invalid/hook"},
                         client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(204))))
    delivered, _ = accepts.send([Event("target", "target:TP1", "TP1 reached", 80_600.0)])
    assert len(delivered) == 1
    store.mark_notified([e["id"] for e in store.trade_events(tid)])
    assert store.trade_events(tid, unnotified_only=True) == []


# ---------- dispatch ----------
def test_no_channel_configured_says_so_and_sends_nothing():
    d = Dispatcher({})
    assert not d.configured
    delivered, problems = d.send([Event("target", "k", "TP1 reached")])
    assert delivered == []
    assert any("No alert channel is configured" in p for p in problems)
    assert any("will go out once a channel exists" in p for p in problems)


def test_the_unconfigured_channels_are_named_with_what_they_need():
    by = {c.name: c for c in Dispatcher({}).channels}
    assert set(by) == {"Telegram", "Discord", "Email", "SMS"}
    assert "TELEGRAM_BOT_TOKEN" in by["Telegram"].reason
    assert "DISCORD_WEBHOOK_URL" in by["Discord"].reason
    assert not by["Email"].configured and "SMTP" in by["Email"].reason


def test_telegram_needs_both_the_token_and_the_chat_id():
    assert not Dispatcher({"TELEGRAM_BOT_TOKEN": "t"}).configured
    assert not Dispatcher({"TELEGRAM_CHAT_ID": "c"}).configured
    assert Dispatcher({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}).configured


def test_telegram_is_sent_to_the_right_endpoint_with_the_headlines():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"ok": True})

    d = Dispatcher({"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "42"},
                   client=httpx.Client(transport=httpx.MockTransport(handler)))
    delivered, problems = d.send([Event("target", "k1", "TP1 reached"),
                                 Event("stop", "k2", "the stop was reached")])
    assert len(delivered) == 2 and problems == []
    assert seen["url"] == "https://api.telegram.org/botabc/sendMessage"
    assert "TP1 reached" in seen["body"] and "42" in seen["body"]


def test_an_urgent_event_is_marked_in_the_message():
    seen = {}
    d = Dispatcher({"DISCORD_WEBHOOK_URL": "https://example.invalid/hook"},
                   client=httpx.Client(transport=httpx.MockTransport(
                       lambda r: (seen.update(body=r.content.decode()), httpx.Response(204))[1])))
    d.send([Event("target", "k1", "TP1 reached"), Event("stop", "k2", "the stop was reached")])
    lines = seen["body"]
    assert "[!] the stop was reached" in lines
    assert "[!] TP1 reached" not in lines, "only the urgent ones are marked"


def test_one_channel_failing_does_not_lose_an_event_the_other_took():
    def handler(request: httpx.Request):
        return httpx.Response(500) if "telegram" in str(request.url) else httpx.Response(204)

    d = Dispatcher({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c",
                    "DISCORD_WEBHOOK_URL": "https://example.invalid/hook"},
                   client=httpx.Client(transport=httpx.MockTransport(handler)))
    delivered, problems = d.send([Event("target", "k", "TP1 reached")])
    assert len(delivered) == 1, "Discord took it"
    assert any("Telegram" in p for p in problems)


def test_an_unreachable_channel_is_reported_rather_than_raising():
    def boom(request):
        raise httpx.ConnectError("no route to host")

    d = Dispatcher({"DISCORD_WEBHOOK_URL": "https://example.invalid/hook"},
                   client=httpx.Client(transport=httpx.MockTransport(boom)))
    delivered, problems = d.send([Event("target", "k", "TP1 reached")])
    assert delivered == [] and any("Discord" in p for p in problems)


def test_sending_nothing_is_not_an_error():
    assert Dispatcher({}).send([]) == ([], [])


# ---------- what survives the per-pass cap ----------
def test_an_urgent_warning_is_never_starved_by_a_crowd_of_reached_targets():
    """Events are appended targets, then stop, then traps, then checkpoints, and the pass is capped at
    eight. Truncating in append order therefore dropped a trap warning whenever eight targets had been
    reached - and because detection is deterministic, it was dropped on every later pass too, so the
    warning was never sent at all rather than merely delayed.

    Note on scope: the stop cannot be starved this way. `hit` needs price at or above a target and
    `stopped` needs it at or below the stop, and the stop sits below the entry which sits below the
    targets, so the two can never both be true for one coherent position."""
    plan_targets = [{"name": f"TP{i}", "price": 80_100.0 + i * 10} for i in range(12)]
    objs = [TimedTarget(f"TP{i}", 80_100.0 + i * 10, "b", NOW + HOUR, 1.0) for i in range(12)]
    p = progress(LONG, ENTRY, 79_000.0, objs, 81_000.0)      # every target reached, not stopped
    assert len(p.hit) == 12, "the fixture really does reach all of them"

    events = detect_events(trade(plan={"targets": plan_targets}), 81_000.0, p, traps=[Trap()],
                           checkpoints=[Checkpoint("4h", NOW - 60_000, "the 4h close")], now_ms=NOW)
    assert len(events) == alerts.MAX_PER_PASS
    kinds = [e.kind for e in events]
    assert "trap" in kinds, "the trap warning survives the cap"
    assert kinds[0] == "trap", "and urgent events are sent first"
    assert kinds.count("target") == alerts.MAX_PER_PASS - 1, "the targets fill what is left"


def test_a_stop_is_first_in_the_pass_when_it_fires():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 78_900.0)
    events = detect_events(trade(), 78_900.0, p,
                           checkpoints=[Checkpoint("4h", NOW - 60_000, "the 4h close")], now_ms=NOW)
    assert events[0].kind == "stop" and events[0].urgent


def test_a_quiet_position_is_unaffected_by_the_ordering():
    p = progress(LONG, ENTRY, 79_000.0, targets(), 80_650.0)
    events = detect_events(trade(), 80_650.0, p, now_ms=NOW)
    assert [e.dedup_key for e in events] == ["target:TP1"]
