"""The 24/7 worker and the shared store: one pass runs the page's steps, and two senders never double-send."""
import os

import pytest

os.environ.setdefault("TI_OFFLINE_FIXTURES", "1")

from core import pipeline, store as store_mod  # noqa: E402
from core.data.offline import fixture_market  # noqa: E402
from core.store import Store  # noqa: E402

import worker  # noqa: E402


class FakeDispatcher:
    sent: list = []

    def __init__(self, secrets):
        pass

    def send(self, events):
        FakeDispatcher.sent += [e.dedup_key for e in events]
        return list(events), []


@pytest.fixture
def two_stores(tmp_path):
    """Two Store objects on one file: two processes, each with its own connection and its own lock."""
    path = tmp_path / "shared.sqlite"
    a, b = Store(path), Store(path)
    yield a, b
    a.close()
    b.close()


def test_one_pass_runs_the_page_steps_and_sends_nothing_on_fixture_data(monkeypatch):
    monkeypatch.setattr(pipeline, "Dispatcher", FakeDispatcher)
    FakeDispatcher.sent = []
    s = Store(":memory:")
    r = worker.one_pass(s, market=fixture_market())
    assert set(r.verdicts) == {"live", "intraday", "weekly", "monthly"}, r.problems
    assert not r.problems, r.problems
    assert FakeDispatcher.sent == [], "fixture data is not the market: never sent"


def test_two_senders_claim_each_market_alert_once(two_stores):
    a, b = two_stores
    for i in range(3):
        a.add_market_alert(1_000 + i, "plan", f"k{i}", f"alert {i}", None)
    first, second = a.claim_market_alerts(2_000), b.claim_market_alerts(2_000)
    assert [r["dedup_key"] for r in first] == ["k0", "k1", "k2"] and second == []
    a.release_market_alerts([first[0]["id"]])
    assert [r["dedup_key"] for r in b.claim_market_alerts(2_001)] == ["k0"], "a refused send goes round again"


def test_two_senders_claim_each_trade_alert_once(two_stores):
    a, b = two_stores
    a.add_trade_event(7, 1_000, "stop", "stop-hit", "stopped", 80_000.0)
    a.add_trade_event(8, 1_000, "target", "t1", "target 1", 81_000.0)
    first, second = a.claim_trade_events(7, 2_000), b.claim_trade_events(7, 2_000)
    assert [r["dedup_key"] for r in first] == ["stop-hit"] and second == []
    assert [r["dedup_key"] for r in b.claim_trade_events(8, 2_000)] == ["t1"], "claims are per trade"
    a.release_trade_events([first[0]["id"]])
    again = b.claim_trade_events(7, 2_001)
    assert [r["dedup_key"] for r in again] == ["stop-hit"], "a refused stop goes round again"
    assert a.mark_notified([again[0]["id"]]) == 1, "a claimed event can be marked delivered"


def test_pipeline_sends_through_the_claim_so_two_passes_send_once(two_stores, monkeypatch):
    a, b = two_stores
    monkeypatch.setattr(pipeline, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(pipeline, "market_events", lambda *args: [])
    FakeDispatcher.sent = []
    a.add_market_alert(1_000, "plan", "watch-1", "WATCH", None)
    m = fixture_market()
    pipeline.market_alerts(a, m, {}, 2_000, {}, True)
    pipeline.market_alerts(b, m, {}, 2_000, {}, True)
    assert FakeDispatcher.sent == ["watch-1"]


def test_a_stale_claim_is_released_for_the_next_pass(two_stores):
    a, b = two_stores
    a.add_market_alert(1_000, "plan", "k", "alert", None)
    assert a.claim_market_alerts(2_000)
    assert b.claim_market_alerts(2_000 + Store.CLAIM_TIMEOUT_MS - 1) == []
    assert b.claim_market_alerts(2_001 + Store.CLAIM_TIMEOUT_MS), "a sender that died mid-send lets go"


def test_turso_rows_read_by_column_name():
    class Cur:
        description = (("id", None), ("kind", None))
        rowcount, lastrowid = 1, 5

        def fetchone(self):
            return (1, "plan")

        def fetchall(self):
            return [(1, "plan"), (2, "trap")]
    c = store_mod._TursoCursor(Cur())
    assert c.fetchone() == {"id": 1, "kind": "plan"}
    assert [r["kind"] for r in c] == ["plan", "trap"] and c.rowcount == 1 and c.lastrowid == 5


def test_without_turso_settings_or_the_client_it_is_the_local_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    assert Store(tmp_path / "a.sqlite").backend == "sqlite"
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.invalid")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "x")
    monkeypatch.setattr(store_mod, "libsql", None)
    assert Store(tmp_path / "b.sqlite").backend == "sqlite", "settings without the client fall back, logged"
    assert Store(":memory:").backend == "sqlite", "tests never reach the network"


def test_turso_upper_cased_keyword_columns_come_back_lowercase():
    """Turso stored reports.plan and trades.plan as PLAN; the code reads row["plan"]."""
    class Cur:
        description = (("id", None), ("PLAN", None))
        rowcount = lastrowid = 0

        def fetchone(self):
            return (1, "{}")

        def fetchall(self):
            return [(1, "{}")]
    c = store_mod._TursoCursor(Cur())
    assert c.fetchone()["plan"] == "{}" and c.fetchall()[0]["plan"] == "{}"
