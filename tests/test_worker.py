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


def test_the_book_report_is_rewritten_every_five_minutes_not_every_pass(monkeypatch):
    """The scoop/sweep report was replaced on every pass: one more page or two of Turso sync a minute (9 Nov)."""
    from core import book_metrics
    monkeypatch.setattr(book_metrics, "report", lambda now_ms: {"available": True, "minutes_in_window": 60, "sweeps": [], "scoops": [],
                                                                "depth": {"bid_025": 1.0, "ask_025": 2.0}, "stamp": now_ms})
    s, r = Store(":memory:"), pipeline.PassResult()
    worker.book_report(s, 1_000_000, r)
    assert s.book_report()["ts_ms"] == 1_000_000
    worker.book_report(s, 1_000_000 + 60_000, r)
    assert s.book_report()["ts_ms"] == 1_000_000, "a minute later: the stored report stands"
    worker.book_report(s, 1_000_000 + worker.BOOK_REPORT_GAP_MS, r)
    assert s.book_report()["ts_ms"] == 1_000_000 + worker.BOOK_REPORT_GAP_MS
    assert r.problems == []
    s.close()


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


def test_turso_secrets_are_found_even_inside_a_section():
    url, tok = "libsql://x.turso.io", "t0k"
    assert store_mod.find_turso({"TURSO_DATABASE_URL": url, "TURSO_AUTH_TOKEN": tok}) == {
        "TURSO_DATABASE_URL": url, "TURSO_AUTH_TOKEN": tok}
    pasted_under_a_section = {"GEMINI_API_KEY": "g", "typesafe": {"key": "k", "TURSO_DATABASE_URL": url,
                                                                  "TURSO_AUTH_TOKEN": tok}}
    assert store_mod.find_turso(pasted_under_a_section) == {"TURSO_DATABASE_URL": url, "TURSO_AUTH_TOKEN": tok}
    assert store_mod.find_turso({"TURSO_DATABASE_URL": url, "TURSO_AUTH_TOKEN": "  "}) == {"TURSO_DATABASE_URL": url}


def test_the_store_says_why_it_is_not_on_turso(tmp_path, monkeypatch):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    assert Store(tmp_path / "a.sqlite").turso_note == "not set: TURSO_DATABASE_URL, TURSO_AUTH_TOKEN"
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://x")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "t")
    monkeypatch.setattr(store_mod, "libsql", None)
    assert "not installed" in Store(tmp_path / "b.sqlite").turso_note
    monkeypatch.setattr(store_mod, "libsql", object())

    def refuse(*a, **k):
        raise ValueError("Hrana: 401 Unauthorized")
    monkeypatch.setattr(store_mod, "_TursoConnection", refuse)
    s = Store(tmp_path / "c.sqlite")
    assert s.backend == "sqlite" and s.turso_note.startswith("could not connect to Turso: ValueError: Hrana: 401")
    s.add_market_alert(1, "plan", "k", "h", None)
    assert s.claim_market_alerts(2), "a failed connection leaves a working local store, not a broken page"


class _FakeLibsql:
    """libsql as the store sees it: connect() records how it was called and returns an in-memory SQLite
    connection with the one extra method a replica has, sync(), which counts its calls."""

    def __init__(self):
        self.calls, self.syncs = [], 0

    def connect(self, *args, **kwargs):
        import sqlite3
        self.calls.append((args, kwargs))
        fake = self

        class Conn:
            def __init__(self):
                self._c = sqlite3.connect(":memory:", check_same_thread=False)

            def execute(self, sql, params=()):
                return self._c.execute(sql, params)

            def executescript(self, script):
                self._c.executescript(script)

            def commit(self):
                self._c.commit()

            def rollback(self):
                self._c.rollback()

            def close(self):
                self._c.close()

            def sync(self):
                fake.syncs += 1
        return Conn()


def test_a_direct_store_keeps_no_replica_and_never_syncs(tmp_path, monkeypatch):
    """Turso meters every page an embedded replica pulls back (3 GB a month free; the account was blocked on
    23 Nov after 250-330 MB a day of it). The worker opens the database directly instead: no local copy, no
    pull, each query a round trip counted as rows read (500 M a month). The dashboard keeps its replica."""
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://quant.example.invalid")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "tok")
    fake = _FakeLibsql()
    monkeypatch.setattr(store_mod, "libsql", fake)
    s = Store(tmp_path / "x.sqlite", direct=True)
    assert s.backend == "turso" and s.direct is True
    assert fake.calls == [(("libsql://quant.example.invalid",), {"auth_token": "tok"})], "the URL alone: no replica path, no sync_url"
    s.sync()
    s.sync(min_gap_s=30)
    assert fake.syncs == 0, "nothing to pull: the queries already run on Turso"
    tid = s.open_trade(opened_ms=1_000, asset="BTC/USDT", side="SHORT", entry=64_000.0, stop=66_000.0, given_stop=66_000.0,
                       stop_verdict="hard invalidation", category="weekly", plan={"desk": "weekly"})
    assert s.trade(tid)["plan"]["desk"] == "weekly" and s.trades(status="open")[0]["entry"] == 64_000.0
    assert s.turso_note is None


def test_a_replica_store_still_syncs_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://quant.example.invalid")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "tok")
    fake = _FakeLibsql()
    monkeypatch.setattr(store_mod, "libsql", fake)
    s = Store(tmp_path / "x.sqlite")
    assert s.backend == "turso" and s.direct is False
    (args, kwargs), = fake.calls
    assert args == (str(tmp_path / "x.turso.db"),) and kwargs == {"sync_url": "libsql://quant.example.invalid", "auth_token": "tok"}
    before = fake.syncs
    s.sync()
    assert fake.syncs == before + 1


def test_the_worker_opens_the_store_direct(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, *args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs
    monkeypatch.setattr(worker, "Store", Recorder)
    worker.open_store()
    assert seen == {"args": (), "kwargs": {"direct": True}}


def test_prune_drops_signals_older_than_a_week_and_keeps_the_latest_read_working():
    """Signals are only ever read newest-first (last_signal); 79,000 of 81,000 rows were older than a week on
    23 Nov, a third of the database that every full replica pull paid for."""
    s = Store(":memory:")
    day = 86_400_000
    now = 100 * day
    for age_days in (30, 8, 7, 6, 1, 0):
        s.add_signal(now - age_days * day, "live", 40, 60_000.0)
    s.add_signal(now - 20 * day, "weekly", 50, 60_000.0)
    assert s.prune(now) == {"signals": 3}, "30, 20 and 8 days old go; exactly 7 days stays"
    assert s.last_signal("live")["ts_ms"] == now and s.last_signal("weekly") is None
    assert s.prune(now) == {"signals": 0}


def test_market_alerts_keyed_and_the_newest_report_use_an_index():
    """On a direct connection every row examined is metered: these two reads scanned the whole table
    (4 x 988 and 5 x 85 rows a pass on 23 Nov)."""
    s = Store(":memory:")
    for i, key in enumerate(["cascade:live:a", "cascade:live:b", "cascade:liver:c", "cascade:intraday:a", "qf:2026-11-23"]):
        s.add_market_alert(1_000 + i, "cascade", key, "h", 1.0)
    assert [r["dedup_key"] for r in s.market_alerts_keyed("cascade:live:")] == ["cascade:live:b", "cascade:live:a"], \
        "a prefix, not a word: cascade:liver: is another desk's key"
    assert [r["dedup_key"] for r in s.market_alerts_keyed("cascade:", limit=2)] == ["cascade:intraday:a", "cascade:liver:c"], "newest first"

    def plan(sql, args):
        return " | ".join(r["detail"] for r in s._conn.execute("EXPLAIN QUERY PLAN " + sql, args).fetchall())
    keyed = plan("SELECT * FROM market_alerts WHERE dedup_key >= ? AND dedup_key < ? ORDER BY ts_ms DESC, id DESC LIMIT ?",
                 ("cascade:live:", "cascade:live:\U0010ffff", 20))
    assert "SEARCH" in keyed and "SCAN market_alerts" not in keyed, keyed
    newest = plan("SELECT id FROM reports ORDER BY ts_ms DESC LIMIT ?", (1,))
    assert "idx_reports_ts" in newest, newest
