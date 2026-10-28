"""The recorded order book: the Gate.io parse, the local snapshot store, and the scoop / sweep metrics."""
from __future__ import annotations

import json

import pytest

from core import book_metrics as bm
from core.book_store import BookStore
from core.data.order_book import BookSnapshot, parse_book, summary

T0 = 1_793_228_400_000
M = 60_000
GATE = {"id": 129283255892, "current": 1793228438.914, "update": 1793228438.908,
        "asks": [{"s": 52061, "p": "81155.1"}, {"s": 2, "p": "81158"}, {"s": 184, "p": "81159.2"}],
        "bids": [{"s": 22508, "p": "81155"}, {"s": 1, "p": "81154.5"}, {"s": 3, "p": "81154.2"}]}


def book(mid: float, ts_ms: int, bid_sizes=None, ask_sizes=None, step: float = 10.0, n: int = 25) -> BookSnapshot:
    """A synthetic book: `n` levels a `step` apart each side, 1 BTC a level unless given."""
    bids = [(mid - step / 2 - k * step, (bid_sizes or {}).get(k, 1.0)) for k in range(n)]
    asks = [(mid + step / 2 + k * step, (ask_sizes or {}).get(k, 1.0)) for k in range(n)]
    return BookSnapshot(ts_ms=ts_ms, book_id=ts_ms // 1000, bids=bids, asks=asks)


def test_gates_book_is_read_in_bitcoin_best_first():
    s = parse_book(GATE, 0.0001)
    assert s.ts_ms == 1793228438914 and s.book_id == 129283255892
    assert s.bids[0] == (81155.0, pytest.approx(2.2508)) and s.asks[0] == (81155.1, pytest.approx(5.2061))
    assert s.mid == pytest.approx(81155.05)
    d = summary(s)
    assert d["best_bid"] == 81155.0 and d["spread_bps"] == pytest.approx(0.0123, abs=1e-3)
    assert d["bid_100"] == pytest.approx(2.2508 + 0.0001 + 0.0003) and d["ask_010"] == pytest.approx(5.2061 + 0.0002 + 0.0184)
    assert -1 <= d["imbalance_025"] <= 1
    shuffled = {**GATE, "asks": list(reversed(GATE["asks"]))}
    assert parse_book(shuffled, 0.0001).asks[0][0] == 81155.1, "sorted best first whatever the order served"
    assert parse_book({"asks": [], "bids": []}, 0.0001).mid is None


def test_the_store_keeps_rows_once_reads_by_minute_and_prunes(tmp_path):
    st = BookStore(tmp_path / "book.sqlite")
    for k in range(25):                                        # 25 snapshots, 5 s apart: two minutes and a bit
        assert st.put(book(80_000, T0 + k * 5000))
    assert not st.put(book(80_000, T0)), "a timestamp is stored once"
    assert st.stats()["rows"] == 25 and st.latest()["ts_ms"] == T0 + 24 * 5000
    rows = st.between(T0, T0 + 3 * M, step_ms=M)
    assert [r["ts_ms"] for r in rows] == [T0, T0 + M, T0 + 2 * M], "the first snapshot of each minute"
    assert len(rows[0]["bids"]) == 25 and rows[0]["bids"][0] == (79_995.0, 1.0) and rows[0]["bid_025"] > 0
    assert len(st.between(T0, T0 + 3 * M)) == 25
    assert st.prune(T0 + M) == 12 and st.stats()["rows"] == 13
    st.close()
    with pytest.raises(FileNotFoundError):
        BookStore(tmp_path / "missing.sqlite", readonly=True)


def test_a_sweep_is_resting_size_price_ran_through_within_a_minute():
    quiet = [book(80_000 + (k % 2) * 2, T0 + k * M) for k in range(6)]                # 0.25 bps wobble
    assert bm.sweeps(quiet) == []
    # up 60 bps in one minute through 5 BTC of asks (1 BTC a level, 10 apart: 48 levels... only 25 kept)
    run = [book(80_000, T0), book(80_480, T0 + M), book(80_482, T0 + 2 * M)]
    ev = bm.sweeps(run)
    assert len(ev) == 1 and ev[0]["side"] == "asks" and ev[0]["direction"] == "up"
    assert ev[0]["cleared_btc"] == 25.0 and ev[0]["move_bps"] == 60.0 and ev[0]["t_ms"] == T0 + M
    thin = [book(80_000, T0, ask_sizes={k: 0.1 for k in range(25)}), book(80_480, T0 + M)]
    assert bm.sweeps(thin) == [], "a move through almost nothing is not a sweep"
    gap = [book(80_000, T0), book(80_480, T0 + 5 * M)]
    assert bm.sweeps(gap) == [], "five minutes apart is a drift, not a sweep"
    down = [book(80_000, T0), book(79_900, T0 + M)]
    assert bm.sweeps(down)[0]["side"] == "bids" and bm.sweeps(down)[0]["direction"] == "down"


def test_a_scoop_is_a_wall_that_holds_under_test_and_its_outcome_is_tracked():
    wall = {0: 40.0}                                            # 40 BTC on the best bid, 1 BTC elsewhere
    # price sits on the wall for three minutes, then leaves it by 15 bps with the wall intact: held
    held = [book(80_000, T0 + k * M, bid_sizes=wall) for k in range(3)] + [book(80_120, T0 + 3 * M, bid_sizes={0: 1.0, 12: 40.0})]
    ev = bm.scoops(held)
    assert len(ev) == 1 and ev[0]["side"] == "bids" and ev[0]["price"] == 79_995.0 and ev[0]["wall_btc"] == 40.0
    assert ev[0]["tested_minutes"] == 3 and ev[0]["outcome"] == "held"
    # the wall is pulled while price is still on it: not absorption
    pulled = [book(80_000, T0 + k * M, bid_sizes=wall) for k in range(2)] + [book(80_000, T0 + 2 * M)]
    assert bm.scoops(pulled)[0]["outcome"] == "pulled after test"
    # price goes through it: swept, not scooped
    swept = [book(80_000, T0 + k * M, bid_sizes=wall) for k in range(2)] + [book(79_900, T0 + 2 * M)]
    assert bm.scoops(swept)[0]["outcome"] == "swept"
    # a wall never tested (price 25 bps away the whole time) is no event
    far = [book(80_000, T0 + k * M, bid_sizes={20: 40.0}) for k in range(4)]
    assert bm.scoops(far) == []
    # still being tested at the end of the window: holding
    live = [book(80_000, T0 + k * M, bid_sizes=wall) for k in range(2)]
    assert bm.scoops(live)[0]["outcome"] == "holding"
    assert bm.scoops([]) == []


def test_the_report_reads_the_last_hour_and_says_when_the_book_is_stale_or_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path / "none"))
    assert bm.report(T0)["available"] is False and "unavailable" in bm.report(T0)["reason"], "no file: said, not crashed"
    st = BookStore(tmp_path / "book.sqlite")
    assert bm.report(T0, st) == {"available": False, "recorder": "empty", "reason": "no order book recorded yet"}
    for k in range(70):
        st.put(book(80_000 + (60 if k == 65 else 0), T0 + k * M, bid_sizes={0: 40.0}))
    rep = bm.report(T0 + 69 * M + 5000, st)
    assert rep["available"] and rep["as_of_ms"] == T0 + 69 * M and rep["minutes_in_window"] == 60
    assert rep["depth"]["bid_025"] > rep["depth"]["ask_025"] and rep["sweeps"] and rep["scoops"]
    assert rep["recorded"]["rows"] == 70 and "Provisional" in rep["method"]
    json.dumps(rep)
    stale = bm.report(T0 + 69 * M + 20 * M, st)
    assert stale["available"] is False and stale["reason"].startswith("last order book snapshot 20 min ago")
    st.close()


def test_the_worker_writes_the_report_to_the_shared_store_and_the_drawer_shows_it(tmp_path, monkeypatch):
    from core import pipeline
    from core.store import Store
    from ui import panels
    import worker
    store = Store(tmp_path / "s.sqlite")
    r = pipeline.PassResult()
    monkeypatch.setattr(bm, "report", lambda now: {"available": False, "recorder": "absent", "reason": "no file"})
    worker.book_report(store, T0, r)
    assert store.book_report() is None and r.problems == [], "no recorder on this machine is normal, not a problem"
    monkeypatch.setattr(bm, "report", lambda now: {"available": False, "recorder": "stale", "reason": "last order book snapshot 20 min ago"})
    worker.book_report(store, T0, r)
    assert r.problems == ["order book: last order book snapshot 20 min ago"], "a recorder gone quiet is"
    rep = {"available": True, "as_of_ms": T0, "depth": {"bid_025": 12.5, "ask_025": 4.0, "bid_100": 30.0, "ask_100": 20.0,
                                                       "imbalance_025": 0.52},
           "sweeps": [{"t_ms": T0, "side": "asks", "direction": "up", "from": 80_000.0, "to": 80_480.0, "move_bps": 60.0,
                       "cleared_btc": 25.0, "seconds": 60}],
           "scoops": [{"side": "bids", "price": 79_995.0, "wall_btc": 40.0, "now_btc": 38.0, "tested_minutes": 3,
                       "outcome": "held", "first_ms": T0, "last_ms": T0}],
           "recorded": {"rows": 70, "since_ms": T0 - 3_600_000, "mb": 0.1}, "method": "x. Provisional."}
    monkeypatch.setattr(bm, "report", lambda now: rep)
    worker.book_report(store, T0 + M, pipeline.PassResult())
    got = store.book_report()
    assert got["ts_ms"] == T0 + M and got["sweeps"][0]["cleared_btc"] == 25.0
    html_ = panels.qf_book_html(got, T0 + 2 * M)
    assert "25.0 BTC" in html_ and "cleared" in html_ and "held" in html_ and "12.5 / 4.0 BTC" in html_
    assert "\n\n" not in html_ and "<pre" not in html_
    assert "not running" in panels.qf_book_html(got, T0 + 30 * M)
    assert "No order-book report yet" in panels.qf_book_html(None, T0)
    store.close()
