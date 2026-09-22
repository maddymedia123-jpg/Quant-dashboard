from datetime import datetime, timezone

import pytest

from core import live_anchor as la
from core.agents.judge import ChoiceResult, JudgeResult
from core.agents.live_recon import LiveReconResult, resolve, score_team
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH
from core.store import Store

H4 = la.WINDOW_MS


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def result(bull_p: float, bear_p: float, trap: ChoiceResult | None = None, ms: int = 0) -> LiveReconResult:
    trap = trap or ChoiceResult(choice="NO_TRAP", probabilities={"NO_TRAP": 0.8})
    bull = score_team(BULLISH, JudgeResult(probabilities={i.id: bull_p for i in ALL_ITEMS}, provider="typesafe"))
    bear = score_team(BEARISH, JudgeResult(probabilities={i.id: bear_p for i in ALL_ITEMS}, provider="typesafe"))
    bias, margin, conf, consulted, override, notes = resolve(bull, bear, trap)
    return LiveReconResult(generated_at=datetime.fromtimestamp(ms / 1000, timezone.utc), bull=bull, bear=bear,
                           trap=trap, bias=bias, margin=margin, confidence=conf, consulted=consulted,
                           override=override, notes=notes)


# ---- window maths ----
def test_windows_align_to_the_utc_four_hour_candle():
    t = int(datetime(2026, 9, 19, 13, 47, 12, tzinfo=timezone.utc).timestamp() * 1000)
    assert la.window_open(t) == int(datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert la.window_close(t) == int(datetime(2026, 9, 19, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close = la.window_close(t)
    assert la.window_open(close) == close, "a run at the close belongs to the new candle"
    assert la.window_open(close - 1) == la.window_open(t)


def test_direction_ignores_the_trap_qualifier():
    assert la.direction("BULL TRAP RISK") == "BULL" and la.direction("BEAR TRAP RISK") == "BEAR"
    assert la.direction("BALANCED") == "BALANCED"


# ---- publishing ----
def test_first_run_of_a_window_anchors_the_summary(store):
    t = 2 * H4 + 60_000
    out = la.publish(store, result(0.8, 0.3, ms=t), t, price=78_000.0)
    assert out.anchored and out.window_open_ms == 2 * H4 and out.window_close_ms == 3 * H4
    assert out.anchor["bias"] == "BULL" and out.anchor["price"] == 78_000.0
    assert out.anchor["domains"]["bullish"]["smc"] == 16.0
    row = store.get_live_anchor(2 * H4)
    assert row["bias"] == "BULL" and row["published_ms"] == t


def test_a_rerun_inside_the_window_never_replaces_the_summary(store):
    t = 2 * H4 + 60_000
    la.publish(store, result(0.8, 0.3, ms=t), t, price=78_000.0)
    later = t + 15 * 60_000
    out = la.publish(store, result(0.2, 0.9, ms=later), later, price=76_000.0)

    assert not out.anchored
    assert out.anchor["bias"] == "BULL" and out.anchor["price"] == 78_000.0, "the pinned summary is untouched"
    assert store.get_live_anchor(2 * H4)["payload"]["bull_points"] == 80.0
    assert [n.kind for n in out.appended] == [la.BIAS_FLIP]
    note = store.side_notes(2 * H4)[0]
    assert "changed from BULL to BEAR" in note["headline"] and note["price"] == 76_000.0
    assert "anchored summary stands" in note["detail"]


def test_the_next_candle_close_publishes_a_new_anchor(store):
    t = 2 * H4 + 60_000
    la.publish(store, result(0.8, 0.3, ms=t), t)
    nxt = 3 * H4 + 1_000
    out = la.publish(store, result(0.2, 0.9, ms=nxt), nxt)
    assert out.anchored and out.window_open_ms == 3 * H4 and out.anchor["bias"] == "BEAR"
    assert store.get_live_anchor()["window_open_ms"] == 3 * H4        # latest
    assert store.get_live_anchor(2 * H4)["bias"] == "BULL"            # history kept
    assert len(store.live_anchors()) == 2


def test_the_same_change_is_only_noted_once_per_window(store):
    t = 2 * H4 + 60_000
    la.publish(store, result(0.8, 0.3, ms=t), t)
    for step in (1, 2, 3):
        out = la.publish(store, result(0.2, 0.9, ms=t + step * 900_000), t + step * 900_000)
    assert out.appended == () and [n.kind for n in out.skipped] == [la.BIAS_FLIP]
    assert len(store.side_notes(2 * H4)) == 1


def test_the_store_refuses_to_overwrite_a_published_anchor(store):
    assert store.put_live_anchor(H4, 2 * H4, H4 + 10, "BULL", 30.0, 0.6, None, 1.0, {"bias": "BULL"})
    assert not store.put_live_anchor(H4, 2 * H4, H4 + 20, "BEAR", -30.0, 0.6, None, 2.0, {"bias": "BEAR"})
    assert store.get_live_anchor(H4)["bias"] == "BULL"


# ---- what counts as substantial ----
def test_a_new_trap_call_files_a_side_note():
    anchor = la.anchor_payload(result(0.8, 0.3))
    trap = ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.75})
    notes = la.detect_side_notes(anchor, result(0.8, 0.35, trap))
    kinds = {n.kind: n for n in notes}
    assert "75%" in kinds[la.TRAP_ALERT].headline
    assert "published with NO_TRAP" in kinds[la.TRAP_ALERT].detail
    silent = la.anchor_payload(result(0.8, 0.3, ChoiceResult(error="timeout")))
    assert "no trap call" in la.detect_side_notes(silent, result(0.8, 0.35, trap))[0].detail
    weak = ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.4})
    assert not any(n.kind == la.TRAP_ALERT for n in la.detect_side_notes(anchor, result(0.8, 0.35, weak)))


def test_a_big_move_that_does_not_flip_the_bias_is_still_reported():
    anchor = la.anchor_payload(result(0.8, 0.3))          # margin +50
    small = la.detect_side_notes(anchor, result(0.75, 0.35))   # margin +40, moved 10
    assert not any(n.kind == la.SCORE_SWING for n in small)
    big = la.detect_side_notes(anchor, result(0.6, 0.45))      # margin +15, moved 35
    swing = next(n for n in big if n.kind == la.SCORE_SWING)
    assert "towards bears" in swing.headline and "+50" in swing.detail and "+15" in swing.detail
    assert not any(n.kind == la.BIAS_FLIP for n in big), "still bullish, so no flip"


def test_a_macro_event_and_a_degraded_scan_are_flagged():
    anchor = la.anchor_payload(result(0.8, 0.3))
    notes = la.detect_side_notes(anchor, result(0.8, 0.3), macro_event="FOMC Rate Decision")
    assert next(n for n in notes if n.kind == la.MACRO_ALERT).dedup_key == "macro:FOMC Rate Decision"

    broken = result(0.8, 0.3)
    broken.bull = score_team(BULLISH, JudgeResult(probabilities={}, error="503"))
    broken.notes = ("The bullish team did not score (503); its 0 is a failure, not a reading.",)
    flagged = la.detect_side_notes(anchor, broken)
    assert any(n.kind == la.DEGRADED for n in flagged)
    assert not any(n.kind == la.DEGRADED for n in la.detect_side_notes(la.anchor_payload(broken), broken))


def test_a_quiet_rerun_files_nothing(store):
    t = 2 * H4 + 60_000
    la.publish(store, result(0.8, 0.3, ms=t), t)
    out = la.publish(store, result(0.78, 0.32, ms=t + 900_000), t + 900_000)
    assert not out.anchored and out.appended == () and out.skipped == ()
    assert store.side_notes(2 * H4) == []


# ---- one anchor per category per window ----
def test_categories_pin_separately_in_the_same_window(store):
    assert store.put_live_anchor(H4, 2 * H4, H4 + 1, "BULL", 30.0, 0.6, None, 1.0, {"bias": "BULL"})
    assert store.put_live_anchor(H4, 2 * H4, H4 + 1, "BEAR", -30.0, 0.6, None, 1.0, {"bias": "BEAR"},
                                 category="intraday")
    assert store.get_live_anchor(H4)["bias"] == "BULL"
    assert store.get_live_anchor(H4, category="intraday")["bias"] == "BEAR"
    assert not store.put_live_anchor(H4, 2 * H4, H4 + 2, "BEAR", -1.0, 0.1, None, 1.0, {},
                                     category="intraday"), "still insert-once within a category"


def test_side_notes_are_kept_per_category(store):
    assert store.add_side_note(H4, H4 + 5, "BIAS_FLIP", "bias:BULL->BEAR", "flip")
    assert store.add_side_note(H4, H4 + 5, "BIAS_FLIP", "bias:BULL->BEAR", "flip", category="weekly")
    assert len(store.side_notes(H4)) == 1 and len(store.side_notes(H4, category="weekly")) == 1


def test_anchors_pinned_before_categories_existed_migrate_to_live(tmp_path):
    import json
    import sqlite3

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE live_anchors (window_open_ms INTEGER PRIMARY KEY, window_close_ms INTEGER NOT NULL,
            published_ms INTEGER NOT NULL, bias TEXT NOT NULL, margin REAL, confidence REAL, trap TEXT,
            price REAL, payload TEXT NOT NULL);
        CREATE TABLE live_side_notes (id INTEGER PRIMARY KEY AUTOINCREMENT, window_open_ms INTEGER NOT NULL,
            ts_ms INTEGER NOT NULL, kind TEXT NOT NULL, dedup_key TEXT NOT NULL, headline TEXT NOT NULL,
            detail TEXT, price REAL);""")
    con.execute("INSERT INTO live_anchors VALUES (?,?,?,?,?,?,?,?,?)",
                (H4, 2 * H4, H4 + 9, "BULL", 20.0, 0.5, "NO_TRAP", 78000.0, json.dumps({"bias": "BULL"})))
    con.execute("INSERT INTO live_side_notes (window_open_ms, ts_ms, kind, dedup_key, headline) VALUES (?,?,?,?,?)",
                (H4, H4 + 99, "BIAS_FLIP", "bias:BULL->BEAR", "flip"))
    con.commit()
    con.close()

    s = Store(path)
    assert s.get_live_anchor(H4)["bias"] == "BULL" and s.get_live_anchor(H4)["payload"] == {"bias": "BULL"}
    assert [n["headline"] for n in s.side_notes(H4)] == ["flip"]
    s.close()
    s2 = Store(path)                                          # migration is idempotent on reopen
    assert len(s2.side_notes(H4)) == 1 and len(s2.live_anchors()) == 1
    s2.close()


def test_publish_pins_each_category_to_its_own_candle(store):
    from core.agents.recon_profiles import PROFILES

    weekly = PROFILES["weekly"]
    tuesday = int(datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    out = la.publish(store, result(0.8, 0.3, ms=tuesday), tuesday, price=81_000.0, profile=weekly)
    monday = int(datetime(2026, 9, 21, tzinfo=timezone.utc).timestamp() * 1000)
    assert out.anchored and out.window_open_ms == monday
    assert out.window_close_ms == monday + 7 * 24 * 3_600_000
    assert store.get_live_anchor(monday, category="weekly")["bias"] == "BULL"
    assert store.get_live_anchor(category="live") is None, "the weekly run does not pin Live Recon"

    later = la.publish(store, result(0.2, 0.9, ms=tuesday + 3_600_000), tuesday + 3_600_000, profile=weekly)
    assert not later.anchored and [n.kind for n in later.appended] == [la.BIAS_FLIP]
    assert len(store.side_notes(monday, category="weekly")) == 1 and store.side_notes(monday) == []
