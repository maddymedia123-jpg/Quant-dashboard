"""The trap ledger: what makes two sightings the same trap, and what settles one exactly once.

A trap is a market event rather than a desk's opinion of one, so the same 4h raid found by the Live desk
and by the Weekly desk is one row. The level is matched on the trap's own scale, because a recomputed
value-area edge drifts between refreshes - measured at $14.79 across five consecutive 1h refreshes, which
the old one-cent window treated as a second trap.
"""
import sqlite3
import threading

import pytest

from core.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


ATR = 400.0          # a plausible 1h ATR, so the invalidation buffer is 200 and the tolerance 100


def declare(s: Store, *, category="live", side="bull_trap", timeframe="1h", level=80_800.0,
            declared_ms=1_000, score=72.0) -> int:
    """Declare a trap the way detection does: the invalidation and target are derived from the level, so
    a drifting level carries its buffer with it rather than moving towards a fixed invalidation."""
    up = side == "bull_trap"
    buffer = ATR / 2
    return s.put_trap(category=category, side=side, timeframe=timeframe, level=level,
                      invalidation=level + buffer if up else level - buffer,
                      plays_out=level - 2 * buffer if up else level + 2 * buffer,
                      declared_ms=declared_ms, score=score, head_call="ENGINEERED_TRAP",
                      evidence=["swept and reclaimed"], notes=[])


# ---------- identity ----------
def test_the_same_raid_is_one_row_however_often_it_is_re_detected(store):
    assert declare(store) == declare(store) == declare(store)
    assert len(store.traps()) == 1


def test_a_level_that_drifts_between_refreshes_is_still_the_same_trap(store):
    """The value-area edge is recomputed from a rolling window, so it moves by dollars between refreshes.
    Anything inside a quarter ATR - half the invalidation distance - is the same raid."""
    first = declare(store, level=80_800.0)
    for drift in (0.01, 5.0, 14.79, 99.0):
        assert declare(store, level=80_800.0 + drift) == first, f"${drift} of drift is the same level"
        assert declare(store, level=80_800.0 - drift) == first
    assert len(store.traps()) == 1


def test_a_genuinely_different_level_is_a_different_trap(store):
    """The tolerance must not be so wide that two real levels collapse into one row."""
    first = declare(store, level=80_800.0)
    second = declare(store, level=80_800.0 + 101.0)      # beyond half the 200-point buffer
    assert second != first and len(store.traps()) == 2


def test_two_desks_that_see_the_same_trap_declare_it_once(store):
    """Weekly and Monthly read the same candles, and Live and Intraday overlap on the 4h. The same 4h
    raid found by any of them is one anchored trap, not one per desk."""
    first = declare(store, category="live", timeframe="4h")
    for desk in ("intraday", "weekly", "monthly"):
        assert declare(store, category=desk, timeframe="4h") == first
    rows = store.traps()
    assert len(rows) == 1 and rows[0]["category"] == "live", "the desk that found it keeps the credit"


def test_the_same_level_on_another_timeframe_is_another_trap(store):
    assert declare(store, timeframe="1h") != declare(store, timeframe="4h")
    assert len(store.traps()) == 2


def test_the_opposite_side_at_the_same_level_is_another_trap(store):
    assert declare(store, side="bull_trap") != declare(store, side="bear_trap")
    assert len(store.traps()) == 2


def test_a_settled_raid_can_be_declared_again_later(store):
    """Dedup covers live traps only. The same level can be raided again next week, and that is a new
    trap - detection is what stops a just-settled one coming straight back."""
    first = declare(store)
    assert store.resolve_trap(first, "played_out", 80_000.0, 2_000) is True
    again = declare(store, declared_ms=9_000)
    assert again != first and len(store.traps()) == 2


# ---------- settling exactly once ----------
def test_a_trap_settles_once_and_keeps_the_first_answer(store):
    """Tested through resolve_trap directly, because settle()'s own active-only query would otherwise
    hide a missing guard here: with the guard removed, this is what changes."""
    trap = declare(store)
    assert store.resolve_trap(trap, "invalidated", 81_050.0, 3_000) is True
    assert store.resolve_trap(trap, "played_out", 79_000.0, 4_000) is False
    row = store.traps()[0]
    assert (row["status"], row["resolved_price"], row["resolved_ms"]) == ("invalidated", 81_050.0, 3_000)


def test_resolving_a_trap_that_does_not_exist_is_false(store):
    assert store.resolve_trap(4242, "invalidated", 80_000.0, 1_000) is False


def test_only_one_of_two_racing_resolutions_wins(store):
    trap = declare(store)
    wins, barrier = [], threading.Barrier(2)

    def resolve(price):
        barrier.wait()
        wins.append(store.resolve_trap(trap, "invalidated", price, 5_000))

    threads = [threading.Thread(target=resolve, args=(p,)) for p in (81_050.0, 81_060.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(wins) == [False, True], "one settlement, one refusal"


# ---------- filters ----------
def test_the_status_filter_selects_only_live_traps(store):
    live = declare(store, timeframe="1h")
    done = declare(store, timeframe="4h")
    store.resolve_trap(done, "invalidated", 81_050.0, 2_000)
    assert [r["id"] for r in store.traps(status="active")] == [live]
    assert [r["id"] for r in store.traps(status="invalidated")] == [done]
    assert len(store.traps()) == 2


def test_a_desk_asks_for_the_traps_on_the_timeframes_it_reads(store):
    """How a desk finds its warnings: by timeframe, so a 1d trap reaches Intraday, Weekly and Monthly."""
    fast = declare(store, timeframe="15m")
    daily = declare(store, timeframe="1d")
    weekly = declare(store, timeframe="1w")
    live_desk = store.traps(timeframes=("15m", "1h", "4h"))
    assert [r["id"] for r in live_desk] == [fast]
    monthly_desk = store.traps(timeframes=("4h", "1d", "1w"))
    assert sorted(r["id"] for r in monthly_desk) == sorted([daily, weekly])
    assert store.traps(timeframes=("1M",)) == []


def test_filters_combine(store):
    declare(store, timeframe="1d")
    done = declare(store, timeframe="1w")
    store.resolve_trap(done, "played_out", 80_000.0, 2_000)
    assert store.traps(timeframes=("1w",), status="active") == []
    assert len(store.traps(timeframes=("1d", "1w"), status="active")) == 1


def test_traps_are_newest_first_and_limited(store):
    ids = [declare(store, timeframe=tf, declared_ms=ms)
           for tf, ms in (("15m", 1_000), ("1h", 2_000), ("4h", 3_000))]
    rows = store.traps(limit=2)
    assert [r["id"] for r in rows] == [ids[2], ids[1]]


# ---------- the row a card is drawn from ----------
def test_a_stored_trap_carries_everything_a_card_needs(store):
    trap = declare(store)
    row = store.traps()[0]
    for key in ("id", "category", "side", "timeframe", "level", "invalidation", "plays_out",
                "declared_ms", "score", "head_call", "evidence", "notes", "status"):
        assert key in row, key
    assert row["id"] == trap and isinstance(row["level"], float) and isinstance(row["declared_ms"], int)
    assert row["evidence"] == '["swept and reclaimed"]', "stored as JSON for the card to parse"


def test_a_failed_declaration_leaves_no_half_written_row(store):
    """The read and the insert share a transaction, so a failure cannot commit a partial trap."""
    with pytest.raises((sqlite3.Error, TypeError, ValueError)):
        store.put_trap(category="live", side="bull_trap", timeframe="1h", level=80_800.0,
                       invalidation=81_000.0, plays_out=80_000.0, declared_ms=1_000, score=1.0,
                       head_call="X", evidence=[object()], notes=[])   # not JSON-serialisable
    assert store.traps() == []
    assert declare(store) is not None, "and the store still works afterwards"
