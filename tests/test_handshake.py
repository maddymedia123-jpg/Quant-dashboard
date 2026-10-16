"""The cross-desk handshake: each category head asks the other three before its conviction is final."""
import sys
from pathlib import Path

import pytest

from core.agents import handshake as hsk
from core.live_anchor import publish
from core.store import Store

sys.path.insert(0, str(Path(__file__).parent))
from test_live_anchor import result  # noqa: E402  - the same war-room result builder the anchor tests use

NOW = 1_790_000_000_000
H = 3_600_000


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def anchor(s, category, bias, closes=NOW + H, confidence=0.7, opens=NOW - H):
    s.put_live_anchor(window_open_ms=opens, window_close_ms=closes, published_ms=NOW, bias=bias, margin=20.0,
                      confidence=confidence, trap=None, price=80_000.0, payload={"bias": bias}, category=category)


def answers(hs):
    return {a.category: a.answer for a in hs.answers}


# ---------- which way a verdict points ----------
def test_a_trap_risk_verdict_leans_against_its_own_word():
    assert hsk.lean("BULL") == "up" and hsk.lean("BEAR") == "down"
    assert hsk.lean("BULL TRAP RISK") == "down" and hsk.lean("BEAR TRAP RISK") == "up"
    assert hsk.lean("BALANCED") is None and hsk.lean("") is None


# ---------- asking ----------
def test_every_other_desk_is_asked_and_named_higher_or_lower(store):
    hs = hsk.consult(store, "intraday", "BULL", NOW)
    assert {a.category: a.relation for a in hs.answers} == {"live": "lower", "weekly": "higher", "monthly": "higher"}
    assert set(answers(hs).values()) == {"silent"} and hs.factor == 1.0


def test_higher_desks_agreeing_raise_conviction(store):
    anchor(store, "weekly", "BULL")
    anchor(store, "monthly", "BEAR TRAP RISK")                 # leans up: agrees with a long
    hs = hsk.consult(store, "intraday", "BULL", NOW)
    assert answers(hs)["weekly"] == answers(hs)["monthly"] == "agrees"
    assert hs.factor == pytest.approx(1.15), "1 + 0.10 + 0.10 = 1.20, held at the 1.15 ceiling"


def test_a_higher_desk_opposing_costs_more_than_one_agreeing_earns(store):
    anchor(store, "weekly", "BULL")
    anchor(store, "monthly", "BEAR")
    hs = hsk.consult(store, "intraday", "BULL", NOW)
    assert hs.factor == pytest.approx(0.90)
    assert any("Against the higher desk: Monthly" in n for n in hs.notes)


def test_a_lower_desk_opposing_is_an_early_warning_not_a_veto(store):
    anchor(store, "live", "BEAR")
    hs = hsk.consult(store, "weekly", "BULL", NOW)
    assert answers(hs)["live"] == "opposes" and hs.factor == pytest.approx(0.95)
    assert any("early warning, not a veto" in n for n in hs.notes)


def test_against_every_higher_desk_is_counter_trend_and_floored(store):
    anchor(store, "intraday", "BEAR")
    anchor(store, "weekly", "BEAR")
    anchor(store, "monthly", "BEAR")
    hs = hsk.consult(store, "live", "BULL", NOW)
    assert hs.factor == pytest.approx(0.5), "1 - 3 x 0.20 = 0.40, held at the 0.5 floor"
    assert any(n.startswith("Counter-trend") for n in hs.notes)


def test_a_closed_candles_verdict_is_silent_not_counted(store):
    anchor(store, "weekly", "BEAR", closes=NOW - 1)
    hs = hsk.consult(store, "intraday", "BULL", NOW)
    assert answers(hs)["weekly"] == "silent" and hs.factor == 1.0
    assert "candle has closed" in next(a.note for a in hs.answers if a.category == "weekly")


def test_a_balanced_desk_is_not_moved(store):
    anchor(store, "weekly", "BEAR")
    hs = hsk.consult(store, "intraday", "BALANCED", NOW)
    assert answers(hs)["weekly"] == "neutral" and hs.factor == 1.0


def test_an_unreadable_ledger_makes_a_desk_silent(store):
    class Broken:
        def get_live_anchor(self, category):
            raise RuntimeError("locked")
    hs = hsk.consult(Broken(), "intraday", "BULL", NOW)
    assert set(answers(hs).values()) == {"silent"} and hs.factor == 1.0


# ---------- finalising ----------
def test_the_handshake_changes_conviction_not_direction(store):
    anchor(store, "weekly", "BEAR")
    res = result(0.9, 0.2)
    assert res.bias.startswith("BULL")
    done = hsk.apply(res, hsk.consult(store, "intraday", res.bias, NOW))
    assert done.bias == res.bias and done.confidence == pytest.approx(res.confidence * 0.8)
    assert done.confidence_before_handshake == res.confidence
    assert done.handshake["answers"] and any("Against the higher desk" in n for n in done.notes)


def test_conviction_never_exceeds_one():
    res = result(1.0, 0.0)
    hs = hsk.Handshake("intraday", res.bias, (), 1.15, ())
    assert hsk.apply(res, hs).confidence <= 1.0


def test_the_pinned_summary_carries_the_exchange(store):
    anchor(store, "weekly", "BULL")
    res = hsk.apply(result(0.9, 0.2), hsk.consult(store, "intraday", "BULL", NOW))
    pub = publish(store, res, NOW, price=80_000.0)
    assert pub.anchored and pub.anchor["handshake"]["factor"] == pytest.approx(1.10)
    assert pub.anchor["confidence"] == pytest.approx(round(res.confidence, 3))
    assert pub.anchor["confidence_before_handshake"] < pub.anchor["confidence"]


# ---------- the other direction ----------
def test_a_desk_this_verdict_opposes_is_told_once(store):
    anchor(store, "weekly", "BULL")
    anchor(store, "live", "BEAR")
    assert hsk.notify(store, "intraday", "BEAR", NOW, price=80_000.0) == ["weekly"]
    notes = store.side_notes(NOW - H, category="weekly")
    assert len(notes) == 1 and notes[0]["kind"] == hsk.HANDSHAKE and "Intraday" in notes[0]["headline"]
    assert "early warning" in notes[0]["detail"], "intraday is lower than weekly"
    assert hsk.notify(store, "intraday", "BEAR", NOW + 60_000) == [], "the same news is not repeated"
    assert store.side_notes(NOW - H, category="live") == [], "a desk that agrees is not told"


def test_a_lower_desk_hears_that_a_higher_one_sets_the_vector(store):
    anchor(store, "live", "BULL")
    hsk.notify(store, "monthly", "BEAR", NOW)
    assert "sets the vector" in store.side_notes(NOW - H, category="live")[0]["detail"]


def test_closed_or_balanced_desks_are_not_told(store):
    anchor(store, "weekly", "BULL", closes=NOW - 1)
    anchor(store, "monthly", "BALANCED")
    assert hsk.notify(store, "intraday", "BEAR", NOW) == []
    anchor(store, "live", "BULL")
    assert hsk.notify(store, "intraday", "BALANCED", NOW) == [], "no direction called, nobody opposed"
    assert store.side_notes(NOW - H, category="live") == []


# ---------- the card ----------
def test_the_summary_card_shows_each_answer_and_the_conviction_change(store):
    from ui import panels
    anchor(store, "weekly", "BEAR")
    res = hsk.apply(result(0.9, 0.2), hsk.consult(store, "intraday", "BULL", NOW))
    pub = publish(store, res, NOW)
    h = panels.anchored_summary_html(pub.anchor, NOW + H, NOW, True)
    assert "Cross-desk handshake" in h and "opposes" in h and "after the handshake (x0.80)" in h
    assert "Weekly" in h and "higher desk" in h
    assert panels.handshake_html({"bias": "BULL"}) == "", "older anchors without a handshake show nothing"


@pytest.mark.parametrize("bias, glyph, tone", [("BULL", "▲", "up"), ("BEAR", "▼", "down"),
                                               ("BULL TRAP RISK", "▼", "down"), ("BALANCED", "◆", "neutral")])
def test_the_pinned_bias_wears_its_direction(bias, glyph, tone):
    """The summary looked biases up as BULLISH/BEARISH; the war room writes BULL/BEAR, so every pinned
    call showed a neutral diamond in a neutral card."""
    from ui import panels
    h = panels.anchored_summary_html({"bias": bias, "confidence": 0.5}, NOW + H, NOW, True)
    assert f"<div class='ti-dir xl {tone}'>{glyph} {bias}</div>" in h
