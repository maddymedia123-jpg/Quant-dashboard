"""The cross-desk handshake: every category head asks the other three before finalising its conviction.

His framework: the four category heads (4H, 24H, Weekly, Monthly) query each other before they commit a
conviction score, and the exchange runs both ways.

It is deterministic, like the Active Trade handshake. Each desk answers with the verdict it has already
anchored for its current candle - read from the ledger, not re-asked of a model - so a desk can never
say one thing on its own screen and another when consulted. An anchor whose candle has closed is not
that desk's current view and stays silent.

What the answers do, and only to conviction (the direction is the war room's evidence, not a vote):

* a HIGHER desk agreeing adds 10% to conviction; a higher desk opposing takes 20% off. The higher
  timeframe sets the vector, so going against it costs more than going with it earns.
* a LOWER desk opposing takes 5% off: it is the engine timeframe, an early warning, not a veto.
* the multiplier is held between 0.5 and 1.15, and conviction never exceeds 100%.

Both ways: when a desk publishes a verdict that opposes another desk's live anchored verdict, a side note
is appended to that other desk's window, so the desk that was overruled hears about it too."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from core.agents.recon_profiles import PROFILES

log = logging.getLogger(__name__)

ORDER = ("live", "intraday", "weekly", "monthly")      # lowest timeframe first
HIGHER_AGREES, HIGHER_OPPOSES, LOWER_OPPOSES = 0.10, -0.20, -0.05
FACTOR_FLOOR, FACTOR_CEIL = 0.5, 1.15
HANDSHAKE = "HANDSHAKE"                                 # the side-note kind


def lean(bias: str) -> str | None:
    """Which way a verdict points. A trap-risk verdict points against its own word: BULL TRAP RISK
    expects the up-move to fail, so it leans down."""
    b = str(bias or "")
    if "BULL" not in b and "BEAR" not in b:
        return None
    return "up" if ("BULL" in b) != ("TRAP RISK" in b) else "down"


@dataclass(frozen=True)
class DeskAnswer:
    category: str
    label: str
    relation: str              # higher | lower
    bias: str
    answer: str                # agrees | opposes | neutral | silent
    confidence: float | None = None
    note: str = ""

    def to_payload(self) -> dict:
        return {"category": self.category, "label": self.label, "relation": self.relation, "bias": self.bias,
                "answer": self.answer, "confidence": self.confidence, "note": self.note}


@dataclass(frozen=True)
class Handshake:
    asked_by: str
    bias: str
    answers: tuple[DeskAnswer, ...]
    factor: float
    notes: tuple[str, ...]

    def to_payload(self) -> dict:
        return {"asked_by": self.asked_by, "factor": round(self.factor, 3),
                "answers": [a.to_payload() for a in self.answers], "notes": list(self.notes)}


def consult(store, category: str, bias: str, now_ms: int) -> Handshake:
    """Ask the other three desks for their current anchored verdict and weigh what they say."""
    me = ORDER.index(category)
    mine = lean(bias)
    answers: list[DeskAnswer] = []
    for other in ORDER:
        if other == category:
            continue
        label = PROFILES[other].label
        relation = "higher" if ORDER.index(other) > me else "lower"
        try:
            anchor = store.get_live_anchor(category=other)
        except Exception as e:  # noqa: BLE001 - a desk we cannot read is silent, not absent
            log.warning("handshake: %s unreadable (%s)", other, e)
            answers.append(DeskAnswer(other, label, relation, "", "silent", note=f"ledger unreadable: {e}"))
            continue
        if not anchor:
            answers.append(DeskAnswer(other, label, relation, "", "silent", note="no anchored verdict yet"))
            continue
        theirs = str(anchor.get("bias") or "")
        if anchor.get("window_close_ms") and anchor["window_close_ms"] <= now_ms:
            answers.append(DeskAnswer(other, label, relation, theirs, "silent", anchor.get("confidence"),
                                      "its candle has closed, so this verdict is not current"))
            continue
        their_lean = lean(theirs)
        answer = ("neutral" if mine is None or their_lean is None else
                  "agrees" if their_lean == mine else "opposes")
        answers.append(DeskAnswer(other, label, relation, theirs, answer, anchor.get("confidence")))

    higher = [a for a in answers if a.relation == "higher"]
    factor = 1.0
    if mine is not None:
        factor += HIGHER_AGREES * sum(a.answer == "agrees" for a in higher)
        factor += HIGHER_OPPOSES * sum(a.answer == "opposes" for a in higher)
        factor += LOWER_OPPOSES * sum(a.answer == "opposes" for a in answers if a.relation == "lower")
    factor = max(FACTOR_FLOOR, min(FACTOR_CEIL, factor))
    return Handshake(category, bias, tuple(answers), factor, tuple(_notes(answers, mine, factor)))


def _names(rows) -> str:
    return ", ".join(f"{a.label} ({a.bias})" for a in rows)


def _notes(answers: list[DeskAnswer], mine: str | None, factor: float) -> list[str]:
    if mine is None:
        return ["No direction called, so the other desks' verdicts do not change conviction."]
    out = []
    higher = [a for a in answers if a.relation == "higher"]
    h_with = [a for a in higher if a.answer == "agrees"]
    h_against = [a for a in higher if a.answer == "opposes"]
    l_against = [a for a in answers if a.relation == "lower" and a.answer == "opposes"]
    if h_against:
        out.append(f"Against the higher desk{'s' if len(h_against) > 1 else ''}: {_names(h_against)}. The higher "
                   "timeframe sets the vector.")
    if higher and len(h_against) == len(higher):
        out.append("Counter-trend: every higher desk with a current verdict reads the other way.")
    if h_with:
        out.append(f"With the higher desk{'s' if len(h_with) > 1 else ''}: {_names(h_with)}.")
    if l_against:
        out.append(f"The lower desk{'s' if len(l_against) > 1 else ''} {_names(l_against)} "
                   f"{'read' if len(l_against) > 1 else 'reads'} the other way: an early warning, not a veto.")
    silent = [a.label for a in answers if a.answer == "silent"]
    if silent:
        out.append(f"No current verdict from {', '.join(silent)}.")
    if abs(factor - 1.0) > 1e-9:
        out.append(f"Cross-desk handshake {'raised' if factor > 1 else 'cut'} conviction by "
                   f"{abs(factor - 1.0):.0%}.")
    return out


def apply(res, hs: Handshake):
    """The war-room result with its conviction finalised by the handshake, and the exchange attached."""
    return replace(res, confidence=min(1.0, res.confidence * hs.factor),
                   confidence_before_handshake=res.confidence,
                   notes=tuple(res.notes) + hs.notes, handshake=hs.to_payload())


def notify(store, category: str, bias: str, now_ms: int, price: float | None = None) -> list[str]:
    """The other half of the handshake: tell each desk whose live verdict this one opposes. Append-only,
    one note per desk, direction and window. Returns the desks that were told."""
    mine = lean(bias)
    if mine is None:
        return []
    told = []
    label = PROFILES[category].label
    for other in ORDER:
        if other == category:
            continue
        try:
            anchor = store.get_live_anchor(category=other)
        except Exception as e:  # noqa: BLE001
            log.warning("handshake notify: %s unreadable (%s)", other, e)
            continue
        if not anchor or (anchor.get("window_close_ms") and anchor["window_close_ms"] <= now_ms):
            continue
        theirs = lean(str(anchor.get("bias") or ""))
        if theirs is None or theirs == mine:
            continue
        higher = ORDER.index(category) > ORDER.index(other)
        wrote = store.add_side_note(
            anchor["window_open_ms"], now_ms, HANDSHAKE, f"handshake:{category}:{mine}",
            f"The {label} desk now reads {bias}, against this desk's {anchor.get('bias')}.",
            ("A higher timeframe sets the vector: weigh this desk's call against it." if higher else
             "A lower timeframe turning first is an early warning for this candle, not a reversal by itself."),
            price, category=other)
        if wrote:
            told.append(other)
    return told
