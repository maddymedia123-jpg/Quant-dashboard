"""The Active Trade desk: ten sub-agents, the Head, and the two handshakes.

From the spec: five bullish and five bearish sub-agents evaluate the position, the Head calls it, and
before anything is presented the four category heads and the TRAP head are consulted. Both consultations
are deterministic here - the desks have already published their verdicts and the traps are already
anchored, so re-asking a model would cost more and could contradict what is on screen.
"""
import asyncio
import json

import httpx
import pytest

from core.agents import trade_desk as td
from core.agents.judge import Judge, JudgeResult
from core.agents.recon_profiles import PROFILES
from core.agents.rubric import BEARISH, BULLISH
from core.agents.settings import LLMSettings
from core.store import Store
from core.trades import LONG, SHORT, StopVerdict, TimedTarget

SETTINGS = LLMSettings(provider="gemini", api_key="k", specialist_model="s", director_model="d")
LIVE = PROFILES["live"]
NOW = 1_760_000_000_000
ENTRY = 80_000.0


def stop(verdict="sound", given=79_000.0, proposed=79_000.0) -> StopVerdict:
    return StopVerdict(LONG, given, proposed, verdict, 79_100.0, "the last swing", ("because",))


def targets() -> list[TimedTarget]:
    return [TimedTarget("TP1", 80_600.0, "the 1 sigma band", NOW + 3_600_000, 1.5, "Fibonacci time window"),
            TimedTarget("TP2", 81_200.0, "the 2 sigma band", NOW + 7_200_000, 3.0, "the second 1h close"),
            TimedTarget("TP3", 81_800.0, "the 3 sigma band", NOW + 10_800_000, 4.5, "the third 1h close")]


def judge_with(handler) -> Judge:
    return Judge(SETTINGS, typesafe_key="ts",
                 client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def responder(for_p=0.8, against_p=0.2, call="TAKE", conf=0.7, seen=None):
    """The supporting team answers first, then the opposing one, then the Head."""
    state = {"n": 0}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        qs = body["questions"]
        if any(q.get("type") == "choice" for q in qs.values()):
            qid = next(iter(qs))
            return httpx.Response(200, json={"answers": {qid: {
                "type": "choice", "choice": call, "confidence": conf, "probabilities": {call: conf}}}})
        state["n"] += 1
        p = for_p if state["n"] == 1 else against_p
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": p} for q in qs}})
    return handler


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def anchor(s: Store, category: str, bias: str, window_close_ms: int = NOW + 3_600_000, confidence=0.7):
    s.put_live_anchor(window_open_ms=NOW - 3_600_000, window_close_ms=window_close_ms, published_ms=NOW,
                      bias=bias, margin=20.0, confidence=confidence, trap=None, price=ENTRY,
                      payload={"bias": bias}, category=category)


# ---------- the ten sub-agents ----------
def test_both_teams_are_asked_about_the_same_position():
    """A trade has a case for it and a case against it; the honest score is the difference, so the same
    five questions go to both sides rather than only to the one the trader hopes is right."""
    seen = []
    v = asyncio.run(td.judge_trade(judge_with(responder(seen=seen)), {"spot": {"price": ENTRY}}, LONG,
                                   ENTRY, stop(), targets(), LIVE, atr=400.0))
    noul_bodies = [b for b in seen if all(q["type"] == "noul" for q in b["questions"].values())]
    assert len(noul_bodies) == 2, "one batched request per side"
    assert all(set(b["questions"]) == {f"trade_{k}" for k in td.DOMAIN_KEYS} for b in noul_bodies)
    assert v.for_points == pytest.approx(80.0) and v.against_points == pytest.approx(20.0)
    assert [d.key for d in v.for_domains] == list(td.DOMAIN_KEYS)


def test_each_sub_agent_is_asked_about_its_own_domain():
    seen = []
    asyncio.run(td.judge_trade(judge_with(responder(seen=seen)), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                               stop(), targets(), LIVE, atr=400.0))
    body = next(b for b in seen if all(q["type"] == "noul" for q in b["questions"].values()))
    texts = {qid: q["instructions"] for qid, q in body["questions"].items()}
    assert len(set(texts.values())) == 5, "five domains, five different questions"
    for qid, word in (("trade_quant", "volatility"), ("trade_auction", "value"), ("trade_delta", "delta"),
                      ("trade_ict", "structure"), ("trade_derivs", "funding")):
        assert word in texts[qid].lower(), f"{qid} is not asking about {word}"
    assert "80,000" in texts["trade_ict"] and "79,000" in texts["trade_ict"], "the position is in the question"


def test_the_position_is_handed_to_the_model():
    captured = {}

    async def fake_score(state, side, items=None, profile=None):
        captured["state"] = state
        raise RuntimeError("stop here")

    j = judge_with(responder())
    j.score = fake_score
    with pytest.raises(RuntimeError):
        asyncio.run(td.judge_trade(j, {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    trade = captured["state"]["active_trade"]
    assert trade["side"] == LONG and trade["entry"] == ENTRY and trade["stop"] == 79_000.0
    assert [t["name"] for t in trade["targets"]] == ["TP1", "TP2", "TP3"]
    assert trade["horizon"] == LIVE.horizon


def test_the_score_is_the_case_for_net_of_the_case_against():
    """50 is a trade with two equally supported sides, which is what it should read as."""
    even = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.5, against_p=0.5)),
                                      {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert even.score == 50.0 and even.margin == 0.0

    good = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.9, against_p=0.1)),
                                      {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert good.score == 90.0 and good.margin == 80.0

    bad = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.1, against_p=0.9)),
                                     {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert bad.score == 10.0 and bad.margin == -80.0

    # asymmetric on purpose: with against = 1 - for, "net of the case against" and "the case for alone"
    # give the same number, so the three cases above cannot tell the two formulas apart
    contested = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.8, against_p=0.8)),
                                           {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(),
                                           LIVE))
    assert contested.for_points == pytest.approx(80.0) and contested.against_points == pytest.approx(80.0)
    assert contested.score == 50.0, "a trade whose opposite is equally supported is not an 80/100 trade"
    assert contested.margin == 0.0


def record_sides(j: Judge, order: list[float]):
    """Capture which team each score() call is for. Both teams get identically worded questions, so the
    request body cannot show this - only the argument can."""
    asked: list[str] = []
    values = list(order)

    async def scoring(state, side, items=None, profile=None):
        asked.append(side)
        return JudgeResult(probabilities={i.id: values[len(asked) - 1] for i in items}, provider="test")

    j.score = scoring
    return asked


def test_the_supporting_team_is_the_bullish_one_for_a_long():
    j = judge_with(responder())
    asked = record_sides(j, [0.9, 0.1])
    v = asyncio.run(td.judge_trade(j, {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert asked == [BULLISH, BEARISH], "the long's case is the bullish one, asked first"
    assert v.for_points == pytest.approx(90.0) and v.against_points == pytest.approx(10.0)


def test_the_supporting_team_is_the_bearish_one_for_a_short():
    j = judge_with(responder())
    asked = record_sides(j, [0.9, 0.1])
    v = asyncio.run(td.judge_trade(j, {"spot": {"price": ENTRY}}, SHORT, ENTRY,
                                   StopVerdict(SHORT, 81_000.0, 81_000.0, "sound", None, "", ()),
                                   targets(), LIVE))
    assert asked == [BEARISH, BULLISH], "a short's case is the bearish one"
    assert v.for_points == pytest.approx(90.0) and v.score == 90.0


def test_both_cases_scoring_high_is_reported_as_the_trap_condition():
    v = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.9, against_p=0.9)),
                                   {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert any("equally well" in n for n in v.notes)


def test_a_weak_case_says_so_against_a_fixed_threshold():
    v = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.5, against_p=0.4)),
                                   {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert td.TAKE_POINTS == 60.0
    assert v.for_points == pytest.approx(50.0)
    assert any("under the 60" in n for n in v.notes)


def test_a_dead_provider_leaves_the_card_unscored_rather_than_confident():
    v = asyncio.run(td.judge_trade(judge_with(lambda r: httpx.Response(503)), {"spot": {"price": ENTRY}},
                                   LONG, ENTRY, stop(), targets(), LIVE))
    assert not v.ok and v.for_points == 0.0 and v.head_call is None
    assert any("did not answer" in n for n in v.notes)


def test_a_probability_outside_the_range_cannot_inflate_the_score():
    async def unclamped(state, side, items=None, profile=None):
        return JudgeResult(probabilities={f"trade_{k}": 6.0 for k in td.DOMAIN_KEYS}, provider="test")

    j = judge_with(responder())
    j.score = unclamped
    v = asyncio.run(td.judge_trade(j, {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert all(d.points <= 20.0 for d in v.for_domains)
    assert 0.0 <= v.score <= 100.0


# ---------- the head ----------
def test_the_head_makes_one_categorical_call_on_the_trade():
    for call in ("TAKE", "WAIT", "STAND_ASIDE"):
        v = asyncio.run(td.judge_trade(judge_with(responder(call=call)), {"spot": {"price": ENTRY}},
                                       LONG, ENTRY, stop(), targets(), LIVE))
        assert v.head_call == call and v.head_confidence == pytest.approx(0.7)
    assert set(td.TRADE_CHOICES) == {"TAKE", "WAIT", "STAND_ASIDE"}


def test_a_silent_head_is_reported_and_does_not_block_the_card():
    def handler(request):
        body = json.loads(request.content)
        if any(q.get("type") == "choice" for q in body["questions"].values()):
            return httpx.Response(500)
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": 0.8}
                                                    for q in body["questions"]}})
    v = asyncio.run(td.judge_trade(judge_with(handler), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                                   stop(), targets(), LIVE))
    assert v.ok and v.head_call is None and v.for_points > 0
    assert any("Head did not answer" in n for n in v.notes)


# ---------- the category handshake ----------
def test_the_four_category_heads_are_consulted_from_their_own_verdicts(store):
    anchor(store, "live", "BULL")
    anchor(store, "intraday", "BEAR")
    anchor(store, "weekly", "BALANCED")
    rows = td.category_alignment(store, LONG, NOW)
    by = {a.category: a for a in rows}
    assert len(rows) == 4, "all four desks, whether or not they have spoken"
    assert by["live"].verdict == "supports" and by["intraday"].verdict == "opposes"
    assert by["weekly"].verdict == "neutral"
    assert by["monthly"].verdict == "silent" and "war room" in by["monthly"].note


def test_the_mirror_reads_the_same_verdicts_the_other_way(store):
    anchor(store, "live", "BULL")
    by = {a.category: a for a in td.category_alignment(store, SHORT, NOW)}
    assert by["live"].verdict == "opposes", "a bullish desk opposes a short"


def test_a_desk_whose_candle_has_closed_does_not_vote(store):
    """An anchor from a window that ended is not that desk's current view, and counting it would let an
    old verdict decide a new trade."""
    anchor(store, "live", "BULL", window_close_ms=NOW - 60_000)
    only = {a.category: a for a in td.category_alignment(store, LONG, NOW)}["live"]
    assert only.verdict == "silent" and only.stale
    assert "not current" in only.note


def test_a_trap_risk_verdict_is_not_read_as_support(store):
    """BULL TRAP RISK means the desk expects the long to fail, so it cannot count for a long."""
    anchor(store, "live", "BULL TRAP RISK")
    only = {a.category: a for a in td.category_alignment(store, LONG, NOW)}["live"]
    assert only.verdict == "opposes"


def test_the_alignment_summary_reads_in_plain_english(store):
    anchor(store, "live", "BULL")
    anchor(store, "intraday", "BULL")
    v = asyncio.run(td.judge_trade(judge_with(responder()), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                                   stop(), targets(), LIVE, store=store, now_ms=NOW, atr=400.0))
    assert v.aligned == "with the higher timeframes"
    assert v.probability > v.score, "aligned desks lift the estimate"

    anchor(store, "weekly", "BEAR")
    v2 = asyncio.run(td.judge_trade(judge_with(responder()), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                                    stop(), targets(), LIVE, store=store, now_ms=NOW, atr=400.0))
    assert v2.aligned == "the desks disagree"
    assert any("against this trade" in n for n in v2.notes)


def test_an_unreadable_ledger_leaves_the_desks_silent_rather_than_raising():
    class Broken:
        def get_live_anchor(self, category="live"):
            raise RuntimeError("database is locked")

    rows = td.category_alignment(Broken(), LONG, NOW)
    assert len(rows) == 4 and all(a.verdict == "silent" for a in rows)
    assert all("unreadable" in a.note for a in rows)


# ---------- the trap handshake ----------
def declare_trap(s: Store, level: float, side="bull_trap", timeframe="1h") -> int:
    return s.put_trap(category="live", side=side, timeframe=timeframe, level=level,
                      invalidation=level + 200, plays_out=level - 800, declared_ms=NOW, score=72.0,
                      head_call="ENGINEERED_TRAP", evidence=["swept and reclaimed"], notes=[])


def test_a_trap_sitting_on_a_target_is_reported(store):
    tid = declare_trap(store, 80_650.0)                       # 50 from TP1 at 80,600
    found = td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 400.0)
    assert [(t.trap_id, t.near) for t in found] == [(tid, "TP1")]
    assert found[0].distance == 50.0


def test_a_trap_sitting_on_the_entry_or_the_stop_is_reported(store):
    declare_trap(store, ENTRY + 100)
    declare_trap(store, 79_050.0, timeframe="4h")
    near = {t.near for t in td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 400.0)}
    assert near == {"entry", "stop"}


def test_a_trap_far_from_every_level_is_not_reported(store):
    declare_trap(store, 60_000.0)
    assert td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 400.0) == ()


def test_a_settled_trap_is_not_consulted(store):
    tid = declare_trap(store, 80_650.0)
    store.resolve_trap(tid, "played_out", 80_000.0, NOW + 1)
    assert td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 400.0) == ()


def test_the_proximity_window_scales_with_volatility(store):
    declare_trap(store, 80_800.0)                              # 200 from TP1
    assert td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 400.0), "half of a 400 ATR reaches it"
    assert td.trap_proximity(store, LONG, ENTRY, 79_000.0, targets(), 100.0) == (), "a quiet tape does not"


def test_a_trap_near_the_position_lowers_the_estimate_and_is_noted(store):
    """Same desks, same scores, one difference: a trap anchored on TP1. The estimate has to move down."""
    anchor(store, "live", "BULL")
    without = asyncio.run(td.judge_trade(judge_with(responder()), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                                         stop(), targets(), LIVE, store=store, now_ms=NOW, atr=400.0))
    assert without.traps == ()

    declare_trap(store, 80_650.0)
    with_trap = asyncio.run(td.judge_trade(judge_with(responder()), {"spot": {"price": ENTRY}}, LONG,
                                           ENTRY, stop(), targets(), LIVE, store=store, now_ms=NOW,
                                           atr=400.0))
    assert with_trap.traps and any("anchored bull trap" in n for n in with_trap.notes)
    assert with_trap.score == without.score, "the agents scored the same; only the trap differs"
    assert with_trap.probability < without.probability, "a trap on the position has to lower the estimate"


# ---------- the probability ----------
def test_the_probability_is_never_presented_as_a_certainty():
    """Nothing here measures how often a setup like this has worked, so a number at the extremes would be
    a lie about what is known."""
    best = asyncio.run(td.judge_trade(judge_with(responder(for_p=1.0, against_p=0.0)),
                                      {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    worst = asyncio.run(td.judge_trade(judge_with(responder(for_p=0.0, against_p=1.0)),
                                       {"spot": {"price": ENTRY}}, LONG, ENTRY, stop(), targets(), LIVE))
    assert best.score == 100.0 and best.probability <= 85.0
    assert worst.score == 0.0 and worst.probability >= 15.0


def test_an_unsound_stop_is_flagged_and_the_score_is_for_the_corrected_one():
    v = asyncio.run(td.judge_trade(judge_with(responder()), {"spot": {"price": ENTRY}}, LONG, ENTRY,
                                   stop(verdict="would be hunted", given=79_200.0, proposed=78_600.0),
                                   targets(), LIVE))
    assert any("would be hunted" in n and "corrected stop" in n for n in v.notes)
