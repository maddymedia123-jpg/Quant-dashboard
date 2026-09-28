"""The TRAP Head: ten sub-agents, the handshake with the category desk, and traps that stay anchored.

From the spec: five bullish and five bearish trap sub-agents across the same five domains; the Head asks
the affected category desk whether the move is an authentic higher-timeframe break before declaring
anything; a declared trap stays anchored to that category until its invalidation is touched or it plays
out. The bearish sub-agents are the ones who hunt bull traps, and vice versa."""
import asyncio
import json

import httpx
import pytest

from core.agents import trap_desk as td
from core.agents.judge import Judge, JudgeResult
from core.agents.recon_profiles import PROFILES
from core.agents.rubric import BEARISH, BULLISH
from core.agents.settings import LLMSettings
from core.store import Store
from core.traps import BEAR_TRAP, BULL_TRAP, TrapCandidate

SETTINGS = LLMSettings(provider="gemini", api_key="k", specialist_model="spec", director_model="dir")
LIVE = PROFILES["live"]


def candidate(side=BULL_TRAP, tf="1h", level=80_800.0, strength=3) -> TrapCandidate:
    up = side == BULL_TRAP
    return TrapCandidate(side=side, timeframe=tf, level=level,
                         invalidation=level + (200 if up else -200),
                         plays_out=level - (600 if up else -600), strength=strength,
                         evidence=("liquidity swept and reclaimed", "delta shows absorption"))


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def judge_with(handler) -> Judge:
    return Judge(SETTINGS, typesafe_key="ts", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def responder(noul=0.8, call="ENGINEERED_TRAP", conf=0.7, seen=None):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        qs = body["questions"]
        if any(q.get("type") == "choice" for q in qs.values()):
            qid = next(iter(qs))
            return httpx.Response(200, json={"answers": {qid: {
                "type": "choice", "choice": call, "confidence": conf,
                "probabilities": {call: conf, "AUTHENTIC_BREAK": round(1 - conf, 3)}}}})
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": noul} for q in qs}})
    return handler


# ---------- the ten sub-agents ----------
def test_the_bearish_sub_agents_are_the_ones_who_hunt_a_bull_trap():
    seen = []
    verdict = asyncio.run(td.judge_candidate(judge_with(responder(seen=seen)), {"spot": {"price": 80_900.0}},
                                             candidate(BULL_TRAP), LIVE))
    noul_body = next(b for b in seen if all(q["type"] == "noul" for q in b["questions"].values()))
    assert set(noul_body["questions"]) == {f"trap_{k}" for k in ("quant", "auction", "delta", "ict", "derivs")}
    assert verdict.side_scored == BEARISH, "a fake break up is the bearish desk's case"
    assert "bull trap" in noul_body["questions"]["trap_delta"]["instructions"].lower()

    bear = asyncio.run(td.judge_candidate(judge_with(responder()), {"spot": {"price": 79_100.0}},
                                          candidate(BEAR_TRAP), LIVE))
    assert bear.side_scored == BULLISH


def test_each_sub_agent_is_asked_about_its_own_domain():
    """Every question interpolates the trap's label, so finding "bull trap" in one of them proves only
    that the label was filled in. The five have to be distinguishable from one another."""
    seen = []
    asyncio.run(td.judge_candidate(judge_with(responder(seen=seen)), {"spot": {"price": 80_900.0}},
                                   candidate(BULL_TRAP), LIVE))
    body = next(b for b in seen if all(q["type"] == "noul" for q in b["questions"].values()))
    texts = {qid: q["instructions"] for qid, q in body["questions"].items()}
    assert len(set(texts.values())) == 5, "five domains, five different questions"
    for qid, word in (("trap_auction", "value area"), ("trap_delta", "delta"), ("trap_ict", "liquidity"),
                      ("trap_derivs", "funding"), ("trap_quant", "volatility")):
        assert word in texts[qid].lower(), f"{qid} is not asking about {word}"


def test_the_candidate_itself_is_handed_to_the_model():
    """The questions name the level, but the judge also gets the candidate in the state it reads; without
    it the model is scoring a trap whose shape it cannot see."""
    captured = {}

    async def fake_score(state, side, items=None, profile=None):
        captured["state"] = state
        raise RuntimeError("stop here")

    j = judge_with(responder())
    j.score = fake_score
    with pytest.raises(RuntimeError):
        asyncio.run(td.judge_candidate(j, {"spot": {"price": 80_900.0}}, candidate(BULL_TRAP), LIVE))
    payload = captured["state"]["trap_candidate"]
    assert payload["type"] == "bull trap" and payload["level"] == 80_800.0
    assert payload["timeframe"] == "1h" and payload["invalidation"] == 81_000.0
    assert payload["evidence"], "including the evidence that produced it"


def test_a_probability_outside_the_range_cannot_inflate_the_score():
    """"Out of 100" is guaranteed twice: the judge clamps what a provider returns, and the desk clamps
    again before turning it into points. Both layers are checked, because the second is what holds if a
    score ever reaches it from anywhere other than the judge's parser."""
    high = asyncio.run(td.judge_candidate(judge_with(responder(noul=3.5)), {"spot": {"price": 80_900.0}},
                                          candidate(), LIVE))
    assert high.score == pytest.approx(100.0) and all(d.points <= 20 for d in high.domains)
    low = asyncio.run(td.judge_candidate(judge_with(responder(noul=-2.0)), {"spot": {"price": 80_900.0}},
                                         candidate(), LIVE))
    assert low.score == pytest.approx(0.0) and not low.declared

    # straight past the parser, with values no provider should ever send
    async def unclamped(state, side, items=None, profile=None):
        return JudgeResult(probabilities={f"trap_{k}": v for k, v in
                                         (("quant", 4.0), ("auction", -1.0), ("delta", 0.5),
                                          ("ict", 12.0), ("derivs", 0.25))}, provider="test")

    j = judge_with(responder())
    j.score = unclamped
    v = asyncio.run(td.judge_candidate(j, {"spot": {"price": 80_900.0}}, candidate(), LIVE))
    assert all(0.0 <= d.points <= 20.0 for d in v.domains), [d.points for d in v.domains]
    assert v.score == pytest.approx(20.0 + 0.0 + 10.0 + 20.0 + 5.0)
    assert v.score <= 100.0


def test_the_declaration_threshold_is_sixty_out_of_a_hundred():
    """Hardcoded, not compared against the constant under test: a threshold moved to 99 would otherwise
    be invisible to the test meant to pin it."""
    assert td.DECLARE_POINTS == 60.0
    under = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.59)),
                                           {"spot": {"price": 80_900.0}}, candidate(), LIVE))
    assert under.score == pytest.approx(59.0) and not under.declared
    over = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.61)),
                                          {"spot": {"price": 80_900.0}}, candidate(), LIVE))
    assert over.score == pytest.approx(61.0) and over.declared


def test_the_five_sub_agents_score_out_of_one_hundred():
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=1.0)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    assert v.score == pytest.approx(100.0)
    assert [d.key for d in v.domains] == ["quant", "auction", "delta", "ict", "derivs"]
    assert all(d.points == 20 for d in v.domains)
    half = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.5)), {"spot": {"price": 80_900.0}},
                                          candidate(), LIVE))
    assert half.score == pytest.approx(50.0)


def test_a_dead_provider_leaves_the_trap_unjudged_rather_than_declared():
    j = judge_with(lambda r: httpx.Response(503))
    v = asyncio.run(td.judge_candidate(j, {"spot": {"price": 80_900.0}}, candidate(), LIVE))
    assert v.status == "unjudged" and v.score == 0.0
    assert not v.declared and "did not answer" in " ".join(v.notes).lower()


# ---------- the handshake ----------
def desk_state(structure=None, position=None, oi=None, price=81_500.0) -> dict:
    """One signal at a time, so the veto rule can be pinned. A fixture that hands over all three at once
    cannot tell "structure and acceptance" from "any one of the three"."""
    return {"spot": {"price": price},
            "smc": {"4h": {"structure": {"available": True, "bias": structure,
                                         "last_event": {"type": "BOS"} if structure else None}}},
            "liquidity": {"1h": {"volume_profile": {"available": True, "price_position": position}}},
            "derivatives": {"futures": {"oi_change_24h_pct": oi}}}


def test_the_desk_can_veto_a_trap_it_reads_as_an_authentic_break():
    """Inbound re-verification: the category desk says this is a real higher-timeframe expansion."""
    authentic = desk_state(structure="bullish", position="above value", oi=8.0)
    view = td.desk_view(authentic, candidate(BULL_TRAP), LIVE)
    assert view.reads == "authentic break"
    assert any("structure is bullish" in r for r in view.reasons)
    assert any("accepting above value" in r for r in view.reasons)

    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), authentic, candidate(), LIVE))
    assert v.score >= 60 and v.head_call == "ENGINEERED_TRAP"
    assert not v.declared and v.status == "watching"
    assert any("desk" in n.lower() for n in v.notes)


def test_the_veto_needs_structure_and_acceptance_together():
    """Half a break is not a break. Either read on its own leaves the trap declarable."""
    only_structure = td.desk_view(desk_state(structure="bullish"), candidate(BULL_TRAP), LIVE)
    assert only_structure.reads != "authentic break" and only_structure.reasons

    only_acceptance = td.desk_view(desk_state(position="above value"), candidate(BULL_TRAP), LIVE)
    assert only_acceptance.reads != "authentic break" and only_acceptance.reasons

    both = td.desk_view(desk_state(structure="bullish", position="above value"), candidate(BULL_TRAP), LIVE)
    assert both.reads == "authentic break"


def test_open_interest_alone_never_vetoes_a_trap():
    """A crowded move is exactly what a trap looks like, so open interest is context and not a veto -
    not even when it is the only thing the desk can see."""
    view = td.desk_view(desk_state(oi=40.0), candidate(BULL_TRAP), LIVE)
    assert view.reads == "engineered or unclear"
    assert any("open interest" in r for r in view.reasons)

    with_oi = td.desk_view(desk_state(structure="bullish", oi=40.0), candidate(BULL_TRAP), LIVE)
    assert with_oi.reads != "authentic break", "open interest cannot complete a veto either"


def test_open_interest_reads_the_same_for_both_sides():
    """Open-interest change is unsigned with respect to direction: a genuine breakdown builds it just as
    a genuine breakout does, so the bear mirror must not flip the comparison."""
    for side in (BULL_TRAP, BEAR_TRAP):
        building = td.desk_view(desk_state(oi=9.0), candidate(side), LIVE)
        assert any("open interest" in r for r in building.reasons), side
        assert td.desk_view(desk_state(oi=-9.0), candidate(side), LIVE).reasons == (), side


def test_the_desk_reads_the_candidate_direction():
    """A bull trap is vetoed by bullish structure, a bear trap by bearish. A desk that ignored the
    candidate's side would read both the same way."""
    bullish = desk_state(structure="bullish", position="above value")
    assert td.desk_view(bullish, candidate(BULL_TRAP), LIVE).reads == "authentic break"
    assert td.desk_view(bullish, candidate(BEAR_TRAP), LIVE).reads == "quiet"

    bearish = desk_state(structure="bearish", position="below value")
    assert td.desk_view(bearish, candidate(BEAR_TRAP), LIVE).reads == "authentic break"
    assert td.desk_view(bearish, candidate(BULL_TRAP), LIVE).reads == "quiet"


def test_the_desk_reads_its_own_higher_timeframe():
    """The veto is a higher-timeframe question: structure on the desk's HTF, acceptance on its MTF."""
    wrong_tf = {"spot": {"price": 81_500.0},
                "smc": {"15m": {"structure": {"available": True, "bias": "bullish", "last_event": None}}},
                "liquidity": {"15m": {"volume_profile": {"price_position": "above value"}}},
                "derivatives": {}}
    assert td.desk_view(wrong_tf, candidate(BULL_TRAP), LIVE).reads == "quiet"


def test_a_desk_with_no_state_at_all_is_quiet_rather_than_raising():
    for empty in (None, {}, {"smc": None, "liquidity": None, "derivatives": None}):
        assert td.desk_view(empty, candidate(BULL_TRAP), LIVE).reads == "quiet"
    assert td.desk_view(desk_state(oi="not a number"), candidate(BULL_TRAP), LIVE).reads == "quiet"


def test_a_quiet_desk_does_not_block_a_declaration():
    quiet = {"spot": {"price": 80_900.0}, "smc": {"4h": {"structure": {"bias": "bearish"}}},
             "liquidity": {"1h": {"volume_profile": {"price_position": "inside value"}}},
             "derivatives": {"futures": {"oi_change_24h_pct": -1.0}}}
    assert td.desk_view(quiet, candidate(BULL_TRAP), LIVE).reads != "authentic break"
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), quiet, candidate(), LIVE))
    assert v.declared and v.status == "active"


def test_a_weak_score_stays_a_watch_even_when_the_head_says_trap():
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.3)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    assert v.score == pytest.approx(30.0) and not v.declared and v.status == "watching"
    assert any("under the" in n for n in v.notes), "and it says why it is only a watch"


def test_the_head_calling_it_authentic_stops_the_declaration():
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9, call="AUTHENTIC_BREAK")),
                                       {"spot": {"price": 80_900.0}}, candidate(), LIVE))
    assert v.head_call == "AUTHENTIC_BREAK" and not v.declared


# ---------- anchored until invalidated ----------
def test_a_declared_trap_persists_and_is_not_duplicated(store):
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    first = td.record(store, v, "live", 1_000)
    again = td.record(store, v, "live", 2_000)
    assert first is not None and again == first, "the same trap is one row, not two"
    rows = store.traps(category="live")
    assert len(rows) == 1 and rows[0]["status"] == "active" and rows[0]["side"] == BULL_TRAP
    assert json.loads(rows[0]["evidence"])[0] == "liquidity swept and reclaimed"


def test_a_trap_stays_anchored_until_price_settles_it(store):
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    td.record(store, v, "live", 1_000)

    assert td.settle(store, price=80_950.0, now_ms=2_000) == []          # still in the raid zone
    assert store.traps(category="live")[0]["status"] == "active"

    done = td.settle(store, price=80_100.0, now_ms=3_000)                 # back inside value: it worked
    assert [(d["side"], d["status"]) for d in done] == [(BULL_TRAP, "played_out")]
    assert store.traps(category="live")[0]["resolved_ms"] == 3_000


def test_a_trap_whose_break_turned_out_real_is_marked_invalidated(store):
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    td.record(store, v, "live", 1_000)
    done = td.settle(store, price=81_200.0, now_ms=4_000)                 # beyond invalidation
    assert [d["status"] for d in done] == ["invalidated"]
    assert td.settle(store, price=81_300.0, now_ms=5_000) == [], "settled once, not every refresh"


def test_watched_candidates_are_not_stored_as_traps(store):
    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.3)), {"spot": {"price": 80_900.0}},
                                       candidate(), LIVE))
    assert td.record(store, v, "live", 1_000) is None
    assert store.traps(category="live") == []
