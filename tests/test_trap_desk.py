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
from core.agents.judge import Judge
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
def test_the_desk_can_veto_a_trap_it_reads_as_an_authentic_break():
    """Inbound re-verification: the category desk says this is a real higher-timeframe expansion."""
    authentic = {"spot": {"price": 81_500.0},
                 "smc": {"4h": {"structure": {"available": True, "bias": "bullish",
                                              "last_event": {"type": "BOS", "direction": "bullish"}}}},
                 "liquidity": {"1h": {"volume_profile": {"available": True, "price_position": "above value"}}},
                 "derivatives": {"futures": {"oi_change_24h_pct": 8.0}}}
    view = td.desk_view(authentic, candidate(BULL_TRAP), LIVE)
    assert view.reads == "authentic break" and view.reasons

    v = asyncio.run(td.judge_candidate(judge_with(responder(noul=0.9)), authentic, candidate(), LIVE))
    assert v.score >= 60 and v.head_call == "ENGINEERED_TRAP"
    assert not v.declared and v.status == "watching"
    assert any("desk" in n.lower() for n in v.notes)


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
    assert v.score < td.DECLARE_POINTS and not v.declared and v.status == "watching"


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
