import asyncio
import json

import httpx
import pytest

from core.agents import live_recon as lr
from core.agents.judge import ChoiceResult, Judge, JudgeResult
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH, DOMAINS
from core.agents.settings import LLMSettings
from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.indicators.category import analyze_category

SETTINGS = LLMSettings(provider="gemini", api_key="k", specialist_model="spec", director_model="dir")


@pytest.fixture(scope="module")
def market_and_analyses():
    m = fixture_market()
    analyses = {k: a for k, c in CATEGORIES.items() if (a := analyze_category(c, m.spot.frames, m.futures)) is not None}
    return m, analyses


def team(side, probs, **kw):
    return lr.score_team(side, JudgeResult(probabilities=probs, provider="typesafe", **kw))


def flat(side, p):
    return team(side, {i.id: p for i in ALL_ITEMS})


# ---- state ----
def test_state_carries_the_three_live_recon_timeframes_and_is_json_safe(market_and_analyses):
    m, analyses = market_and_analyses
    state = lr.build_state(m, analyses)
    json.dumps(state)
    assert set(state["categories"]) == {"live", "intraday", "weekly"}
    assert state["timeframes_read"] == {"live": "15m", "intraday": "1h", "weekly": "4h"}
    assert state["spot"]["price"] and "macro_calendar" in state


# ---- scoring ----
def test_team_score_breaks_down_into_the_five_domain_agents():
    t = flat(BULLISH, 1.0)
    assert t.points == 100.0 and t.verdict == "high confluence"
    assert [d.agent_no for d in t.domains] == [1, 2, 3, 4, 5]
    assert all(d.points == d.max_points == 20 for d in t.domains)
    assert sum(w for d in t.domains for _, w, _ in d.items) == 100
    half = flat(BEARISH, 0.5)
    assert half.points == 50.0 and half.verdict == "moderate" and half.domains[0].pct == 50.0


def test_a_failed_judgment_scores_zero_and_says_so_rather_than_reading_zero():
    t = team(BULLISH, {}, error="502 upstream")
    assert t.points == 0.0 and not t.ok and t.missing == ()
    _, _, _, _, _, notes = lr.resolve(t, flat(BEARISH, 0.5), ChoiceResult(choice="NO_TRAP"))
    assert any("failure, not a reading" in n for n in notes)


def test_missing_checklist_answers_are_named_in_the_notes():
    probs = {i.id: 0.5 for i in ALL_ITEMS}
    probs.pop("macro_release")
    t = team(BULLISH, probs)
    assert t.missing == ("macro_release",)
    _, _, _, _, _, notes = lr.resolve(t, flat(BEARISH, 0.5), ChoiceResult(choice="NO_TRAP"))
    assert any("missing 1 checklist answers" in n and "Macro" in n for n in notes)


# ---- bias and consult policy ----
def test_bias_needs_a_ten_point_margin():
    assert lr.bias_for(75.0, 40.0) == ("BULL", 35.0)
    assert lr.bias_for(40.0, 75.0) == ("BEAR", -35.0)
    assert lr.bias_for(62.0, 55.0)[0] == "BALANCED"
    assert lr.bias_for(50.0, 40.0)[0] == "BULL"   # exactly ten points is decisive


def test_head_consults_the_trap_agent_on_conflict_or_indecision():
    assert lr.needs_consult(80.0, 70.0) is True      # both teams strong
    assert lr.needs_consult(55.0, 52.0) is True      # no decisive margin
    assert lr.needs_consult(75.0, 30.0) is False     # one-sided and decisive


def test_trap_probability_overrides_a_direction():
    bull, bear = flat(BULLISH, 0.8), flat(BEARISH, 0.3)
    trap = ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.72, "GENUINE_MOVE": 0.2})
    bias, margin, conf, consulted, override, notes = lr.resolve(bull, bear, trap)
    assert bias == "BULL TRAP RISK" and "72%" in override
    assert margin == pytest.approx(50.0) and conf > 0
    # below the threshold the direction stands
    weak = ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.45})
    assert lr.resolve(bull, bear, weak)[0] == "BULL"
    # a bull trap does not touch a bearish read
    assert lr.resolve(flat(BULLISH, 0.3), flat(BEARISH, 0.8), trap)[0] == "BEAR"


def test_balanced_scores_give_no_direction_and_zero_confidence():
    bias, margin, conf, consulted, override, notes = lr.resolve(
        flat(BULLISH, 0.65), flat(BEARISH, 0.65), ChoiceResult(choice="BULL_TRAP", probabilities={"BULL_TRAP": 0.8}))
    assert bias == "BALANCED" and conf == 0.0 and override is None and consulted
    assert any("Both teams score above 60" in n for n in notes)
    assert any("under the 10-point threshold" in n for n in notes)
    assert any("bull trap" in n for n in notes)


def test_a_silent_trap_agent_is_reported_and_changes_nothing():
    bias, _, _, _, override, notes = lr.resolve(flat(BULLISH, 0.9), flat(BEARISH, 0.2),
                                                ChoiceResult(error="timeout"))
    assert bias == "BULL" and override is None
    assert any("trap agent did not answer" in n for n in notes)


# ---- end to end, no network ----
def test_run_live_recon_makes_three_calls_on_one_shared_state(market_and_analyses):
    m, analyses = market_and_analyses
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        if "trap" in body["questions"]:
            return httpx.Response(200, json={"model": "jev-latest", "answers": {"trap": {
                "type": "choice", "choice": "BULL_TRAP", "confidence": 0.78, "probabilities": {
                    "BULL_TRAP": 0.7, "BEAR_TRAP": 0.1, "GENUINE_MOVE": 0.15, "NO_TRAP": 0.05}}}})
        side = body["questions"]["smc_bos"]["instructions"]
        p = 0.9 if "bullish" in side else 0.2
        return httpx.Response(200, json={"model": "jev-latest",
                                         "answers": {q: {"type": "noul", "noul": p} for q in body["questions"]},
                                         "usage": {"input_tokens": 1000, "output_tokens": 50}})

    j = Judge(SETTINGS, typesafe_key="ts", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    res = asyncio.run(lr.run_live_recon(j, m, analyses))

    assert len(seen) == 3
    states = [json.dumps(b["state"], sort_keys=True) for b in seen]
    assert len(set(states)) == 1, "all eleven agents must judge the same state"
    assert res.bull.points == pytest.approx(90.0) and res.bear.points == pytest.approx(20.0)
    assert res.bias == "BULL TRAP RISK" and res.consulted is False and res.override
    assert res.trap.choice == "BULL_TRAP" and res.trap.confidence == 0.78 and res.leader is res.bull
    assert res.prompt_tokens == 2000 and res.completion_tokens == 100
    assert not res.degraded and res.generated_at.tzinfo is not None


def test_run_live_recon_survives_a_dead_provider(market_and_analyses):
    m, analyses = market_and_analyses
    j = Judge(SETTINGS, typesafe_key="ts",
              client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    res = asyncio.run(lr.run_live_recon(j, m, analyses))
    assert res.bias == "BALANCED" and res.confidence == 0.0 and res.degraded
    assert res.bull.error and res.bear.error and not res.trap.ok
    assert sum("failure, not a reading" in n for n in res.notes) == 2


def test_the_rubric_covers_every_domain_exactly_once():
    assert len(DOMAINS) == 5 and sum(d.max_points for d in DOMAINS) == 100
