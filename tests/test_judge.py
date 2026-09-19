import asyncio
import json

import httpx

from core.agents.judge import Judge, parse_gemini_probabilities, parse_typesafe
from core.agents.llm import LLMResult
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH, team_score
from core.agents.settings import LLMSettings

SETTINGS = LLMSettings(provider="gemini", api_key="k", specialist_model="spec", director_model="dir")
STATE = {"price": 78000, "categories": {"live": {"direction": "BEARISH"}}}


def test_parsers_read_probabilities_and_clamp():
    ts = {"nouls": {"smc_bos": {"noul": 0.82}, "smc_ob": {"noul": 1.4}, "bad": {"noul": "x"}}}
    assert parse_typesafe(ts) == {"smc_bos": 0.82, "smc_ob": 1.0}
    assert parse_gemini_probabilities({"probabilities": {"a": 0.5, "b": -3, "c": None}}) == {"a": 0.5, "b": 0.0}
    assert parse_gemini_probabilities({"a": {"probability": 0.25}}) == {"a": 0.25}


def test_typesafe_backend_sends_one_batched_request_per_side():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("Authorization")
        body = json.loads(request.content)
        seen["body"] = body
        return httpx.Response(200, json={"nouls": {qid: {"noul": 0.75} for qid in body["questions"]},
                                         "usage": {"input_tokens": 900, "output_tokens": 60}})

    j = Judge(SETTINGS, typesafe_key="ts-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    res = asyncio.run(j.score(STATE, BULLISH))
    assert res.ok and res.provider == "typesafe" and len(res.probabilities) == len(ALL_ITEMS)
    assert abs(team_score(res.probabilities) - 75.0) < 1e-9
    assert seen["auth"] == "Bearer ts-key" and seen["body"]["model"] == "jev-latest"
    assert seen["body"]["state"] == STATE
    assert all(q["type"] == "noul" for q in seen["body"]["questions"].values())
    assert "bullish" in seen["body"]["questions"]["smc_bos"]["instructions"]
    assert res.prompt_tokens == 900 and res.completion_tokens == 60


def test_side_changes_the_question_wording():
    captured = []

    def handler(request: httpx.Request):
        captured.append(json.loads(request.content)["questions"]["mtf_triple"]["instructions"])
        return httpx.Response(200, json={"nouls": {}})

    j = Judge(SETTINGS, typesafe_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(j.score(STATE, BULLISH))
    asyncio.run(j.score(STATE, BEARISH))
    assert "point UP" in captured[0] and "point DOWN" in captured[1]


def test_gemini_fallback_used_when_there_is_no_typesafe_key():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        async def complete_json(self, system, user, model, max_tokens=2048, temperature=0.2):
            self.calls.append((system, user, temperature))
            data = {"probabilities": {i.id: 0.5 for i in ALL_ITEMS}}
            return LLMResult(json.dumps(data), data, 120, 40, None, 0.1, model, "gemini")

    llm = FakeLLM()
    j = Judge(SETTINGS, typesafe_key=None, llm=llm)
    res = asyncio.run(j.score(STATE, BEARISH))
    assert res.ok and res.provider == "gemini" and abs(team_score(res.probabilities) - 50.0) < 1e-9
    system, user, temp = llm.calls[0]
    assert "probability from 0 to 1" in system and temp == 0.0
    assert "smc_bos" in user and "bearish" in system


def test_failures_return_an_empty_result_instead_of_raising():
    def boom(request: httpx.Request):
        return httpx.Response(500, json={"error": "upstream"})

    j = Judge(SETTINGS, typesafe_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    res = asyncio.run(j.score(STATE, BULLISH))
    assert not res.ok and res.error and res.probabilities == {}
    assert not asyncio.run(Judge(SETTINGS, typesafe_key=None, llm=None).score(STATE, BULLISH)).ok


# The exact request/response shapes from https://docs.typesafe.ai/api.md, so a mock cannot drift
# from the live contract without this failing.
DOCUMENTED_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "department": {"type": "choice", "choice": "technical",
                       "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                       "confidence": 0.82},
    },
    "usage": {"input_tokens": 312, "output_tokens": 48},
}


def test_parsers_read_the_documented_answers_envelope():
    from core.agents.judge import parse_typesafe_choice

    assert parse_typesafe(DOCUMENTED_RESPONSE) == {"is_urgent": 0.92}, "nouls live under answers"
    c = parse_typesafe_choice(DOCUMENTED_RESPONSE, "department")
    assert c.choice == "technical" and c.confidence == 0.82
    assert c.probabilities == {"billing": 0.08, "technical": 0.85, "sales": 0.07}
    assert parse_typesafe_choice(DOCUMENTED_RESPONSE, "absent").choice is None


def test_questions_are_built_the_way_the_api_documents_them():
    seen = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.update(body)
        qid = next(iter(body["questions"]))
        return httpx.Response(200, json={"model": "jev-latest", "answers": {
            q: ({"type": "choice", "choice": "BULL_TRAP", "probabilities": {"BULL_TRAP": 0.7}, "confidence": 0.6}
                if body["questions"][q]["type"] == "choice" else {"type": "noul", "noul": 0.6})
            for q in body["questions"]}, "usage": {"input_tokens": 10, "output_tokens": 2}})

    j = Judge(SETTINGS, typesafe_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    res = asyncio.run(j.score(STATE, BULLISH))
    q = seen["questions"]["smc_bos"]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"} and q["instructions"]
    assert res.ok and res.probabilities["smc_bos"] == 0.6

    c = asyncio.run(j.classify(STATE, "trap", "Is this a trap?", {"BULL_TRAP": "up is fake", "NO_TRAP": "no trap"}))
    assert c.ok and c.choice == "BULL_TRAP" and c.confidence == 0.6 and c.provider == "typesafe"
    assert seen["questions"]["trap"]["criteria"] == {"BULL_TRAP": "up is fake", "NO_TRAP": "no trap"}
