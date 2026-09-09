import asyncio

import httpx
import pytest

from core.agents.llm import LLMClient, extract_json, parse_gemini, parse_openrouter
from core.agents.settings import LLMSettings, load_settings


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure. {"a": {"b": [1,2]}} trailing') == {"a": {"b": [1, 2]}}
    assert extract_json('{"s": "brace } inside string"}') == {"s": "brace } inside string"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_gemini_and_openrouter_shapes():
    g = {"candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
    assert parse_gemini(g) == ('{"ok": true}', 10, 5)
    o = {"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001}}
    assert parse_openrouter(o) == ('{"ok": true}', 7, 3, 0.0001)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = load_settings()
    assert s.provider == "gemini" and s.specialist_model.startswith("gemini") and s.director_model.startswith("gemini")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    assert load_settings().provider == "openrouter"


def test_client_retries_then_repairs_json():
    calls = {"n": 0}

    def handler(request: httpx.Request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        if calls["n"] == 2:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{\"fixed\": true}"}]}}], "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2}})

    s = LLMSettings(provider="gemini", api_key="k", specialist_model="m", director_model="m")
    c = LLMClient(s, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), backoff=(0, 0))
    r = asyncio.run(c.complete_json("sys", "user", "m"))
    assert r.data == {"fixed": True} and r.error is None and calls["n"] == 3
    assert r.prompt_tokens == 3 and r.completion_tokens == 3


def test_client_returns_error_result_on_hard_failure():
    def handler(request: httpx.Request):
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    s = LLMSettings(provider="openrouter", api_key="k", specialist_model="m", director_model="m")
    c = LLMClient(s, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), backoff=(0, 0))
    r = asyncio.run(c.complete_json("sys", "user", "m"))
    assert r.data is None and r.error and "400" in r.error
