"""Overloaded providers: a busy model hands the call to the next one, and a down TypeSafe to the LLM judge."""
import asyncio
import json

import httpx

from core.agents.judge import TYPESAFE_URL, Judge
from core.agents.llm import LLMClient
from core.agents.rubric import BULLISH
from core.agents.settings import LLMSettings

OK = {"candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}],
      "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
BUSY = {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}


def _settings():
    return LLMSettings(provider="gemini", api_key="k", specialist_model="lite", director_model="flash")


def _client(handler):
    return LLMClient(_settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), backoff=(0, 0))


def _model(request):
    return request.url.path.split("/models/")[1].split(":")[0]


def test_an_overloaded_model_hands_the_call_to_the_next():
    seen = []

    def handler(request):
        seen.append(_model(request))
        return httpx.Response(503, json=BUSY) if _model(request) == "flash" else httpx.Response(200, json=OK)

    r = asyncio.run(_client(handler).complete_json("s", "u", "flash"))
    assert r.data == {"ok": True} and r.model == "lite" and r.error is None
    assert seen == ["flash", "flash", "flash", "lite"], "three tries on the busy model, then the next one"


def test_the_order_is_requested_then_configured_then_spares():
    assert _client(lambda r: None).models_for("lite") == ["lite", "flash", "gemini-3.5-flash"]
    assert _client(lambda r: None).models_for("flash") == ["flash", "lite", "gemini-3.5-flash"]


def test_when_every_model_is_busy_the_error_names_them_all():
    r = asyncio.run(_client(lambda req: httpx.Response(503, json=BUSY)).complete_json("s", "u", "flash"))
    assert r.data is None and "overloaded on every model tried" in r.error
    assert "flash (503)" in r.error and "lite (503)" in r.error and "gemini-3.5-flash (503)" in r.error


def test_a_refused_key_does_not_try_other_models():
    seen = []

    def handler(request):
        seen.append(_model(request))
        return httpx.Response(403, json={"error": {"message": "API key not valid"}})

    r = asyncio.run(_client(handler).complete_json("s", "u", "flash"))
    assert r.data is None and "403" in r.error and seen == ["flash"], "another model would not fix a bad key"


def test_a_timeout_also_moves_on():
    def handler(request):
        if _model(request) == "flash":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=OK)

    assert asyncio.run(_client(handler).complete_json("s", "u", "flash")).model == "lite"


def test_the_json_repair_retry_uses_the_model_that_answered():
    seen = []

    def handler(request):
        m = _model(request)
        seen.append(m)
        if m == "flash":
            return httpx.Response(503, json=BUSY)
        n = seen.count("lite")
        text = "not json" if n == 1 else "{\"fixed\": true}"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    r = asyncio.run(_client(handler).complete_json("s", "u", "flash"))
    assert r.data == {"fixed": True} and seen[-2:] == ["lite", "lite"]


# ---------- the war-room judge ----------
def _judge(ts_status, llm_ok=True):
    def ts_handler(request):
        return httpx.Response(ts_status, json={"detail": "upstream error"})

    def llm_handler(request):
        body = json.loads(request.content)
        prompt = body["contents"][0]["parts"][0]["text"]
        if "OPTIONS" in prompt:
            data = {"choice": "NO_TRAP", "probabilities": {"NO_TRAP": 0.7, "BULL_TRAP": 0.2, "BEAR_TRAP": 0.1}}
        else:
            ids = [line.split('"')[1] for line in prompt.split("CONDITIONS")[1].splitlines() if line.startswith('- "')]
            data = {"probabilities": {i: 0.6 for i in ids}}
        return httpx.Response(200 if llm_ok else 503, json={"candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}}]})

    llm = _client(llm_handler)
    return Judge(_settings(), typesafe_key="ts", client=httpx.AsyncClient(transport=httpx.MockTransport(ts_handler)),
                 llm=llm)


def test_typesafe_down_falls_back_to_the_llm_judge_and_says_so():
    res = asyncio.run(_judge(503).score({"x": 1}, BULLISH))
    assert res.ok and res.provider.startswith("gemini (TypeSafe failed: ") and "503" in res.provider


def test_typesafe_down_falls_back_for_the_trap_call_too():
    res = asyncio.run(_judge(502).classify({"x": 1}, "trap", "Is it a trap?",
                                           {"NO_TRAP": "no", "BULL_TRAP": "bull", "BEAR_TRAP": "bear"}))
    assert res.ok and res.choice == "NO_TRAP" and "TypeSafe failed" in res.provider


def test_with_no_fallback_a_typesafe_failure_is_still_reported():
    j = Judge(_settings(), typesafe_key="ts",
              client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, json={}))))
    res = asyncio.run(j.score({"x": 1}, BULLISH))
    assert not res.ok and "503" in res.error


def test_both_down_is_a_failure_naming_both():
    res = asyncio.run(_judge(503, llm_ok=False).score({"x": 1}, BULLISH))
    assert not res.ok and "overloaded on every model tried" in res.error


def test_the_typesafe_url_is_unchanged():
    assert TYPESAFE_URL == "https://api.typesafe.ai/v1/systemone"


def test_a_spent_daily_quota_moves_on_without_retrying():
    """Gemini's free tier: 20 requests per model per day. Retrying that in seconds is pointless."""
    seen = []
    quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}

    def handler(request):
        seen.append(_model(request))
        return httpx.Response(429, json=quota) if _model(request) == "flash" else httpx.Response(200, json=OK)

    r = asyncio.run(_client(handler).complete_json("s", "u", "flash"))
    assert r.model == "lite" and seen == ["flash", "lite"]


def test_a_per_minute_429_is_still_retried():
    seen = []

    def handler(request):
        seen.append(_model(request))
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED", "details": [
                {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}})
        return httpx.Response(200, json=OK)

    r = asyncio.run(_client(handler).complete_json("s", "u", "flash"))
    assert r.model == "flash" and seen == ["flash", "flash"]
