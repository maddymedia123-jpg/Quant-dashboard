"""Provider-agnostic JSON-mode chat client: Gemini generateContent or OpenRouter chat completions."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from core.agents.settings import LLMSettings

log = logging.getLogger(__name__)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {429, 500, 502, 503, 504}
# Gemini models are overloaded on and off ("This model is currently experiencing high demand", 503), and
# not all at once: on 16 Oct flash-lite failed while 3.7-flash answered, then the reverse. So a model that
# is still overloaded after its retries hands the call to the next one rather than failing the agent.
GEMINI_SPARES = ("gemini-3.5-flash",)


def _overloaded(e: Exception) -> bool:
    """A busy or unreachable provider, which another model may get past. A refused key does not."""
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in RETRY_STATUS
    return isinstance(e, (httpx.TimeoutException, httpx.TransportError))


@dataclass
class LLMResult:
    text: str
    data: dict | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_s: float
    model: str
    provider: str
    error: str | None = None


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON object in response")


def parse_gemini(payload: dict) -> tuple[str, int, int]:
    cands = payload.get("candidates") or []
    if not cands:
        raise ValueError(f"gemini returned no candidates: {json.dumps(payload)[:300]}")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = payload.get("usageMetadata", {})
    return text, int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))


def parse_openrouter(payload: dict) -> tuple[str, int, int, float | None]:
    if "error" in payload and not payload.get("choices"):
        raise ValueError(f"openrouter error: {payload['error']}")
    text = payload["choices"][0]["message"]["content"] or ""
    u = payload.get("usage", {})
    cost = u.get("cost")
    return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), (float(cost) if cost is not None else None)


class LLMClient:
    def __init__(self, settings: LLMSettings, client: httpx.AsyncClient | None = None, backoff: tuple[float, float] = (3.0, 8.0)):
        self.s = settings
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_s, connect=10.0))
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._backoff = backoff

    async def aclose(self):
        await self._client.aclose()

    def _request(self, system: str, user: str, model: str, max_tokens: int, temperature: float) -> tuple[str, dict, dict]:
        if self.s.provider == "gemini":
            url = f"{GEMINI_BASE}/models/{model}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"},
            }
            return url, body, {"x-goog-api-key": self.s.api_key, "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True},
        }
        return OPENROUTER_URL, body, {
            "Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/maddymedia123-jpg/Quant-dashboard", "X-Title": "Trap Intelligence",
        }

    async def _once(self, system: str, user: str, model: str, max_tokens: int, temperature: float) -> tuple[str, int, int, float | None]:
        url, body, headers = self._request(system, user, model, max_tokens, temperature)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = await self._client.post(url, json=body, headers=headers)
                # a spent DAILY quota (Gemini free tier: 20 requests per model per day) does not come back
                # in seconds, so waiting and retrying only delays handing the call to the next model
                if r.status_code == 429 and "PerDay" in r.text:
                    r.raise_for_status()
                if r.status_code in RETRY_STATUS and attempt < 2:
                    await asyncio.sleep(self._backoff[attempt])
                    continue
                r.raise_for_status()
                payload = r.json()
                if self.s.provider == "gemini":
                    t, pt, ct = parse_gemini(payload)
                    return t, pt, ct, None
                return parse_openrouter(payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(self._backoff[attempt])
                    continue
        raise last_exc or RuntimeError("llm request failed")

    def models_for(self, model: str) -> list[str]:
        """The requested model first, then the provider's other configured models, then the spares."""
        spares = GEMINI_SPARES if self.s.provider == "gemini" else ()
        return list(dict.fromkeys([model, self.s.specialist_model, self.s.director_model, *spares]))

    async def _once_any(self, system: str, user: str, model: str, max_tokens: int,
                        temperature: float) -> tuple[str, int, int, float | None, str]:
        """`_once` on the requested model, moving to the next only when the provider is overloaded."""
        tried: list[str] = []
        for m in self.models_for(model):
            try:
                text, pt, ct, cost = await self._once(system, user, m, max_tokens, temperature)
                if tried:
                    log.warning("llm: %s overloaded, answered by %s", ", ".join(tried), m)
                return text, pt, ct, cost, m
            except Exception as e:  # noqa: BLE001 - decide below whether another model can help
                if not _overloaded(e):
                    raise
                code = getattr(getattr(e, "response", None), "status_code", None) or type(e).__name__
                tried.append(f"{m} ({code})")
        raise RuntimeError(f"{self.s.provider} overloaded on every model tried: {', '.join(tried)}")

    async def complete_json(self, system: str, user: str, model: str, max_tokens: int = 2048, temperature: float = 0.2) -> LLMResult:
        t0 = time.perf_counter()
        pt_sum = ct_sum = 0
        cost_sum: float | None = None
        async with self._sem:
            try:
                text, pt, ct, cost, model = await self._once_any(system, user, model, max_tokens, temperature)
                pt_sum += pt
                ct_sum += ct
                cost_sum = cost
                try:
                    data = extract_json(text)
                except ValueError:
                    text, pt, ct, cost = await self._once(
                        system, user + "\n\nYour previous reply was not valid JSON. Return ONLY valid JSON matching the schema.",
                        model, max_tokens, temperature)
                    pt_sum += pt
                    ct_sum += ct
                    if cost is not None:
                        cost_sum = (cost_sum or 0.0) + cost
                    data = extract_json(text)
                return LLMResult(text, data, pt_sum, ct_sum, cost_sum, time.perf_counter() - t0, model, self.s.provider)
            except Exception as e:  # noqa: BLE001 - agent boundary
                log.warning("llm call failed (%s): %s", model, e)
                return LLMResult("", None, pt_sum, ct_sum, cost_sum, time.perf_counter() - t0, model, self.s.provider, error=str(e)[:300])
