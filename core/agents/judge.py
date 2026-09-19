"""Typed judgments for the Live Recon rubric.

One interface, two backends:

* TypeSafe (System One / Jev) returns a calibrated probability per question and batches every question
  against one shared state in a single request, which is what makes an 18-item checklist affordable.
* Gemini is the fallback so the rubric works before a TypeSafe key exists. It is asked for the same
  0-1 probabilities in JSON mode.

Either way the model only answers "does this condition hold". The weights and the arithmetic live in
`core.agents.rubric`, so a team score is auditable and weights can change without new inference."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from core.agents.rubric import ALL_ITEMS, RubricItem
from core.agents.settings import LLMSettings

log = logging.getLogger(__name__)
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"


@dataclass
class JudgeResult:
    probabilities: dict[str, float] = field(default_factory=dict)
    provider: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.probabilities)


def _clamp(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


def parse_typesafe(payload: dict) -> dict[str, float]:
    """Read the noul probability for each question id."""
    out: dict[str, float] = {}
    nouls = payload.get("answers") or payload.get("nouls") or {}
    for qid, ans in nouls.items():
        value = ans.get("noul") if isinstance(ans, dict) else ans
        p = _clamp(value)
        if p is not None:
            out[qid] = p
    return out


def parse_gemini_probabilities(data: dict) -> dict[str, float]:
    block = data.get("probabilities") if isinstance(data.get("probabilities"), dict) else data
    out: dict[str, float] = {}
    for qid, value in block.items():
        if isinstance(value, dict):
            value = value.get("probability", value.get("noul"))
        p = _clamp(value)
        if p is not None:
            out[qid] = p
    return out


@dataclass
class ChoiceResult:
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None    # how concentrated the distribution is, per the API
    provider: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.choice is not None


def parse_typesafe_choice(payload: dict, qid: str) -> ChoiceResult:
    """Both primitives answer under "answers"; "choices" is accepted too, harmlessly."""
    answers = payload.get("answers") or payload.get("choices") or {}
    block = answers.get(qid) or {}
    probs = {k: _clamp(v) for k, v in (block.get("probabilities") or {}).items()}
    return ChoiceResult(choice=block.get("choice"), confidence=_clamp(block.get("confidence")),
                        probabilities={k: v for k, v in probs.items() if v is not None})


class Judge:
    """Scores rubric items for one side (bullish or bearish) against one shared state."""

    def __init__(self, settings: LLMSettings, typesafe_key: str | None = None,
                 client: httpx.AsyncClient | None = None, llm=None, timeout_s: float = 60.0):
        self.s = settings
        self.typesafe_key = typesafe_key
        self._client = client
        self._llm = llm
        self._timeout = timeout_s

    @property
    def provider(self) -> str:
        return "typesafe" if self.typesafe_key else self.s.provider

    async def score(self, state: dict, side: str, items: tuple[RubricItem, ...] = ALL_ITEMS) -> JudgeResult:
        t0 = time.perf_counter()
        try:
            if self.typesafe_key:
                probs, pt, ct = await self._typesafe(state, side, items)
            else:
                probs, pt, ct = await self._fallback(state, side, items)
            return JudgeResult(probabilities=probs, provider=self.provider, latency_s=time.perf_counter() - t0,
                               prompt_tokens=pt, completion_tokens=ct)
        except Exception as e:  # noqa: BLE001 - judge boundary: a failure must not stop the run
            log.warning("judge failed (%s): %s", self.provider, e)
            return JudgeResult(provider=self.provider, latency_s=time.perf_counter() - t0, error=str(e)[:300])

    async def classify(self, state: dict, qid: str, instructions: str, criteria: dict[str, str]) -> ChoiceResult:
        """One categorical judgment with a probability per option (the War Room trap call)."""
        try:
            if self.typesafe_key:
                body = {"state": state, "model": TYPESAFE_MODEL,
                        "questions": {qid: {"type": "choice", "instructions": instructions, "criteria": criteria}}}
                client = self._client or httpx.AsyncClient(timeout=self._timeout)
                try:
                    r = await client.post(TYPESAFE_URL, json=body, headers={"Authorization": f"Bearer {self.typesafe_key}"})
                    r.raise_for_status()
                    payload = r.json()
                finally:
                    if self._client is None:
                        await client.aclose()
                res = parse_typesafe_choice(payload, qid)
                res.provider = "typesafe"
                return res
            if self._llm is None:
                raise RuntimeError("no TypeSafe key and no LLM client for the classifier")
            opts = "\n".join(f'- "{k}": {v}' for k, v in criteria.items())
            system = ("You are a calibrated market judge. Choose exactly one option and give a probability from 0 to 1 "
                      "for every option, summing to about 1. Use only the data given. Do not explain. Return JSON as "
                      '{"choice": "<option>", "probabilities": {"<option>": <0-1>, ...}}.')
            user = (f"DATA (JSON):\n{json.dumps(state, separators=(',', ':'), default=str)}\n\n"
                    f"QUESTION\n{instructions}\n\nOPTIONS\n{opts}\n\nReturn JSON only.")
            res = await self._llm.complete_json(system, user, self.s.specialist_model, max_tokens=500, temperature=0.0)
            if res.data is None:
                raise RuntimeError(res.error or "classifier returned no JSON")
            probs = {k: _clamp(v) for k, v in (res.data.get("probabilities") or {}).items()}
            choice = res.data.get("choice")
            if choice not in criteria:
                choice = max(probs, key=probs.get) if probs else None
            return ChoiceResult(choice=choice, probabilities={k: v for k, v in probs.items() if v is not None},
                                provider=self.s.provider)
        except Exception as e:  # noqa: BLE001 - judge boundary
            log.warning("classify failed (%s): %s", self.provider, e)
            return ChoiceResult(provider=self.provider, error=str(e)[:300])

    # ---- backends ----
    async def _typesafe(self, state: dict, side: str, items) -> tuple[dict[str, float], int, int]:
        questions = {
            i.id: {"type": "noul",
                   "instructions": f"Judging the {side} case for BTC over the next four hours: {i.question(side)}",
                   "criteria": {"true": "The data shows this condition holding right now.",
                                "false": "The data does not show it, or cannot answer it."}}
            for i in items
        }
        body = {"state": state, "model": TYPESAFE_MODEL, "questions": questions}
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            r = await client.post(TYPESAFE_URL, json=body, headers={"Authorization": f"Bearer {self.typesafe_key}"})
            r.raise_for_status()
            payload = r.json()
        finally:
            if self._client is None:
                await client.aclose()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        return parse_typesafe(payload), int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)

    async def _fallback(self, state: dict, side: str, items) -> tuple[dict[str, float], int, int]:
        if self._llm is None:
            raise RuntimeError("no TypeSafe key and no LLM client for the fallback judge")
        listing = "\n".join(f'- "{i.id}": {i.question(side)}' for i in items)
        system = (
            "You are a calibrated market judge. For each condition, return the probability from 0 to 1 that the "
            f"condition currently holds for BTC on the stated data, judged for the {side} case over the next four "
            "hours. Use only the data given; when the data cannot answer a condition, return 0.5. Do not explain. "
            'Return JSON exactly as {"probabilities": {"<id>": <0-1 number>, ...}} with one entry per condition id.'
        )
        user = f"DATA (JSON):\n{json.dumps(state, separators=(',', ':'), default=str)}\n\nCONDITIONS\n{listing}\n\nReturn JSON only."
        res = await self._llm.complete_json(system, user, self.s.specialist_model, max_tokens=1200, temperature=0.0)
        if res.data is None:
            raise RuntimeError(res.error or "fallback judge returned no JSON")
        return parse_gemini_probabilities(res.data), res.prompt_tokens, res.completion_tokens
