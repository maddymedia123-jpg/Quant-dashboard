"""Three-phase pipeline: specialists + volatility in parallel → desk syntheses → accuracy → Director.

Every agent call is isolated: a failure produces an AgentRun with ok=False and the pipeline continues."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ValidationError

from core.agents import context
from core.agents.llm import LLMClient
from core.agents.prompts import SPECIALISTS, schema_for, system_prompt, user_prompt
from core.agents.schemas import AgentRun, TrapReport
from core.agents.settings import LLMSettings
from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis

ProgressFn = Callable[[str, str], None]
SPECIALIST_IDS = tuple(SPECIALISTS.keys())
WINDOW = "Live 15m · Intraday 1h · Weekly 4h · Monthly 1d"
MAX_TOKENS = {"specialist": 1500, "desk": 2000, "aux": 1500, "director": 4000}


def _noop(_agent_id: str, _status: str) -> None:
    return None


async def _call(client: LLMClient, agent_id: str, model: str, payload: dict, progress: ProgressFn, max_tokens: int) -> tuple[BaseModel | None, AgentRun]:
    schema = schema_for(agent_id)
    system, user = system_prompt(agent_id), user_prompt(agent_id, payload)
    progress(agent_id, "running")
    res = await client.complete_json(system, user, model, max_tokens=max_tokens)
    pt, ct, cost, lat, err = res.prompt_tokens, res.completion_tokens, res.cost_usd, res.latency_s, res.error
    obj = None
    if res.data is not None:
        try:
            obj = schema.model_validate(res.data)
        except ValidationError as e:
            fix = user + f"\n\nYour previous JSON failed validation:\n{str(e)[:700]}\nFix those fields and return ONLY valid JSON."
            res2 = await client.complete_json(system, fix, model, max_tokens=max_tokens)
            pt += res2.prompt_tokens
            ct += res2.completion_tokens
            lat += res2.latency_s
            if res2.cost_usd is not None:
                cost = (cost or 0.0) + res2.cost_usd
            if res2.data is not None:
                try:
                    obj = schema.model_validate(res2.data)
                except ValidationError as e2:
                    err = f"schema validation failed twice: {str(e2)[:200]}"
            else:
                err = res2.error or "repair call returned no JSON"
    if obj is not None and agent_id in SPECIALISTS:
        s = SPECIALISTS[agent_id]
        obj.agent_id, obj.desk, obj.domain = agent_id, s["desk"], s["domain"]
    if obj is not None and agent_id.startswith("DESK_"):
        obj.desk = agent_id.split("_")[1].lower()
    run = AgentRun(agent_id=agent_id, ok=obj is not None, model=model, latency_s=round(lat, 2), prompt_tokens=pt,
                   completion_tokens=ct, cost_usd=cost, error=None if obj is not None else (err or "no JSON returned"))
    progress(agent_id, "ok" if obj is not None else "failed")
    return obj, run


async def run_pipeline(client: LLMClient, m: MarketSnapshot, analyses: dict[str, CategoryAnalysis], settings: LLMSettings,
                       raw_metrics: list[tuple[str, str, str]] | None = None, progress: ProgressFn | None = None) -> TrapReport:
    progress = progress or _noop
    raw_metrics = raw_metrics or []
    t0 = time.perf_counter()
    runs: list[AgentRun] = []
    spec_model, dir_model = settings.specialist_model, settings.director_model

    # Phase 1: ten specialists + volatility agent, concurrently (client semaphore bounds parallelism)
    tasks = [_call(client, aid, spec_model, context.payload_for(aid, m, analyses), progress, MAX_TOKENS["specialist"]) for aid in SPECIALIST_IDS]
    tasks.append(_call(client, "VOLATILITY", spec_model, context.volatility_payload(m, analyses), progress, MAX_TOKENS["aux"]))
    phase1 = await asyncio.gather(*tasks)
    briefs = []
    for obj, run in phase1[:-1]:
        runs.append(run)
        if obj is not None:
            briefs.append(obj)
    volatility, vrun = phase1[-1]
    runs.append(vrun)

    # Phase 2: desk syntheses
    desks = {}
    desk_tasks = []
    for desk in ("bullish", "bearish"):
        mine = [b for b in briefs if b.desk == desk]
        failed = [aid for aid in SPECIALIST_IDS if SPECIALISTS[aid]["desk"] == desk and aid not in {b.agent_id for b in mine}]
        if mine:
            desk_tasks.append((desk, _call(client, f"DESK_{desk.upper()}", spec_model, context.desk_payload(desk, mine, failed), progress, MAX_TOKENS["desk"])))
        else:
            runs.append(AgentRun(agent_id=f"DESK_{desk.upper()}", ok=False, model=spec_model, latency_s=0.0, error="no specialist briefs for this desk"))
    if desk_tasks:
        results = await asyncio.gather(*[t for _, t in desk_tasks])
        for (desk, _), (obj, run) in zip(desk_tasks, results):
            runs.append(run)
            if obj is not None:
                desks[desk] = obj

    failed_ids = [r.agent_id for r in runs if not r.ok]

    # Accuracy / correlation agent
    accuracy = None
    if desks:
        accuracy, arun = await _call(client, "ACCURACY", spec_model, context.accuracy_payload(desks, volatility, analyses, m.unavailable()), progress, MAX_TOKENS["aux"])
        runs.append(arun)
    else:
        runs.append(AgentRun(agent_id="ACCURACY", ok=False, model=spec_model, latency_s=0.0, error="skipped: no desk reports"))

    # Phase 3: Director
    director = None
    if desks:
        dpayload = context.director_payload(m, analyses, desks, volatility, accuracy, raw_metrics, failed_ids)
        director, drun = await _call(client, "DIRECTOR", dir_model, dpayload, progress, MAX_TOKENS["director"])
        runs.append(drun)
        if director is None and dir_model != spec_model:
            # primary Director model unavailable (5xx / timeout / quota): retry once on the specialist model
            progress("DIRECTOR", f"retrying on {spec_model}")
            director, drun2 = await _call(client, "DIRECTOR", spec_model, dpayload, progress, MAX_TOKENS["director"])
            drun2.agent_id = "DIRECTOR (fallback model)"
            runs.append(drun2)
    else:
        runs.append(AgentRun(agent_id="DIRECTOR", ok=False, model=dir_model, latency_s=0.0, error="skipped: no desk reports"))

    now = datetime.now(timezone.utc)
    return TrapReport(
        report_id=f"TIR-{now:%Y%m%d-%H%M%S}", generated_at=now, window=WINDOW, provider=settings.provider,
        briefs=briefs, desks=desks, volatility=volatility, accuracy=accuracy, director=director, runs=runs,
        wall_time_s=round(time.perf_counter() - t0, 1), unavailable_domains=m.unavailable(), raw_metrics=raw_metrics,
    )


def run_pipeline_sync(client: LLMClient, m: MarketSnapshot, analyses: dict[str, CategoryAnalysis], settings: LLMSettings,
                      raw_metrics: list[tuple[str, str, str]] | None = None, progress: ProgressFn | None = None) -> TrapReport:
    async def _go():
        try:
            return await run_pipeline(client, m, analyses, settings, raw_metrics, progress)
        finally:
            await client.aclose()
    return asyncio.run(_go())
