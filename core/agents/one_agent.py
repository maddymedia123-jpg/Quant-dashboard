"""The 1-agent Director: one model reads the whole state and writes the whole report, plan as numbers included.

The fifteen-agent pipeline hands slices of the same data down a chain - ten specialists, two desks, an
accuracy check, the Director - and each hand-off loses context; five of its agents must argue the bearish
side whatever the data says. Here one model sees everything at once, writes the bull case and the bear
case itself (each as strong as the evidence honestly allows), and returns the same report sections plus
`plan`: the trade plan as numbers that code can watch for alerts.

One call per report instead of fifteen, so the free Gemini tier allows about twenty reports a day instead
of one. The answer is validated (scenario probabilities, plan geometry); a failure is sent back once with
the validation error, then reported as failed rather than shown half-built."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from pydantic import ValidationError

from core.agents import context
from core.agents.llm import LLMClient
from core.agents.prompts import (DISCIPLINE_TEXT, HORIZON_TEXT, PLAN_TEXT, PREAMBLE, REGIME_TEXT, SHARED_RULES,
                                 WINDOWS_TEXT)
from core.agents.schemas import (
    AgentRun, DirectorReport, OneAgentOutput, TrapReport, schema_hint,
)

MAX_TOKENS = 16_000     # the model thinks before it answers, and thinking counts against this
AGENT_ID = "DIRECTOR_1"
# Measured 20 Oct: 3.8 Flash reading ~13k tokens and thinking took about 100 s, so the shared 90 s timeout
# threw away a good answer and the retry made the report take 190 s. One call deserves a longer wait.
TIMEOUT_S = 240.0


def client_for(settings) -> LLMClient:
    from dataclasses import replace
    return LLMClient(replace(settings, timeout_s=max(settings.timeout_s, TIMEOUT_S)))

ONE_AGENT_TEXT = ("""You are the DIVISION HEAD, codename DIRECTOR, working alone: you read the WHOLE state below and write the entire report yourself. Ruthlessly objective; never take a side before the evidence does.
1. BULL CASE and BEAR CASE (bull_case with desk="bullish", bear_case with desk="bearish"): argue each side as strongly as the evidence HONESTLY allows. A side with little support gets a low conviction and says so - do not invent a case. Each gets convergence, divergence, conviction 1-10, a best_setup and acknowledged_risks.
2. CONTRADICTION ANALYSIS: metrics that read both ways; say which reading has the stronger evidence and why.
3. CONVERGENCE: signals both cases accept; the highest-confidence evidence.
4. BLIND SPOTS: what matters but is weak, missing or UNAVAILABLE.
5. SCENARIOS: 2-4, probabilities summing to 100, each with the price path and an explicit invalidation level.
6. FINAL TRAP CLASSIFICATION: BULL_TRAP (pump into resistance, fail, reverse down), BEAR_TRAP (dump into support, fail, reverse up), NO_TRAP (genuine move backed by flow, funding and OI), RANGE_TRAP (both sides liquidated before a trend).
7. trade_plan (prose, as before) AND """ + PLAN_TEXT + """
8. category_summaries: one paragraph each for live, intraday, weekly and monthly, in plain English, quoted levels only. executive_summary is three sentences at most. data_quality_flags: missing, stale or estimated inputs.
9. """ + HORIZON_TEXT + """
10. """ + REGIME_TEXT + """
READING THE DATA: cvd_5bars is REAL taker flow on the BTC perpetual (market buys minus market sells, BTC); if it says ESTIMATE it is not CVD. Each desk's forecast carries a track_record: when lean_right is not above price_rose_anyway, its lean has no edge - use its range and level odds, not its lean. COT is weekly context, not intraday timing. """ + WINDOWS_TEXT + """ TIMING: use turning_window (measured US-session turning hours); its kill_zones carry the spec's three sessions with measured odds - only New York beat chance, so never time a turn on the Asian or London session; the Fibonacci reversal_window in each category was tested and timed turns no better than chance - never use it for timing.
""" + DISCIPLINE_TEXT)

ONE_AGENT_RULES = SHARED_RULES.replace(
    "3. Quote the figures you rely on in raw_metrics exactly as given (same units), so the Director can audit you.",
    "3. Quote the figures you rely on exactly as given (same units), so a reader can audit you.").replace(
    "4. Argue your desk's side as hard as the evidence allows, then state the single strongest argument AGAINST your thesis in key_risk_to_thesis.",
    "4. Every claim names the figure behind it.").replace(
    "5. Conviction is 1-10. If your domain's data is unavailable, conviction is at most 3.",
    "5. Conviction is 1-10, and a case built on unavailable data is at most 3.")


def director_state(m, analyses: dict) -> dict:
    """The whole state in one payload: every desk's brief, cascade and forecast, all derivatives, options
    expiries, flow and sentiment, the macro read and the raw metric table."""
    from core.agents.recon_profiles import PROFILES
    from core.indicators.cascade import classify
    from core.indicators.forecast import desk_forecast, to_state

    state = context.common_payload(m, analyses)
    deriv, missing = context.derivatives_payload(m)
    state["derivatives"] = deriv
    for domain in ("options", "flow", "sentiment"):
        data, gaps = context.domain_payload(domain, m, analyses)
        missing += gaps
        if domain == "options" and "options" in data:
            state["options_expiries"] = data["options"].get("expiries", [])
        if domain == "flow":
            state["whales"] = data.get("whales")
            state["taker_buy_sell_ratio"] = data.get("taker_buy_sell_ratio")
        if domain == "sentiment":
            state["sentiment"] = data.get("sentiment")
            state["funding_by_exchange"] = data.get("funding_by_exchange")
            state["unusual_flow_radar"] = data.get("unusual_flow_radar")
    state.update(context.desk_blocks(m, analyses))
    state["unavailable"] = sorted(set(m.unavailable()) | set(missing))
    return state


def system_prompt() -> str:
    return (f"AGENT_ID: {AGENT_ID}\n\n{PREAMBLE}\n\n{ONE_AGENT_TEXT}\n\n{ONE_AGENT_RULES}\n\n"
            f"Return ONLY a JSON object that validates against this JSON schema:\n{schema_hint(OneAgentOutput)}")


def user_prompt(state: dict) -> str:
    body = {k: v for k, v in state.items() if k != "unavailable"}
    return (f"DATA (JSON):\n{json.dumps(body, separators=(',', ':'), default=str)}\n\n"
            f"UNAVAILABLE: {json.dumps(state.get('unavailable', []))}\n\nReturn JSON only.")


def tag_plan(plan: PlanLevels | None, regime: dict | None) -> PlanLevels | None:
    """A sided plan made while an event regime is active carries the event's title and half the
    Director's risk_pct and max_leverage (the Director gives its normal size: rule 10). The geometry is
    the Director's: the rule asks it for the wider stop. The alerts and the Pulse read the tag
    (core.plan_alerts.regime_line). A stand-aside is left alone. The tag is code's: outside a regime a
    model-written event_regime is cleared, so no plan claims a regime the calendar does not hold."""
    if plan is None or plan.side == "NONE":
        return plan
    if not (regime or {}).get("active"):
        return plan if not plan.event_regime else plan.model_copy(update={"event_regime": ""})
    half = lambda x: None if x is None else round(x / 2, 2)  # noqa: E731
    return plan.model_copy(update={"event_regime": str(regime.get("title") or "major event"),
                                   "risk_pct": half(plan.risk_pct), "max_leverage": half(plan.max_leverage)})


_PRICE = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,6}")


def plan_notes(plan: PlanLevels | None, scenarios) -> list[str]:
    """Code's check of the plan against the Director's own words. TIR1-20261105-153911: the 55% scenario died
    on a 1-hour close below 75,700 while the plan's stop sat at 75,350, so the trade outlived its thesis by
    350 dollars. The first price in the primary (first) scenario's invalidation is read; a LONG stop below
    it, or a SHORT stop above it, is named. No price in the words, no plan, no stop: nothing to say."""
    if plan is None or plan.side == "NONE" or plan.stop is None or not scenarios:
        return []
    words = str(getattr(scenarios[0], "invalidation", "") or "")
    m = _PRICE.search(words)
    if not m:
        return []
    inv = float(m.group(0).replace(",", ""))
    gap = inv - plan.stop if plan.side == "LONG" else plan.stop - inv
    if gap <= 0:
        return []
    return [f"the stop {plan.stop:,.0f} sits {gap:,.0f} {'below' if plan.side == 'LONG' else 'above'} the primary "
            f"scenario's invalidation ({words.strip()}): the trade outlives its thesis; the stop belongs at the "
            "invalidation, or the invalidation at the stop"]


def _validate(data: dict | None, horizons: dict | None = None,
              regime: dict | None = None) -> tuple[OneAgentOutput | None, DirectorReport | None, str | None]:
    if data is None:
        return None, None, "no JSON returned"
    try:
        out = OneAgentOutput.model_validate(data)
        out = out.model_copy(update={"plan": tag_plan(out.plan, regime)})
        from core.indicators.horizons import plan_check
        check = None
        if horizons and horizons.get("framing"):
            try:
                check = plan_check(out.plan, horizons)
            except Exception as e:  # noqa: BLE001 - the check is an addition to the report, never its failure
                check = {"verdict": "unknown", "note": f"plan check failed: {type(e).__name__}: {e}"}
        director = DirectorReport(
            executive_summary=out.executive_summary, trap_classification=out.trap_classification,
            classification_rationale=out.classification_rationale, contradictions=out.contradictions,
            convergence=out.convergence, blind_spots=out.blind_spots, scenarios=out.scenarios,
            trade_plan=out.trade_plan, category_summaries=out.category_summaries,
            horizon_framing=out.horizon_framing, horizon_check=check, plan=out.plan,
            plan_notes=plan_notes(out.plan, out.scenarios))
        return out, director, None
    except (ValidationError, ValueError) as e:
        return None, None, str(e)[:900]


async def run_one_agent(client: LLMClient, m, analyses: dict, model: str,
                        raw_metrics: list[tuple[str, str, str]] | None = None,
                        snapshot_id: str | None = None) -> TrapReport:
    t0 = time.perf_counter()
    state = director_state(m, analyses)
    system, user = system_prompt(), user_prompt(state)
    res = await client.complete_json(system, user, model, max_tokens=MAX_TOKENS)
    pt, ct, used, err = res.prompt_tokens, res.completion_tokens, res.model or model, res.error
    regime = (state.get("macro_read") or {}).get("event_regime")
    out, director, verr = _validate(res.data, state.get("horizons"), regime) if res.data is not None else (None, None, err)
    if out is None and res.data is not None:
        # one correction round: the validation error goes back with the same data
        fix = user + f"\n\nYour previous JSON failed validation:\n{verr}\nFix those fields and return ONLY valid JSON."
        res2 = await client.complete_json(system, fix, used, max_tokens=MAX_TOKENS)
        pt, ct = pt + res2.prompt_tokens, ct + res2.completion_tokens
        out, director, verr = _validate(res2.data, state.get("horizons"), regime) if res2.data is not None else (None, None, res2.error)
    run = AgentRun(agent_id=AGENT_ID, ok=out is not None, model=used, latency_s=round(time.perf_counter() - t0, 2),
                   prompt_tokens=pt, completion_tokens=ct, cost_usd=None,
                   error=None if out is not None else (verr or err or "no JSON returned"))
    now = datetime.now(timezone.utc)
    return TrapReport(
        report_id=f"TIR1-{now:%Y%m%d-%H%M%S}", generated_at=now,
        asset="BTC perpetual (BTC_USDT)" if "perp" in (m.spot.symbol or "") else "BTC/USD spot",
        window="Live Recon 1h · Intraday 4h · Weekly 1D · Monthly 1W", provider=getattr(getattr(client, "s", None), "provider", "unknown"),
        desks=({"bullish": out.bull_case.model_copy(update={"desk": "bullish"}),
                "bearish": out.bear_case.model_copy(update={"desk": "bearish"})} if out else {}),
        director=director, runs=[run], wall_time_s=round(time.perf_counter() - t0, 1),
        unavailable_domains=m.unavailable(), raw_metrics=raw_metrics or [], mode="1-agent",
        plan=out.plan if out else None, snapshot_id=snapshot_id)


def run_one_agent_sync(client: LLMClient, m, analyses: dict, model: str, raw_metrics=None,
                       snapshot_id: str | None = None) -> TrapReport:
    import asyncio

    async def _go():
        try:
            return await run_one_agent(client, m, analyses, model, raw_metrics, snapshot_id)
        finally:
            await client.aclose()
    return asyncio.run(_go())
