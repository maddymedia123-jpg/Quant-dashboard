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
import time
from datetime import datetime, timezone

from pydantic import ValidationError

from core.agents import context
from core.agents.llm import LLMClient
from core.agents.prompts import PREAMBLE, SHARED_RULES
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

ONE_AGENT_TEXT = """You are the DIVISION HEAD, codename DIRECTOR, working alone: you read the WHOLE state below and write the entire report yourself. Ruthlessly objective; never take a side before the evidence does.
1. BULL CASE and BEAR CASE (bull_case with desk="bullish", bear_case with desk="bearish"): argue each side as strongly as the evidence HONESTLY allows. A side with little support gets a low conviction and says so - do not invent a case. Each gets convergence, divergence, conviction 1-10, a best_setup and acknowledged_risks.
2. CONTRADICTION ANALYSIS: metrics that read both ways; say which reading has the stronger evidence and why.
3. CONVERGENCE: signals both cases accept; the highest-confidence evidence.
4. BLIND SPOTS: what matters but is weak, missing or UNAVAILABLE.
5. SCENARIOS: 2-4, probabilities summing to 100, each with the price path and an explicit invalidation level.
6. FINAL TRAP CLASSIFICATION: BULL_TRAP (pump into resistance, fail, reverse down), BEAR_TRAP (dump into support, fail, reverse up), NO_TRAP (genuine move backed by flow, funding and OI), RANGE_TRAP (both sides liquidated before a trend).
7. trade_plan (prose, as before) AND plan (numbers, watched by code for alerts):
   - side LONG, SHORT or NONE. Choose NONE, with why_none, when there is no edge: a plan is not required.
   - zone_low/zone_high: the entry zone, built from levels that appear in the data (supports, resistances, VWAP, value area, gamma walls, liquidity pools). Never enter at market.
   - confirm_timeframe and confirm_close: the CLOSED candle that confirms. SHORT: a candle on confirm_timeframe closes below confirm_close (inside or under the zone). LONG: closes above it.
   - taker_ratio_limit (optional): SHORT requires taker buy/sell below it; LONG above it.
   - confirmations: the same rules in plain words, one per line.
   - stop beyond the zone (SHORT above zone_high, LONG below zone_low) at a structural level; targets in order of distance, each a level from the data.
   - expires_hours: how long the idea is valid if price never reaches the zone. risk_pct and max_leverage for this volatility regime.
8. category_summaries: one paragraph each for live, intraday, weekly and monthly, in plain English, quoted levels only. executive_summary is three sentences at most. data_quality_flags: missing, stale or estimated inputs.
READING THE DATA: cvd_5bars is REAL taker flow on the BTC perpetual (market buys minus market sells, BTC); if it says ESTIMATE it is not CVD. Each desk's forecast carries a track_record: when lean_right is not above price_rose_anyway, its lean has no edge - use its range and level odds, not its lean. COT is weekly context, not intraday timing."""

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
    state["cascade"], state["forecast"] = {}, {}
    gamma = m.options.gamma if m.options.available else None
    for key, profile in PROFILES.items():
        try:
            bias_tf, anchor_tf, engine_tfs = profile.cascade
            state["cascade"][key] = classify(m.spot.frames, bias_tf, anchor_tf, engine_tfs).to_state()
        except Exception as e:  # noqa: BLE001 - one desk's read must not take the payload down
            state["cascade"][key] = {"available": False, "note": f"{type(e).__name__}: {e}"}
        a = analyses.get(key)
        if a is not None:
            try:
                f, tr = desk_forecast(a, gamma, m.spot.frames.get(a.chart_tf))
                state["forecast"][key] = to_state(f, tr)
            except Exception as e:  # noqa: BLE001
                state["forecast"][key] = {"available": False, "note": f"{type(e).__name__}: {e}"}
    state["unavailable"] = sorted(set(m.unavailable()) | set(missing))
    return state


def system_prompt() -> str:
    return (f"AGENT_ID: {AGENT_ID}\n\n{PREAMBLE}\n\n{ONE_AGENT_TEXT}\n\n{ONE_AGENT_RULES}\n\n"
            f"Return ONLY a JSON object that validates against this JSON schema:\n{schema_hint(OneAgentOutput)}")


def user_prompt(state: dict) -> str:
    body = {k: v for k, v in state.items() if k != "unavailable"}
    return (f"DATA (JSON):\n{json.dumps(body, separators=(',', ':'), default=str)}\n\n"
            f"UNAVAILABLE: {json.dumps(state.get('unavailable', []))}\n\nReturn JSON only.")


def _validate(data: dict | None) -> tuple[OneAgentOutput | None, DirectorReport | None, str | None]:
    if data is None:
        return None, None, "no JSON returned"
    try:
        out = OneAgentOutput.model_validate(data)
        director = DirectorReport(
            executive_summary=out.executive_summary, trap_classification=out.trap_classification,
            classification_rationale=out.classification_rationale, contradictions=out.contradictions,
            convergence=out.convergence, blind_spots=out.blind_spots, scenarios=out.scenarios,
            trade_plan=out.trade_plan, category_summaries=out.category_summaries)
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
    out, director, verr = _validate(res.data) if res.data is not None else (None, None, err)
    if out is None and res.data is not None:
        # one correction round: the validation error goes back with the same data
        fix = user + f"\n\nYour previous JSON failed validation:\n{verr}\nFix those fields and return ONLY valid JSON."
        res2 = await client.complete_json(system, fix, used, max_tokens=MAX_TOKENS)
        pt, ct = pt + res2.prompt_tokens, ct + res2.completion_tokens
        out, director, verr = _validate(res2.data) if res2.data is not None else (None, None, res2.error)
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
