"""Step 5: the AI writes the Quant Forecast's game plan in plain English, from the plan's numbers.

The model never makes a number. It is handed the computed plan - direction, odds, windows, high and low,
invalidation, the levels - and asked for three pieces of text: the one-sentence directive the top panel
and the bot carry (Hammad's "core plan": one direct sentence), and one line each for Scenario A and B. The
direction is not its to choose: the regime comes from the math, and the plan's scoring is of that regime.
It is shown no jargon it could echo (the lean's reads stay in the drawer) and no track record.

Every reply is checked before it is used (`check`): each price must be one of the plan's own (to within
PRICE_TOL), each clock time one of the plan's times in UTC or PKT, each percentage one of its odds; it must
not name the opposite direction, and it must not use trader jargon (the plan_alerts list). A reply that
fails any check, or a model that is down or slow, leaves the code-written text in place: a plan is never
late and never carries a number the math did not produce.

Calls: one per plan, so at most LOCK_AFTER_INVALIDATIONS a day - well inside Gemini's free tier. The plan's
row is written before the call, so only the process that won the row makes it.
Never called on offline fixture data."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace

from core import quant_forecast as qf
from core.plan_alerts import BANNED_WORDS

log = logging.getLogger(__name__)

TIMEOUT_S = 20.0                 # per request
TOTAL_S = 45.0                   # the whole write-up, retries and spare models included: the worker's alerts wait on it
PRICE_TOL = 0.0002               # 0.02%: about $16 at $80,000 - rounding, not a different level
MAX_SUMMARY_CHARS = 220          # one desk-wire sentence
MAX_LINE_CHARS = 260
LEVEL_WORDS = {"gamma flip": "hedging flip level", "call wall": "upper hedging wall", "put wall": "lower hedging wall",
               "POC": "busiest price of the last 3 days", "value high": "top of the last 3 days' value",
               "value low": "bottom of the last 3 days' value"}

SYSTEM = """You write the daily BTC desk wire for a trading dashboard whose readers are not traders.
You are given a plan that has already been computed. Your job is only to put it in clear, punchy, plain English.

Rules - a reply that breaks any of them is thrown away:
- Use ONLY the prices, times and percentages given. Write prices exactly as given (for example $80,954) and
  times exactly as given, with both clocks (for example 14:00 UTC (19:00 PKT)). Never round, estimate or add
  a number of your own. The directive needs no numbers at all.
- Keep the direction you are given. Do not hedge it into its opposite, and do not invent a new forecast.
- No trader jargon. Never use these words: gamma, CVD, RSI, Fibonacci, delta, option, VWAP, taker, EMA,
  liquidity, sweep, VAH, VAL, order flow, funding, open interest, squeeze. Say what they mean instead
  ("buyers were paying up", "more traders are piling in").
- Never call it certain.

Return JSON only:
{"summary": "ONE direct sentence, under 160 characters, saying what price will do today - for example 'Quiet chop early, the morning dip holds support, then an afternoon run toward the high.'",
 "scenario_a": "one sentence: the main path",
 "scenario_b": "one sentence: what happens if it fails"}"""


def facts(plan: dict, live: dict | None, at) -> dict:
    """What the model is shown: the plan's numbers, already formatted the way they must be written."""
    from core.qf_alerts import REGIME_WORD, window_label
    money = lambda x: f"${x:,.0f}"  # noqa: E731
    out = {
        "direction": REGIME_WORD[plan["regime"]],
        "price_now": money(plan["origin"]),
        "plan_ends": at(plan["end_ms"]),
        "odds_main_path": f"{plan['p_scenario_a']:.0%}",
        "odds_alternative": f"{1 - plan['p_scenario_a']:.0%}",
        "likely_high": f"{money(plan['projected_high']['price'])} around {at(plan['projected_high']['t_ms'])}",
        "likely_low": f"{money(plan['projected_low']['price'])} around {at(plan['projected_low']['t_ms'])}",
        "cancelled_if": [f"a 4-hour candle closes {side} {money(v)}" for side, v in plan["invalidation"].items()],
        "quiet_windows": [f"{window_label(w['start_ms'], w['end_ms'])}, range {money(w['lower'])} to {money(w['upper'])}"
                          for w in plan.get("windows") or [] if w["phase"] == "chop"],
        "busy_windows": [window_label(w["start_ms"], w["end_ms"]) for w in plan.get("windows") or []
                         if w["phase"] == "expansion"],
        "levels": [f"{LEVEL_WORDS.get(lv['label'], lv['label'])} {money(lv['price'])}: {lv['p_touch']:.0%} chance "
                   "price trades there today" for lv in plan.get("levels") or [] if lv["p_touch"] >= 0.05][:6],
        "draft_written_by_code": {"summary": qf.wire_line(plan), "scenario_a": plan["scenario_a"],
                                  "scenario_b": plan["scenario_b"]},
    }
    if plan.get("replaces"):
        out["note"] = "This replaces an earlier plan today that was cancelled when price broke its line."
    return out


def _allowed(plan: dict, live: dict | None) -> tuple[list[float], set[str], set[int]]:
    from core.qf_alerts import _clock
    prices = [plan["origin"], plan["projected_high"]["price"], plan["projected_low"]["price"],
              *plan["invalidation"].values(), *(plan.get("range_68") or [])]
    prices += [w[k] for w in plan.get("windows") or [] for k in ("lower", "upper")]
    prices += [lv["price"] for lv in plan.get("levels") or []]
    prices += [v for v in (plan.get("value_area") or {}).values() if v]
    times = {plan["created_ms"], plan["end_ms"], plan["projected_high"]["t_ms"], plan["projected_low"]["t_ms"]}
    times |= {w[k] for w in plan.get("windows") or [] for k in ("start_ms", "end_ms")}
    clocks = {c for t in times for c in _clock(int(t))}
    p = plan["p_scenario_a"]
    pcts = {round(p * 100), round((1 - p) * 100), 68}
    pcts |= {round(lv["p_touch"] * 100) for lv in plan.get("levels") or []}
    return prices, clocks, pcts


PRICE_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?([kK])(?![A-Za-z]))?")   # "$80.8k" yes; "$80,000 keeps" no
CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
PCT_RE = re.compile(r"(\d{1,3})(?:\.\d+)?\s?%")
OPPOSITE = {"bullish": ("bearish", "leans down"), "bearish": ("bullish", "leans up"),
            "neutral": ("bullish", "bearish")}


def check(text: str, plan: dict, live: dict | None = None) -> list[str]:
    """Why this text may not be shown, or [] when it may."""
    prices, clocks, pcts = _allowed(plan, live)
    problems = []
    for num, k in PRICE_RE.findall(text):
        x = float(num.replace(",", "")) * (1000 if k else 1)
        if not any(abs(x - p) <= PRICE_TOL * p for p in prices if p):
            problems.append(f"price ${x:,.0f} is not in the plan")
    for h, m in CLOCK_RE.findall(text):
        if f"{int(h):02d}:{m}" not in clocks:
            problems.append(f"time {h}:{m} is not in the plan")
    for n in PCT_RE.findall(text):
        if int(n) not in pcts:
            problems.append(f"{n}% is not one of the plan's odds")
    low = text.lower()
    for w in OPPOSITE.get(plan["regime"], ()):
        if w in low:
            problems.append(f"says '{w}' on a {plan['regime']} plan")
    for w in BANNED_WORDS:
        # whole words: "ema" is jargon, "remains" is not
        if re.search(rf"\b{re.escape(w.strip())}s?\b", low):
            problems.append(f"jargon: '{w.strip()}'")
    return problems


def _ask(settings, system: str, user: str) -> dict | None:
    from core.agents.llm import LLMClient

    async def go():
        client = LLMClient(replace(settings, timeout_s=TIMEOUT_S))
        try:
            r = await asyncio.wait_for(client.complete_json(system, user, settings.director_model, max_tokens=900,
                                                            temperature=0.3), TOTAL_S)
        finally:
            await client.aclose()
        if r.error or not r.data:
            raise RuntimeError(r.error or "empty reply")
        return {"data": r.data, "model": r.model}

    return asyncio.run(go())


def write(plan: dict, live: dict | None = None, settings=None, ask=None) -> dict | None:
    """The checked AI text for this plan: {"summary", "scenario_a", "scenario_b", "model"}, or a dict with
    "rejected" saying why the code's own text stands. None when there is nothing to ask (no key, offline,
    a locked day). `ask` replaces the model call in tests."""
    from core.data.market import _offline
    from core.qf_alerts import at_label
    if plan.get("regime") == "locked" or not plan.get("available"):
        return None
    if ask is None:
        if _offline():
            return None
        if settings is None:
            from core.agents.settings import load_settings
            settings = load_settings()
        if settings is None:
            return None
        ask = lambda s, u: _ask(settings, s, u)  # noqa: E731
    user = "The computed plan:\n" + json.dumps(facts(plan, live, at_label), indent=1, ensure_ascii=False)
    try:
        got = ask(SYSTEM, user)
    except Exception as e:  # noqa: BLE001 - the code's text stands; the plan is not held up
        log.warning("quant forecast write-up failed: %s", e)
        return {"rejected": f"the model did not answer ({type(e).__name__}: {str(e)[:120]})"}
    data = got.get("data") or {}
    out = {k: str(data.get(k) or "").strip() for k in ("summary", "scenario_a", "scenario_b")}
    if not all(out.values()):
        return {"rejected": "the reply was missing a part"}
    if len(out["summary"]) > MAX_SUMMARY_CHARS or max(len(out["scenario_a"]), len(out["scenario_b"])) > MAX_LINE_CHARS:
        return {"rejected": "the reply was too long"}
    problems = check(" ".join(out.values()), plan, live)
    if problems:
        log.warning("quant forecast write-up rejected: %s", "; ".join(problems[:4]))
        return {"rejected": "; ".join(problems[:3])}
    out["model"] = got.get("model") or ""
    return out
