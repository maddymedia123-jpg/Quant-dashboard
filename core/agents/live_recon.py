"""Live Recon war room: two five-agent teams, one trap consult, one head verdict.

The client's model is eleven agents per timeframe: five bullish domain agents, five bearish domain
agents and a trap agent the Head of the War Room consults. Each domain agent is worth 20 points, so a
team carries 100.

Here the ten domain agents are one batched judgment per side (`core.agents.judge`) against the same
market state, because ten separate prompts would answer the same eighteen questions ten times over.
The scoring, the bias, the consult policy and the override live in this module as arithmetic, so every
number in the report can be recomputed from the probabilities without asking a model again."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.agents.context import common_payload
from core.agents.recon_profiles import LIVE, ReconProfile
from core.indicators import liquidity as lq
from core.indicators import smc
from core.agents.judge import ChoiceResult, Judge, JudgeResult
from core.agents.rubric import (
    BEARISH, BULLISH, DOMAINS, MAX_TEAM_POINTS, Domain,
    domain_score, missing_items, team_score, verdict,
)

log = logging.getLogger(__name__)

# A side needs this much more than the other before the head calls a direction at all.
DECISIVE_MARGIN = 10.0
# Both teams scoring this high means both cases are well supported: the classic trap setup.
CONFLICT_POINTS = 60.0
# How sure the trap agent must be before the head overrides a direction.
TRAP_OVERRIDE_P = 0.60

TRAP_OPTIONS: dict[str, str] = {
    "BULL_TRAP": "Upside strength is being manufactured: buyside liquidity is being taken to fill "
                 "sell orders, and the move up is likely to fail and reverse down.",
    "BEAR_TRAP": "Downside weakness is being manufactured: sellside liquidity is being taken to fill "
                 "buy orders, and the move down is likely to fail and reverse up.",
    "GENUINE_MOVE": "The current move is real continuation, supported by order flow and structure "
                    "rather than a liquidity raid.",
    "NO_TRAP": "There is no trap and no decisive move: the market is ranging or waiting on a catalyst.",
}
def trap_instructions(profile: ReconProfile | None = None) -> str:
    p = profile or LIVE
    return ("Both a bullish and a bearish team have argued their case on this BTC market state. Judge what "
            f"the setup over the next {p.horizon} actually is, paying attention to liquidity sweeps, CVD "
            "absorption, open interest and whether structure confirms the move or only price does.")


TRAP_INSTRUCTIONS = trap_instructions(LIVE)


@dataclass
class DomainBreakdown:
    key: str
    agent_no: int
    title: str
    points: float
    max_points: int
    items: tuple[tuple[str, int, float], ...] = ()   # (item id, weight, probability)

    @property
    def pct(self) -> float:
        return 100.0 * self.points / self.max_points if self.max_points else 0.0


@dataclass
class TeamScore:
    side: str
    points: float = 0.0
    domains: tuple[DomainBreakdown, ...] = ()
    provider: str = ""
    latency_s: float = 0.0
    missing: tuple[str, ...] = ()
    error: str | None = None

    @property
    def verdict(self) -> str:
        return verdict(self.points)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class LiveReconResult:
    generated_at: datetime
    bull: TeamScore
    bear: TeamScore
    trap: ChoiceResult
    bias: str = "BALANCED"
    margin: float = 0.0
    confidence: float = 0.0
    consulted: bool = False
    override: str | None = None
    notes: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unavailable: tuple[str, ...] = field(default_factory=tuple)
    category: str = "live"

    @property
    def leader(self) -> TeamScore | None:
        if self.bias.startswith("BULL"):
            return self.bull
        if self.bias.startswith("BEAR"):
            return self.bear
        return None

    @property
    def degraded(self) -> bool:
        return not (self.bull.ok and self.bear.ok) or bool(self.bull.missing or self.bear.missing)


LIVE_TIMEFRAMES = LIVE.timeframes


def build_state(m, analyses: dict, now_ms: int | None = None, profile: ReconProfile | None = None) -> dict:
    """The shared state both teams and the trap agent judge.

    Live Recon reads 15m, 1h and 4h, which are the live, intraday and weekly analyses. The rest of the
    snapshot (spot, derivatives, calendar) comes along in the common payload, availability flags and
    all, so an agent can see that a field is missing instead of guessing at it.

    The SMC and liquidity blocks matter most: without them the structure, order block, gap, sweep,
    delta and volume-profile questions in the rubric have nothing to read, and a judge that cannot see
    the evidence correctly scores near zero. Every one of them is computed here from candles."""
    profile = profile or LIVE
    state = common_payload(m, analyses)
    wanted = profile.analysis_keys
    state["categories"] = {k: v for k, v in state.get("categories", {}).items() if k in wanted}
    state["timeframes_read"] = dict(wanted)
    state["war_room"] = profile.label
    state["horizon"] = profile.horizon

    now = now_ms if now_ms is not None else int(m.generated_at.timestamp() * 1000)
    oi_history = list(getattr(m.futures, "oi_history", []) or [])
    state["smc"], state["liquidity"] = {}, {}
    for tf in profile.timeframes:
        df = m.spot.frames.get(tf)
        if df is None or df.empty:
            state["smc"][tf] = {"available": False, "note": f"no {tf} candles"}
            state["liquidity"][tf] = {"available": False, "note": f"no {tf} candles"}
            continue
        state["smc"][tf] = smc.summarise(df)
        state["liquidity"][tf] = lq.summarise(df, oi_history, now, windows=profile.timeframes)
    return state


def score_team(side: str, res: JudgeResult) -> TeamScore:
    probs = res.probabilities
    breakdown = tuple(
        DomainBreakdown(
            key=d.key, agent_no=d.agent_no, title=d.title,
            points=domain_score(d, probs), max_points=d.max_points,
            items=tuple((i.id, i.weight, float(probs.get(i.id, 0.0))) for i in d.items),
        )
        for d in DOMAINS
    )
    return TeamScore(side=side, points=team_score(probs), domains=breakdown, provider=res.provider,
                     latency_s=res.latency_s, missing=tuple(missing_items(probs)) if res.ok else (),
                     error=res.error)


def bias_for(bull: float, bear: float) -> tuple[str, float]:
    """Direction and its margin. Below DECISIVE_MARGIN the head does not call a side."""
    margin = bull - bear
    if margin >= DECISIVE_MARGIN:
        return "BULL", margin
    if -margin >= DECISIVE_MARGIN:
        return "BEAR", margin
    return "BALANCED", margin


def needs_consult(bull: float, bear: float) -> bool:
    """The head consults the trap agent when the teams conflict or neither is decisive."""
    return (bull >= CONFLICT_POINTS and bear >= CONFLICT_POINTS) or abs(bull - bear) < DECISIVE_MARGIN


def _domain(key: str) -> Domain:
    return next(d for d in DOMAINS if d.key == key)


def resolve(bull: TeamScore, bear: TeamScore, trap: ChoiceResult) -> tuple[str, float, float, bool, str | None, tuple[str, ...]]:
    """Turn two team scores and the trap call into the head's verdict. Pure arithmetic and policy."""
    base, margin = bias_for(bull.points, bear.points)
    consulted = needs_consult(bull.points, bear.points)
    notes: list[str] = []
    override: str | None = None

    p = trap.probabilities if trap.ok else {}
    bull_trap, bear_trap = p.get("BULL_TRAP", 0.0), p.get("BEAR_TRAP", 0.0)
    bias = base

    if base == "BULL" and bull_trap >= TRAP_OVERRIDE_P:
        bias, override = "BULL TRAP RISK", f"bull trap probability {bull_trap:.0%} overrides the long bias"
    elif base == "BEAR" and bear_trap >= TRAP_OVERRIDE_P:
        bias, override = "BEAR TRAP RISK", f"bear trap probability {bear_trap:.0%} overrides the short bias"
    elif base == "BALANCED" and trap.ok and trap.choice in ("BULL_TRAP", "BEAR_TRAP"):
        notes.append(f"No decisive team score; trap agent reads the setup as {trap.choice.replace('_', ' ').lower()}.")

    winner = max(bull.points, bear.points)
    confidence = 0.0 if bias == "BALANCED" else min(1.0, (winner / MAX_TEAM_POINTS) * (0.5 + min(abs(margin), 40.0) / 80.0))

    if bull.points >= CONFLICT_POINTS and bear.points >= CONFLICT_POINTS:
        notes.append(f"Both teams score above {CONFLICT_POINTS:.0f}: the two cases are equally well supported, "
                     "which is the condition a trap is built in.")
    if abs(margin) < DECISIVE_MARGIN:
        notes.append(f"Margin {abs(margin):.1f} points is under the {DECISIVE_MARGIN:.0f}-point threshold, "
                     "so no direction is called.")
    for team in (bull, bear):
        if team.error:
            notes.append(f"The {team.side} team did not score ({team.error}); its 0 is a failure, not a reading.")
        elif team.missing:
            missing_titles = ", ".join(sorted({_domain(k.split('_')[0]).title for k in team.missing
                                               if k.split('_')[0] in {d.key for d in DOMAINS}}))
            notes.append(f"The {team.side} team is missing {len(team.missing)} checklist answers"
                         + (f" ({missing_titles})" if missing_titles else "") + "; those items scored 0.")
    if not trap.ok:
        notes.append("The trap agent did not answer, so no trap override was applied.")

    return bias, margin, confidence, consulted, override, tuple(notes)


async def run_live_recon(judge: Judge, m, analyses: dict, now: datetime | None = None,
                         profile: ReconProfile | None = None) -> LiveReconResult:
    """One war-room pass: both teams and the trap consult, in parallel, on one shared state.

    Live Recon by default; any category's profile runs the same eleven agents on its own timeframes."""
    profile = profile or LIVE
    state = build_state(m, analyses, profile=profile)
    bull_res, bear_res, trap = await asyncio.gather(
        judge.score(state, BULLISH, profile=profile),
        judge.score(state, BEARISH, profile=profile),
        judge.classify(state, "trap", trap_instructions(profile), TRAP_OPTIONS),
    )
    bull, bear = score_team(BULLISH, bull_res), score_team(BEARISH, bear_res)
    bias, margin, confidence, consulted, override, notes = resolve(bull, bear, trap)
    return LiveReconResult(
        generated_at=now or datetime.now(timezone.utc),
        bull=bull, bear=bear, trap=trap,
        bias=bias, margin=margin, confidence=confidence, consulted=consulted, override=override, notes=notes,
        prompt_tokens=bull_res.prompt_tokens + bear_res.prompt_tokens,
        completion_tokens=bull_res.completion_tokens + bear_res.completion_tokens,
        category=profile.category,
    )
