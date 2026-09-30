"""The TRAP department: ten sub-agents, a Head that consults the category desk, and anchored traps.

The spec's shape, kept faithfully:

* Five bullish and five bearish trap sub-agents across the same five domains. The bearish specialists
  hunt bull traps and the bullish ones hunt bear traps, so a candidate is scored by the side whose case
  the trap would prove - one batched request of five domain questions, out of 100 points.
* The Head then makes one categorical call: engineered trap, authentic break, or unclear.
* Inbound re-verification before anything is declared: the affected category desk is asked, in code,
  whether this is an authentic higher-timeframe break rather than an engineered raid. A desk that reads
  a real expansion vetoes the declaration and the trap stays a watch.
* A declared trap is anchored to its category until price settles it - the invalidation is touched, or
  it plays out back inside value. Nothing expires on a timer.

Scores and policy are arithmetic here; the model only judges whether each domain's evidence supports the
trap, and makes the one categorical call."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.agents.judge import Judge
from core.agents.recon_profiles import LIVE, ReconProfile
from core.agents.rubric import BEARISH, BULLISH, DOMAIN_BY_KEY, RubricItem
from core.traps import BULL_TRAP, TrapCandidate

log = logging.getLogger(__name__)

DOMAIN_KEYS = ("quant", "auction", "delta", "ict", "derivs")
POINTS_PER_DOMAIN = 20
DECLARE_POINTS = 60.0          # below this the desk watches rather than declares
OI_BUILDING_PCT = 5.0          # 24h open-interest change that counts as the crowd piling in

TRAP_CHOICES = {
    "ENGINEERED_TRAP": "The move is manufactured: liquidity was taken to fill the other side, and price "
                       "is likely to fail back through the level.",
    "AUTHENTIC_BREAK": "The move is a real break or regime shift: the level gave way and price is "
                       "accepting beyond it.",
    "UNCLEAR": "The evidence does not settle whether this is a raid or a real break.",
}

# What each trap sub-agent is asked about its own domain, for the side whose case the trap would prove.
DOMAIN_QUESTIONS = {
    "quant": "Volatility and timeframe statistics support a {label} here: the move is stretched or "
             "mean-reverting rather than a trending expansion.",
    "auction": "The auction supports a {label}: price failed to accept beyond the value area edge at "
               "{level:,.0f} and is being rejected back towards value.",
    "delta": "Order flow supports a {label}: delta disagrees with the move through {level:,.0f}, so the "
             "break is being absorbed rather than driven.",
    "ict": "Liquidity and structure support a {label}: the move through {level:,.0f} took resting "
           "liquidity and failed to hold it, leaving a swing failure.",
    "derivs": "Positioning supports a {label}: funding, open interest or liquidations show the crowd on "
              "the wrong side of the move through {level:,.0f}.",
}


@dataclass(frozen=True)
class DomainPoints:
    key: str
    title: str
    points: float
    probability: float


@dataclass
class DeskView:
    """What the affected category desk says when the TRAP Head asks."""
    reads: str                 # authentic break | engineered or unclear | quiet
    reasons: tuple[str, ...] = ()


@dataclass
class TrapVerdict:
    candidate: TrapCandidate
    side_scored: str
    score: float = 0.0
    domains: tuple[DomainPoints, ...] = ()
    head_call: str | None = None
    head_confidence: float | None = None
    desk: DeskView = field(default_factory=lambda: DeskView("quiet"))
    status: str = "watching"   # active | watching | unjudged
    notes: tuple[str, ...] = ()
    provider: str = ""

    @property
    def declared(self) -> bool:
        return self.status == "active"


def _items(candidate: TrapCandidate) -> tuple[RubricItem, ...]:
    """One question per domain, phrased for this candidate. Identical on both sides: the side being
    scored already says which case the trap would prove."""
    out = []
    for key in DOMAIN_KEYS:
        text = DOMAIN_QUESTIONS[key].format(label=candidate.label, level=candidate.level)
        out.append(RubricItem(f"trap_{key}", POINTS_PER_DOMAIN, text, text))
    return tuple(out)


def desk_view(state: dict, candidate: TrapCandidate, profile: ReconProfile | None = None) -> DeskView:
    """The inbound half of the handshake, in code.

    A trap and a real expansion look the same for one candle. The desk's own reads settle it: structure
    broken in the direction of the move on its higher timeframe, price accepting beyond value, and open
    interest building with it all say the level genuinely gave way."""
    p = profile or LIVE
    _ltf, mtf, htf = p.timeframes
    state = state if isinstance(state, dict) else {}
    up = candidate.side == BULL_TRAP
    strong: list[str] = []          # the two reads that distinguish an expansion from a raid
    weak: list[str] = []            # context that is just as consistent with either

    structure = ((state.get("smc") or {}).get(htf) or {}).get("structure") or {}
    bias = structure.get("bias")
    if bias == ("bullish" if up else "bearish"):
        event = (structure.get("last_event") or {}).get("type")
        strong.append(f"{htf} structure is {bias}" + (f" on a confirmed {event}" if event else ""))

    position = (((state.get("liquidity") or {}).get(mtf) or {}).get("volume_profile") or {}).get("price_position")
    if position == ("above value" if up else "below value"):
        strong.append(f"price is accepting {position} rather than being rejected")

    # Open interest *change* is unsigned with respect to direction: a genuine breakdown builds open
    # interest exactly as a genuine breakout does, so the same test serves both sides. It is context
    # only - crowding is as consistent with a trap as with a real break - so it never vetoes.
    oi = ((state.get("derivatives") or {}).get("futures") or {}).get("oi_change_24h_pct")
    try:
        if oi is not None and float(oi) > OI_BUILDING_PCT:
            weak.append(f"open interest is building into the move ({float(oi):+.1f}% over 24h)")
    except (TypeError, ValueError):
        pass

    reasons = tuple(strong + weak)
    # Structure *and* acceptance together is the pair that vetoes: one on its own is half a break.
    if len(strong) >= 2:
        return DeskView("authentic break", reasons)
    return DeskView("engineered or unclear" if reasons else "quiet", reasons)


async def judge_candidate(judge: Judge, state: dict, candidate: TrapCandidate,
                          profile: ReconProfile | None = None) -> TrapVerdict:
    """Score one candidate with the five trap sub-agents, take the Head's call, then ask the desk."""
    p = profile or LIVE
    side = BEARISH if candidate.side == BULL_TRAP else BULLISH
    items = _items(candidate)
    trap_state = dict(state)
    trap_state["trap_candidate"] = {
        "type": candidate.label, "timeframe": candidate.timeframe, "level": candidate.level,
        "invalidation": candidate.invalidation, "evidence": list(candidate.evidence),
    }

    scored = await judge.score(trap_state, side, items=items, profile=p)
    head = await judge.classify(
        trap_state, "trap_call",
        f"A {candidate.label} may be forming on the {candidate.timeframe} chart at {candidate.level:,.0f}. "
        f"The evidence is: {'; '.join(candidate.evidence)}. Judge what this move actually is over the next "
        f"{p.horizon}.", TRAP_CHOICES)

    notes: list[str] = []
    if not scored.ok:
        notes.append(f"The trap sub-agents did not answer ({scored.error or 'no probabilities'}); "
                     "the candidate stands on its deterministic evidence only.")
        return TrapVerdict(candidate=candidate, side_scored=side, status="unjudged", notes=tuple(notes),
                           provider=judge.provider, desk=desk_view(state, candidate, p))

    domains = tuple(
        DomainPoints(key=k, title=DOMAIN_BY_KEY[k].title,
                     points=POINTS_PER_DOMAIN * max(0.0, min(1.0, scored.probabilities.get(f"trap_{k}", 0.0))),
                     probability=float(scored.probabilities.get(f"trap_{k}", 0.0)))
        for k in DOMAIN_KEYS
    )
    score = sum(d.points for d in domains)
    view = desk_view(state, candidate, p)

    if not head.ok:
        notes.append(f"The TRAP Head did not answer ({head.error or 'no call'}).")
    if view.reads == "authentic break":
        notes.append(f"The {p.label} desk reads this as an authentic break, so it stays a watch: "
                     + "; ".join(view.reasons))
    if score < DECLARE_POINTS:
        notes.append(f"Sub-agents score {score:.0f}/100, under the {DECLARE_POINTS:.0f} needed to declare.")

    declared = (score >= DECLARE_POINTS and head.ok and head.choice == "ENGINEERED_TRAP"
                and view.reads != "authentic break")
    return TrapVerdict(candidate=candidate, side_scored=side, score=score, domains=domains,
                       head_call=head.choice, head_confidence=head.confidence, desk=view,
                       status="active" if declared else "watching", notes=tuple(notes),
                       provider=judge.provider)


def record(store, verdict: TrapVerdict, category: str, now_ms: int) -> int | None:
    """Anchor a declared trap to its category. Returns the trap id, or None when it is only a watch."""
    if not verdict.declared:
        return None
    c = verdict.candidate
    return store.put_trap(category=category, side=c.side, timeframe=c.timeframe, level=c.level,
                          invalidation=c.invalidation, plays_out=c.plays_out, declared_ms=now_ms,
                          score=verdict.score, head_call=verdict.head_call or "",
                          evidence=list(c.evidence), notes=list(verdict.notes))


def settle(store, price: float, now_ms: int) -> list[dict]:
    """Resolve anchored traps that price has now settled. Nothing expires on a timer."""
    done = []
    for row in store.traps(status="active"):
        c = TrapCandidate(side=row["side"], timeframe=row["timeframe"], level=row["level"],
                          invalidation=row["invalidation"], plays_out=row["plays_out"], strength=0)
        outcome = c.resolved_by(price)
        if outcome and store.resolve_trap(row["id"], outcome, price, now_ms):
            done.append({**row, "status": outcome, "resolved_price": price})
    return done
