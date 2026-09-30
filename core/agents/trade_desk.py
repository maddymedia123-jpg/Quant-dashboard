"""The Active Trade desk: ten sub-agents, a Head, and both handshakes the spec asks for.

The spec's shape, kept faithfully:

* Five bullish and five bearish Active Trade sub-agents evaluate the position against the same five
  domains the war rooms use. Both sides are asked, because a trade has a case for it and a case against
  it and the honest score is the difference - one batched request per side.
* The Head then makes one categorical call on the trade itself: take it, wait, or stand aside.
* *Category consultation*: alignment with the Live Recon, Intraday, Weekly and Monthly heads. This reads
  each desk's own published anchor rather than asking the model again - the desks have already voted, and
  re-asking would both cost more and risk a different answer to the one on screen.
* *TRAP consultation*: whether the entry, the stop or any target sits on an active trap's level. Also
  deterministic, against the anchored trap ledger.

The score and the probability are arithmetic here. The model judges conditions; it is never asked for a
number that the card then presents as a probability."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.agents.judge import Judge
from core.agents.recon_profiles import PROFILES, ReconProfile
from core.agents.rubric import BEARISH, BULLISH, DOMAIN_BY_KEY, RubricItem
from core.trades import LONG, StopVerdict, TimedTarget

log = logging.getLogger(__name__)

DOMAIN_KEYS = ("quant", "auction", "delta", "ict", "derivs")
POINTS_PER_DOMAIN = 20
MAX_POINTS = POINTS_PER_DOMAIN * len(DOMAIN_KEYS)
# Below this the case for the trade is not there; the Head is still asked, so the card can say why.
TAKE_POINTS = 60.0
# A trap counts as sitting on the trade when its level is within this fraction of an ATR of a price the
# trade depends on. Half an ATR is the same distance the trap desk invalidates on.
TRAP_NEAR_ATR = 0.5

TRADE_CHOICES = {
    "TAKE": "The framework supports the position as entered: the evidence is behind it and the levels "
            "are where they need to be.",
    "WAIT": "The idea is sound but this entry is not: price, timing or the stop need to be different.",
    "STAND_ASIDE": "The framework is against this position; taking it means trading against the weight "
                   "of the evidence.",
}

# Each domain asks the same question of two opposed directions, the way the war-room rubric does with
# "point UP" and "point DOWN". That mirroring is what makes the two team scores a case FOR and a case
# AGAINST rather than two readings of the same proposition - see _items for why it has to be direction
# and not "the trade" that flips.
DOMAIN_QUESTIONS: dict[str, tuple[str, str]] = {
    "quant": (
        "Volatility and timeframe statistics favour upside from {entry:,.0f} over {horizon}: the regime "
        "and the sigma channels leave room to travel higher rather than being stretched above value.",
        "Volatility and timeframe statistics favour downside from {entry:,.0f} over {horizon}: the regime "
        "and the sigma channels leave room to travel lower rather than being stretched below value."),
    "auction": (
        "The auction favours upside from {entry:,.0f}: value is being accepted higher, and attempts to "
        "trade below it are being rejected.",
        "The auction favours downside from {entry:,.0f}: value is being accepted lower, and attempts to "
        "trade above it are being rejected."),
    "delta": (
        "Order flow favours upside from {entry:,.0f}: delta is confirming buying rather than being "
        "absorbed by sellers.",
        "Order flow favours downside from {entry:,.0f}: delta is confirming selling rather than being "
        "absorbed by buyers."),
    "ict": (
        "Liquidity and structure favour upside from {entry:,.0f}: structure is bullish on the higher "
        "timeframe, and the liquidity price is most likely to reach for next sits above.",
        "Liquidity and structure favour downside from {entry:,.0f}: structure is bearish on the higher "
        "timeframe, and the liquidity price is most likely to reach for next sits below."),
    "derivs": (
        "Positioning favours upside from {entry:,.0f}: funding, open interest and liquidations leave the "
        "crowd short and the fuel for a squeeze higher.",
        "Positioning favours downside from {entry:,.0f}: funding, open interest and liquidations leave "
        "the crowd long and the fuel for a squeeze lower."),
}


@dataclass(frozen=True)
class DomainPoints:
    key: str
    title: str
    points: float
    probability: float


@dataclass(frozen=True)
class DeskAlignment:
    """One category head's answer when the Active Trade Head asks."""
    category: str
    label: str
    bias: str
    verdict: str             # supports | opposes | neutral | silent
    confidence: float | None = None
    stale: bool = False
    note: str = ""


@dataclass(frozen=True)
class TrapProximity:
    trap_id: int
    side: str
    timeframe: str
    level: float
    near: str                # entry | stop | TP1 | TP2 | TP3
    distance: float


@dataclass
class TradeVerdict:
    side: str
    for_points: float = 0.0
    against_points: float = 0.0
    for_domains: tuple[DomainPoints, ...] = ()
    against_domains: tuple[DomainPoints, ...] = ()
    head_call: str | None = None
    head_confidence: float | None = None
    alignment: tuple[DeskAlignment, ...] = ()
    traps: tuple[TrapProximity, ...] = ()
    notes: tuple[str, ...] = ()
    provider: str = ""
    ok: bool = True

    @property
    def score(self) -> float:
        """The trade's score out of 100: the case for it, net of the case against, rebased."""
        return round(max(0.0, min(100.0, 50.0 + (self.for_points - self.against_points) / 2.0)), 1)

    @property
    def margin(self) -> float:
        return round(self.for_points - self.against_points, 1)

    @property
    def probability(self) -> float:
        """A confluence-weighted estimate, NOT a backtested frequency.

        It is built from the agents' own probabilities and the desks' alignment, and it is deliberately
        kept away from the extremes: nothing here measures how often a setup like this has worked, so a
        number near 0 or 100 would be a lie about what is known. The card says as much."""
        base = self.score / 100.0
        supporting = sum(1 for a in self.alignment if a.verdict == "supports")
        opposing = sum(1 for a in self.alignment if a.verdict == "opposes")
        tilt = 0.04 * (supporting - opposing)
        if self.traps:
            tilt -= 0.05
        return round(max(0.15, min(0.85, base + tilt)) * 100, 1)

    @property
    def aligned(self) -> str:
        counts = {"supports": 0, "opposes": 0, "neutral": 0, "silent": 0}
        for a in self.alignment:
            counts[a.verdict] = counts.get(a.verdict, 0) + 1
        if counts["supports"] and not counts["opposes"]:
            return "with the higher timeframes"
        if counts["opposes"] and not counts["supports"]:
            return "against the higher timeframes"
        if counts["opposes"] and counts["supports"]:
            return "the desks disagree"
        return "the desks have no live verdict"


def _items(entry: float, horizon: str) -> tuple[RubricItem, ...]:
    """One question per domain, asked of both directions.

    A `RubricItem` carries two phrasings and `judge.score` picks by side: the bullish team is asked the
    first, the bearish team the second. Mirroring on *direction* therefore gives the for-and-against
    symmetry for free, on both sides of the market:

        LONG   supporting = bullish -> "favours upside"   = the case FOR the trade
               opposing   = bearish -> "favours downside" = the case AGAINST it
        SHORT  supporting = bearish -> "favours downside" = the case FOR the trade
               opposing   = bullish -> "favours upside"   = the case AGAINST it

    The earlier version handed the *same* string to both teams, phrased in favour of the position, so
    `against_points` was the opposing persona's probability that the trade was supported - not the case
    against it. The score, being their difference, measured disagreement between two prompt preambles
    rather than for-versus-against, which meant a strong trade and a hopeless one could both read 50.

    The stop is deliberately not in the question text: it is side-specific, so putting it there would
    break the mirror. The judge gets it in `state["active_trade"]` instead, along with the targets."""
    out = []
    for key in DOMAIN_KEYS:
        up, down = DOMAIN_QUESTIONS[key]
        out.append(RubricItem(f"trade_{key}", POINTS_PER_DOMAIN,
                              up.format(entry=entry, horizon=horizon),
                              down.format(entry=entry, horizon=horizon)))
    return tuple(out)


def _domains(probabilities: dict[str, float]) -> tuple[DomainPoints, ...]:
    return tuple(
        DomainPoints(key=k, title=DOMAIN_BY_KEY[k].title,
                     points=POINTS_PER_DOMAIN * max(0.0, min(1.0, probabilities.get(f"trade_{k}", 0.0))),
                     probability=float(probabilities.get(f"trade_{k}", 0.0)))
        for k in DOMAIN_KEYS)


def category_alignment(store, side: str, now_ms: int) -> tuple[DeskAlignment, ...]:
    """The inbound handshake with the four category heads, read from what they have already published.

    A desk whose anchored candle has closed is reported as stale rather than counted: an anchor from a
    window that ended hours ago is not that desk's current view, and treating it as one would let an old
    verdict vote on a new trade."""
    long = side == LONG
    out: list[DeskAlignment] = []
    for category, profile in PROFILES.items():
        try:
            anchor = store.get_live_anchor(category=category)
        except Exception as e:  # noqa: BLE001 - a desk we cannot read is silent, not absent
            log.warning("category alignment: %s unreadable (%s)", category, e)
            out.append(DeskAlignment(category, profile.label, "", "silent", note=f"ledger unreadable: {e}"))
            continue
        if not anchor:
            out.append(DeskAlignment(category, profile.label, "", "silent",
                                     note="no anchored verdict yet; run its war room"))
            continue
        bias = str(anchor.get("bias") or "")
        stale = bool(anchor.get("window_close_ms") and anchor["window_close_ms"] <= now_ms)
        # A trap-risk verdict inverts the desk's direction rather than cancelling it: BULL TRAP RISK
        # means the desk expects the up-move to fail, so it is a bearish read - it opposes a long and
        # supports a short. Collapsing both to "opposes" reported the desk as against the very trade it
        # favoured, and tilted the probability the wrong way.
        leaning_up = ("BULL" in bias) != ("TRAP RISK" in bias)
        directional = ("BULL" in bias) or ("BEAR" in bias)
        supports = directional and (leaning_up == long)
        opposes = directional and (leaning_up != long)
        verdict = "silent" if stale else ("supports" if supports else "opposes" if opposes else "neutral")
        out.append(DeskAlignment(category, profile.label, bias, verdict,
                                 anchor.get("confidence"), stale,
                                 "its candle has closed, so this verdict is not current" if stale else ""))
    return tuple(out)


def trap_proximity(store, side: str, entry: float, stop: float, targets: list[TimedTarget],
                   atr: float) -> tuple[TrapProximity, ...]:
    """The inbound handshake with the TRAP desk: does an anchored trap sit on a price this trade needs?"""
    try:
        rows = store.traps(status="active", limit=50)
    except Exception as e:  # noqa: BLE001
        log.warning("trap consultation unavailable: %s", e)
        return ()
    watch = [("entry", entry), ("stop", stop)] + [(t.name, t.price) for t in targets]
    window = max(atr * TRAP_NEAR_ATR, 1.0)
    out: list[TrapProximity] = []
    for row in rows:
        try:
            level = float(row["level"])
        except (TypeError, ValueError, KeyError):
            continue
        for name, price in watch:
            if abs(level - price) <= window:
                out.append(TrapProximity(int(row.get("id") or 0), str(row.get("side") or ""),
                                         str(row.get("timeframe") or ""), level, name,
                                         round(abs(level - price), 2)))
                break
    return tuple(out)


async def judge_trade(judge: Judge, state: dict, side: str, entry: float, stop: StopVerdict,
                      targets: list[TimedTarget], profile: ReconProfile, store=None,
                      now_ms: int = 0, atr: float = 0.0) -> TradeVerdict:
    """Both teams on the position, then the Head's call, then the two handshakes."""
    trade_state = dict(state)
    trade_state["active_trade"] = {
        "side": side, "entry": entry, "stop": stop.stop, "stop_verdict": stop.verdict,
        "targets": [{"name": t.name, "price": t.price, "due": t.when, "reward_r": t.reward_r}
                    for t in targets],
        "horizon": profile.horizon,
    }
    items = _items(entry, profile.horizon)
    supporting = BULLISH if side == LONG else BEARISH
    opposing = BEARISH if side == LONG else BULLISH

    for_res = await judge.score(trade_state, supporting, items=items, profile=profile)
    against_res = await judge.score(trade_state, opposing, items=items, profile=profile)

    notes: list[str] = []
    alignment = category_alignment(store, side, now_ms) if store is not None else ()
    traps = trap_proximity(store, side, entry, stop.stop, targets, atr) if store is not None else ()

    if not for_res.ok or not against_res.ok:
        problem = for_res.error or against_res.error or "no probabilities"
        notes.append(f"The trade sub-agents did not answer ({problem}); the card stands on its "
                     "deterministic levels only, and carries no score.")
        return TradeVerdict(side=side, alignment=alignment, traps=traps, notes=tuple(notes),
                            provider=judge.provider, ok=False)

    for_domains, against_domains = _domains(for_res.probabilities), _domains(against_res.probabilities)
    for_points = sum(d.points for d in for_domains)
    against_points = sum(d.points for d in against_domains)

    head = await judge.classify(
        trade_state, "trade_call",
        f"A {side.lower()} is open from {entry:,.0f} with its stop at {stop.stop:,.0f} and targets at "
        + ", ".join(f"{t.price:,.0f} by {t.when}" for t in targets)
        + f". The case for it scores {for_points:.0f}/100 and the case against it {against_points:.0f}/100."
        + f" Judge the position over {profile.horizon}.", TRADE_CHOICES)

    if not head.ok:
        notes.append(f"The Active Trade Head did not answer ({head.error or 'no call'}).")
    if for_points < TAKE_POINTS:
        notes.append(f"The case for the trade scores {for_points:.0f}/100, under the "
                     f"{TAKE_POINTS:.0f} the framework asks for.")
    if against_points >= TAKE_POINTS and for_points >= TAKE_POINTS:
        notes.append("Both cases score above 60: the evidence supports the trade and its opposite "
                     "equally well, which is the condition a trap is built in.")
    if not stop.sound and stop.given is not None:
        notes.append(f"The stop as given is {stop.verdict}; the score above is for the position with the "
                     f"corrected stop at {stop.stop:,.0f}.")
    for t in traps:
        notes.append(f"An anchored {t.side.replace('_', ' ')} on the {t.timeframe} sits {t.distance:,.0f} "
                     f"from this trade's {t.near} at {t.level:,.0f}.")
    for a in alignment:
        if a.verdict == "opposes":
            notes.append(f"The {a.label} desk reads {a.bias}, against this trade.")

    return TradeVerdict(side=side, for_points=for_points, against_points=against_points,
                        for_domains=for_domains, against_domains=against_domains,
                        head_call=head.choice, head_confidence=head.confidence,
                        alignment=alignment, traps=traps, notes=tuple(notes), provider=judge.provider)
