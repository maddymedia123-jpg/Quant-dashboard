"""The desk scoring rubric: the client's 100-point checklist as data.

Five sub-agents per desk, as the enhanced spec names them - Quant/Statistics, Auction Market & Volume
Profile, Order Flow & Delta, ICT & Liquidity, On-Chain & Derivatives - each worth 20 points. Macro and
news moved out to the global engine in that spec, so they are no longer a desk sub-agent.

Each item is a yes/no judgment about the market state, asked from one side's point of view, and every
one of them is answerable from evidence the deterministic layer actually computes: the volatility
matrix, the volume profile, cumulative delta, the SMC reads, and the derivatives feeds. The model only
judges whether the condition holds; the weights and the arithmetic stay here in code, so a team score
can be audited line by line and a weight can change without re-running any inference.

Item ids are the accuracy report's memory. RUBRIC_VERSION changes whenever they do, so scores from an
older checklist are never compared against items that did not exist when they were made."""
from __future__ import annotations

from dataclasses import dataclass

from core.agents.recon_profiles import LIVE, ReconProfile

BULLISH, BEARISH = "bullish", "bearish"

# 1: SMC / Liquidity / MTF / Quant / Macro. 2: the spec's five sub-agents.
RUBRIC_VERSION = 2


@dataclass(frozen=True)
class RubricItem:
    id: str
    weight: int
    bullish: str   # the condition, phrased for the bullish case
    bearish: str   # same condition, phrased for the bearish case

    def question(self, side: str, profile: "ReconProfile | None" = None) -> str:
        text = self.bullish if side == BULLISH else self.bearish
        return text.format(**(profile or LIVE).fmt)


@dataclass(frozen=True)
class Domain:
    key: str
    agent_no: int
    title: str
    checklist: tuple[str, ...]      # diagnostic steps, templated on {ltf}/{mtf}/{htf}/{horizon}
    items: tuple[RubricItem, ...]

    @property
    def max_points(self) -> int:
        return sum(i.weight for i in self.items)

    def steps(self, profile: "ReconProfile | None" = None) -> tuple[str, ...]:
        """The diagnostic checklist, filled in for one category's timeframes."""
        fmt = (profile or LIVE).fmt
        return tuple(step.format(**fmt) for step in self.checklist)


DOMAINS: tuple[Domain, ...] = (
    Domain(
        key="quant", agent_no=1, title="Quant & Statistics",
        checklist=(
            "Read the Hurst exponent: does this market persist or mean-revert right now?",
            "Compare current ATR with its own recent distribution.",
            "Check where price sits in the 1, 2 and 3 sigma channels.",
            "Verify trend agreement across {ltf}, {mtf} and {htf}, and RSI for momentum or divergence.",
        ),
        items=(
            RubricItem("quant_mtf", 5,
                       "All three timeframes ({ltf}, {mtf}, {htf}) point UP.",
                       "All three timeframes ({ltf}, {mtf}, {htf}) point DOWN."),
            RubricItem("quant_memory", 5,
                       "The Hurst reading shows trending memory rather than mean reversion, so an upside "
                       "move should persist over the next {horizon}.",
                       "The Hurst reading shows trending memory rather than mean reversion, so a downside "
                       "move should persist over the next {horizon}."),
            RubricItem("quant_bands", 4,
                       "Price is working the upper sigma channel in a way that favours continuation higher "
                       "rather than exhaustion.",
                       "Price is working the lower sigma channel in a way that favours continuation lower "
                       "rather than exhaustion."),
            RubricItem("quant_ema", 3,
                       "The EMA ribbon is stacked bullish on both the {mtf} and {htf}.",
                       "The EMA ribbon is stacked bearish on both the {mtf} and {htf}."),
            RubricItem("quant_momentum", 3,
                       "RSI momentum supports upside and is not contradicted by a bearish divergence.",
                       "RSI momentum supports downside and is not contradicted by a bullish divergence."),
        ),
    ),
    Domain(
        key="auction", agent_no=2, title="Auction Market & Volume Profile",
        checklist=(
            "Locate the point of control and the value area high and low.",
            "Decide whether price is accepting or rejecting value at its current position.",
            "Identify which way the auction is rotating inside the range.",
            "Look for excess or rejection at the far edge of value.",
        ),
        items=(
            RubricItem("auction_value", 6,
                       "Price is accepting value at or above the value area high rather than being rejected "
                       "from it.",
                       "Price is accepting value at or below the value area low rather than being rejected "
                       "from it."),
            RubricItem("auction_poc", 5,
                       "The point of control is acting as support beneath price.",
                       "The point of control is acting as resistance above price."),
            RubricItem("auction_rotation", 5,
                       "The auction is rotating up from the value area low towards the point of control or "
                       "the value area high.",
                       "The auction is rotating down from the value area high towards the point of control "
                       "or the value area low."),
            RubricItem("auction_excess", 4,
                       "There is excess or rejection at the low end of value, marking sellers as finished "
                       "there.",
                       "There is excess or rejection at the high end of value, marking buyers as finished "
                       "there."),
        ),
    ),
    Domain(
        key="delta", agent_no=3, title="Order Flow & Delta",
        checklist=(
            "Compare cumulative delta with price over the recent window.",
            "Decide whether either side is absorbing rather than driving.",
            "Read taker flow for who is paying the spread.",
            "Note that delta here is a candle close-position proxy, not tick tape.",
        ),
        items=(
            RubricItem("delta_confirm", 8,
                       "Cumulative delta is rising with price, so buyers are driving rather than chasing.",
                       "Cumulative delta is falling with price, so sellers are driving rather than chasing."),
            RubricItem("delta_absorption", 7,
                       "Delta shows bullish absorption: price held or fell while delta rose, so sellers are "
                       "being absorbed.",
                       "Delta shows bearish absorption: price held or rose while delta fell, so buyers are "
                       "being absorbed."),
            RubricItem("delta_taker", 5,
                       "Taker flow leans to the buy side, paying the spread to get long.",
                       "Taker flow leans to the sell side, paying the spread to get short."),
        ),
    ),
    Domain(
        key="ict", agent_no=4, title="ICT & Liquidity",
        checklist=(
            "Identify the {htf} structural state and the most recent break of structure or change of "
            "character.",
            "Locate buyside and sellside liquidity pools and whether either was swept and reclaimed.",
            "Find the nearest unmitigated order block and any unfilled fair value gap.",
            "Place price in premium or discount of the dealing range.",
        ),
        items=(
            RubricItem("ict_structure", 6,
                       "A confirmed {htf} break of structure or change of character points UP.",
                       "A confirmed {htf} break of structure or change of character points DOWN."),
            RubricItem("ict_sweep", 5,
                       "A sweep of sellside liquidity was reclaimed, leaving a swing failure that favours "
                       "upside.",
                       "A sweep of buyside liquidity was reclaimed, leaving a swing failure that favours "
                       "downside."),
            RubricItem("ict_ob", 4,
                       "Price is testing or reacting to an unmitigated {mtf}/{htf} demand order block.",
                       "Price is testing or reacting to an unmitigated {mtf}/{htf} supply order block."),
            RubricItem("ict_fvg", 3,
                       "A clear unfilled fair value gap sits ABOVE price as an upside magnet.",
                       "A clear unfilled fair value gap sits BELOW price as a downside magnet."),
            RubricItem("ict_pd", 2,
                       "Price is in discount of the dealing range, where longs are priced well.",
                       "Price is in premium of the dealing range, where shorts are priced well."),
        ),
    ),
    Domain(
        key="derivs", agent_no=5, title="On-Chain & Derivatives",
        checklist=(
            "Read funding against its own recent mean.",
            "Track the net change in open interest over {ltf}, {mtf} and {htf}.",
            "Compare long and short liquidations over the last day.",
            "Check options positioning: max pain and the put/call skew.",
        ),
        items=(
            RubricItem("derivs_funding", 5,
                       "Funding is not overheated on the long side, so an upside move is not crowded.",
                       "Funding is not overheated on the short side, so a downside move is not crowded."),
            RubricItem("derivs_oi", 5,
                       "Open interest is expanding in a way that indicates aggressive long positioning.",
                       "Open interest is expanding in a way that indicates aggressive short positioning."),
            RubricItem("derivs_liquidations", 4,
                       "Recent liquidations flushed longs rather than shorts, clearing the way higher.",
                       "Recent liquidations flushed shorts rather than longs, clearing the way lower."),
            RubricItem("derivs_options", 3,
                       "Options positioning (max pain and skew) sits above price, pulling it up.",
                       "Options positioning (max pain and skew) sits below price, pulling it down."),
            RubricItem("derivs_ls", 3,
                       "The long/short account ratio is not stretched long, leaving room to squeeze up.",
                       "The long/short account ratio is not stretched short, leaving room to squeeze down."),
        ),
    ),
)

DOMAIN_BY_KEY = {d.key: d for d in DOMAINS}
ALL_ITEMS = tuple(i for d in DOMAINS for i in d.items)
MAX_TEAM_POINTS = sum(d.max_points for d in DOMAINS)


def domain_score(domain: Domain, probabilities: dict[str, float]) -> float:
    """Points for one domain: each item's weight times the probability its condition holds."""
    return sum(i.weight * max(0.0, min(1.0, probabilities.get(i.id, 0.0))) for i in domain.items)


def team_score(probabilities: dict[str, float]) -> float:
    return sum(domain_score(d, probabilities) for d in DOMAINS)


def missing_items(probabilities: dict[str, float]) -> list[str]:
    return [i.id for i in ALL_ITEMS if i.id not in probabilities]


def verdict(points: float) -> str:
    """The client's reading of the 100-point scale."""
    if points > 70:
        return "high confluence"
    if points < 50:
        return "neutral or conflicting"
    return "moderate"
