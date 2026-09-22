"""Live Recon scoring rubric: the client's 100-point checklist as data.

Five domain agents per side, each worth 20 points, split into weighted checklist items. Each item is a
yes/no judgment about the market state, asked from one side's point of view (bullish or bearish). The
model only judges whether each condition holds; the weights and the arithmetic stay here in code, so a
team score can be audited line by line and weights can change without re-running any inference."""
from __future__ import annotations

from dataclasses import dataclass

from core.agents.recon_profiles import LIVE, ReconProfile

BULLISH, BEARISH = "bullish", "bearish"


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
        key="smc", agent_no=1, title="Market Structure & SMC",
        checklist=(
            "Identify the {htf} structural state: expansion, retracement or consolidation.",
            "Locate the nearest unmitigated premium/discount order block.",
            "Map active fair value gaps on the {mtf} and {ltf} charts.",
            "Verify whether a change of character occurred within the last 1-3 candles on {ltf}.",
        ),
        items=(
            RubricItem("smc_bos", 5,
                       "A confirmed {htf} break of structure or change of character points UP.",
                       "A confirmed {htf} break of structure or change of character points DOWN."),
            RubricItem("smc_ob", 5,
                       "Price is testing or reacting to an unmitigated {mtf}/{htf} demand order block.",
                       "Price is testing or reacting to an unmitigated {mtf}/{htf} supply order block."),
            RubricItem("smc_fvg", 5,
                       "A clear unfilled fair value gap sits ABOVE price as an upside magnet or entry zone.",
                       "A clear unfilled fair value gap sits BELOW price as a downside magnet or entry zone."),
            RubricItem("smc_ltf", 5,
                       "The {ltf} structure is clean, making higher highs and higher lows.",
                       "The {ltf} structure is clean, making lower highs and lower lows."),
        ),
    ),
    Domain(
        key="liquidity", agent_no=2, title="Liquidity & Order Flow",
        checklist=(
            "Locate buyside and sellside liquidity targets.",
            "Confirm whether CVD is expanding with price or showing absorption or divergence.",
            "Identify the point of control and value area limits.",
            "Track the net change in open interest over {ltf}, {mtf} and {htf}.",
        ),
        items=(
            RubricItem("liq_sweep", 5,
                       "A recent sweep of equal lows or the previous day low was rejected, favouring upside.",
                       "A recent sweep of equal highs or the previous day high was rejected, favouring downside."),
            RubricItem("liq_cvd", 5,
                       "Cumulative volume delta confirms buying, or shows bullish absorption against falling price.",
                       "Cumulative volume delta confirms selling, or shows bearish absorption against rising price."),
            RubricItem("liq_profile", 5,
                       "Price sits where the volume profile supports upside: above value or holding a high volume node.",
                       "Price sits where the volume profile supports downside: below value or rejected at a high volume node."),
            RubricItem("liq_oi", 5,
                       "Open interest is expanding in a way that indicates aggressive long positioning.",
                       "Open interest is expanding in a way that indicates aggressive short positioning."),
        ),
    ),
    Domain(
        key="mtf", agent_no=3, title="Multi-Timeframe Alignment",
        checklist=(
            "Verify the {htf} trend direction.",
            "Verify the {mtf} trend direction.",
            "Verify the {ltf} trend direction.",
            "Measure how much of the three timeframes agree.",
        ),
        items=(
            RubricItem("mtf_triple", 8,
                       "All three timeframes ({ltf}, {mtf}, {htf}) point UP.",
                       "All three timeframes ({ltf}, {mtf}, {htf}) point DOWN."),
            RubricItem("mtf_double", 6,
                       "The {mtf} and {htf} point UP while the {ltf} pulls back into discount.",
                       "The {mtf} and {htf} point DOWN while the {ltf} pulls back into premium."),
            RubricItem("mtf_fib", 6,
                       "Price is at a Fibonacci retracement or time zone that supports an upside turn.",
                       "Price is at a Fibonacci retracement or time zone that supports a downside turn."),
        ),
    ),
    Domain(
        key="quant", agent_no=4, title="Quantitative Volatility",
        checklist=(
            "Compare current ATR with its 20-period average.",
            "Verify EMA alignment on the {mtf} and {htf}.",
            "Check the standard deviation band boundaries.",
            "Audit RSI and StochRSI for momentum locks or hidden divergences.",
        ),
        items=(
            RubricItem("quant_bands", 5,
                       "Price is pushing the upper volatility band in a way that favours continuation higher.",
                       "Price is pushing the lower volatility band in a way that favours continuation lower."),
            RubricItem("quant_ema", 5,
                       "The EMA ribbon is stacked bullish on both the {mtf} and {htf}.",
                       "The EMA ribbon is stacked bearish on both the {mtf} and {htf}."),
            RubricItem("quant_momentum", 5,
                       "RSI momentum supports upside and is not contradicted by a bearish divergence.",
                       "RSI momentum supports downside and is not contradicted by a bullish divergence."),
            RubricItem("quant_atr", 5,
                       "ATR is expanding, confirming an upside move rather than quiet consolidation.",
                       "ATR is expanding, confirming a downside move rather than quiet consolidation."),
        ),
    ),
    Domain(
        key="macro", agent_no=5, title="Macro & Financial News",
        checklist=(
            "Scan the economic calendar for tier-1 releases in the next {ltf}, {mtf} and {htf}.",
            "Read sentiment indicators for market-moving shifts.",
            "Assess broader market positioning and funding conditions.",
            "Flag any high-impact event landing mid-candle.",
        ),
        items=(
            RubricItem("macro_release", 8,
                       "The nearest high-impact release or its outcome favours upside.",
                       "The nearest high-impact release or its outcome favours downside."),
            RubricItem("macro_sentiment", 6,
                       "Current sentiment and positioning support upside over the next {horizon}.",
                       "Current sentiment and positioning support downside over the next {horizon}."),
            RubricItem("macro_clear", 6,
                       "No imminent high-impact event threatens the upside case inside the next {horizon}.",
                       "No imminent high-impact event threatens the downside case inside the next {horizon}."),
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
