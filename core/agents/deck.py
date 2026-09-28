"""The Institutional Quantitative Matrix: each sub-agent's readings, taken from the judges' own state.

The enhanced spec puts an expandable deck on every category tab showing what all five sub-agents saw.
Building it from the same state dictionary the judges are handed means the screen and the score can
never disagree: if a number is not in the state, the agent could not have used it, and the deck shows
it empty rather than filling the gap.

Each row is (label, value, kind) - the kind says how the value should be rendered, so the formatting
stays in the UI while the choice of evidence stays here."""
from __future__ import annotations

from dataclasses import dataclass

from core.agents.recon_profiles import LIVE, ReconProfile
from core.agents.rubric import DOMAIN_BY_KEY

Row = tuple[str, object, str]      # label, value (or None), kind: pct | usd | num | text


@dataclass(frozen=True)
class DeckAgent:
    agent_no: int
    key: str
    title: str
    timeframe: str
    rows: tuple[Row, ...]
    note: str = ""


def _get(d: dict | None, *path, default=None):
    cur = d or {}
    for step in path:
        if not isinstance(cur, dict) or step not in cur:
            return default
        cur = cur[step]
    return cur if cur is not None else default


def sub_agent_deck(state: dict, profile: ReconProfile | None = None) -> list[DeckAgent]:
    """Five agents, in the spec's order, read off one category's state."""
    p = profile or LIVE
    ltf, mtf, htf = p.timeframes
    vol = _get(state, "volatility", mtf, default={})
    liq = _get(state, "liquidity", mtf, default={})
    smc = _get(state, "smc", htf, default={})          # structure is read on the higher timeframe
    fut = _get(state, "derivatives", "futures", default={})
    opts = _get(state, "derivatives", "options", default={})
    liqs = _get(state, "derivatives", "liquidations", default={})
    missing = list(_get(state, "unavailable", default=[]) or [])

    def agent(key: str, timeframe: str, rows: list[Row], note: str = "") -> DeckAgent:
        d = DOMAIN_BY_KEY[key]
        return DeckAgent(d.agent_no, d.key, d.title, timeframe, tuple(rows), note)

    pool_b, pool_s = _get(liq, "pools", "buyside", default={}), _get(liq, "pools", "sellside", default={})
    ob = _get(smc, "order_blocks", "nearest_demand") or _get(smc, "order_blocks", "nearest_supply") or {}
    gap = _get(smc, "fair_value_gaps", "unfilled_above") or _get(smc, "fair_value_gaps", "unfilled_below") or {}
    last = _get(smc, "structure", "last_event", default={})

    deck = [
        agent("quant", mtf, [
            ("Hurst", _get(vol, "hurst"), "num"),
            ("Memory", _get(vol, "memory", default="—"), "text"),
            ("Regime", _get(vol, "regime", default="—"), "text"),
            ("Realized vol", _get(vol, "realized_vol_pct"), "pct"),
            ("Implied vol", _get(vol, "implied_vol_pct"), "pct"),
            ("3 sigma channel", _get(vol, "bands", "sigma3_up"), "usd"),
        ], note=_get(vol, "note", default="")),
        agent("auction", mtf, [
            ("Point of control", _get(liq, "volume_profile", "point_of_control"), "usd"),
            ("Value area high", _get(liq, "volume_profile", "value_area_high"), "usd"),
            ("Value area low", _get(liq, "volume_profile", "value_area_low"), "usd"),
            ("Price vs value", _get(liq, "volume_profile", "price_position", default="—"), "text"),
        ]),
        agent("delta", mtf, [
            ("Delta read", _get(liq, "cumulative_delta", "state", default="—"), "text"),
            ("Delta, last 20 bars", _get(liq, "cumulative_delta", "change_over_20_bars"), "num"),
            ("Price, last 20 bars", _get(liq, "cumulative_delta", "price_change_over_20_bars"), "usd"),
            ("Taker buy/sell", _get(fut, "taker_buy_sell_ratio"), "num"),
        ], note="delta is a candle close-position proxy, not tick tape"),
        agent("ict", htf, [
            ("Structure", _get(smc, "structure", "bias", default="unclear"), "text"),
            ("Last break", f"{_get(last, 'type', default='—')} {_get(last, 'direction', default='')}".strip(), "text"),
            ("Nearest block", _get(ob, "low"), "usd"),
            ("Unfilled gap", _get(gap, "low"), "usd"),
            ("Range position", _get(smc, "premium_discount", "zone", default="—"), "text"),
            ("Buyside pool", _get(pool_b, "price"), "usd"),
            ("Sellside pool", _get(pool_s, "price"), "usd"),
        ]),
        agent("derivs", mtf, [
            ("Funding", _get(fut, "funding_rate"), "rate"),
            ("Funding 7d mean", _get(fut, "funding_7d_mean"), "rate"),
            ("Open interest 24h", _get(fut, "oi_change_24h_pct"), "pct"),
            ("Long/short ratio", _get(fut, "long_short_ratio"), "num"),
            ("Longs liquidated 24h", _get(liqs, "long_usd"), "usd"),
            ("Shorts liquidated 24h", _get(liqs, "short_usd"), "usd"),
            ("Options max pain", _get(opts, "max_pain"), "usd"),
            ("IV skew", _get(opts, "iv_skew"), "pct"),
        ], note=("no data from " + ", ".join(missing)) if missing else ""),
    ]
    return deck
