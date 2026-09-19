"""Smart Money Concepts on candles, built from hand-made series where the answer is known.

Every fixture here is constructed so the expected structure is unambiguous: a swing sequence with a
named break, one gap with known bounds, one order block at a known index. Nothing is asserted against
live data, so a change in market conditions cannot turn these green or red by accident."""
import pandas as pd
import pytest

from core.indicators import smc


def candles(rows) -> pd.DataFrame:
    """rows: (open, high, low, close[, volume]) -> a frame shaped like the market data."""
    out = []
    for i, r in enumerate(rows):
        o, h, l, c = r[:4]
        v = r[4] if len(r) > 4 else 1.0
        out.append({"timestamp": 1_700_000_000_000 + i * 900_000, "open": o, "high": h, "low": l,
                    "close": c, "volume": v})
    return pd.DataFrame(out)


def flat(bars: int, price: float, vol: float = 1.0) -> list[tuple]:
    """Overlapping candles: no imbalance, no structural break - just length."""
    return [(price, price + 1.0, price - 1.0, price, vol) for _ in range(bars)]


def leg(start: float, end: float, bars: int, vol: float = 1.0) -> list[tuple]:
    """A clean directional run from start to end."""
    step = (end - start) / bars
    rows = []
    for i in range(bars):
        o = start + step * i
        c = o + step
        rows.append((o, max(o, c) + 1.0, min(o, c) - 1.0, c, vol))
    return rows


# ---------- swings ----------
def test_swings_alternate_high_and_low_in_time_order():
    df = candles(leg(100, 120, 8) + leg(120, 105, 8) + leg(105, 130, 8))
    swings = smc.swings(df, left=2, right=2)
    assert [s.kind for s in swings] == ["high", "low"] or [s.kind for s in swings][:2] == ["high", "low"]
    assert all(swings[i].index < swings[i + 1].index for i in range(len(swings) - 1))
    assert all(s.kind != swings[i + 1].kind for i, s in enumerate(swings[:-1])), "no two highs in a row"


# ---------- break of structure and change of character ----------
def test_a_higher_high_after_an_uptrend_is_a_bullish_break_of_structure():
    df = candles(leg(100, 120, 8) + leg(120, 110, 6) + leg(110, 125, 8))
    s = smc.structure(df)
    assert s.last_event is not None
    assert s.last_event.kind == "BOS" and s.last_event.direction == "bullish"
    assert s.bias == "bullish" and s.available


def test_a_break_of_the_prior_low_after_higher_highs_is_a_bearish_change_of_character():
    df = candles(leg(100, 120, 8) + leg(120, 110, 6) + leg(110, 124, 8) + leg(124, 105, 8))
    s = smc.structure(df)
    assert s.last_event.kind == "CHOCH" and s.last_event.direction == "bearish"
    assert s.bias == "bearish"
    assert s.last_event.price == pytest.approx(110.0, abs=1.5), "broke the swing low that held the uptrend"


def test_structure_reports_unavailable_rather_than_guessing_on_thin_data():
    s = smc.structure(candles(leg(100, 101, 4)))   # 4 bars, under the 20-bar floor
    assert not s.available and s.last_event is None and s.bias == "unclear"


# ---------- fair value gaps ----------
def test_a_three_candle_imbalance_is_a_fair_value_gap_with_exact_bounds():
    rows = flat(4, 103) + [(100, 102, 99, 101), (103, 112, 103, 111), (114, 116, 113, 115),
                           (115, 117, 111, 116)] + flat(4, 116)
    gaps = smc.fair_value_gaps(candles(rows))
    bull = [g for g in gaps if g.direction == "bullish"]
    assert len(bull) == 1
    g = bull[0]
    assert (g.low, g.high) == (102.0, 113.0), "gap runs from candle 1 high to candle 3 low"
    assert g.filled is False and g.index == 5


def test_a_gap_price_has_traded_back_through_is_marked_filled():
    rows = flat(4, 103) + [(100, 102, 99, 101), (103, 112, 103, 111), (114, 116, 113, 115),
                           (115, 117, 111, 116)] + leg(116, 99, 8)
    gaps = smc.fair_value_gaps(candles(rows))
    assert [g.filled for g in gaps if g.direction == "bullish"] == [True]
    assert smc.nearest_unfilled(gaps, price=116.0, direction="bullish") is None


def test_bearish_gaps_sit_above_price_and_are_found_by_nearest_unfilled():
    rows = flat(4, 119) + [(120, 121, 118, 119), (118, 118, 101, 102), (99, 100, 97, 98),
                           (98, 102, 97, 99)] + flat(4, 99)
    gaps = smc.fair_value_gaps(candles(rows))
    bear = [g for g in gaps if g.direction == "bearish"]
    assert len(bear) == 1 and (bear[0].low, bear[0].high) == (100.0, 118.0)
    near = smc.nearest_unfilled(gaps, price=99.0, direction="bearish")
    assert near is bear[0] and near.low > 99.0


# ---------- order blocks ----------
def test_the_down_candle_before_an_up_impulse_that_breaks_structure_is_a_demand_block():
    rows = leg(100, 120, 8) + leg(120, 110, 6) + [(110, 110.5, 108, 108.5)] + leg(108.5, 126, 8)
    blocks = smc.order_blocks(candles(rows))
    demand = [b for b in blocks if b.kind == "demand"]
    assert demand, "an impulse that breaks structure must leave a demand block"
    b = demand[-1]
    assert (b.low, b.high) == (108.0, 110.5) and b.mitigated is False


def test_an_order_block_price_has_returned_into_is_mitigated():
    rows = (leg(100, 120, 8) + leg(120, 110, 6) + [(110, 110.5, 108, 108.5)]
            + leg(108.5, 126, 8) + leg(126, 109, 8))
    blocks = smc.order_blocks(candles(rows))
    demand = [b for b in blocks if b.kind == "demand"]
    assert demand and demand[-1].mitigated is True


# ---------- premium and discount ----------
def test_price_in_the_top_of_the_dealing_range_is_premium():
    df = candles(leg(100, 200, 10) + leg(200, 180, 4))
    pd_ = smc.premium_discount(df)
    assert pd_.equilibrium == pytest.approx(150.0, abs=2.0)
    assert pd_.zone == "premium" and 0.5 < pd_.position <= 1.0

    df2 = candles(leg(200, 100, 10) + leg(100, 115, 4))
    assert smc.premium_discount(df2).zone == "discount"


# ---------- the state the judge reads ----------
def test_the_summary_is_json_safe_and_says_what_is_unavailable():
    import json

    df = candles(leg(100, 120, 8) + leg(120, 110, 6) + leg(110, 125, 8))
    out = smc.summarise(df)
    json.dumps(out)
    assert out["structure"]["bias"] == "bullish"
    assert "fair_value_gaps" in out and "order_blocks" in out and "premium_discount" in out

    thin = smc.summarise(candles(leg(100, 101, 3)))   # under the 20-bar floor
    assert thin["structure"]["available"] is False
