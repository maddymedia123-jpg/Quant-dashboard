"""Liquidity and order flow on candles, from hand-made series with known answers.

Covers the four rubric questions the deterministic layer could not answer: was a pool of equal highs
or lows swept and rejected, does cumulative delta confirm or diverge from price, where is the volume
profile's point of control and value area, and how has open interest moved per timeframe."""
import pandas as pd
import pytest

from core.indicators import liquidity as lq
from tests.test_smc import candles, flat, leg


# ---------- equal highs and lows ----------
def test_repeated_highs_within_tolerance_form_one_liquidity_pool():
    rows = leg(98, 100, 3) + [(100, 110, 99, 101), (101, 110.2, 100, 102), (102, 109.9, 100, 103)] + leg(103, 104, 3)
    pools = lq.liquidity_pools(candles(rows), tolerance_pct=0.5)
    highs = [p for p in pools if p.side == "buyside"]
    assert len(highs) == 1
    p = highs[0]
    assert p.price == pytest.approx(110.0, abs=0.2) and p.touches == 3
    assert p.swept is False


def test_a_wick_through_a_pool_that_closes_back_inside_is_a_sweep():
    rows = (flat(3, 100) + [(100, 110, 99, 101), (101, 110.1, 100, 102)] + flat(3, 103)
            + [(103, 113, 102, 104)])                     # wicks above 110, closes back under it
    pools = lq.liquidity_pools(candles(rows), tolerance_pct=0.5)
    swept = [p for p in pools if p.swept]
    assert len(swept) == 1 and swept[0].side == "buyside" and swept[0].broken is False
    assert swept[0].sweep_bars_ago == 0 and swept[0].reclaimed is True


def test_a_break_that_closes_beyond_the_pool_is_not_a_sweep():
    rows = (flat(3, 100) + [(100, 110, 99, 101), (101, 110.1, 100, 102)] + flat(3, 103)
            + [(103, 114, 102, 113)])                     # closes above the pool: a break, not a raid
    pools = lq.liquidity_pools(candles(rows), tolerance_pct=0.5)
    buyside = [p for p in pools if p.side == "buyside"]
    assert buyside and buyside[0].swept is False and buyside[0].broken is True


def test_sellside_pools_are_found_under_price():
    rows = flat(3, 110) + [(110, 111, 100, 109), (109, 110, 100.1, 108)] + flat(3, 108)
    pools = lq.liquidity_pools(candles(rows), tolerance_pct=0.5)
    sell = [p for p in pools if p.side == "sellside"]
    assert sell and sell[0].price == pytest.approx(100.0, abs=0.2)


# ---------- cumulative volume delta ----------
def test_delta_splits_each_candle_by_where_it_closed_in_its_range():
    up = candles([(100, 110, 100, 110, 10.0)])            # closes on the high: all buying
    assert lq.cumulative_delta(up).last == pytest.approx(10.0)
    down = candles([(110, 110, 100, 100, 10.0)])          # closes on the low: all selling
    assert lq.cumulative_delta(down).last == pytest.approx(-10.0)
    mid = candles([(105, 110, 100, 105, 10.0)])           # closes mid range: neutral
    assert lq.cumulative_delta(mid).last == pytest.approx(0.0, abs=1e-9)


def test_price_rising_while_delta_falls_is_flagged_as_bearish_absorption():
    rising_sellers = [(100 + i, 101 + i, 99.9 + i, 100.2 + i, 10.0) for i in range(12)]
    cvd = lq.cumulative_delta(candles(rising_sellers), lookback=10)
    assert cvd.price_change > 0 and cvd.delta_change < 0
    assert cvd.state == "bearish absorption"


def test_delta_agreeing_with_price_is_confirmation():
    strong = [(100 + i, 101.5 + i, 100 + i, 101.4 + i, 10.0) for i in range(12)]
    cvd = lq.cumulative_delta(candles(strong), lookback=10)
    assert cvd.state == "confirming" and cvd.delta_change > 0


def test_cumulative_delta_is_labelled_as_a_candle_proxy_not_real_tape():
    cvd = lq.cumulative_delta(candles(flat(5, 100)))
    assert "proxy" in cvd.method.lower() and cvd.available


# ---------- volume profile ----------
def test_the_point_of_control_is_the_price_that_traded_the_most_volume():
    rows = flat(10, 100, vol=1.0) + flat(30, 105, vol=5.0) + flat(10, 110, vol=1.0)
    vp = lq.volume_profile(candles(rows), bins=24)
    assert vp.available
    assert vp.poc == pytest.approx(105.0, abs=1.0)
    assert vp.value_area_low <= vp.poc <= vp.value_area_high
    assert vp.value_area_low > 100.0 and vp.value_area_high < 110.0, "70% of volume sits around 105"


def test_the_value_area_holds_about_seventy_percent_of_traded_volume():
    rows = leg(100, 140, 40, vol=2.0)
    vp = lq.volume_profile(candles(rows), bins=20)
    assert 0.60 <= vp.value_area_share <= 0.80
    assert vp.position in ("above value", "inside value", "below value")


def test_volume_profile_says_unavailable_without_volume():
    rows = [(100, 101, 99, 100, 0.0) for _ in range(30)]
    vp = lq.volume_profile(candles(rows))
    assert not vp.available and vp.poc is None


# ---------- open interest per timeframe ----------
def test_open_interest_change_is_measured_per_window_from_samples():
    now = 1_700_000_000_000
    hist = [(now - 4 * 3_600_000, 100.0), (now - 3_600_000, 110.0), (now - 900_000, 120.0), (now, 125.0)]
    oi = lq.oi_changes(hist, now_ms=now)
    assert oi["15m"] == pytest.approx((125 / 120 - 1) * 100, abs=0.01)
    assert oi["1h"] == pytest.approx((125 / 110 - 1) * 100, abs=0.01)
    assert oi["4h"] == pytest.approx(25.0, abs=0.01)


def test_open_interest_windows_without_a_sample_report_none_rather_than_zero():
    now = 1_700_000_000_000
    oi = lq.oi_changes([(now - 600_000, 100.0), (now, 105.0)], now_ms=now)
    assert oi["15m"] == pytest.approx(5.0, abs=0.01)
    assert oi["1h"] is None and oi["4h"] is None, "no sample that old: unknown, not unchanged"
    assert lq.oi_changes([], now_ms=now) == {"15m": None, "1h": None, "4h": None}


# ---------- the state the judge reads ----------
def test_the_summary_is_json_safe_and_marks_missing_pieces():
    import json

    df = candles(flat(10, 100) + leg(100, 120, 20) + flat(10, 120))
    out = lq.summarise(df, oi_history=[], now_ms=1_700_000_000_000)
    json.dumps(out)
    assert out["cumulative_delta"]["available"] is True
    assert out["volume_profile"]["available"] is True
    assert out["open_interest"] == {"15m": None, "1h": None, "4h": None}
    assert "pools" in out and isinstance(out["pools"]["buyside"], (dict, type(None)))


def test_a_far_older_sample_cannot_answer_a_short_window():
    """Sparse sampling must not let a 3-hour-old reading be reported as a 15-minute change."""
    now = 1_700_000_000_000
    stale = [(now - 3 * 3_600_000, 100.0), (now, 130.0)]
    oi = lq.oi_changes(stale, now_ms=now)
    assert oi["15m"] is None and oi["1h"] is None, "no sample near those windows"
    assert oi["4h"] == pytest.approx(30.0, abs=0.01), "3h old is close enough to answer 4h"


def test_the_sample_used_must_bracket_the_window():
    now = 1_700_000_000_000
    # 50 minutes old: too old for 15m, close enough for 1h, too recent to call 4h
    hist = [(now - 50 * 60_000, 100.0), (now, 110.0)]
    oi = lq.oi_changes(hist, now_ms=now)
    assert oi["15m"] is None and oi["4h"] is None
    assert oi["1h"] == pytest.approx(10.0, abs=0.01)
