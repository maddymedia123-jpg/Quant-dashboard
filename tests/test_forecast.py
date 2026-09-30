"""The forecast envelope drawn as candles, which the spec's architecture tree asks of every desk.

"Forecast: 1h candles / 4h horizon" up to "1W candles / 1M horizon", drawn in his sketch as "Futuristic
Projected Candles (Dotted Outlines)".

What these tests pin is that nothing is invented. There is no forecast of any candle's shape here: each
bar is the volatility envelope at that step - body half a sigma either side of the drift path, wicks a
full sigma, sigma widening as the square root of the steps ahead. If that ever turns into four made-up
OHLC numbers, these fail.
"""
import math

import pytest

from core.indicators import volatility as vol

PRICE = 80_000.0
SIGMA_PCT = 2.0          # 2% over the whole horizon, as volatility_profile reports it


def bars(**over):
    kwargs = {"price": PRICE, "sigma_pct": SIGMA_PCT, "horizon_bars": 12, "drift": 0.0}
    kwargs.update(over)
    return vol.forecast_candles(**kwargs)


# ---------- the shape of the envelope ----------
def test_one_bar_per_horizon_bar():
    assert len(bars(horizon_bars=4)) == 4
    assert [b["step"] for b in bars(horizon_bars=4)] == [1, 2, 3, 4]


def test_the_envelope_widens_with_the_square_root_of_the_steps_ahead():
    """Random-walk scaling - the same assumption the Hurst reading measures departures from. A linear
    or constant envelope would be a different claim about the market."""
    out = bars(horizon_bars=16, drift=0.0)
    widths = [b["high"] - b["low"] for b in out]
    assert widths == sorted(widths), "it has to widen, not narrow"
    # the last bar's width over the fourth's should be sqrt(16/4) = 2
    assert widths[-1] / widths[3] == pytest.approx(2.0, rel=0.01)
    sigma = PRICE * SIGMA_PCT / 100.0
    assert widths[-1] == pytest.approx(2 * sigma * vol.FORECAST_WICK_SIGMA, rel=1e-6)


def test_the_body_sits_inside_the_wicks_on_every_bar():
    for b in bars(horizon_bars=20, drift=0.6):
        assert b["low"] <= min(b["open"], b["close"]), b
        assert b["high"] >= max(b["open"], b["close"]), b


def test_the_body_is_half_a_sigma_and_the_wick_a_full_one():
    """The numbers the legend claims. If these constants drift, the tooltip becomes a lie."""
    assert (vol.FORECAST_BODY_SIGMA, vol.FORECAST_WICK_SIGMA) == (0.5, 1.0)
    out = bars(horizon_bars=1, drift=0.0)
    sigma = PRICE * SIGMA_PCT / 100.0
    assert out[0]["high"] - PRICE == pytest.approx(sigma * 1.0)
    assert abs(out[0]["close"] - PRICE) == pytest.approx(sigma * 0.5)


def test_each_bar_opens_where_the_last_one_closed():
    out = bars(horizon_bars=8, drift=0.4)
    assert out[0]["open"] == pytest.approx(PRICE), "the first opens at the live price"
    for previous, nxt in zip(out, out[1:]):
        assert nxt["open"] == pytest.approx(previous["close"]), "no gaps in a continuous envelope"


# ---------- the drift is the direction engine's, not a price target ----------
def test_a_flat_direction_gives_a_symmetric_envelope():
    out = bars(horizon_bars=10, drift=0.0)
    for b in out:
        centre = (b["high"] + b["low"]) / 2
        assert centre == pytest.approx(PRICE, abs=0.02), "no drift, no lean"


def test_a_bullish_drift_leans_the_envelope_up_and_a_bearish_one_down():
    up = bars(horizon_bars=10, drift=1.0)
    down = bars(horizon_bars=10, drift=-1.0)
    assert up[-1]["close"] > PRICE and down[-1]["close"] < PRICE
    assert up[-1]["high"] > up[-1]["low"] > PRICE - PRICE * SIGMA_PCT / 100
    # and they are mirror images about the price
    assert (up[-1]["close"] - PRICE) == pytest.approx(PRICE - down[-1]["close"], rel=1e-6)


def test_the_drift_cannot_exceed_the_envelope_it_is_drawn_in():
    """Confidence leans the path by at most the body width - half a sigma - so the envelope stays the
    claim and the drift only decides where inside it the path sits."""
    sigma = PRICE * SIGMA_PCT / 100.0
    out = bars(horizon_bars=10, drift=1.0)
    lean = (out[-1]["high"] + out[-1]["low"]) / 2 - PRICE
    assert lean == pytest.approx(sigma * vol.FORECAST_BODY_SIGMA, rel=1e-6)


def test_a_confident_bull_still_shows_downside_in_the_envelope():
    """The envelope must never become one-sided: a forecast that cannot go against the trade is not a
    forecast, it is an advert."""
    out = bars(horizon_bars=12, drift=1.0)
    assert out[-1]["low"] < PRICE, "the downside wick survives full confidence"


# ---------- refusing to draw rather than inventing ----------
@pytest.mark.parametrize("kwargs", [
    {"price": None}, {"price": 0}, {"sigma_pct": None}, {"sigma_pct": 0}, {"sigma_pct": -1},
    {"horizon_bars": 0}, {"horizon_bars": -5},
])
def test_nothing_is_drawn_without_the_inputs_to_draw_it(kwargs):
    assert bars(**kwargs) == []


def test_the_bar_count_is_capped_so_a_long_horizon_stays_legible():
    out = bars(horizon_bars=200, max_bars=24)
    assert len(out) == 24
    sigma = PRICE * SIGMA_PCT / 100.0
    # the cap compresses the steps rather than truncating the envelope: the last bar is still the full
    # horizon sigma, so a capped chart does not understate the range
    assert (out[-1]["high"] - out[-1]["low"]) == pytest.approx(2 * sigma, rel=1e-6)


def test_the_envelope_scales_with_the_measured_sigma():
    quiet = bars(sigma_pct=0.5, horizon_bars=10)
    wild = bars(sigma_pct=5.0, horizon_bars=10)
    assert (wild[-1]["high"] - wild[-1]["low"]) == pytest.approx(
        10 * (quiet[-1]["high"] - quiet[-1]["low"]), rel=1e-6)


# ---------- what the chart receives ----------
def test_every_desk_gets_forecast_bars_on_its_own_timeframe_and_horizon():
    import os
    import tempfile

    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category
    from ui.charts import chart_payload

    m = fixture_market()
    for key, cfg in CATEGORIES.items():
        a = analyze_category(cfg, m.spot.frames, m.futures)
        if a is None:
            continue
        payload = chart_payload(a, dark=False)
        forecast = payload["forecast"]
        assert forecast, f"{key} has no forecast bars"
        assert len(forecast) == min(cfg.horizon_bars, 24), key
        assert a.chart_tf in payload["forecastLabel"], key
        assert a.vol.horizon_label in payload["forecastLabel"], key

        interval = payload["forecast"][1]["time"] - payload["forecast"][0]["time"]
        assert interval > 0 and forecast[0]["time"] > payload["now"], "the bars sit in the future"
        widths = [b["high"] - b["low"] for b in forecast]
        assert widths[-1] > widths[0], f"{key}: the envelope must widen"


def test_the_chart_legend_says_what_the_bars_are():
    """The one thing that stops these reading as predicted candles."""
    import os
    import tempfile

    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category
    from ui.charts import build_chart_html

    m = fixture_market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    html = build_chart_html(a, dark=False)
    assert "forecast envelope" in html
    assert "Not a prediction" in html, "the tooltip has to say what it is not"
    assert "half a " in html and "sigma" in html

    # hollow, not solid: filled candles in the live colours would read as real bars, which is the one
    # thing these must never do
    block = html.split("P.forecast")[1][:600]
    assert "upColor: 'rgba(0,0,0,0)'" in block and "downColor: 'rgba(0,0,0,0)'" in block, block[:200]
    assert "borderVisible: true" in block, "the outline is what makes them readable at all"
    assert "T.up" not in block and "T.down" not in block, "never the live candle colours"
