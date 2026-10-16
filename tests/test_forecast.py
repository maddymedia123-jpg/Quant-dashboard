"""The forecast pane (his Section 8): one median path per desk, with its range, its odds and its record.

What these pin: the path is the median of simulated paths and moves only as far as the measured
momentum allows; the range is honest about its width; every level carries the chance price trades there;
the gamma regime only touches the desks the options cover; and the track record scores the same model
against what happened rather than describing it."""
import math
import os
import tempfile

import numpy as np
import pytest

from core.indicators import forecast as fc

H = 3_600_000
NOW = 1_790_000_000_000


def walk(n=600, drift=0.0, sigma=0.004, seed=1, start=80_000.0):
    r = np.random.default_rng(seed).normal(drift, sigma, n)
    return start * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def build(closes=None, horizon=4, levels=None, gamma=None, tf="1h"):
    closes = walk() if closes is None else closes
    return fc.build(closes, float(closes[-1]), NOW, H, horizon, tf, levels, gamma)


# ---------- the path ----------
def test_one_candle_per_horizon_bar_each_an_hour_apart_after_now():
    f = build(horizon=6)
    assert f.available and len(f.steps) == 6 and f.bar_ms == H
    assert [s.time_ms for s in f.steps] == [NOW + k * H for k in range(1, 7)]
    assert all(s.low <= min(s.open, s.close) <= max(s.open, s.close) <= s.high for s in f.steps)


def test_the_path_starts_at_the_price_and_its_candles_join_up():
    f = build()
    assert f.steps[0].open == pytest.approx(f.price, rel=1e-9)
    for a, b in zip(f.steps, f.steps[1:]):
        assert b.open == pytest.approx(a.close, rel=2e-3), "each candle opens near where the last one closed"


def test_the_range_widens_with_the_horizon_and_holds_the_path():
    f = build(horizon=8)
    widths = [s.band_high - s.band_low for s in f.steps]
    assert widths == sorted(widths) and widths[-1] > 2 * widths[0], "sqrt(8) is about 2.8 times one bar"
    assert all(s.band_low < s.close < s.band_high for s in f.steps)
    assert f.end_low < f.end_median < f.end_high


def test_a_random_walk_gets_no_lean():
    """Hurst 0.5 means no memory: the momentum term is multiplied by 2H - 1, so roughly nothing."""
    f = build(walk(seed=3))
    assert abs(f.p_up - 0.5) < 0.06 and abs(f.end_median / f.price - 1) < 0.002


def test_the_drift_never_runs_away():
    """A straight-line rally gives a huge momentum reading; the cap keeps the path inside a quarter sigma
    per bar, so the range still reaches below the price."""
    rally = walk(drift=0.004, sigma=0.002, seed=4)
    drift, sigma, _h = fc.drift_and_sigma(rally, 4)
    assert abs(drift) <= fc.MAX_DRIFT_SIGMA * sigma + 1e-15
    f = build(rally)
    assert f.end_low < f.price, "a forecast that cannot go against the move is an advert"


def test_the_same_inputs_draw_the_same_path():
    assert [s.close for s in build().steps] == [s.close for s in build().steps]


def test_too_little_history_is_said_not_drawn():
    f = build(walk(n=20))
    assert not f.available and "candles" in f.reason


# ---------- the odds ----------
def test_each_level_carries_the_chance_of_trading_there_and_nearer_is_likelier():
    c = walk()
    p = float(c[-1])
    f = build(c, levels=[("near ceiling", p * 1.003), ("far ceiling", p * 1.02), ("near floor", p * 0.997)])
    odds = {lv.label: lv for lv in f.levels}
    assert odds["near ceiling"].side == "above" and odds["near floor"].side == "below"
    assert odds["near ceiling"].p_touch > odds["far ceiling"].p_touch > 0
    assert [lv.price for lv in f.levels] == sorted([lv.price for lv in f.levels], reverse=True)


def test_two_names_for_one_price_are_one_level():
    c = walk()
    p = float(c[-1])
    f = build(c, levels=[("floor (8x)", p * 0.99), ("put wall", p * 0.99)])
    assert len(f.levels) == 1 and f.levels[0].label == "floor (8x) · put wall"


def test_gamma_narrows_or_widens_only_the_desks_the_options_cover():
    base = build()
    pos, neg = build(gamma="positive"), build(gamma="negative")
    width = lambda f: f.end_high - f.end_low  # noqa: E731
    assert width(pos) < width(base) < width(neg)
    assert pos.gamma_regime == "positive" and pos.gamma_scale == fc.GAMMA_SCALE["positive"]
    weekly = build(gamma="positive", tf="1d")
    assert weekly.gamma_scale == 1.0 and weekly.gamma_regime is None, "the expiries do not span a week"


# ---------- the track record ----------
def test_on_a_random_walk_the_range_holds_about_68_percent_and_the_lean_is_a_coin_flip():
    tr = fc.track_record(walk(n=3000, seed=11), 4, origins=2000)
    assert tr.origins == 2000
    assert 0.62 < tr.band_hit < 0.75
    assert 0.45 < tr.direction_hit < 0.55, "no memory to exploit, and the record must say so"


def test_the_record_scores_what_actually_happened():
    """A series that only goes up: every horizon closed higher, so the base rate is 1."""
    up = 80_000 * np.exp(np.cumsum(np.full(600, 0.001) + np.random.default_rng(2).normal(0, 1e-5, 600)))
    tr = fc.track_record(up, 4, origins=100)
    assert tr.base_up == 1.0 and tr.direction_hit == 1.0


# ---------- the desks ----------
@pytest.fixture(scope="module")
def market():
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())
    from core.data.offline import fixture_market
    return fixture_market()


def test_every_desk_forecasts_its_own_horizon(market):
    from core.config import CATEGORIES
    from core.indicators.category import analyze_category
    for key, cfg in CATEGORIES.items():
        a = analyze_category(cfg, market.spot.frames, market.futures)
        f, tr = fc.desk_forecast(a, market.options.gamma, market.spot.frames.get(a.chart_tf))
        assert f.available and len(f.steps) == cfg.horizon_bars, key
        assert f.steps[0].time_ms > int(a.chart_df["timestamp"].iloc[-1]), "the path sits after the last candle"
        assert tr.origins > 0 and tr.band_hit is not None, key
        labels = " ".join(lv.label for lv in f.levels)
        assert ("wall" in labels or "flip" in labels) == (a.chart_tf in fc.GAMMA_TIMEFRAMES), key


def test_the_forecast_chart_is_clean_and_reads_utc_then_pkt(market):
    import json
    import re

    from core.config import CATEGORIES
    from core.indicators.category import analyze_category
    from ui.charts import build_forecast_html, forecast_payload
    a = analyze_category(CATEGORIES["live"], market.spot.frames, market.futures)
    f, _tr = fc.desk_forecast(a, market.options.gamma)
    p = forecast_payload(a, f, dark=False)
    assert len(p["candles"]) == 40 and len(p["path"]) == len(f.steps)
    assert p["path"][0]["time"] > p["now"] and p["bandHigh"][0]["time"] == p["now"]
    assert all("%" in lv["title"] for lv in p["levels"]), "every level line carries its odds"
    h = build_forecast_html(a, f, dark=False)
    json.loads(re.search(r"const P = (\{.*?\});\n", h, re.S).group(1))
    assert "' UTC (' + hm(k) + ' PKT - '" in h, "the crosshair reads UTC first, PKT in brackets"
    assert "EMA" not in h and "RSI" not in h, "no indicators on the forecast pane"
    assert "upColor: 'rgba(0,0,0,0)'" in h.split("const path")[1][:200], "the path is hollow"


def test_the_main_chart_no_longer_carries_a_second_forecast(market):
    from core.config import CATEGORIES
    from core.indicators.category import analyze_category
    from ui.charts import build_chart_html, chart_payload
    a = analyze_category(CATEGORIES["live"], market.spot.frames, market.futures)
    p = chart_payload(a, dark=False)
    assert "forecast" not in p and "projection" not in p
    assert "P.forecast" not in build_chart_html(a, dark=False)


def test_the_odds_card_says_what_the_path_is_and_how_it_has_done(market):
    from core.config import CATEGORIES
    from core.indicators.category import analyze_category
    from ui import panels
    a = analyze_category(CATEGORIES["live"], market.spot.frames, market.futures)
    f, tr = fc.desk_forecast(a, market.options.gamma, market.spot.frames.get("1h"))
    h = panels.forecast_html(f, tr, a.vol.horizon_label)
    assert "Forecast · next 4 hours" in h and "Most likely path ends near" in h
    assert "68% range" in h and "chance it ends higher" in h and "Chance price trades at each level" in h
    assert h.count(" UTC (") >= len(f.steps) + 1, "every time is UTC first, PKT in brackets"
    assert "median of 4,000 simulated paths, not a promise" in h
    assert "Track record" in h and "(aim: 68%)" in h


def test_the_record_verdict_follows_the_numbers():
    from ui import panels
    f = build()
    flat = fc.TrackRecord(500, 0.71, 500, 0.50, 0.54)
    assert "no proven edge" in panels.forecast_html(f, flat, "next 4h")
    good = fc.TrackRecord(500, 0.70, 500, 0.60, 0.52)
    assert "beat simply betting on the usual direction" in panels.forecast_html(f, good, "next 4h")
    assert math.isfinite(f.end_median)


def test_the_cap_holds_even_when_memory_is_strong(monkeypatch):
    """With H forced to 1 the momentum passes straight through, and only the cap stands in its way."""
    monkeypatch.setattr(fc, "hurst", lambda closes: 1.0)
    drift, sigma, _h = fc.drift_and_sigma(walk(drift=0.004, sigma=0.002, seed=4), 4)
    assert drift == pytest.approx(fc.MAX_DRIFT_SIGMA * sigma)


def test_the_range_is_sigma_times_root_horizon():
    """Each bar is split into sub-steps whose variances add up to one bar's: the 68% range at the
    horizon is +/- one sigma * sqrt(bars), whatever the number of sub-steps."""
    c = walk(seed=5)
    f = build(c, horizon=9)
    half_width = math.log(f.end_high / f.end_low) / 2
    assert half_width == pytest.approx(f.sigma_per_bar * 3, rel=0.06)


def test_trading_at_a_level_is_likelier_than_finishing_beyond_it():
    """The chance of touching a level one sigma away is about twice the chance of closing beyond it
    (the reflection principle): reading the finish as the touch would halve every number in the table."""
    c = walk(seed=6)
    p = float(c[-1])
    probe = build(c, horizon=4)
    one_sigma = p * math.exp(probe.sigma_per_bar * 2)
    f = build(c, horizon=4, levels=[("one sigma up", one_sigma)])
    assert 0.25 < f.levels[0].p_touch < 0.40


def test_the_record_centres_the_range_on_the_drift(monkeypatch):
    """A series climbing exactly as fast as the model's drift ends dead centre every time."""
    monkeypatch.setattr(fc, "drift_and_sigma", lambda closes, h: (0.01, 0.001, 0.6))
    up = 80_000 * np.exp(np.arange(600) * 0.01)
    assert fc.track_record(up, 4, origins=50).band_hit == 1.0


def test_the_record_scores_the_lean_not_the_market(monkeypatch):
    monkeypatch.setattr(fc, "drift_and_sigma", lambda closes, h: (-0.001, 0.01, 0.4))
    up = 80_000 * np.exp(np.arange(600) * 0.001)
    tr = fc.track_record(up, 4, origins=50)
    assert tr.base_up == 1.0 and tr.direction_hit == 0.0, "leaning down in a market that only rose"


def test_unreachable_levels_stay_off_the_chart_and_the_table():
    from types import SimpleNamespace as NS

    from ui import panels
    from ui.charts import forecast_payload
    c = walk()
    p = float(c[-1])
    f = build(c, levels=[("near", p * 1.002), ("call wall", p * 1.12)])
    assert {lv.label for lv in f.levels} == {"near", "call wall"}
    h = panels.forecast_html(f, None, "next 4h")
    assert "near" in h and "call wall" not in h
    import pandas as pd
    df = pd.DataFrame({"timestamp": [NOW], "open": [p], "high": [p], "low": [p], "close": [p]})
    payload = forecast_payload(NS(chart_df=df, chart_tf="1h"), f, dark=False)
    assert [lv["title"].split(" ")[0] for lv in payload["levels"]] == ["near"]
