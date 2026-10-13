"""The 3-tier timeframe cascade: bias above, anchor, engine below, and the A/B/C state it implies."""
import numpy as np
import pandas as pd
import pytest

from core.indicators.cascade import MIN_CANDLES, anchored_vwap, classify, direction

H = 3_600_000
T0 = 1_790_000_000_000 - 1_790_000_000_000 % (24 * H)      # a day boundary: the open of the bias candle


def series(step_ms, values, end_ms):
    """Candles ending at end_ms (the last one still forming), closes following `values`."""
    n = len(values)
    ts = [end_ms - (n - 1 - i) * step_ms for i in range(n)]
    v = np.asarray(values, dtype=float)
    prev = np.concatenate([[v[0]], v[:-1]])
    return pd.DataFrame({"timestamp": ts, "open": prev, "high": np.maximum(prev, v) + 0.5,
                         "low": np.minimum(prev, v) - 0.5, "close": v, "volume": 1.0})


def frames(bias, anchor, fast, slow):
    return {"1d": series(24 * H, bias, T0), "4h": series(4 * H, anchor, T0 + 8 * H),
            "15m": series(H // 4, fast, T0 + 8 * H), "1h": series(H, slow, T0 + 8 * H)}


UP = list(np.linspace(100, 200, 120))
DOWN = list(np.linspace(200, 100, 120))
FLAT = [100.0] * 120


def test_trend_reads_as_a_direction_and_flat_as_none():
    assert direction(series(H, UP, T0))[0] == "bullish"
    assert direction(series(H, DOWN, T0))[0] == "bearish"
    assert direction(series(H, FLAT, T0))[0] == "neutral"
    assert direction(series(H, UP[:MIN_CANDLES - 1], T0))[0] == "neutral", "too few candles, no read"


def test_bias_and_engine_aligned_is_full_confluence():
    c = classify(frames(UP, UP, UP, UP), "1d", "4h", ("15m", "1h"))
    assert c.state == "A" and c.engine_direction == "bullish" and not c.pullback


def test_a_bearish_bias_with_a_bearish_engine_is_confluence_too():
    assert classify(frames(DOWN, DOWN, DOWN, DOWN), "1d", "4h", ("15m", "1h")).state == "A"


def test_an_engine_against_the_bias_without_acceptance_is_a_pullback_with_bands_below_price():
    # a long decline on the engine, but since the bias candle opened price has been rising - so the hours
    # since then close ABOVE VWAP: against the bias, not yet accepted
    fast = list(np.linspace(200, 120, 160)) + list(np.linspace(120, 130, 32))
    slow = list(np.linspace(200, 120, 60)) + list(np.linspace(120, 130, 8))
    c = classify(frames(UP, UP, fast, slow), "1d", "4h", ("15m", "1h"))
    assert c.state == "B" and c.engine_direction != "bullish" and c.closes_beyond_vwap == 0
    px = 200.0
    assert c.pullback["timeframe"] == "4h" and c.pullback["p95"] <= c.pullback["p80"] <= c.pullback["p50"] <= px


def test_two_closes_below_vwap_against_the_bias_is_a_control_shift():
    fast = list(np.linspace(200, 130, 160)) + list(np.linspace(130, 100, 32))
    slow = list(np.linspace(200, 130, 60)) + list(np.linspace(130, 100, 8))
    c = classify(frames(UP, UP, fast, slow), "1d", "4h", ("15m", "1h"))
    assert c.state == "C" and c.closes_beyond_vwap >= 2
    assert c.vwap is not None and 100 < c.vwap < 130


def test_closes_from_before_vwaps_start_are_not_counted():
    # every hour BEFORE the bias candle opened closed far below; since it opened, price sits on VWAP
    slow = list(np.linspace(200, 50, 60)) + [120.0] * 8
    fast = list(np.linspace(200, 50, 160)) + [120.0] * 32
    c = classify(frames(UP, UP, fast, slow), "1d", "4h", ("15m", "1h"))
    assert c.closes_beyond_vwap == 0 and c.state != "C"


def test_no_bias_vector_is_its_own_state():
    c = classify(frames(FLAT, UP, UP, UP), "1d", "4h", ("15m", "1h"))
    assert c.state == "N" and "range edges" in c.playbook


def test_a_bias_without_enough_candles_says_so():
    f = frames(UP, UP, UP, UP)
    f["1d"] = f["1d"].tail(MIN_CANDLES - 1)
    c = classify(f, "1d", "4h", ("15m", "1h"))
    assert c.state == "?" and "1d" in c.playbook


def test_vwap_is_volume_weighted_from_its_start():
    df = pd.DataFrame({"timestamp": [0, 1, 2], "high": [10.0, 20, 30], "low": [10.0, 20, 30],
                       "close": [10.0, 20, 30], "volume": [100.0, 1, 3]})
    assert anchored_vwap(df, 1) == pytest.approx((20 * 1 + 30 * 3) / 4)
    assert anchored_vwap(df.assign(volume=0.0), 0) is None, "no volume, no VWAP"


def test_the_judges_get_the_same_cascade_the_desk_shows():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.live_recon import build_state
    from core.agents.recon_profiles import PROFILES
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category

    m = fixture_market()
    a = {k: x for k, c in CATEGORIES.items() if (x := analyze_category(c, m.spot.frames, m.futures))}
    for key, p in PROFILES.items():
        shown = classify(m.spot.frames, *p.cascade).to_state()
        assert build_state(m, a, profile=p)["cascade"] == shown, key


def test_each_desk_has_a_bias_one_step_above_its_anchor():
    from core.agents.recon_profiles import PROFILES
    assert {k: p.cascade for k, p in PROFILES.items()} == {
        "live": ("1d", "4h", ("15m", "1h")), "intraday": ("1w", "1d", ("1h", "4h")),
        "weekly": ("1M", "1w", ("4h", "1d")), "monthly": ("3M", "1M", ("1d", "1w"))}


def test_the_panel_reports_the_state_its_bands_and_escapes():
    from ui import panels
    fast = list(np.linspace(200, 120, 160)) + list(np.linspace(120, 130, 32))
    slow = list(np.linspace(200, 120, 60)) + list(np.linspace(120, 130, 8))
    h = panels.cascade_html(classify(frames(UP, UP, fast, slow), "1d", "4h", ("15m", "1h")))
    assert "State B · Pullback" in h and "Pullback bands" in h and "ti-card warn" in h
    assert "Anchor · the candle being called" in h and "▲ bullish" in h
    assert "Engine · where change shows first" in h


# ---------- the exact boundaries of a control shift, with directions and VWAP held fixed ----------
def _rigged(monkeypatch, dirs: dict, slow_closes: list, before: list | None = None, vwap_level=100.0):
    """Frames whose direction is set per timeframe and whose VWAP is exactly `vwap_level`. `slow_closes` are
    the 1h closes since the bias candle opened - the last of them still forming; `before` are 1h closes from
    earlier, which must not count."""
    from core.indicators import cascade as cz

    start = T0
    def df(tf, ts, closes):
        d = pd.DataFrame({"timestamp": ts, "open": closes, "high": closes, "low": closes, "close": closes,
                          "volume": 1.0})
        d.attrs["tf"] = tf
        return d

    before = before or []
    slow_ts = [start - (len(before) - i) * H for i in range(len(before))] + [start + i * H for i in range(len(slow_closes))]
    f = {"1d": df("1d", [start - (20 - i) * 24 * H for i in range(21)], [100.0] * 21),
         "4h": df("4h", [start - (20 - i) * 4 * H for i in range(21)], [100.0] * 21),
         "15m": df("15m", [start + i * H // 4 for i in range(20)], [vwap_level] * 20),
         "1h": df("1h", slow_ts, [float(c) for c in before + slow_closes])}
    monkeypatch.setattr(cz, "direction", lambda d: (dirs[d.attrs["tf"]], "MIXED", "unclear") if d is not None else ("neutral", "MIXED", "unclear"))
    return classify(f, "1d", "4h", ("15m", "1h"))


BEARISH_ENGINE = {"1d": "bullish", "4h": "bullish", "15m": "bearish", "1h": "bearish"}


def test_two_completed_closes_below_vwap_is_a_control_shift(monkeypatch):
    c = _rigged(monkeypatch, BEARISH_ENGINE, [101, 99, 98, 97])            # 97 is still forming
    assert c.vwap == 100.0 and c.closes_beyond_vwap == 2 and c.state == "C"


def test_one_completed_close_is_not_acceptance_and_the_forming_candle_does_not_count(monkeypatch):
    c = _rigged(monkeypatch, BEARISH_ENGINE, [101, 99, 50])                 # only 99 is a completed close below
    assert c.closes_beyond_vwap == 1 and c.state == "B"


def test_an_engine_against_the_bias_without_acceptance_stays_a_pullback(monkeypatch):
    c = _rigged(monkeypatch, BEARISH_ENGINE, [101, 102, 103])
    assert c.engine_direction == "bearish" and c.closes_beyond_vwap == 0 and c.state == "B"


def test_closes_from_before_the_vwap_start_are_never_counted(monkeypatch):
    c = _rigged(monkeypatch, BEARISH_ENGINE, [99, 120], before=[90, 90, 90])
    assert c.closes_beyond_vwap == 1 and c.state == "B"


def test_trend_and_structure_pointing_opposite_ways_give_no_direction(monkeypatch):
    from core.indicators import cascade as cz

    class Stack:
        state = "BULL"

    class Struct:
        available, bias = True, "bearish"

    monkeypatch.setattr(cz, "ema_stack", lambda df, periods=None: Stack())
    monkeypatch.setattr(cz.smc, "structure", lambda df: Struct())
    assert direction(series(H, UP, T0))[0] == "neutral"
    Struct.bias = "unclear"
    assert direction(series(H, UP, T0))[0] == "bullish", "one vote and an abstention is a direction"
