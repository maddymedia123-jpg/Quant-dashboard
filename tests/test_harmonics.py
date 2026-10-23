"""Harmonic patterns: the zigzag, each pattern's ratios, the three stages, and his confluence rule."""
import numpy as np
import pandas as pd
import pytest

from core.indicators import harmonics as hm

H = 3_600_000
T0 = 20_000 * 24 * H

# X A B C D for each pattern, bullish (X a low). Each fits its own table and the CD rule.
SHAPES = {
    "Gartley": (100, 110, 103.82, 107.64, 102.14),
    "Bat": (100, 110, 105.5, 108.5, 101.14),
    "Butterfly": (100, 110, 102.14, 107.0, 97.0),
    "Crab": (100, 110, 105.0, 109.43, 93.82),
    "Cypher": (100, 110, 104.0, 113.0, 102.78),
}


def _path(points, bars=10, lead=108.0, lead_bars=70, tail=None):
    """Closes walking straight between the points; a flat lead-in; an optional tail after the last."""
    closes = [lead] * lead_bars
    prev = lead
    for p in list(points) + ([tail] if tail is not None else []):
        closes += list(np.linspace(prev, p, bars + 1)[1:])
        prev = p
    closes.append(closes[-1])                                 # the forming candle: never read
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": [T0 + i * H for i in range(len(c))], "open": o,
                         "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c,
                         "volume": 0.0})                     # no volume: no VWAP or value area to confuse


def _mirror(points):
    return tuple(200 - p for p in points)


def _named(found, name, stage=None):
    return [h for h in found if h.name == name and (stage is None or h.stage == stage)]


# ---------- the zigzag ----------
def test_the_zigzag_finds_each_turn_and_alternates():
    df = _path(SHAPES["Gartley"], tail=108.0).iloc[:-1]
    pivots, current = hm.zigzag(df, atr_value=float(hm.atr_series(df).iloc[-1]))
    prices = [round(p.price, 1) for p in pivots]
    assert prices[-5:] == pytest.approx([99.8, 110.2, 103.6, 107.8, 101.9], abs=0.15)
    assert all(a.kind != b.kind for a, b in zip(pivots, pivots[1:]))
    assert current.kind == "high", "the leg up from D is still running"


def test_moves_smaller_than_the_threshold_are_not_pivots():
    df = _path((100, 101, 100.5, 101.2, 100.6)).iloc[:-1]
    pivots, _ = hm.zigzag(df, atr_value=2.0)
    assert len(pivots) <= 1


# ---------- each pattern ----------
@pytest.mark.parametrize("name", list(SHAPES))
def test_each_pattern_is_recognised_when_complete(name):
    found = hm.find(_path(SHAPES[name], tail=SHAPES[name][4] + 8), "1h")
    hit = _named(found, name, "completed")
    assert hit and hit[0].side == "bullish" and hit[0].prz[0] - 1 <= SHAPES[name][4] <= hit[0].prz[1] + 1


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_mirror_image_is_the_bearish_pattern(name):
    pts = _mirror(SHAPES[name])
    found = hm.find(_path(pts, lead=92.0, tail=pts[4] - 8), "1h")
    assert _named(found, name, "completed") and _named(found, name, "completed")[0].side == "bearish"


def test_ratios_outside_every_table_give_nothing():
    # AB at 0.70 of XA fits no pattern's B (the later swings may form something else; this XABCD does not)
    found = hm.find(_path((100, 110, 103.0, 107.0, 101.0), tail=109.0), "1h")
    assert not [h for h in found if h.stage == "completed"]


def test_the_reversal_zone_needs_both_rules_to_agree():
    # Gartley X-A-B-C, but a C so shallow that the CD rule's zone misses the 0.786 retracement
    assert hm._prz("Gartley", 100, 110, 103.82, 104.4, True) is None
    lo, hi = hm._prz("Gartley", 100, 110, 103.82, 107.64, True)
    assert 101.8 < lo < hi < 102.5


# ---------- stages ----------
def test_price_arriving_in_the_zone_is_in_zone():
    found = hm.find(_path(SHAPES["Gartley"]), "1h")
    assert _named(found, "Gartley", "in_zone")


def test_a_leg_short_of_the_zone_is_forming_with_its_distance():
    x, a, b, c, _ = SHAPES["Gartley"]
    found = hm.find(_path((x, a, b, c, 105.0)), "1h")
    g = _named(found, "Gartley", "forming")
    assert g and g[0].distance_pct < 0 and g[0].targets[0] > g[0].prz[1]


def test_a_close_through_the_far_edge_kills_the_setup():
    x, a, b, c, _ = SHAPES["Gartley"]
    found = hm.find(_path((x, a, b, c, 99.0)), "1h")
    assert not _named(found, "Gartley", "in_zone")


def test_targets_are_the_382_and_618_of_the_ad_leg():
    h = _named(hm.find(_path(SHAPES["Gartley"], tail=110.0), "1h"), "Gartley", "completed")[0]
    a, d = h.points[1].price, h.points[4].price
    assert h.targets == pytest.approx((d + 0.382 * (a - d), d + 0.618 * (a - d)))


# ---------- his rule ----------
def test_valid_only_beside_a_vwap_value_area_or_gamma_level():
    df = _path(SHAPES["Gartley"], tail=110.0)
    bare = _named(hm.find(df, "1h"), "Gartley", "completed")[0]
    assert not bare.valid and "Unconfirmed: no VWAP" in bare.headline
    gamma = {"available": True, "put_wall": 102.0, "call_wall": 130.0}
    g = _named(hm.find(df, "1h", gamma), "Gartley", "completed")[0]
    assert g.valid and g.confluence == ("put wall",) and "Valid: put wall in the zone" in g.headline


def test_a_level_far_from_the_zone_does_not_count():
    g = _named(hm.find(_path(SHAPES["Gartley"], tail=110.0), "1h", {"available": True, "put_wall": 90.0}),
               "Gartley", "completed")[0]
    assert not g.valid


def test_vwap_and_value_area_levels_are_offered():
    df = _path(SHAPES["Gartley"], tail=110.0)
    df["volume"] = 10.0
    names = [n for n, _ in hm.levels_for(df, "1h", None)]
    assert names[:1] == ["1d VWAP"] and {"point of control", "value area high", "value area low"} <= set(names)


# ---------- where it applies ----------
def test_nothing_below_an_hour_is_read():
    assert hm.find(_path(SHAPES["Gartley"], tail=110.0), "15m") == []
    assert set(hm.find_all({"15m": None, "1h": None}, ("15m", "1h", "4h"))) == {"1h", "4h"}


def test_the_forming_candle_is_never_read():
    df = _path(SHAPES["Gartley"], tail=110.0)
    df.loc[df.index[-1], ["low", "close"]] = [50.0, 50.0]
    assert _named(hm.find(df, "1h"), "Gartley", "completed")


def test_the_structure_agent_gets_patterns_on_its_hour_and_above():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.live_recon import build_state
    from core.agents.recon_profiles import PROFILES
    from core.data.offline import fixture_market
    st = build_state(fixture_market(), {}, profile=PROFILES["live"])
    assert isinstance(st["smc"]["1h"]["harmonics"], list) and "harmonics" not in st["smc"]["15m"]


def test_the_deck_names_the_first_valid_pattern():
    from core.agents.deck import _harmonic
    assert _harmonic({"harmonics": []}) == "none"
    rows = [{"side": "bearish", "pattern": "Bat", "stage": "forming", "valid": False},
            {"side": "bullish", "pattern": "Gartley", "stage": "in_zone", "valid": True}]
    assert _harmonic({"harmonics": rows}) == "bullish Gartley in zone"
    assert _harmonic({"harmonics": rows[:1]}) == "bearish Bat forming (unconfirmed)"


def test_the_card_shows_stage_validity_levels_and_track_record():
    from ui import panels
    df = _path(SHAPES["Gartley"], tail=110.0)
    found = {"1h": hm.find(df, "1h", {"available": True, "put_wall": 102.0}), "4h": []}
    h = panels.harmonics_html(found)
    assert "Bullish Gartley" in h and "completed" in h and "valid</span>" in h and "put wall" in h
    assert "137 setups" in h and "ti-card up" in h
    assert "No Gartley, Bat" in panels.harmonics_html({"1h": [], "4h": []})


# ---------- the spec's RSI-divergence check at D ----------
def _legs(waypoints, lead=108.0, lead_bars=70):
    """Like _path, but each leg names its own candle count: (price, candles) pairs."""
    closes, prev = [lead] * lead_bars, lead
    for p, n in waypoints:
        closes += list(np.linspace(prev, p, n + 1)[1:])
        prev = p
    closes.append(closes[-1])
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": [T0 + i * H for i in range(len(c))], "open": o,
                         "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c,
                         "volume": 0.0})


def _grind_into_d():
    """The Gartley, but D is reached by a fast drop to 102.6 and then a 30-candle grind of alternating
    candles to a marginal new low at 102.14: price makes its lower low below B, RSI does not (measured:
    RSI 44.0 at D against 37.3 at B)."""
    x, a, b, c, _ = SHAPES["Gartley"]
    grind = [(p + (0.45 if i % 2 == 0 else 0.0), 1) for i, p in enumerate(np.linspace(102.6, 102.14, 30))]
    return [(x, 10), (a, 10), (b, 10), (c, 10), (102.6, 4)] + grind


def test_a_grind_to_a_marginal_low_in_the_zone_is_a_regular_bullish_divergence():
    g = _named(hm.find(_legs(_grind_into_d()), "1h"), "Gartley", "in_zone")[0]
    assert g.points[3].price > g.points[2].price, "B is the earlier low D is compared with, C a high"
    assert g.rsi_divergence is True and g.confirmed_by == ("rsi_divergence",)
    assert "Regular bullish RSI divergence at D" in g.rsi_read and g.rsi_read in g.headline
    st = g.to_state()
    assert st["rsi_divergence"] is True and st["confirmed_by"] == ["rsi_divergence"] and not st["valid"]


def test_the_mirror_grind_is_a_regular_bearish_divergence():
    df = _legs([(200 - p, n) for p, n in _grind_into_d()], lead=92.0)
    g = _named(hm.find(df, "1h"), "Gartley", "in_zone")[0]
    assert g.side == "bearish" and g.rsi_divergence is True and "Regular bearish" in g.rsi_read


def test_a_straight_drop_into_the_zone_has_no_divergence():
    # one steady leg from C to D: RSI makes its lower low with price (measured: 28.4 at D, 37.3 at B)
    g = _named(hm.find(_path(SHAPES["Gartley"]), "1h"), "Gartley", "in_zone")[0]
    assert g.rsi_divergence is False and "No RSI divergence" in g.rsi_read
    assert g.to_state()["confirmed_by"] == []
    c = _named(hm.find(_path(SHAPES["Gartley"], tail=110.0), "1h"), "Gartley", "completed")[0]
    assert c.rsi_divergence is False


def test_a_d_that_does_not_extend_past_b_cannot_be_a_regular_divergence():
    pts = (hm.Pivot(0, 95.0, "low"), hm.Pivot(2, 110.0, "high"), hm.Pivot(5, 100.0, "low"),
           hm.Pivot(7, 105.0, "high"))
    r = np.array([50.0] * 5 + [20.0] + [50.0] * 3 + [60.0])           # RSI far higher at D than at B
    ok, why = hm.rsi_divergence(pts, hm.Pivot(9, 101.0, "low"), True, r)
    assert ok is False and "not a lower low" in why
    assert hm.rsi_divergence(pts, hm.Pivot(9, 99.0, "low"), True, r)[0] is True
    ok, why = hm.rsi_divergence(pts, hm.Pivot(9, 99.0, "low"), True, np.full(10, np.nan))
    assert ok is None and "not defined" in why


def test_a_forming_pattern_has_no_d_to_check():
    x, a, b, c, _ = SHAPES["Gartley"]
    g = _named(hm.find(_path((x, a, b, c, 105.0)), "1h"), "Gartley", "forming")[0]
    assert g.rsi_divergence is None and g.rsi_read == "" and g.to_state()["rsi_divergence"] is None


def test_both_confirmations_are_named_and_the_level_rule_is_unchanged():
    g = _named(hm.find(_legs(_grind_into_d()), "1h", {"available": True, "put_wall": 102.2}), "Gartley",
               "in_zone")[0]
    assert g.valid and g.confluence == ("put wall",) and g.confirmed_by == ("level", "rsi_divergence")


def test_the_deck_and_card_name_the_divergence():
    from core.agents.deck import _harmonic
    from ui import panels
    row = {"side": "bullish", "pattern": "Gartley", "stage": "in_zone", "valid": False, "rsi_divergence": True}
    assert _harmonic({"harmonics": [row]}) == "bullish Gartley in zone (unconfirmed) + RSI divergence"
    h = panels.harmonics_html({"1h": hm.find(_legs(_grind_into_d()), "1h")})
    assert "RSI divergence</span>" in h and "Regular bullish RSI divergence at D" in h
    plain = panels.harmonics_html({"1h": hm.find(_path(SHAPES["Gartley"]), "1h")})
    assert "RSI divergence</span>" not in plain and "No RSI divergence" in plain
