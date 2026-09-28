"""Deterministic trap detection: what the TRAP desk is allowed to call a candidate.

The spec's bull trap is a fake breakout above resting liquidity into a negative delta divergence; the
bear trap is the mirror below value with absorption. Detection reads the same state the desk agents get,
so a trap is never asserted without the evidence that produced it. Levels come from the data, and a
candidate needs a trigger plus at least one supporting factor - one signal on its own is not a trap."""
import pytest

from core import traps


def state(**over) -> dict:
    """A neutral state: nothing swept, delta flat, price inside value."""
    base = {
        "spot": {"price": 80_000.0},
        "volatility": {"1h": {"available": True, "atr": 400.0, "bands": {
            "mean": 80_000.0, "sigma1_up": 80_500.0, "sigma1_dn": 79_500.0,
            "sigma2_up": 81_000.0, "sigma2_dn": 79_000.0, "sigma3_up": 81_500.0, "sigma3_dn": 78_500.0}}},
        "liquidity": {"1h": {
            "pools": {"buyside": {"price": 80_800.0, "touches": 3, "swept": False, "reclaimed": False,
                                  "broken": False, "sweep_bars_ago": None},
                      "sellside": {"price": 79_200.0, "touches": 3, "swept": False, "reclaimed": False,
                                   "broken": False, "sweep_bars_ago": None},
                      "recent_sweep": None},
            "cumulative_delta": {"available": True, "state": "flat", "change_over_20_bars": 0.0,
                                 "price_change_over_20_bars": 0.0},
            "volume_profile": {"available": True, "point_of_control": 80_000.0, "value_area_high": 80_600.0,
                               "value_area_low": 79_400.0, "price_position": "inside value",
                               "rotation": "balanced inside value",
                               "excess": {"low": 0, "high": 0, "last_low_bars_ago": None,
                                          "last_high_bars_ago": None}},
            "open_interest": {"15m": 0.0, "1h": 0.0, "4h": 0.0}}},
        "smc": {"1h": {"structure": {"available": True, "bias": "unclear", "last_event": None},
                       "premium_discount": {"available": True, "zone": "equilibrium", "position": 0.5}}},
        "derivatives": {"futures": {"funding_rate": 0.0001, "funding_7d_mean": 0.0001,
                                    "oi_change_24h_pct": 0.0}},
    }
    for path, value in over.items():
        node = base
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return base


TFS = ("1h",)


# ---------- nothing on a quiet tape ----------
def test_a_quiet_market_produces_no_candidates():
    assert traps.find_candidates(state(), TFS) == []


def test_a_sweep_on_its_own_is_not_enough():
    """One signal is a data point. The spec's trap is a sweep plus flow that disagrees with the break."""
    swept = state(**{"liquidity.1h.pools": {
        "buyside": {"price": 80_800.0, "touches": 4, "swept": True, "reclaimed": True, "broken": False,
                    "sweep_bars_ago": 1},
        "sellside": {"price": 79_200.0, "touches": 3, "swept": False, "reclaimed": False, "broken": False,
                     "sweep_bars_ago": None},
        "recent_sweep": {"side": "buyside", "price": 80_800.0, "bars_ago": 1}}})
    assert traps.find_candidates(swept, TFS) == []


# ---------- the spec's bull trap ----------
def bull_trap_state() -> dict:
    return state(**{
        "liquidity.1h.pools": {
            "buyside": {"price": 80_800.0, "touches": 4, "swept": True, "reclaimed": True, "broken": False,
                        "sweep_bars_ago": 1},
            "sellside": {"price": 79_200.0, "touches": 3, "swept": False, "reclaimed": False,
                         "broken": False, "sweep_bars_ago": None},
            "recent_sweep": {"side": "buyside", "price": 80_800.0, "bars_ago": 1}},
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -900.0,
                                          "price_change_over_20_bars": 450.0},
        "derivatives.futures": {"funding_rate": 0.0004, "funding_7d_mean": 0.0001,
                                "oi_change_24h_pct": 6.0},
        "smc.1h.premium_discount": {"available": True, "zone": "premium", "position": 0.86},
    })


def test_a_swept_high_with_absorption_is_a_bull_trap_candidate():
    found = traps.find_candidates(bull_trap_state(), TFS)
    assert len(found) == 1
    c = found[0]
    assert c.side == "bull_trap" and c.timeframe == "1h"
    assert c.level == 80_800.0, "the level is the liquidity that was taken"
    assert c.invalidation > c.level, "above the raid the break was real"
    assert c.plays_out < c.level, "it pays off when price returns into value"
    assert c.strength >= 3
    joined = " ".join(c.evidence).lower()
    assert "swept" in joined and "absorption" in joined
    assert any("funding" in e.lower() for e in c.evidence)
    assert any("open interest" in e.lower() for e in c.evidence)


def test_the_invalidation_sits_a_fraction_of_an_atr_beyond_the_raid():
    """Pinning the relationship, not the constant: the buffer has to be wide enough that noise does not
    invalidate a live trap, and the constant carries that decision."""
    c = traps.find_candidates(bull_trap_state(), TFS)[0]
    assert c.invalidation == pytest.approx(c.level + traps.INVALIDATION_ATR * 400.0)
    assert traps.INVALIDATION_ATR >= 0.5, "tighter than this and ordinary noise resolves the trap"


def test_acceptance_above_value_with_selling_also_qualifies():
    """The other bull-trap shape in the spec: price holding above value while delta drains."""
    s = state(**{
        "liquidity.1h.volume_profile": {"available": True, "point_of_control": 80_000.0,
                                        "value_area_high": 80_600.0, "value_area_low": 79_400.0,
                                        "price_position": "above value",
                                        "rotation": "rotating up out of value",
                                        "excess": {"low": 0, "high": 2, "last_low_bars_ago": None,
                                                   "last_high_bars_ago": 1}},
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -700.0,
                                          "price_change_over_20_bars": 300.0},
        "spot.price": 80_750.0,
    })
    found = traps.find_candidates(s, TFS)
    assert [c.side for c in found] == ["bull_trap"]
    assert found[0].level == 80_600.0, "the value area high is the level being faked"


# ---------- the mirror ----------
def test_a_swept_low_with_bullish_absorption_is_a_bear_trap_candidate():
    s = state(**{
        "liquidity.1h.pools": {
            "buyside": {"price": 80_800.0, "touches": 3, "swept": False, "reclaimed": False,
                        "broken": False, "sweep_bars_ago": None},
            "sellside": {"price": 79_200.0, "touches": 5, "swept": True, "reclaimed": True,
                         "broken": False, "sweep_bars_ago": 2},
            "recent_sweep": {"side": "sellside", "price": 79_200.0, "bars_ago": 2}},
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bullish absorption",
                                          "change_over_20_bars": 800.0,
                                          "price_change_over_20_bars": -400.0},
        "smc.1h.premium_discount": {"available": True, "zone": "discount", "position": 0.12},
    })
    found = traps.find_candidates(s, TFS)
    assert len(found) == 1 and found[0].side == "bear_trap"
    c = found[0]
    assert c.level == 79_200.0 and c.invalidation < c.level and c.plays_out > c.level
    assert any("discount" in e.lower() for e in c.evidence)


def test_a_clean_break_beyond_the_pool_is_not_a_trap():
    """Broken, not swept: price closed beyond the liquidity and kept it. That is the real thing."""
    s = bull_trap_state()
    s["liquidity"]["1h"]["pools"]["buyside"].update({"swept": False, "reclaimed": False, "broken": True})
    s["liquidity"]["1h"]["pools"]["recent_sweep"] = None
    assert [c.side for c in traps.find_candidates(s, TFS)] == []


# ---------- robustness ----------
def test_detection_survives_missing_pieces():
    thin = {"spot": {"price": 80_000.0}, "liquidity": {"1h": {}}, "smc": {"1h": {}},
            "volatility": {"1h": {}}, "derivatives": {}}
    assert traps.find_candidates(thin, TFS) == []
    assert traps.find_candidates({}, TFS) == []


def test_candidates_are_reported_per_timeframe_strongest_first():
    s = bull_trap_state()
    s["liquidity"]["4h"] = s["liquidity"]["1h"]
    s["smc"]["4h"] = s["smc"]["1h"]
    s["volatility"]["4h"] = s["volatility"]["1h"]
    found = traps.find_candidates(s, ("1h", "4h"))
    assert {c.timeframe for c in found} == {"1h", "4h"}
    assert found == sorted(found, key=lambda c: -c.strength)
