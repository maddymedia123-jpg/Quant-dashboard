"""Deterministic trap detection: what the TRAP desk is allowed to call a candidate.

The spec's bull trap is a fake breakout above resting liquidity into a negative delta divergence; the
bear trap is the mirror below value with absorption. Detection reads the same state the desk agents get,
so a trap is never asserted without the evidence that produced it. Levels come from the data, and a
candidate needs a trigger plus at least one *independent* supporting factor - one signal on its own is
not a trap, and the same reading counted twice is still one signal.

A candidate must also be live: the raid recent, the level reclaimed, and the question not already
answered by the price that produced it. Those three gates are what stop the ledger filling with traps
that settle themselves on the next page load.

The fixtures below are built to be physically coherent, because an impossible tape proves nothing. Note
that the value area comes from the 200-bar volume profile while premium/discount comes from the 100-bar
dealing range, so price can sit inside value and in the premium of the range at the same time - the two
readings are over different windows.
"""
import pytest

from core import traps


def state(**over) -> dict:
    """A neutral state: nothing swept, delta flat, price inside value at the point of control."""
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


def swept_pools(side: str, bars_ago: int = 1) -> dict:
    """Prior highs (or lows) taken and reclaimed; the untouched side left alone."""
    raided = {"price": 80_800.0 if side == "buyside" else 79_200.0, "touches": 4, "swept": True,
              "reclaimed": True, "broken": False, "sweep_bars_ago": bars_ago}
    quiet = {"price": 79_200.0 if side == "buyside" else 80_800.0, "touches": 3, "swept": False,
             "reclaimed": False, "broken": False, "sweep_bars_ago": None}
    pools = {"buyside": raided if side == "buyside" else quiet,
             "sellside": quiet if side == "buyside" else raided,
             "recent_sweep": {"side": side, "price": raided["price"], "bars_ago": bars_ago}}
    return pools


def bull_trap_state(bars_ago: int = 1, price: float = 80_540.0) -> dict:
    """The textbook swing failure: the 80,800 highs are taken, price falls back inside value.

    Price at 80,540 is back below the swept pool and back inside the 79,400-80,600 value area, while
    sitting at 0.86 of a 79,000-80,800 dealing range - premium on the 100-bar window, inside value on
    the 200-bar profile. Delta drained while price rose, funding is above its weekly mean, open interest
    built into the move, and 1h structure is bearish, so the break runs against it.
    """
    return state(**{
        "spot.price": price,
        "liquidity.1h.pools": swept_pools("buyside", bars_ago),
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -900.0,
                                          "price_change_over_20_bars": 450.0},
        "derivatives.futures": {"funding_rate": 0.0004, "funding_7d_mean": 0.0001,
                               "oi_change_24h_pct": 6.0},
        "smc.1h.premium_discount": {"available": True, "zone": "premium", "position": 0.86},
        "smc.1h.structure": {"available": True, "bias": "bearish",
                             "last_event": {"type": "CHoCH", "direction": "bearish"}},
    })


def bear_trap_state(bars_ago: int = 2, price: float = 79_460.0) -> dict:
    """The mirror: the 79,200 lows are swept, price reclaims back into value at 0.14 of the range."""
    return state(**{
        "spot.price": price,
        "liquidity.1h.pools": swept_pools("sellside", bars_ago),
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bullish absorption",
                                          "change_over_20_bars": 800.0,
                                          "price_change_over_20_bars": -400.0},
        "derivatives.futures": {"funding_rate": -0.0002, "funding_7d_mean": 0.0001,
                               "oi_change_24h_pct": 5.0},
        "smc.1h.premium_discount": {"available": True, "zone": "discount", "position": 0.14},
        "smc.1h.structure": {"available": True, "bias": "bullish",
                             "last_event": {"type": "BOS", "direction": "bullish"}},
    })


# ---------- nothing on a quiet tape ----------
def test_a_quiet_market_produces_no_candidates():
    assert traps.find_candidates(state(), TFS) == []


def test_a_sweep_on_its_own_is_not_enough():
    """One signal is a data point. The spec's trap is a sweep plus flow that disagrees with the break."""
    swept = state(**{"spot.price": 80_540.0, "liquidity.1h.pools": swept_pools("buyside")})
    assert traps.find_candidates(swept, TFS) == []


def test_a_clean_break_beyond_the_pool_is_not_a_trap():
    """Broken, not swept: price closed beyond the liquidity and kept it. That is the real thing."""
    s = bull_trap_state()
    s["liquidity"]["1h"]["pools"]["buyside"].update({"swept": False, "reclaimed": False, "broken": True})
    s["liquidity"]["1h"]["pools"]["recent_sweep"] = None
    assert traps.find_candidates(s, TFS) == []


# ---------- the spec's bull trap ----------
def test_a_swept_high_with_absorption_is_a_bull_trap_candidate():
    found = traps.find_candidates(bull_trap_state(), TFS)
    assert len(found) == 1
    c = found[0]
    assert c.side == "bull_trap" and c.timeframe == "1h"
    assert c.level == 80_800.0, "the level is the liquidity that was taken"
    assert c.invalidation > c.level, "above the raid the break was real"
    assert c.plays_out < c.level, "it pays off when price returns into value"
    assert c.plays_out == 80_000.0, "the point of control is the target, not an arbitrary distance"
    assert c.strength == len(c.evidence) == 6, "trigger plus five independent factors, counted exactly"
    joined = " ".join(c.evidence).lower()
    for expected in ("swept and reclaimed", "absorption", "funding", "open interest", "premium",
                     "against bearish structure"):
        assert expected in joined, f"the {expected} factor is missing from the evidence"


def test_the_invalidation_sits_between_half_and_one_atr_beyond_the_raid():
    """Pinning the band, not one constant: too tight and noise resolves a live trap, too wide and the
    trap risks more than the move it is trading. A floor on its own let 2.0 - four times the intended
    buffer - pass unnoticed."""
    c = traps.find_candidates(bull_trap_state(), TFS)[0]
    assert c.invalidation == pytest.approx(c.level + traps.INVALIDATION_ATR * 400.0)
    assert 0.5 <= traps.INVALIDATION_ATR <= 1.0


def test_the_payoff_is_never_nearer_than_the_invalidation():
    """A card that risks 154 points to make 68 advertises a losing trade. When the point of control sits
    inside the invalidation buffer it is pushed out to at least an equal-distance target."""
    near_poc = state(**{
        # a quiet 1h (ATR 120) whose value area sits just under the swept 80,800 highs, so the point of
        # control is only 40 points below the level while the invalidation buffer is 60
        "spot.price": 80_770.0,
        "volatility.1h": {"available": True, "atr": 120.0, "bands": {
            "mean": 80_500.0, "sigma1_up": 80_650.0, "sigma1_dn": 80_350.0,
            "sigma2_up": 80_900.0, "sigma2_dn": 80_100.0, "sigma3_up": 81_050.0, "sigma3_dn": 79_950.0}},
        "liquidity.1h.pools": swept_pools("buyside"),
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -900.0,
                                          "price_change_over_20_bars": 450.0},
        "liquidity.1h.volume_profile": {"available": True, "point_of_control": 80_760.0,
                                        "value_area_high": 80_780.0, "value_area_low": 80_300.0,
                                        "price_position": "inside value", "rotation": "balanced",
                                        "excess": {"low": 0, "high": 0, "last_low_bars_ago": None,
                                                   "last_high_bars_ago": None}},
        "derivatives.futures": {"funding_rate": 0.0004, "funding_7d_mean": 0.0001,
                               "oi_change_24h_pct": 6.0},
    })
    c = traps.find_candidates(near_poc, TFS)[0]
    risk, reward = c.invalidation - c.level, c.level - c.plays_out
    assert reward >= risk > 0, "the point of control was nearer than the buffer and should be pushed out"
    assert c.plays_out == 80_740.0, "half an ATR below the level, not the 80,760 point of control"


def test_acceptance_beyond_value_with_selling_also_qualifies():
    """The other bull-trap shape in the spec: price holding above value while delta drains. Here the
    delta read IS the trigger, so it must not be counted a second time as its own support."""
    s = state(**{
        "spot.price": 80_700.0,
        "liquidity.1h.volume_profile": {"available": True, "point_of_control": 80_000.0,
                                        "value_area_high": 80_600.0, "value_area_low": 79_400.0,
                                        "price_position": "above value",
                                        "rotation": "rotating up out of value",
                                        "excess": {"low": 0, "high": 2, "last_low_bars_ago": None,
                                                   "last_high_bars_ago": 1}},
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -700.0,
                                          "price_change_over_20_bars": 300.0},
        "derivatives.futures": {"funding_rate": 0.0005, "funding_7d_mean": 0.0001,
                               "oi_change_24h_pct": 7.0},
    })
    found = traps.find_candidates(s, TFS)
    assert [c.side for c in found] == ["bull_trap"]
    c = found[0]
    assert c.level == 80_600.0, "the value area high is the level being faked"
    restated = [e for e in c.evidence if e.lower().startswith("cumulative delta shows")]
    assert restated == [], "the delta read fired the trigger; repeating it is not a second signal"
    assert c.strength == len(c.evidence) == 3


def test_the_delta_trigger_needs_support_that_is_not_itself():
    """Strip the independent factors and the acceptance shape has one signal, so it is not a candidate."""
    s = state(**{
        "spot.price": 80_700.0,
        "liquidity.1h.volume_profile": {"available": True, "point_of_control": 80_000.0,
                                        "value_area_high": 80_600.0, "value_area_low": 79_400.0,
                                        "price_position": "above value", "rotation": "rotating up",
                                        "excess": {"low": 0, "high": 0, "last_low_bars_ago": None,
                                                   "last_high_bars_ago": None}},
        "liquidity.1h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -700.0,
                                          "price_change_over_20_bars": 300.0},
    })
    assert traps.find_candidates(s, TFS) == [], "delta alone, twice over, is still one signal"


# ---------- the gates that keep a candidate live ----------
def test_a_stale_raid_is_not_a_candidate():
    """"Immediately reclaimed" is a handful of bars. Without this bound a level swept 189 bars ago -
    days, on the 1h - still produced a card reading "swept and reclaimed 189 bars ago"."""
    assert traps.find_candidates(bull_trap_state(bars_ago=traps.MAX_SWEEP_BARS), TFS), "5 bars is fresh"
    assert traps.find_candidates(bull_trap_state(bars_ago=traps.MAX_SWEEP_BARS + 1), TFS) == []
    assert traps.find_candidates(bull_trap_state(bars_ago=189), TFS) == []


def test_a_break_still_in_progress_is_not_a_trap():
    """A fake break up is only a fake once price is back below the high it took. While price is still
    above the swept level the break is in progress, and calling it a trap is a guess."""
    assert traps.find_candidates(bull_trap_state(price=80_900.0), TFS) == []
    assert traps.find_candidates(bear_trap_state(price=79_100.0), TFS) == []


def test_a_candidate_is_never_born_already_settled():
    """settle() runs on every page load, so a candidate that its own price already resolves would be
    anchored and closed seconds later - 35% of them were, most logged as "the break was real"."""
    for price in (81_100.0, 79_900.0):          # beyond the invalidation, and already at the target
        for s in (bull_trap_state(price=price), bear_trap_state(price=price)):
            for c in traps.find_candidates(s, TFS):
                assert c.resolved_by(price) is None, f"{c.label} at {price} was born settled"


def test_the_bull_trap_candidate_is_live_at_the_price_that_found_it():
    c = traps.find_candidates(bull_trap_state(), TFS)[0]
    assert c.resolved_by(80_540.0) is None
    assert c.resolved_by(c.invalidation) == "invalidated"
    assert c.resolved_by(c.plays_out) == "played_out"


# ---------- the mirror ----------
def test_a_swept_low_with_bullish_absorption_is_a_bear_trap_candidate():
    found = traps.find_candidates(bear_trap_state(), TFS)
    assert len(found) == 1 and found[0].side == "bear_trap"
    c = found[0]
    assert c.level == 79_200.0 and c.invalidation < c.level and c.plays_out > c.level
    joined = " ".join(c.evidence).lower()
    for expected in ("swept and reclaimed", "bullish absorption", "discount", "against bullish structure"):
        assert expected in joined, f"the {expected} factor is missing"
    assert "funding at -0.020%" in joined, "funding below its mean traps crowded shorts"


# ---------- evidence branches that were never exercised ----------
def test_a_stretched_price_is_reported_as_beyond_the_sigma_channel():
    """The 2-sigma factor: dead in every earlier fixture because the bands sat above every price."""
    s = bull_trap_state(price=80_540.0)
    s["volatility"]["1h"]["bands"].update({"mean": 79_800.0, "sigma2_up": 80_400.0,
                                           "sigma3_up": 80_700.0})
    c = traps.find_candidates(s, TFS)[0]
    assert any("2 sigma" in e for e in c.evidence)
    assert c.strength == 7, "one factor more than the same tape inside the channel"


def test_the_counter_structure_factor_needs_the_opposing_bias():
    """A bull trap runs against bearish structure. Bullish structure agrees with the break, so the
    factor must not fire - and neither an unclear nor a missing bias counts."""
    with_bias = traps.find_candidates(bull_trap_state(), TFS)[0]
    assert any("against bearish structure" in e for e in with_bias.evidence)

    agreeing = bull_trap_state()
    agreeing["smc"]["1h"]["structure"] = {"available": True, "bias": "bullish", "last_event": None}
    c = traps.find_candidates(agreeing, TFS)[0]
    assert not any("structure" in e for e in c.evidence)
    assert c.strength == with_bias.strength - 1


def test_a_missing_volatility_block_falls_back_to_a_sane_buffer():
    """The ATR fallback was uncovered, so a 100x wrong constant passed unnoticed. Half a percent of
    price is the right order of magnitude for a BTC 1h ATR; 40% is not."""
    assert 0.001 <= traps.DEFAULT_ATR_PCT <= 0.02
    s = bull_trap_state()
    s["volatility"]["1h"] = {"available": False, "note": "no 1h candles"}
    c = traps.find_candidates(s, TFS)[0]
    expected = 80_540.0 * traps.DEFAULT_ATR_PCT * traps.INVALIDATION_ATR   # a fraction of price, not the level
    assert c.invalidation == pytest.approx(c.level + expected)
    assert c.invalidation - c.level < 0.02 * c.level, "the fallback buffer stays a sane fraction of price"


# ---------- robustness and ordering ----------
def test_detection_survives_missing_pieces():
    thin = {"spot": {"price": 80_000.0}, "liquidity": {"1h": {}}, "smc": {"1h": {}},
            "volatility": {"1h": {}}, "derivatives": {}}
    assert traps.find_candidates(thin, TFS) == []
    assert traps.find_candidates({}, TFS) == []
    assert traps.find_candidates(None, TFS) == []


def test_candidates_are_reported_strongest_first_across_timeframes():
    """The timeframes carry genuinely different evidence, so the order has to be earned. A symmetric
    fixture made this assertion unfailable: equal strengths satisfy any ordering under a stable sort."""
    s = bull_trap_state()
    weaker = state(**{
        "spot.price": 80_540.0,
        "liquidity.4h.pools": swept_pools("buyside"),
        "liquidity.4h.cumulative_delta": {"available": True, "state": "bearish absorption",
                                          "change_over_20_bars": -500.0,
                                          "price_change_over_20_bars": 200.0},
    })
    s["liquidity"]["4h"] = weaker["liquidity"]["4h"]
    s["smc"]["4h"] = {"structure": {"available": True, "bias": "unclear", "last_event": None},
                      "premium_discount": {"available": True, "zone": "equilibrium", "position": 0.5}}
    s["volatility"]["4h"] = weaker["volatility"]["1h"]

    found = traps.find_candidates(s, ("1h", "4h"))
    assert {c.timeframe for c in found} == {"1h", "4h"}
    strengths = [c.strength for c in found]
    assert strengths[0] > strengths[-1], "the fixture must make the two timeframes distinguishable"
    assert strengths == sorted(strengths, reverse=True)
    assert found[0].timeframe == "1h", "the 1h has five supporting factors, the 4h has one"
