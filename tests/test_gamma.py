"""Dealer gamma regime: the Black-Scholes gamma, the profile built from it, and where it reaches the page."""
import json
import math
import pathlib

import pytest

from core.data.deribit_options import gamma_regime, parse_book_summary
from core.indicators import gamma as gx

FX = pathlib.Path(__file__).parent / "fixtures"
DAY = 86_400_000
NOW = gx._expiry_ms("01OCT26") - 7 * DAY        # one week before a 01 Oct expiry


def _opt(strike, kind, oi, iv=50.0, code="01OCT26"):
    return {"instrument_name": f"BTC-{code}-{strike}-{kind}", "open_interest": oi, "mark_iv": iv}


# ---------- the formula ----------
def test_gamma_matches_the_closed_form_at_the_money():
    s, t, v = 100_000.0, 7 / 365, 0.5
    expect = math.exp(-0.5 * (0.5 * v * math.sqrt(t)) ** 2) / (math.sqrt(2 * math.pi) * s * v * math.sqrt(t))
    assert gx.bs_gamma(s, s, t, v) == pytest.approx(expect)


def test_gamma_peaks_at_the_strike_and_falls_away_from_it():
    g = [gx.bs_gamma(s, 100_000, 7 / 365, 0.5) for s in (80_000, 95_000, 100_000, 105_000, 120_000)]
    assert g[2] > g[1] > g[0] and g[2] > g[3] > g[4]


def test_degenerate_inputs_give_zero_not_an_error():
    assert gx.bs_gamma(0, 1, 1, 1) == gx.bs_gamma(1, 1, 0, 1) == gx.bs_gamma(1, 1, 1, 0) == 0.0


def test_expiry_settles_at_eight_utc():
    assert gx._expiry_ms("01OCT26") % DAY == 8 * 3_600_000
    assert gx._expiry_ms("garbage") is None


# ---------- the chain ----------
def test_options_beyond_thirty_days_or_already_settling_are_left_out():
    rows = gx.chain_rows([_opt(100000, "C", 10), _opt(100000, "C", 10, code="25DEC26"),
                          _opt(100000, "C", 10, code="24SEP26")], NOW)
    assert [r["expiry"] for r in rows] == ["01OCT26"], "the December is too far and the 24 Sep has settled"


def test_rows_without_iv_or_open_interest_are_dropped_not_guessed():
    rows = gx.chain_rows([_opt(100000, "C", 0), _opt(100000, "P", 5, iv=None), {"instrument_name": "junk"},
                          _opt(100000, "P", 5)], NOW)
    assert len(rows) == 1 and rows[0]["kind"] == "P"


# ---------- the profile ----------
def test_a_call_heavy_book_is_positive_gamma_and_its_walls_are_the_heaviest_strikes():
    chain = [_opt(105000, "C", 900), _opt(110000, "C", 300), _opt(95000, "P", 400), _opt(90000, "P", 100)]
    p = gx.profile(chain, 100_000.0, NOW)
    assert p.available and p.regime == "positive" and p.net_gex_usd > 0
    assert (p.call_wall, p.put_wall) == (105000.0, 95000.0)


def test_a_put_heavy_book_is_negative_gamma():
    p = gx.profile([_opt(100000, "P", 900), _opt(105000, "C", 100)], 100_000.0, NOW)
    assert p.regime == "negative" and p.net_gex_usd < 0


def test_the_zero_gamma_level_sits_between_put_and_call_concentrations():
    # puts at 90k, calls at 110k, equal weight: net gamma is negative low, positive high, zero near 100k
    p = gx.profile([_opt(90000, "P", 500), _opt(110000, "C", 500)], 100_000.0, NOW)
    assert p.zero_gamma is not None and 95_000 < p.zero_gamma < 105_000
    below = gx._gex(gx.chain_rows([_opt(90000, "P", 500), _opt(110000, "C", 500)], NOW), p.zero_gamma - 2000)[0]
    above = gx._gex(gx.chain_rows([_opt(90000, "P", 500), _opt(110000, "C", 500)], NOW), p.zero_gamma + 2000)[0]
    assert below < 0 < above, "the flip really is a sign change"


def test_a_book_with_no_crossing_says_so():
    p = gx.profile([_opt(100000, "C", 500)], 100_000.0, NOW)
    assert p.zero_gamma is None and "does not cross zero" in p.note


def test_gex_scales_with_open_interest():
    a = gx.profile([_opt(100000, "C", 100)], 100_000.0, NOW).net_gex_usd
    b = gx.profile([_opt(100000, "C", 300)], 100_000.0, NOW).net_gex_usd
    assert b == pytest.approx(3 * a)


def test_no_spot_or_no_usable_options_is_unavailable_and_says_why():
    assert not gx.profile([_opt(100000, "C", 5)], None, NOW).available
    p = gx.profile([_opt(100000, "C", 5, code="25DEC26")], 100_000.0, NOW)
    assert not p.available and "30 days" in p.note


def test_the_judges_get_the_assumption_with_the_numbers():
    s = gx.profile([_opt(90000, "P", 500), _opt(110000, "C", 500)], 100_000.0, NOW).to_state()
    assert s["assumption"].startswith("naive dealer sign") and "profile" not in s
    assert set(s) >= {"regime", "zero_gamma_level", "call_wall", "put_wall", "max_gamma_strike", "net_gex_usd_per_1pct"}


# ---------- the recorded Deribit chain ----------
def test_the_recorded_chain_is_timed_from_its_own_quote_not_the_clock():
    raw = json.loads((FX / "deribit_book_summary.json").read_text(encoding="utf-8"))
    g = parse_book_summary(raw).gamma
    assert g["available"] and g["regime"] in ("positive", "negative")
    assert g["expiries"][0] == "9SEP26", "the chain was quoted in September 2026; the clock today would drop these"
    assert abs(g["spot"] - raw["result"][0]["estimated_delivery_price"]) < 1


def test_a_broken_chain_fails_the_regime_only():
    raw = json.loads((FX / "deribit_book_summary.json").read_text(encoding="utf-8"))
    assert gamma_regime({"result": "not a list"})["available"] is False
    snap = parse_book_summary(raw)
    assert snap.max_pain is not None, "max pain still computed beside the regime"


# ---------- what the page and the judges see ----------
def test_the_bar_names_the_regime_its_playbook_and_its_levels():
    from ui import panels
    g = {"available": True, "regime": "positive", "zero_gamma_level": 81_404, "call_wall": 90_000, "put_wall": 80_000,
         "max_gamma_strike": 90_000, "net_gex_usd_per_1pct": 23_500_000, "expiries": ["17OCT26"], "options_used": 300}
    h = panels.gamma_bar_html(g, 81_822)
    assert "POSITIVE GAMMA · BALANCE" in h and "do not trust breakouts" in h
    assert "$80,000" in h and "$81,404" in h and "$90,000" in h and "+$23.5M" in h
    assert "price above the flip by $418" in h and "naive dealer sign" in h
    neg = panels.gamma_bar_html({**g, "regime": "negative", "net_gex_usd_per_1pct": -5e6}, 80_000)
    assert "NEGATIVE GAMMA · IMBALANCE" in neg and "-$5.0M" in neg and "below the flip" in neg


def test_the_bar_says_why_when_there_is_no_regime():
    from ui import panels
    assert "options chain unavailable" in panels.gamma_bar_html(None)
    assert "no index price" in panels.gamma_bar_html({"available": False, "note": "no index price in the options chain"})


def test_the_judges_receive_the_regime_inside_the_options_data():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.live_recon import build_state
    from core.agents.recon_profiles import PROFILES
    from core.data.offline import fixture_market
    st = build_state(fixture_market(), {}, profile=PROFILES["live"])
    assert st["derivatives"]["options"]["gamma"]["regime"] in ("positive", "negative")


# ---------- one profile per desk, from that desk's own expiries (his v15, Section 2) ----------
def test_each_desk_builds_its_walls_from_only_the_options_settling_in_its_window():
    tomorrow, next_week, quarterly = "24SEP26", "30SEP26", "25DEC26"
    now = gx._expiry_ms("24SEP26") - 12 * 3_600_000                       # 20:00 UTC the evening before
    chain = [_opt(98_000, "P", 400, code=tomorrow), _opt(102_000, "C", 300, code=tomorrow),
             _opt(95_000, "P", 5_000, code=next_week), _opt(105_000, "C", 4_000, code=next_week),
             _opt(90_000, "P", 90_000, code=quarterly), _opt(110_000, "C", 80_000, code=quarterly)]
    by = gx.desk_profiles(chain, 100_000.0, now)
    assert by["live"]["expiries"] == [tomorrow] and by["live"]["put_wall"] == 98_000
    assert by["weekly"]["expiries"] == [tomorrow, next_week] and by["weekly"]["put_wall"] == 95_000
    assert quarterly in by["monthly"]["expiries"], "the monthly desk sees the quarterly a 30-day cut drops"
    assert quarterly not in gx.profile(chain, 100_000.0, now).to_state()["expiries"]
    assert [by[k]["horizon_days"] for k in ("live", "intraday", "weekly", "monthly")] == [1, 3, 8, 100]


def test_desk_windows_are_the_same_ones_max_pain_is_read_from():
    from core.indicators.desk_derivs import WINDOWS
    assert gx.DESK_DAYS == {k: w.expiry_days for k, w in WINDOWS.items()}


def test_the_book_carries_a_profile_per_desk_and_a_bad_chain_takes_nothing_down():
    from core.data.deribit_options import gamma_by_desk
    snap = parse_book_summary(json.loads((FX / "deribit_book_summary.json").read_text()))
    assert set(snap.gamma_by_desk) == {"live", "intraday", "weekly", "monthly"}
    assert gamma_by_desk({"result": None}) == {}


def test_a_desk_reads_its_own_gamma_and_falls_back_to_the_30_day_one():
    from types import SimpleNamespace as NS
    from core.indicators.desk_derivs import gamma_for
    own, old = {"regime": "negative", "available": True}, {"regime": "positive", "available": True}
    m = NS(options=NS(available=True, gamma=old, gamma_by_desk={"weekly": own}))
    assert gamma_for("weekly", m) is own
    assert gamma_for("live", m) is old and gamma_for(None, m) is old
    assert gamma_for("weekly", NS(options=NS(available=False))) is None


def test_the_bar_names_whose_expiries_it_is_built_from():
    from ui import panels
    g = {"available": True, "regime": "negative", "horizon_days": 8, "expiries": ["x"], "options_used": 3}
    h = panels.gamma_bar_html(g, 100_000.0, "Weekly")
    assert "Weekly: options settling in the next 8 days" in h and "next 8 days" in h
    assert "settling" not in panels.gamma_bar_html(g, 100_000.0)
