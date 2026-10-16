"""Qualified value-area breaks: the three conditions, what follows a break, and where the read is shown."""
import numpy as np
import pandas as pd
import pytest

from core.indicators import vwap_break as vb

H = 3_600_000
DAY = 86_400_000
T0 = 20_000 * DAY                     # a UTC midnight
RANGE = 216                           # nine days of hourly range, so the break opens a new UTC day


def _bar(c, o=None, hi=None, lo=None, v=10.0):
    o = c if o is None else o
    return {"open": o, "high": max(o, c) + 0.5 if hi is None else hi, "low": min(o, c) - 0.5 if lo is None else lo,
            "close": c, "volume": v}


def _frame(after=(), brk=None):
    """A balanced hourly range around 100, one break candle, what follows it, then a forming candle."""
    rng = np.random.default_rng(7)
    rows = [_bar(float(100 + rng.uniform(-1, 1))) for _ in range(RANGE)]
    rows[-1] = _bar(100.0)                                         # the close before the break is inside value
    rows.append(brk or _bar(106.0, o=101.0, lo=100.8, v=30.0))
    rows += list(after)
    rows.append(_bar(rows[-1]["close"]))                          # forming: never read
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", [T0 + i * H for i in range(len(df))])
    return df


# ---------- the three conditions ----------
def test_all_three_conditions_make_a_qualified_break_awaiting_its_pullback():
    r = vb.read(_frame(after=[_bar(108.0, o=106.0, lo=106.5), _bar(109.0, o=108.0, lo=107.8)]), "1h")
    assert r.state == "qualified" and r.direction == "up" and r.bars_ago == 2
    assert all(ok for _, ok in r.checks) and r.volume_ratio == pytest.approx(3.0)
    assert 100 < r.edge < 102, "the value area high of the range, not of the break"


def test_a_climactic_break_is_measured_against_the_value_before_it():
    """With enough volume the break candle would swallow the value area if it were counted in it, and the
    biggest breakouts would read as no break at all."""
    r = vb.read(_frame(brk=_bar(106.0, o=101.0, lo=100.8, v=5000.0), after=[_bar(108.0, o=106.0, lo=106.5)]), "1h")
    assert r.state == "qualified" and r.edge < 102


def test_a_break_without_volume_expansion_is_unqualified():
    r = vb.read(_frame(brk=_bar(106.0, o=101.0, lo=100.8, v=12.0)), "1h")
    assert r.state == "unqualified" and dict(r.checks)["volume_expansion"] is False
    assert "trap risk" in r.headline


def test_a_close_that_vwap_has_not_accepted_is_unqualified():
    # a huge upper wick puts the day's VWAP above the close: price closed outside value but below VWAP
    r = vb.read(_frame(brk=_bar(106.0, o=101.0, hi=140.0, lo=100.8, v=30.0)), "1h")
    assert r.state == "unqualified" and dict(r.checks) == {"close_outside_value": True, "volume_expansion": True,
                                                         "vwap_acceptance": False}
    assert r.vwap > 106


def test_the_vwap_is_anchored_at_the_open_of_the_next_timeframe_up():
    assert vb.VWAP_ANCHOR == {"1h": "1d", "4h": "1w", "1d": "1M"}
    t = T0 + 5 * DAY + 13 * H
    assert vb.period_start(t, "1d") == T0 + 5 * DAY
    monday = vb.period_start(t, "1w")
    assert (monday - vb.EPOCH_MONDAY_MS) % (7 * DAY) == 0 and 0 <= t - monday < 7 * DAY
    assert vb.period_start(1_791_000_000_000, "1M") == 1_790_812_800_000, "1 Oct 2026 00:00 UTC"


# ---------- after the break ----------
def test_the_first_retest_of_the_edge_that_holds_is_the_pullback_entry():
    r = vb.read(_frame(after=[_bar(108.0, o=106.0, lo=106.5), _bar(104.0, o=108.0, lo=101.0),
                              _bar(107.0, o=104.0, lo=103.5)]), "1h")
    assert r.state == "pullback" and r.pullback_bars_ago == 1 and "the entry" in r.headline


def test_a_close_back_inside_value_fails_the_break():
    r = vb.read(_frame(after=[_bar(108.0, o=106.0, lo=106.5), _bar(100.0, o=108.0)]), "1h")
    assert r.state == "failed" and "a bull trap" in r.headline


def test_the_forming_candle_is_never_read():
    df = _frame(after=[_bar(108.0, o=106.0, lo=106.5)])
    df.loc[df.index[-1], ["close", "low"]] = [99.0, 98.0]          # a forming candle back inside value
    assert vb.read(df, "1h").state == "qualified"


def test_a_downside_break_mirrors_the_upside():
    rows = _frame(brk=_bar(94.0, o=99.0, hi=99.2, v=30.0), after=[_bar(92.0, o=94.0, hi=93.5)])
    r = vb.read(rows, "1h")
    assert r.state == "qualified" and r.direction == "down" and 98 < r.edge < 100


def test_a_range_with_no_close_outside_value_is_none():
    df = _frame(brk=_bar(100.0, v=10.0))
    df.loc[df.index[-vb.SEARCH_BARS - 2:], ["open", "high", "low", "close"]] = [100.0, 100.3, 99.7, 100.0]
    assert vb.read(df, "1h").state == "none", "every recent close sits at the point of control"
    df.loc[df.index[-3], "close"] = 104.0                          # one poke outside, on no volume
    assert vb.read(df, "1h").state == "unqualified"


# ---------- where it applies ----------
def test_nothing_below_an_hour_is_read():
    r = vb.read(_frame(), "15m")
    assert not r.available and "1h, 4h, 1d only" in r.note
    assert set(vb.read_all({"15m": _frame(), "1h": _frame()}, ("15m", "1h", "4h"))) == {"1h", "4h"}


def test_too_few_candles_says_so():
    assert vb.read(_frame().head(20), "1h").state == "too_few"


def test_the_judges_get_the_read_on_the_desks_hour_and_above_only():
    import os
    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    from core.agents.deck import sub_agent_deck
    from core.agents.live_recon import build_state
    from core.agents.recon_profiles import PROFILES
    from core.data.offline import fixture_market
    st = build_state(fixture_market(), {}, profile=PROFILES["live"])
    assert "qualified_break" in st["liquidity"]["1h"] and "qualified_break" in st["liquidity"]["4h"]
    assert "qualified_break" not in st["liquidity"]["15m"]
    auction = next(a for a in sub_agent_deck(st, PROFILES["live"]) if a.agent_no == 2)
    row = next(r for r in auction.rows if r[0] == "Value break")
    assert row[1] == st["liquidity"][auction.timeframe]["qualified_break"]["read"]


# ---------- the card ----------
def test_the_card_shows_each_timeframe_with_its_conditions():
    from ui import panels
    reads = {"1h": vb.read(_frame(brk=_bar(106.0, o=101.0, lo=100.8, v=12.0)), "1h"),
             "4h": vb.read(_frame(after=[_bar(108.0, o=106.0, lo=106.5), _bar(104.0, o=108.0, lo=101.0)]), "1h"),
             "1d": vb.BreakRead("1d", False, state="too_few", note="fewer than 40 completed 1d candles")}
    h = panels.value_breaks_html(reads)
    assert "Value-area breaks" in h and "trap risk" in h and "first pullback held" in h
    assert "□ volume expansion" in h and "■ close outside value" in h
    assert "fewer than 40 completed 1d candles" in h and "Read on 1H, 4H and 1D only" in h


def test_a_failed_break_is_coloured_against_its_direction():
    from ui import panels
    failed_down = vb.BreakRead("4h", True, state="failed", direction="down", edge=81_700, vwap=83_150,
                               volume_ratio=3.7, bars_ago=11, checks=(("close_outside_value", True),))
    h = panels.value_breaks_html({"4h": failed_down})
    assert "<td class='up'>Break below value failed back inside value: a bear trap</td>" in h
    assert h.startswith("<div class=\"ti-card up\"")
