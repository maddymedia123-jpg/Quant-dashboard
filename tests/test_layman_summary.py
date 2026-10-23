"""His Section 6: every desk ends with the same standardized layman summary, built from its own numbers."""
from types import SimpleNamespace as NS

from core import layman_summary as ls
from ui import panels

NOON = 1_790_035_200_000 - (1_790_035_200_000 % 86_400_000) + 6 * 3_600_000   # 06:00 UTC


def lvl(kind, price):
    return NS(kind=kind, price=price)


def desk(direction="BEARISH", confidence=0.7, invalidation=82_600.0, tf="1h", horizon="next 4h", atr=200.0,
         levels=None):
    levels = levels if levels is not None else [lvl("support", 80_800.0), lvl("support", 80_200.0),
                                                lvl("resistance", 81_600.0), lvl("resistance", 82_300.0)]
    return NS(price=81_100.0, chart_tf=tf, levels=levels,
              direction=NS(direction=direction, confidence=confidence, invalidation=invalidation),
              vol=NS(atr=atr, exp_low=79_900.0, exp_high=82_900.0, horizon_label=horizon))


DERIVS = {"taker_delta": {"over": "4h", "delta_btc": -138.4}, "funding": {"latest_8h_rate": 0.00004},
          "open_interest": [{"over": "1h", "change_pct": 0.1}, {"over": "4h", "change_pct": -1.1}]}


def test_a_bearish_desk_sells_the_ceiling_with_the_stop_above_and_targets_below():
    s = ls.lines(desk(), DERIVS, 32, NOON)
    assert s["bias"] == "Bearish (strong) / Sell a rally into $81,600"
    assert s["entry"] == "$81,550 – $81,650"                      # a quarter ATR either side
    assert s["invalidation"] == "$82,600"                         # the desk's own, beyond the zone
    assert (s["target1"], s["target2"]) == ("$80,800", "$80,200")
    assert s["condition"] == "A 1-hour candle closing above $82,600 cancels the short."


def test_a_bullish_desk_mirrors_it():
    s = ls.lines(desk("BULLISH", 0.4, 80_000.0, tf="1d", horizon="next 7 days"), DERIVS, 60, NOON)
    assert s["bias"] == "Bullish (moderate) / Buy a dip into $80,800"
    assert s["invalidation"] == "$80,000" and (s["target1"], s["target2"]) == ("$81,600", "$82,300")
    assert s["condition"] == "A daily candle closing below $80,000 cancels the long."
    assert s["summary"].startswith("Buyers have the edge for the next 7 days")


def test_an_invalidation_inside_the_zone_is_pushed_past_it():
    f = ls.framework(desk(invalidation=81_620.0))                  # inside the $81,550-81,650 zone
    assert f.invalidation == 81_650.0 + 0.5 * 200.0


def test_sideways_takes_his_example_from_the_nearer_edge_and_names_both_ways_out():
    s = ls.lines(desk("NEUTRAL", 0.1, None), DERIVS, 50, NOON)     # $300 above the floor, $500 below the ceiling
    assert s["bias"] == "Range-Bound / Conditional Long on Floor Retest at $80,800"
    assert "picks a side on a 1-hour close above $81,600 (up) or below $80,650 (down)" in s["summary"]
    near_top = desk("NEUTRAL", 0.1, None, levels=[lvl("support", 80_200.0), lvl("resistance", 81_300.0)])
    assert ls.lines(near_top, DERIVS, 50, NOON)["bias"].startswith("Range-Bound / Conditional Short on Ceiling Retest")


def test_the_summary_tells_mood_flow_and_positions_in_words():
    s = ls.lines(desk(), DERIVS, 32, NOON)["summary"]
    assert "The crowd's mood is fear (32/100)." in s
    assert "more selling than buying at market (net 138 BTC sold)" in s
    assert "buyers pay a small premium to stay in" in s and "open positions are shrinking (-1.1% in 4h)" in s
    assert s.count(". ") <= 3, "one to three sentences, as his format asks"
    bare = ls.lines(desk(), None, None, NOON)["summary"]
    assert "mood" not in bare and "market" not in bare, "missing feeds leave the sentence out, not a guess"


def test_window_one_is_the_measured_hours_and_window_two_says_why_it_is_empty():
    s = ls.lines(desk(), DERIVS, 50, NOON)
    assert s["window1"].startswith("17:00 – 21:00 PKT") and "(12:00–16:00 UTC)" in s["window1"]
    assert "happening now" in ls.lines(desk(), DERIVS, 50, NOON + 7 * 3_600_000)["window1"]
    assert "no better than chance" in s["window2"]


def test_no_history_says_so_instead_of_inventing_levels():
    s = ls.lines(desk(atr=None), DERIVS, 50, NOON)
    assert s["entry"] == "—" and s["target1"] == "—" and s["condition"] == "No clear line yet."


def test_the_box_follows_his_format_in_order_with_no_indicator_words():
    h = panels.layman_summary_html(desk(), "Live Recon", DERIVS, 32, NOON)
    order = ["📌 LAYMAN SUMMARY", "📊 Quantitative Swing Trade Framework", "Directional Bias:", "Exact Entry Zone:",
             "Hard Invalidation Level:", "Target 1:", "Target 2:", "⏳ Time Reversal Windows &amp; Timeline (PKT)",
             "Window 1 (Active Peak Window):", "Window 2 (Cycle Confluence):", "❌ Bias Invalidation Condition:"]
    at = [h.index(x) for x in order]
    assert at == sorted(at)
    for word in ("EMA", "RSI", "σ", "Fibonacci", "CVD", "squeeze", "moving average", "funding"):
        assert word.lower() not in h.lower(), word
