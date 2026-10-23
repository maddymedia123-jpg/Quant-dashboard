"""The Plain-English read: everyday words, the call first, jargon left to the detail cards."""
from types import SimpleNamespace as NS

import pandas as pd

from ui import panels


def lvl(kind, price, touches):
    return NS(kind=kind, price=price, touches=touches)


def analysis(direction="BEARISH", confidence=0.76, invalidation=81_926.0, levels=None, rsi=46.0, regime="NORMAL",
             exp=(80_752.0, 82_371.0), tf="1h", horizon="next 4h", patterns=(), divergences=(), reversal=None,
             squeeze=(None, "NONE")):
    if levels is None:
        levels = [lvl("support", 80_985.0, 4), lvl("support", 80_410.0, 2), lvl("resistance", 81_655.0, 7)]
    return NS(direction=NS(direction=direction, confidence=confidence, invalidation=invalidation),
              vol=NS(exp_low=exp[0] if exp else None, exp_high=exp[1] if exp else None, regime=regime,
                     horizon_label=horizon),
              squeeze=NS(score=squeeze[0], direction=squeeze[1]), levels=levels, chart_tf=tf, price=81_557.0,
              rsi=pd.Series([50.0, rsi]), patterns=list(patterns), divergences=list(divergences), reversal=reversal)


def read(a):
    return dict(panels.layman_lines(a))


def test_the_live_recon_example_reads_as_agreed():
    r = read(analysis())
    assert r["What's happening"] == "Strong lean lower. BTC is $81,557 on the 1h chart."
    assert r["Levels that matter"] == ("Floor $80,985 (held 4 times), ceiling $81,655 (rejected 7 times). "
                                       "Normal range for the next 4 hours: $80,752 to $82,371.")
    assert r["What changes the view"] == ("A 1h close above $81,926, or the moving averages turning up, flips this "
                                          "higher. Losing $80,985 opens the way toward $80,410.")
    assert "Also worth noting" not in r, "an ordinary day is three lines"


def test_confidence_is_a_word_never_a_percent():
    for c, word in ((0.2, "Slight"), (0.34, "Slight"), (0.35, "Moderate"), (0.59, "Moderate"), (0.6, "Strong"),
                    (0.84, "Strong"), (0.85, "Very strong")):
        assert read(analysis(confidence=c))["What's happening"].startswith(f"{word} lean lower.")
    assert "%" not in panels.layman_html(analysis())


def test_neutral_names_both_ways_out():
    r = read(analysis("NEUTRAL", 0.14, None, tf="4h", horizon="next 24h"))
    assert r["What's happening"].startswith("No clear side.") and "stuck between its floor and ceiling" in r["What's happening"]
    assert r["What changes the view"] == ("A 4h close below $80,985 tilts it lower, above $81,655 higher, "
                                          "if the moving averages agree.")
    assert "next 24 hours" in r["Levels that matter"]


def test_bullish_losing_the_floor_is_the_flip():
    r = read(analysis("BULLISH", 0.47, 80_700.0, tf="1w", horizon="next month"))
    assert r["What's happening"] == "Moderate lean higher. BTC is $81,557 on the weekly chart."
    assert r["What changes the view"] == ("A weekly close below $80,700 (under the $80,985 floor), or the moving "
                                          "averages turning down, flips this lower.")


def test_extras_only_when_they_matter():
    assert "Also worth noting" not in read(analysis(rsi=55, squeeze=(39, "LONG_SQUEEZE")))
    r = read(analysis(rsi=28, squeeze=(58, "LONG_SQUEEZE")))
    assert r["Also worth noting"] == ("RSI 28, oversold, the drop is stretched; crowded longs (58/100) raise the odds "
                                      "of a sharp shakeout lower.")
    assert "overbought" in read(analysis(rsi=74))["Also worth noting"]
    assert "Heavily crowded shorts (81/100) raise the odds of a sharp pop higher" in read(analysis(squeeze=(81, "SHORT_SQUEEZE")))["Also worth noting"]


def test_fibonacci_windows_are_not_told_to_the_reader():
    """Tested on a year of hourly candles, they timed turns no better than a random hour."""
    for status in ("UPCOMING", "ACTIVE"):
        rw = NS(status=status, bias="bearish", confluence=["x"])
        assert "Also worth noting" not in read(analysis(reversal=rw)), status


def test_patterns_and_divergences_in_words():
    pat = NS(name="Double Top", status="forming", bias="bearish", breakout=80_900.0, target=79_800.0)
    div = NS(kind="regular", direction="bearish", status="confirmed")
    note = read(analysis(patterns=[pat], divergences=[div]))["Also worth noting"]
    assert note.startswith("A double top is forming, bearish, a break of $80,900 points to $79,800; ")
    assert "price made a higher high without momentum behind it (confirmed bearish divergence)" in note


def test_missing_levels_and_range_still_read():
    r = read(analysis("NEUTRAL", 0.1, None, levels=[], exp=None))
    assert r["Levels that matter"] == "Not enough history yet for levels or a range."
    assert r["What changes the view"] == "A clean break either way, backed by the moving averages, picks a side."
    r = read(analysis("NEUTRAL", 0.1, None, levels=[]))
    assert r["Levels that matter"].startswith("No clear floor or ceiling nearby. Normal range")
    r = read(analysis(levels=[lvl("resistance", 81_655.0, 1)]))
    assert "No clear floor below; ceiling $81,655 (rejected once)" in r["Levels that matter"]
    assert r["What changes the view"].endswith("There is no clear floor beneath to lean on.")


def test_a_wide_or_tight_market_says_so():
    assert "wider than usual" in read(analysis(regime="HIGH"))["Levels that matter"]
    assert "tighter than usual" in read(analysis(regime="LOW"))["Levels that matter"]


def test_no_jargon_reaches_the_reader():
    h = panels.layman_html(analysis(rsi=28, squeeze=(58, "LONG_SQUEEZE")))
    for word in ("EMA", "σ", "sigma", "stack", "anchored", "Fibonacci", "touches"):
        assert word not in h, word


def test_an_before_a_vowel():
    pat = NS(name="Inverse Head and Shoulders", status="forming", bias="bullish", breakout=None, target=None)
    assert read(analysis(patterns=[pat]))["Also worth noting"].startswith("An inverse head and shoulders is forming")
