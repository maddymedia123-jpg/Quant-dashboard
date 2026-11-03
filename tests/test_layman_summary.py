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
    assert s["condition"] == "A 4-hour candle closing above $82,600 cancels the short.", "his 4-hour close rule"


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
    assert ("tilts lower on a 4-hour close below the $80,800 floor and higher above the $81,600 ceiling; "
            "the long setup is cancelled below $80,650") in s["summary"]
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


def test_the_box_follows_his_v22_layout_in_order_with_no_indicator_words():
    h = panels.layman_summary_html(desk(), "Live Recon", DERIVS, 32, NOON)
    order = ["🧭 BIG PICTURE", "🕒 TIMELINE", "Now: What's happening:", "What to expect:", "Turning window:",
             "🎯 BOUNCE ZONE", "Sell $81,550 – $81,650", "Why:", "📈 TARGETS", "Target 1 (the first low to expect):",
             "Target 2:", "Target 3:", "🛡 SAFETY LINE", "✅ VERIFIED CONFLUENCES · 0 of 7 agree, 3 needed",
             "📣 DIRECTIVE", "WAIT. 0 of 7 confluences agree", "⚠️ WARNING"]
    at = [h.index(x) for x in order]
    assert at == sorted(at), order
    assert "\n\n" not in h and "<pre" not in h
    for word in ("EMA", "RSI", "σ", "Fibonacci", "CVD", "squeeze score", "moving average", "funding", "gamma"):
        assert word.lower() not in h.lower(), word


def evidence_for_a_long(**extra):
    """Three families behind the $80,800 floor: a pattern in the zone, a put wall in it, extreme fear."""
    from core.confluence import Evidence
    return Evidence(harmonics=(NS(name="Gartley", side="bullish", stage="in_zone", prz=(80_700.0, 80_820.0),
                               targets=(81_600.0, 82_300.0), invalidation=80_400.0),),
                    derivs={**DERIVS, "gamma": {"put_wall": 80_800.0, "call_wall": 84_000.0}}, fear_greed=18, **extra)


def test_the_gate_opens_the_directive_with_three_targets_and_closes_it_naming_what_is_missing():
    bull = desk("BULLISH", 0.6, 80_000.0)
    s = ls.lines(bull, DERIVS, 18, NOON, evidence_for_a_long())
    assert s["verdict"] == "TRADE" and s["score"] == "3 of 7 agree"
    assert s["directive"] == ("BUY $80,750 – $80,850. Targets $81,600, $82,300, $82,900. "
                               "Out on a 4-hour close below $80,000.")
    assert s["target3"] == "$82,900" and "backed by a Gartley in its zone" in s["zone_why"]
    assert "What to expect: a bounce from the zone toward $81,600, then $82,300" in s["phase_now"]
    assert [c["agrees"] for c in s["confluences"]] == [True, True, None, True, None, None, None]
    assert s["safety"].startswith("$80,000. A 4-hour candle closing below $80,000 cancels the long.")
    bare = ls.lines(bull, None, None, NOON)
    assert bare["verdict"] == "WAIT" and bare["directive"].startswith("WAIT. 0 of 7 confluences agree, 3 are needed. Missing: ")
    assert "nothing else backs it yet" in bare["zone_why"]
    # the mood and derivatives the caller already passes count as evidence even without an Evidence object
    assert ls.lines(bull, {"funding": {"latest_8h_rate": -0.0002}}, 15, NOON)["confluences"][3]["agrees"] is True
    h = panels.layman_summary_html(bull, "Live Recon", DERIVS, 18, NOON, evidence_for_a_long())
    assert "Buy $80,750 – $80,850" in h and "✔ A pattern completing at the zone: a Gartley" in h and "BUY $80,750" in h
    assert "Target 1 (the first high to expect): $81,600" in h


def test_a_named_anomaly_trap_shows_in_the_summary_and_on_its_own_card():
    bull = desk("BULLISH", 0.6, 80_000.0)
    ev = evidence_for_a_long(futures=NS(long_account_pct=0.40, long_short_ratio=None), price_change_24h_pct=-2.3,
                             liquidity={"1h": {"cumulative_delta": {"state": "bullish absorption"}, "pools": {}, "volume_profile": {}}})
    s = ls.lines(bull, DERIVS, 18, NOON, ev)
    assert s["anomaly"]["name"] == "Bullish squeeze trap" and s["anomaly"]["status"] == "set"
    assert s["confluences"][3]["note"].startswith("Bullish squeeze trap (set)")
    h = panels.layman_summary_html(bull, "Live Recon", DERIVS, 18, NOON, ev)
    assert "🪤 ANOMALY TRAP" in h and h.index("🧭 BIG PICTURE") < h.index("🪤 ANOMALY TRAP") < h.index("🕒 TIMELINE")
    from core.confluence import Evidence
    calm = Evidence(**{**ev.__dict__, "fear_greed": 50})          # the bundle carries the mood; the bare argument is the fallback
    assert "🪤 ANOMALY TRAP" not in panels.layman_summary_html(bull, "Live Recon", DERIVS, 50, NOON, calm)
    from core.indicators.anomaly_trap import from_evidence
    card = panels.anomaly_trap_html(from_evidence(ev, "1h"))
    assert card.startswith("<div class=\"ti-card up") and "Anomaly trap · Bullish squeeze trap · SET" in card
    assert "the shorts are the crowd that gets forced out" in card and "\n\n" not in card


def test_the_weekend_switches_the_words_to_sweep_and_rejection_and_flags_the_desk():
    bull = desk("BULLISH", 0.6, 80_000.0)
    saturday = NOON + 4 * 86_400_000
    week = ls.lines(bull, DERIVS, 18, NOON, evidence_for_a_long())
    wk = ls.lines(bull, DERIVS, 18, saturday, evidence_for_a_long())
    assert week["weekend"] is None and wk["weekend"]["weekend"] is True and wk["weekend"]["ends_utc"] == "Sun 22:00 UTC"
    assert week["verdict"] == wk["verdict"] == "TRADE"
    assert week["directive"].endswith("Out on a 4-hour close below $80,000.")
    assert wk["directive"].endswith("Out on a 4-hour close below $80,000. Weekend: trade the sweep and rejection at the zone, not breakout continuation.")
    assert "What to expect: weekend playbook: a sweep through the zone and a rejection back inside the range" in wk["phase_now"]
    assert "the first target $81,600 is the realistic one until weekday volume returns" in wk["phase_now"]
    h = panels.layman_summary_html(bull, "Live Recon", DERIVS, 18, saturday, evidence_for_a_long())
    assert "🛋 WEEKEND TAPE" in h and h.index("🧭 BIG PICTURE") < h.index("🛋 WEEKEND TAPE") < h.index("🕒 TIMELINE")
    assert "🛋 WEEKEND TAPE" not in panels.layman_summary_html(bull, "Live Recon", DERIVS, 18, NOON, evidence_for_a_long())
    from core.indicators.weekend import read as weekend_read
    card = panels.weekend_html(weekend_read(saturday))
    assert card.startswith("<div class=\"ti-card warn") and "Weekend tape</h4>" in card and "sweep and rejection" in card
    assert panels.weekend_html(weekend_read(NOON)) is None
    assert "Weekend tape ahead" in panels.weekend_html(weekend_read(NOON + 3 * 86_400_000 + 12 * 3_600_000))


def test_the_warning_names_the_next_print_the_dollar_and_the_weekend():
    bull = desk("BULLISH", 0.6, 80_000.0)
    ev = evidence_for_a_long(prints=(NS(title="FOMC rate decision", time_ms=NOON + 5 * 3_600_000, state="positioned",
                                        runup_pct=2.3),),
                             rates={"available": True, "DTWEXBGS": {"value": 120.0, "change_1w": 0.9}})
    w = ls.lines(bull, DERIVS, 18, NOON, ev)["warning"]
    assert w.startswith("Watch: a high-impact US print, FOMC rate decision, lands at 11:00 UTC (16:00 PKT) Tue 22 Sep, "
                        "5 hours away; the event window is open")
    assert "moved +2.3% into it" in w and "the dollar has risen 0.8% this week, a headwind for a long" in w
    saturday = NOON + 4 * 86_400_000
    assert "weekend tape" in ls.lines(bull, DERIVS, 18, saturday)["warning"]
    friday_evening = NOON + 3 * 86_400_000 + 12 * 3_600_000          # Fri 18:00 UTC
    assert "the weekend tape starts at Fri 21:00 UTC (02:00 PKT), in 3 hours" in ls.lines(bull, DERIVS, 50, friday_evening)["warning"]
    assert ls.lines(bull, DERIVS, 50, NOON)["warning"].startswith("Nothing scheduled")
    # a bigger level on the path that is too far out to bar the trade is named here instead
    walled = ls.lines(bull, DERIVS, 18, NOON, evidence_for_a_long(htf_levels=(("weekly level", 81_200.0),)))
    assert walled["verdict"] == "TRADE" and walled["bar"] is None
    assert walled["warning"].startswith("Watch: a weekly level at $81,200 stands between the zone and Target 1; expect a pause there")
    barred = ls.lines(bull, DERIVS, 18, NOON, evidence_for_a_long(htf_levels=(("weekly level", 80_910.0),)))
    assert barred["verdict"] == "WAIT" and barred["bar"]["price"] == 80_910.0
    assert barred["directive"].startswith("WAIT. 3 of 7 confluences agree, but a weekly level at $80,910 sits just above the zone")
    assert "What to expect: no trade yet. WAIT: 3 of 7" in barred["phase_now"]
    # the window is five days either side (owner, 4 Nov): three days out warns on every desk, six does not
    soon = evidence_for_a_long(prints=(NS(title="CPI", time_ms=NOON + 3 * 86_400_000, state="flat"),))
    for tf, horizon in (("1h", "next 4h"), ("1w", "next 30 days")):
        w3 = ls.lines(desk("BULLISH", 0.6, 80_000.0, tf=tf, horizon=horizon), DERIVS, 50, NOON, soon)["warning"]
        assert "CPI, lands at 06:00 UTC (11:00 PKT) Fri 25 Sep, 3 days away" in w3, (tf, w3)
    late = evidence_for_a_long(prints=(NS(title="CPI", time_ms=NOON + 6 * 86_400_000, state="scheduled"),))
    assert ls.lines(bull, DERIVS, 50, NOON, late)["warning"].startswith("Nothing scheduled")


def test_kill_zones_are_shown_with_their_measured_odds_and_the_dead_ones_said_to_be():
    s = ls.lines(desk(), DERIVS, 50, NOON)["kill_zones"]
    assert "New York session 18:00–22:00 PKT: 30% of turns, 1.8x chance" in s
    assert "Asian session (05:00–09:00 PKT) and London open (12:00–15:00 PKT): no better than chance" in s
    assert "Kill zones, measured:" in panels.layman_summary_html(desk(), "Live Recon", DERIVS, 50, NOON)


def test_the_weekly_desk_keeps_its_own_candle_for_the_safety_line():
    s = ls.lines(desk("BULLISH", 0.4, 80_000.0, tf="1d", horizon="next 7 days"), DERIVS, 60, NOON)
    assert s["condition"] == "A daily candle closing below $80,000 cancels the long."
    assert ls.rule_tf("1h") == "4h" and ls.rule_tf("4h") == "4h" and ls.rule_tf("1w") == "1w"


def test_the_two_turning_windows_are_said_to_be_one_afternoon():
    s = ls.lines(desk(), DERIVS, 50, NOON)["kill_zones"]
    assert "overlaps it by three, so they are one afternoon, not two" in s
