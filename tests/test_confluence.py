"""His v22 Section 9: a trade needs three of seven independent confluences at the zone; fewer is a WAIT that
names what is missing, two against it is a WAIT that names the conflict."""
from types import SimpleNamespace as NS

from core import confluence as cf
from core.layman_summary import framework
from tests.test_layman_summary import DERIVS, NOON, desk, lvl

ON = NOON + 7 * 3_600_000          # 13:00 UTC, inside the measured 12:00-16:00 turning window
H = 3_600_000


def long_zone():
    """A bullish desk buying the $80,800 floor: zone $80,750 - $80,850."""
    return framework(desk("BULLISH", 0.6, 80_000.0))


def full_evidence():
    """Every family pointing up at the floor."""
    return cf.Evidence(
        harmonics=(NS(name="Gartley", side="bullish", stage="in_zone", prz=(80_700.0, 80_820.0),
                      targets=(81_600.0, 82_300.0), invalidation=80_400.0),),
        derivs={**DERIVS, "gamma": {"put_wall": 80_800.0, "call_wall": 83_000.0},
                "funding": {"latest_8h_rate": -0.0002}},
        fear_greed=18,
        futures=NS(long_account_pct=0.40, long_short_ratio=None),
        breaks=(NS(timeframe="1h", available=True, state="failed", direction="down", edge=80_800.0),),
        liquidity={"1h": {"cumulative_delta": {"state": "bullish absorption"},
                          "pools": {"buyside": None, "sellside": None, "recent_sweep": None},
                          "volume_profile": {"available": True, "value_area_low": 80_000.0, "value_area_high": 82_000.0}}},
        prints=(NS(title="CPI", time_ms=NOON - 3 * H, state="spike_faded", trapped="shorts"),),
        price_change_24h_pct=-2.3,
        rates={"available": True, "DTWEXBGS": {"value": 120.0, "change_1w": -0.6},
               "DGS2": {"value": 3.6, "change_1w": -0.15}},
    )


def test_every_family_reads_its_own_evidence_and_seven_of_seven_opens_the_trade():
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), full_evidence(), ON)
    assert [f.key for f in g.families] == ["pattern", "options", "flow", "positioning", "structure", "macro", "timing"]
    assert all(f.side == "LONG" for f in g.families), [(f.key, f.side, f.note) for f in g.families]
    assert g.verdict == "TRADE" and g.side == "LONG" and len(g.agreeing) == 7 and not g.opposing
    notes = " ".join(f.note for f in g.families)
    assert "Gartley" in notes and "$80,800" in notes and "turning window is on now" in notes


def test_fewer_than_three_waits_and_names_what_is_missing():
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), cf.Evidence(), NOON)
    assert g.verdict == "WAIT" and len(g.agreeing) == 0
    assert g.reason.startswith("WAIT: 0 of 7 confluences agree, 3 are needed")
    assert "a pattern completing at the zone" in g.reason and "the turning window" in g.reason
    assert all(f.side == "NONE" for f in g.families)
    assert "next turning window" in g.families[-1].note


def test_three_agreeing_is_enough_and_one_against_is_said_not_fatal():
    ev = cf.Evidence(harmonics=full_evidence().harmonics, derivs=full_evidence().derivs, fear_greed=18,
                     rates={"available": True, "DTWEXBGS": {"value": 120.0, "change_1w": 0.9}})
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), ev, NOON)
    assert g.verdict == "TRADE" and [f.key for f in g.agreeing] == ["pattern", "options", "positioning"]
    assert [f.key for f in g.opposing] == ["macro"] and "the dollar" in g.reason


def test_two_families_against_the_zone_wait_on_the_conflict():
    ev = full_evidence()
    ev = cf.Evidence(**{**ev.__dict__,
                        "harmonics": (NS(name="Bat", side="bearish", stage="completed", prz=(80_760.0, 80_840.0),
                                         targets=(80_000.0, 79_500.0), invalidation=81_200.0),),
                        "derivs": {**ev.derivs, "gamma": {"put_wall": 78_000.0, "call_wall": 80_820.0}}})
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), ev, ON)
    assert g.verdict == "WAIT" and len(g.agreeing) == 5 and [f.key for f in g.opposing] == ["pattern", "options"]
    assert g.reason.startswith("WAIT: a pattern completing at the zone and an options wall at the zone point the other way")


def test_extreme_fear_and_a_crowded_short_read_as_a_bullish_trigger():
    """Hammad's note: positioning is contrarian - the crowd that is trapped is the one that gets forced out."""
    fam = cf.positioning(18, NS(long_account_pct=0.38, long_short_ratio=None), None)
    assert fam.side == "LONG" and "extreme fear" in fam.note and "crowded short" in fam.note
    fam = cf.positioning(82, NS(long_account_pct=None, long_short_ratio=1.8), {"funding": {"latest_8h_rate": 0.0005}})
    assert fam.side == "SHORT" and "extreme greed" in fam.note and "crowded long" in fam.note
    fam = cf.positioning(15, NS(long_account_pct=0.70, long_short_ratio=None), None)
    assert fam.side == "NONE" and "both ways" in fam.note
    assert cf.positioning(50, NS(long_account_pct=0.52, long_short_ratio=None), None).side == "NONE"


def test_structure_reads_a_failed_break_as_the_trap_it_is_then_sweeps_then_the_value_edge():
    zone, atr = (80_750.0, 80_850.0), 200.0
    failed_up = (NS(timeframe="4h", available=True, state="failed", direction="up", edge=81_600.0),)
    assert cf.structure("4h", zone, atr, failed_up, None).side == "SHORT"
    swept = {"4h": {"cumulative_delta": {}, "pools": {"sellside": {"price": 80_790.0, "swept": True, "reclaimed": True},
                                                       "buyside": None, "recent_sweep": None}, "volume_profile": {}}}
    fam = cf.structure("4h", zone, atr, (), swept)
    assert fam.side == "LONG" and "swept" in fam.note and "reclaimed" in fam.note
    edge = {"4h": {"cumulative_delta": {}, "pools": {}, "volume_profile": {"available": True, "value_area_low": 80_830.0,
                                                                            "value_area_high": 82_000.0}}}
    assert cf.structure("4h", zone, atr, (), edge).side == "LONG"
    assert cf.structure("4h", zone, atr, (), None).side == "NONE"


def test_timing_counts_only_while_the_measured_window_is_on():
    assert cf.timing("LONG", NOON).side == "NONE"
    on = cf.timing("LONG", ON)
    assert on.side == "LONG" and "12:00" in on.note and "17:00" in on.note, "UTC and PKT both named"


def test_no_level_is_no_setup():
    a = desk("BULLISH", 0.6, None, atr=None)
    g = cf.read(framework(a), a, full_evidence(), ON)
    assert g.verdict == "NO SETUP" and g.side == "NONE"


def test_a_weekly_level_just_beyond_the_zone_bars_the_path_but_one_inside_the_zone_is_the_entry():
    """Hammad's note b: test the path from the zone toward Target 1, not the zone itself."""
    a, f = desk("BULLISH", 0.6, 80_000.0), long_zone()          # zone $80,750 - $80,850, Target 1 $81,600, ATR 200
    ev = full_evidence()
    assert cf.read(f, a, ev, ON).verdict == "TRADE"
    barred = cf.read(f, a, cf.Evidence(**{**ev.__dict__, "htf_levels": (("weekly level", 80_910.0),)}), ON)
    assert barred.verdict == "WAIT" and barred.bar.price == 80_910.0 and len(barred.agreeing) == 7
    assert barred.reason == ("WAIT: 7 of 7 confluences agree, but a weekly level at $80,910 sits just above the zone, "
                             "between it and Target 1: the path is barred until a close above it, or until that level "
                             "is the zone.")
    inside = cf.read(f, a, cf.Evidence(**{**ev.__dict__, "htf_levels": (("monthly level", 80_800.0), ("weekly level", 80_850.0))}), ON)
    assert inside.verdict == "TRADE" and inside.bar is None, "a level inside or at the edge of the zone is the entry, not a wall"
    far = cf.read(f, a, cf.Evidence(**{**ev.__dict__, "htf_levels": (("weekly level", 81_200.0),)}), ON)
    assert far.verdict == "TRADE" and far.bar is None, "farther than half an ATR: named in the warning, not a bar"
    beyond = cf.read(f, a, cf.Evidence(**{**ev.__dict__, "htf_levels": (("weekly level", 81_700.0), ("weekly level", 80_600.0))}), ON)
    assert beyond.verdict == "TRADE", "past Target 1, or behind the zone, is not on the path"
    # when the gate is already a WAIT the bar is still said
    thin = cf.read(f, a, cf.Evidence(htf_levels=(("weekly level", 80_910.0),)), NOON)
    assert thin.verdict == "WAIT" and thin.reason.startswith("WAIT: 0 of 7") and thin.reason.endswith("Also a weekly level at $80,910 sits just above the zone, between it and Target 1: the path is barred until a close above it, or until that level is the zone.")


def test_the_bar_mirrors_for_a_short_and_reads_only_the_desks_above():
    zone, atr = (81_550.0, 81_650.0), 200.0                       # a short at the $81,600 ceiling, Target 1 $80,800
    assert cf.htf_bar("SHORT", zone, atr, 80_800.0, (("monthly level", 81_480.0),)).price == 81_480.0
    assert cf.htf_bar("SHORT", zone, atr, 80_800.0, (("monthly level", 81_600.0), ("monthly level", 81_700.0))) is None
    assert cf.htf_bar("SHORT", zone, atr, 80_800.0, (("monthly level", 81_400.0),)) is None   # 0.75 ATR: warning only
    assert [p for _, p in cf.levels_before_target("SHORT", zone, atr, 80_800.0,
                                                   (("monthly level", 81_400.0), ("weekly level", 81_000.0), ("weekly level", 80_700.0)))] == [81_400.0, 81_000.0]
    assert cf.HIGHER == {"live": ("weekly", "monthly"), "intraday": ("weekly", "monthly"), "weekly": ("monthly",), "monthly": ()}


def test_a_named_anomaly_trap_speaks_for_the_positioning_family():
    """His Section 5.3: extreme fear plus a crowded short, with the flow absorbing, is the Bullish squeeze trap."""
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), full_evidence(), ON)
    assert g.trap is not None and g.trap.name == "Bullish squeeze trap" and g.trap.status == "set"
    pos = g.families[3]
    assert pos.side == "LONG" and pos.note.startswith("Bullish squeeze trap (set): extreme fear (18/100) + a crowded short")
    assert "the shorts are the crowd that gets forced out" in pos.note
    # without the mood there is no trap, and the family falls back to its votes
    quiet = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), cf.Evidence(**{**full_evidence().__dict__, "fear_greed": 50}), ON)
    assert quiet.trap is None and quiet.families[3].note.startswith("a crowded short")
    # a flush trap points the other way and opposes a long zone
    greedy = cf.Evidence(fear_greed=82, futures=NS(long_account_pct=0.66, long_short_ratio=None),
                         derivs={"open_interest": [{"over": "4h", "change_pct": 2.0}]}, price_change_24h_pct=1.5)
    g2 = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), greedy, NOON)
    assert g2.trap.name == "Liquidation flush trap" and g2.families[3].side == "SHORT" and [f.key for f in g2.opposing] == ["positioning"]
    # no zone at all still reports the trap
    a = desk("BULLISH", 0.6, None, atr=None)
    assert cf.read(framework(a), a, full_evidence(), ON).trap is not None


def test_the_notes_carry_no_indicator_words():
    g = cf.read(long_zone(), desk("BULLISH", 0.6, 80_000.0), full_evidence(), ON)
    text = (g.reason + " ".join(f.note + f.label for f in g.families)).lower()
    # "squeeze" on its own is allowed: his Section 5.3 names the Bullish squeeze trap; the indicator is the squeeze score
    for word in ("ema", "rsi", "σ", "fibonacci", "cvd", "squeeze score", "moving average", "funding", "gamma", "delta"):
        assert word not in text, word
