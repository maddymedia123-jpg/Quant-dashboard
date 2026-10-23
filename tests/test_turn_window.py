"""The spec's three kill zones, carried with the odds they measured (research/kill_zones.py, 23 Oct 2026)."""
import json

import pytest

from core.indicators import turn_window as tw

H = 3_600_000
DAY = 24 * H
MIDNIGHT = 1_792_713_600_000          # Fri 23 Oct 2026 00:00 UTC
assert MIDNIGHT % DAY == 0


def test_the_zones_are_the_spec_hours_in_utc():
    got = {z.key: (z.start_hour_utc, z.end_hour_utc) for z in tw.zones()}
    assert got == {"asian": (0, 4), "london": (7, 10), "new_york": (13, 17)}


def test_measured_odds_are_internally_consistent():
    """Canary for a typo in the table: share = turns / 353 and times chance = share / (hours / 24)."""
    for z in tw.zones():
        assert z.share_of_turns == pytest.approx(z.turns / tw.GATE_TURNS, abs=0.0006), z.key
        assert z.times_chance == pytest.approx(z.share_of_turns / (z.hours / 24), abs=0.006), z.key


def test_only_new_york_beats_chance():
    flags = {z.key: z.near_chance for z in tw.zones()}
    assert flags == {"asian": True, "london": True, "new_york": False}
    ny = next(z for z in tw.zones() if z.key == "new_york")
    assert ny.times_chance == 1.80 and ny.older_times_chance == 1.74 and ny.big_times_chance == 1.59


def test_readings_say_no_edge_for_near_chance_zones_and_quote_the_odds():
    by = {z.key: z for z in tw.zones()}
    assert "no edge" in by["asian"].reading and "0.95x" in by["asian"].reading
    assert "no edge" in by["london"].reading
    ny = by["new_york"].reading
    assert "no edge" not in ny and "30%" in ny and "1.8x" in ny
    assert "13:00–17:00 UTC (18:00–22:00 PKT)" in ny


def test_pkt_is_utc_plus_five_and_wraps_past_midnight():
    by = {z.key: z for z in tw.zones()}
    assert by["asian"].hours_label == "00:00–04:00 UTC (05:00–09:00 PKT)"
    assert by["london"].hours_label == "07:00–10:00 UTC (12:00–15:00 PKT)"
    late = tw.Zone("x", "x", 20, 23, 0, 0.0, 0.0, 0.0, 0.0)
    assert late.hours_label == "20:00–23:00 UTC (01:00–04:00 PKT)"


def test_next_zones_soonest_first_with_the_one_on_now_marked():
    at = MIDNIGHT + 8 * H                          # 08:00 UTC: London is on, Asian has ended
    ws = tw.next_zones(at)
    assert [w.zone.key for w in ws] == ["london", "new_york", "asian"]
    assert [w.active for w in ws] == [True, False, False]
    assert ws[0].start_ms == MIDNIGHT + 7 * H and ws[0].end_ms == MIDNIGHT + 10 * H
    assert ws[2].start_ms == MIDNIGHT + DAY, "Asian ended at 04:00, so tomorrow's"
    assert ws[1].label.startswith("13:00–17:00 UTC (18:00–22:00 PKT - Fri 23 Oct")
    assert ws[1].odds == ws[1].zone.reading


def test_zone_end_hour_is_exclusive():
    ws = {w.zone.key: w for w in tw.next_zones(MIDNIGHT + 17 * H)}
    assert not ws["new_york"].active and ws["new_york"].start_ms == MIDNIGHT + DAY + 13 * H
    assert {w.zone.key: w for w in tw.next_zones(MIDNIGHT + 17 * H - 1)}["new_york"].active


def test_single_window_api_is_unchanged():
    w = tw.next_window(MIDNIGHT + 6 * H)
    assert not w.active and w.label.startswith("12:00–16:00 UTC (17:00–21:00 PKT")
    assert tw.next_window(MIDNIGHT + 13 * H).active
    assert "35%" in w.odds and w.ends == "16:00 UTC (21:00 PKT)"


def test_agents_see_all_three_zones_with_odds_and_the_no_edge_flags():
    s = tw.to_state(MIDNIGHT + 2 * H)
    json.dumps(s)                                  # goes into the agent payload as JSON
    assert s["share_of_turns"] == 0.35 and "Fibonacci" in s["note"]
    zs = {z["zone"]: z for z in s["kill_zones"]}
    assert set(zs) == {"Asian session", "London open", "New York session"}
    assert zs["Asian session"]["active_now"] and zs["Asian session"]["near_chance"]
    assert zs["New York session"]["times_chance"] == 1.80 and not zs["New York session"]["near_chance"]
    assert "Binance" in s["kill_zones_measured_on"] and "never" in s["kill_zones_note"]
