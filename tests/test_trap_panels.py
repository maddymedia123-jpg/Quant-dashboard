"""The TRAP cards, console and desk warning: the HTML a trader actually reads.

These panels had no unit test, which mattered twice over. They go out through st.markdown with HTML
enabled, so every string from the database has to be escaped; and they are a safety warning, so a card
that renders the wrong number, loses its evidence, or quietly says "nothing anchored" while traps are
live is worse than no card at all.
"""
import re

import pytest

from ui import panels


def text(h: str) -> str:
    """What the reader sees: the card's words with the markup removed. The design sets a figure as a
    label over a value, so "Invalidated above" and "$78,900" are separate elements in the HTML."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h)).strip()

NOW = 1_760_000_000_000
HOUR = 3_600_000


def row(**over) -> dict:
    """A stored trap as `Store.traps()` hands it back: evidence is a JSON string, times are epoch ms."""
    base = {"id": 1, "category": "weekly", "side": "bull_trap", "timeframe": "1d", "level": 78_600.0,
            "invalidation": 78_900.0, "plays_out": 78_000.0, "declared_ms": NOW - 3 * HOUR,
            "score": 72.0, "head_call": "ENGINEERED_TRAP",
            "evidence": '["buyside liquidity swept and reclaimed", "delta shows absorption"]',
            "notes": "[]", "status": "active", "resolved_ms": None, "resolved_price": None}
    base.update(over)
    return base


class Candidate:
    """The shape find_candidates returns, as the console reads it."""

    def __init__(self, side="bull_trap", timeframe="4h", level=78_400.0, strength=3, evidence=None):
        self.side, self.timeframe, self.level, self.strength = side, timeframe, level, strength
        self.evidence = tuple(evidence or ("liquidity swept and reclaimed", "funding beyond its mean"))


# ---------- the card ----------
def test_the_card_names_the_trap_its_level_and_what_settles_it():
    h = text(panels.trap_card_html(row(), NOW, "Weekly"))
    assert "bull trap" in h and "1d at $78,600" in h
    assert "Invalidated above $78,900" in h, "a bull trap dies above the raid"
    assert "Pays off back at $78,000" in h
    assert "declared 3h 00m ago" in h
    assert "sub-agents 72/100" in h
    assert "Weekly desk" in h
    for fact in ("buyside liquidity swept and reclaimed", "delta shows absorption"):
        assert fact in h, "a trap is never asserted without the evidence that produced it"


def test_a_bear_trap_dies_below_the_raid():
    h = text(panels.trap_card_html(row(side="bear_trap", level=78_000.0, invalidation=77_700.0,
                                       plays_out=78_600.0), NOW))
    assert "bear trap" in h and "Invalidated below $77,700" in h
    assert "Invalidated above" not in h


def test_the_invalidation_and_the_target_are_not_interchangeable():
    """Swapping them would advertise the stop as the target. Both numbers are pinned to their own words."""
    h = text(panels.trap_card_html(row(), NOW))
    assert "Invalidated above $78,900" in h and "Pays off back at $78,000" in h
    assert "Invalidated above $78,000" not in h and "Pays off back at $78,900" not in h


def test_a_settled_card_says_how_it_ended():
    h = panels.trap_card_html(row(status="invalidated", resolved_price=79_050.0,
                                  resolved_ms=NOW - HOUR), NOW)
    assert "invalidated at $79,050" in h


# ---------- values a real database can hold ----------
def test_a_missing_score_reads_as_a_dash_not_as_zero():
    """This module's own rule: a dash means no feed, not zero. "sub-agents 0/100" reads as a trap that
    failed the 60-point gate it had to clear to be declared at all."""
    h = panels.trap_card_html(row(score=None), NOW)
    assert "sub-agents —/100" in h and "sub-agents 0/100" not in h


def test_a_missing_declared_time_does_not_crash_the_card():
    """`.get(key, default)` returns the default only when the key is absent, never for a stored NULL."""
    for bad in (None, "", "not a time"):
        h = panels.trap_card_html(row(declared_ms=bad), NOW)
        assert "declared — ago" in h
    assert "declared — ago" in (panels.trap_warning_html([row(declared_ms=None)], NOW) or "")


def test_a_missing_level_reads_as_a_dash():
    h = panels.trap_card_html(row(level=None, invalidation=None, plays_out=None), NOW)
    assert "$0" not in h and "None" not in h


def test_an_unrecognised_side_is_not_silently_called_a_bear_trap():
    """The branch is `== "bull_trap"`, so anything unknown used to be described as a bear trap and
    printed as the literal word None."""
    h = panels.trap_card_html(row(side=None), NOW)
    assert "None" not in h and "unknown side" in h
    assert "Invalidated below" not in h and "Invalidated above" not in h


def test_unreadable_evidence_says_so_rather_than_vanishing():
    assert "could not be read" in panels.trap_card_html(row(evidence="{not json"), NOW)
    assert "swept" in panels.trap_card_html(row(evidence=["swept and reclaimed"]), NOW), "or a real list"
    assert panels.trap_card_html(row(evidence=None), NOW)


def test_the_age_reads_hours_and_minutes_the_right_way_round():
    assert "0h 45m" in panels.trap_card_html(row(declared_ms=NOW - 45 * 60_000), NOW)
    assert "2h 05m" in panels.trap_card_html(row(declared_ms=NOW - 125 * 60_000), NOW)
    assert "1d 1h" in panels.trap_card_html(row(declared_ms=NOW - 25 * HOUR), NOW)
    assert "0h 00m" in panels.trap_card_html(row(declared_ms=NOW + HOUR), NOW), "clock skew, not a negative"


# ---------- escaping ----------
@pytest.mark.parametrize("field", ["side", "timeframe", "status", "category"])
def test_every_stored_string_on_a_card_is_escaped(field):
    payload = "<script>alert(1)</script>"
    h = panels.trap_card_html(row(**{field: payload, "status": "played_out"}), NOW, payload)
    assert "<script>" not in h, f"{field} reached the page as markup"


def test_stored_evidence_is_escaped():
    h = panels.trap_card_html(row(evidence='["<img src=x onerror=alert(1)>"]'), NOW)
    assert "<img" not in h and "&lt;img" in h


def test_the_console_escapes_the_candidates_and_the_settled_line():
    h = panels.trap_watch_html([Candidate(evidence=("<b>spoofed</b>", "x"), timeframe="<i>4h</i>")], NOW)
    h += panels.trap_settled_html([row(status="<script>x</script>", side="bear_trap", category="<b>c</b>")], NOW)
    h += panels.trap_console_html([row()], [], NOW, {"<i>x</i>": "<i>x</i>"}, plan=[("<i>x</i>", "<i>1h</i>")])
    assert "<b>spoofed</b>" not in h and "<i>4h</i>" not in h and "<script>" not in h
    assert "<b>c</b>" not in h and "<i>x</i>" not in h and "<i>1h</i>" not in h


def test_the_card_draws_price_between_its_two_settling_levels():
    h = panels.trap_card_html(row(), NOW, price_now=78_180.0)
    # a bull trap pays off at 78,000 (left) and dies at 78,900 (right): 78,180 is a fifth of the way along,
    # off-centre so a bar measured from the wrong end reads 80% and fails
    assert "left:20%" in h and "Now $78,180" in h
    assert "ti-track-now" not in panels.trap_card_html(row(), NOW), "no price, no marker"


def test_the_card_shows_the_judgement_stored_with_the_trap():
    detail = ('{"domains": [{"title": "Quant & Statistics", "points": 15}], "head_confidence": 0.74, '
              '"desk_reads": "engineered or unclear", "desk_reasons": ["no acceptance above value"]}')
    h = panels.trap_card_html(row(detail=detail), NOW)
    assert "Head: engineered, 74%" in h and "15/20" in h and "Quant &amp; Statistics" in h
    assert "Desk says:</strong> engineered or unclear - no acceptance above value" in h
    bare = panels.trap_card_html(row(detail="{broken"), NOW)
    assert "Desk says" not in bare and "Head: engineered" in bare, "an unreadable detail is omitted, not fatal"


def test_the_sweep_plan_names_each_desk_and_the_one_that_scans_nothing():
    h = text(panels.sweep_plan_html([("live", "15m"), ("live", "1h"), ("weekly", "1w")],
                                    {"live": "Live Recon", "weekly": "Weekly", "monthly": "Monthly"}))
    assert "15m · 1h → Live Recon" in h and "1w → Weekly" in h and "none → Monthly" in h


def test_the_warning_escapes_its_row():
    h = panels.trap_warning_html([row(timeframe="<script>x</script>")], NOW)
    assert "<script>" not in h


# ---------- the console ----------
def test_an_empty_console_says_nothing_is_anchored_in_its_own_words():
    h = panels.trap_console_html([], [], NOW)
    assert "No trap is anchored." in h
    assert "<span class='n'>0</span><span class='ti-label'>Anchored</span>" in h
    assert "0 anchored" in h


def test_the_console_counts_what_is_anchored_and_what_is_watched():
    h = panels.trap_console_html([row(), row(id=2, timeframe="4h")], [Candidate()], NOW,
                                 settled=[row(id=3, status="played_out")])
    assert "2 anchored" in h and "1 candidate(s) on watch" in h and "1 recently settled" in h
    assert "No trap is anchored." not in h, "it must not claim nothing is anchored while two are"


def test_a_watch_candidate_is_never_counted_as_anchored():
    h = panels.trap_console_html([], [Candidate(), Candidate(timeframe="1h")], NOW)
    assert "0 anchored" in h and "2 candidate(s) on watch" in h


def test_the_watch_list_is_dated_and_a_stale_scan_is_called_out():
    """Candidates live in session state and survive every auto-refresh, so an undated watch list read as
    current three quarters of an hour after the scan that produced it."""
    fresh = panels.trap_watch_html([Candidate()], NOW, swept_ms=NOW - 5 * 60_000)
    assert "0h 05m ago" in fresh and "Sweep again" not in fresh

    stale = panels.trap_watch_html([Candidate()], NOW, swept_ms=NOW - 47 * 60_000)
    assert "0h 47m ago" in stale and "Sweep again" in stale


def test_the_settled_line_says_which_desk_each_trap_belonged_to():
    h = panels.trap_settled_html([row(status="played_out"),
                                  row(id=2, category="live", timeframe="1h", status="invalidated")],
                                 NOW, {"weekly": "Weekly", "live": "Live Recon"})
    assert "Weekly" in h and "Live Recon" in h
    assert "played out" in h and "invalidated" in h


# ---------- the desk warning ----------
def test_no_warning_when_the_desk_has_no_live_trap():
    assert panels.trap_warning_html([], NOW) is None
    assert panels.trap_warning_html([row(status="played_out")], NOW) is None


def test_the_warning_carries_the_level_and_the_invalidation():
    h = panels.trap_warning_html([row()], NOW)
    assert "1d at $78,600" in h and "invalidated above $78,900" in h
    assert "1d at $78,900" not in h, "the level and the invalidation must not be swapped"


def test_a_bear_trap_warning_dies_below_the_raid():
    h = panels.trap_warning_html([row(side="bear_trap", level=78_000.0, invalidation=77_700.0, plays_out=78_600.0)], NOW)
    assert "invalidated below $77,700" in h and "invalidated above" not in h


def test_the_warning_reports_how_many_more_it_is_not_showing():
    rows = [row(id=i, timeframe=tf) for i, tf in enumerate(("15m", "1h", "4h", "1d", "1w"), start=1)]
    h = panels.trap_warning_html(rows, NOW)
    assert "and 2 more" in h
