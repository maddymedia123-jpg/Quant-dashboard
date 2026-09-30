"""Pakistan time on screen, UTC underneath.

The trader reads PKT; the machine reasons in UTC because every candle this system aligns to is a UTC
candle. These tests pin both halves of that, and in particular that the second half did not move:
converting the boundary arithmetic would mean our "daily" window no longer matched any exchange's daily
candle, and the accuracy ledger scores at exact candle closes.
"""
import datetime as dt

import pytest

from core import trades
from ui import panels


def at(year, month, day, hour, minute=0) -> int:
    return int(dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc).timestamp() * 1000)


# ---------- the offset ----------
def test_pakistan_time_is_utc_plus_five():
    assert trades.LOCAL_OFFSET_MS == 5 * 3_600_000
    assert trades.LOCAL_LABEL == "PKT"


def test_the_offset_is_the_same_in_january_and_july():
    """Pakistan has had no daylight saving since 2009, which is why a fixed offset is exact and no
    timezone database is needed - `zoneinfo` would only work here because `tzdata` happens to be
    installed, and that is not a declared dependency."""
    winter = trades.fmt_when(at(2026, 1, 15, 0))
    summer = trades.fmt_when(at(2026, 7, 15, 0))
    assert winter.startswith("05:00 PKT") and summer.startswith("05:00 PKT")


# ---------- what a trader reads ----------
def test_an_actionable_time_shows_pakistan_first_and_utc_after():
    """PKT first because it is the zone the trader lives in; UTC kept because the candle closes this
    schedule is built from are UTC-aligned, and dropping it would hide which candle a checkpoint is."""
    assert trades.fmt_when(at(2026, 10, 2, 0)) == "05:00 PKT Fri 02 Oct / 00:00 UTC"
    assert trades.fmt_when(at(2026, 10, 1, 12)) == "17:00 PKT Thu 01 Oct / 12:00 UTC"


def test_a_time_that_rolls_the_date_shows_both_dates():
    """22:00 UTC Thursday is 03:00 PKT Friday. Showing one date would make the other zone wrong."""
    out = trades.fmt_when(at(2026, 10, 1, 22))
    assert out == "03:00 PKT Fri 02 Oct / 22:00 UTC Thu 01 Oct"


def test_the_utc_date_is_left_off_when_it_is_the_same_day():
    out = trades.fmt_when(at(2026, 10, 1, 12))
    assert out.count("Thu 01 Oct") == 1, "no need to repeat the date for both zones"


def test_a_pakistan_only_form_exists_for_tight_spaces():
    assert trades.fmt_when(at(2026, 10, 1, 22), both=False) == "03:00 PKT Fri 02 Oct"


def test_no_date_is_still_no_date():
    assert trades.fmt_when(None) == "no date"


# ---------- the arithmetic did NOT move ----------
def test_candle_boundaries_are_still_utc_aligned():
    """The whole point of doing this as a display change. If a boundary moved to PKT midnight, our
    daily window would no longer be any exchange's daily candle."""
    now = at(2026, 10, 1, 13, 37)
    assert trades.next_close(now, "1d") == at(2026, 10, 2, 0), "the daily still closes at 00:00 UTC"
    assert trades.next_close(now, "4h") == at(2026, 10, 1, 16), "the 4h still closes on a UTC boundary"
    weekly = trades.next_close(now, "1w")
    d = dt.datetime.fromtimestamp(weekly / 1000, dt.timezone.utc)
    assert (d.weekday(), d.hour) == (0, 0), "the weekly still opens Monday 00:00 UTC"


def test_a_utc_boundary_is_simply_read_in_pakistan_time():
    """The same instant, both ways round - 00:00 UTC is 05:00 PKT, and that is the daily close."""
    close = trades.next_close(at(2026, 10, 1, 13), "1d")
    shown = trades.fmt_when(close)
    assert shown.startswith("05:00 PKT") and "00:00 UTC" in shown


def test_the_monitoring_schedule_reads_in_both_zones():
    schedule = trades.monitoring_schedule(at(2026, 10, 1, 13), ("4h", "1d"))
    assert schedule, "there is something to check"
    for checkpoint in schedule:
        assert "PKT" in checkpoint.when and "UTC" in checkpoint.when


def test_durations_are_unaffected_because_they_are_the_same_in_any_zone():
    """Ages and countdowns need no conversion, and must not accidentally get one."""
    now = at(2026, 10, 1, 13)
    assert panels._age(now - 90 * 60_000, now) == "1h ago"
    assert panels._trap_age(now - 45 * 60_000, now) == "0h 45m"


# ---------- the clock ----------
def test_the_clock_shows_both_zones():
    """Asserted on exact code fragments, not on loose substrings: "PKT is UTC+5" also appears in a
    comment in this file, so matching the bare phrase passed with the label deleted."""
    html = panels.clock_html(dark=False)
    assert "getElementById('ti-pkt').textContent =" in html, "the PKT line is written"
    assert "getElementById('ti-utc').textContent =" in html, "and so is the UTC line"
    assert "' UTC  (PKT is UTC+5)'" in html, "the relationship is stated in the rendered text"


def test_the_clock_ticks_rather_than_being_rendered_once():
    """Streamlit only redraws on a rerun, so a server-rendered time would sit frozen at the page-load
    moment - a stale clock is worse than none.

    Note the limit of these tests: nothing here executes the JavaScript, so they pin the code that is
    shipped rather than its behaviour in a browser."""
    html = panels.clock_html(dark=False)
    assert "setInterval(tick, 1000)" in html and "new Date()" in html
    assert "now.getTime() + 5*3600*1000" in html, "the offset is added, not subtracted"


def test_the_clock_has_no_server_rendered_time_in_it():
    """If a formatted time leaked into the markup it would be the stale one."""
    html = panels.clock_html(dark=False)
    import re
    assert not re.search(r">\s*\d{2}:\d{2}", html), "no baked-in clock face"


@pytest.mark.parametrize("dark", [True, False])
def test_the_clock_is_legible_in_both_themes(dark):
    html = panels.clock_html(dark=dark)
    assert "__TEXT__" not in html, "the colour placeholder is filled in"
    assert "color:#" in html.replace(" ", "")
