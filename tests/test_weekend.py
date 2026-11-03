"""His weekend microstructure modifier: Friday 21:00 UTC to Sunday 22:00 UTC is weekend tape, flagged everywhere
and worded as sweep and rejection at the range edges instead of breakout continuation."""
from datetime import datetime, timezone

from core.indicators import weekend as wk


def at(y, mo, d, h, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


def test_the_weekend_runs_from_friday_us_close_to_sunday_cme_open():
    # 2026-09-25 is a Friday
    assert wk.read(at(2026, 9, 25, 20, 59)).active is False
    assert wk.read(at(2026, 9, 25, 21, 0)).active is True
    assert wk.read(at(2026, 9, 26, 12)).active is True            # Saturday
    assert wk.read(at(2026, 9, 27, 21, 59)).active is True        # Sunday evening
    assert wk.read(at(2026, 9, 27, 22, 0)).active is False        # CME open
    assert wk.read(at(2026, 9, 22, 6)).phase == "weekday"         # Tuesday


def test_the_words_name_the_end_in_utc_and_pkt_and_the_playbook():
    w = wk.read(at(2026, 9, 26, 12))
    assert w.phase == "weekend" and w.note.startswith("Weekend tape until Sun 22:00 UTC (03:00 PKT)")
    assert "sweep and rejection at the range edges, not breakout continuation" in w.note
    s = w.to_state()
    assert s["weekend"] is True and s["ends_utc"] == "Sun 22:00 UTC" and "sweep and rejection" in s["playbook"]


def test_late_friday_is_flagged_as_starting():
    w = wk.read(at(2026, 9, 25, 17, 30))
    assert w.active is False and w.phase == "starting" and "in 4 hours" in w.note and "late-Friday break is suspect" in w.note
    assert wk.read(at(2026, 9, 25, 20, 30)).note.startswith("The weekend tape starts at Fri 21:00 UTC (02:00 PKT), in under an hour:")
    assert ", in 1 hour:" in wk.read(at(2026, 9, 25, 19, 50)).note
    s = wk.read(at(2026, 9, 22, 6)).to_state()
    assert s["weekend"] is False and s["starts_utc"] == "Fri 21:00 UTC" and s["note"] == ""
