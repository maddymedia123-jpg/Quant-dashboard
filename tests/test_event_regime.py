"""Event regime (election-day fix 3): which prints make one, and when its window is open."""
from types import SimpleNamespace as NS

from core.indicators import event_regime as er

H = 3_600_000
T = 1_793_200_000_000 - 1_793_200_000_000 % (24 * H) + 15 * H + 16 * 60_000     # a 15:16 UTC


def ev(title, t_ms, impact="High", country="USD"):
    return {"title": title, "country": country, "impact": impact, "time_ms": t_ms}


def test_the_window_runs_from_a_day_before_to_two_days_after_a_major_print():
    vote = ev("Congressional Elections", T)
    assert er.read([vote], T - 24 * H - 1).active is False
    ahead = er.read([vote], T - 24 * H)
    assert ahead.active and ahead.phase == "ahead" and ahead.until_ms == T + 48 * H
    assert ahead.note.startswith("Congressional Elections prints at 15:16 UTC (20:16 PKT) ") and "24 hours away" in ahead.note
    react = er.read([vote], T + 23 * H)                                   # 4 Nov 14:16 UTC, the hour of the spike
    assert react.phase == "reaction" and "printed 23 hours ago (15:16 UTC" in react.note
    assert "the reaction is being traded until 15:16 UTC (20:16 PKT)" in react.note
    assert er.read([vote], T + 48 * H).active is False
    assert er.read([vote], T + 48 * H - 1).active is True
    assert er.read([], T) == er.NONE and er.NONE.to_state() == {"active": False}
    assert set(react.to_state()) == {"active", "title", "time_ms", "phase", "until_ms", "note"}


def test_only_high_impact_usd_prints_that_name_an_election_or_the_fed_make_a_regime():
    assert er.read([ev("CPI m/m", T)], T + H).active is False
    assert er.read([ev("Non-Farm Employment Change", T)], T + H).active is False
    assert er.read([ev("Congressional Elections", T, impact="Medium")], T + H).active is False
    assert er.read([ev("ECB Press Conference", T, country="EUR")], T + H).active is False
    assert er.read([ev("General Election", T, country="GBP")], T + H).active is False
    for title in ("FOMC Statement", "Federal Funds Rate", "Fed Chair Powell Speaks", "FOMC Press Conference",
                  "Presidential Election"):
        assert er.read([ev(title, T)], T + H).title == title, title
    # the macro engine's reads (title + time, already High USD) are read the same way
    reads = (NS(title="FOMC Statement + Federal Funds Rate", time_ms=T, state="reacting"),)
    assert er.read(reads, T + H).title == "FOMC Statement + Federal Funds Rate"
    assert er.read([{"title": "FOMC Statement", "time_ms": None, "impact": "High", "country": "USD"}], T).active is False


def test_two_major_prints_in_one_window_are_one_regime_that_ends_with_the_later():
    both = [ev("FOMC Statement", T), ev("FOMC Press Conference", T + 30 * 60_000)]
    r = er.read(both, T + H)
    assert r.title == "FOMC Press Conference" and r.until_ms == T + 30 * 60_000 + 48 * H
    assert er.read(both, T - 2 * H).title == "FOMC Press Conference", "the window that ends last names the regime"
