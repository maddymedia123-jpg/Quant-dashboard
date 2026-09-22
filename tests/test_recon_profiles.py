"""One war room per category: each reads its own three timeframes and anchors on its own candle."""
from datetime import datetime, timezone

import pytest

from core.agents import recon_profiles as rp
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH, DOMAINS


def ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


def test_every_category_has_a_profile_with_three_ascending_timeframes():
    assert set(rp.PROFILES) == {"live", "intraday", "weekly", "monthly"}
    order = ["15m", "1h", "4h", "1d", "1w"]
    for p in rp.PROFILES.values():
        assert len(p.timeframes) == 3
        idx = [order.index(tf) for tf in p.timeframes]
        assert idx == sorted(idx) and len(set(idx)) == 3, f"{p.category} timeframes must ascend"
    assert rp.PROFILES["live"].timeframes == ("15m", "1h", "4h"), "the client's spec, unchanged"


# ---------- anchor windows ----------
def test_live_anchors_on_the_utc_four_hour_candle():
    p = rp.PROFILES["live"]
    t = ms(2026, 9, 22, 13, 47)
    assert p.window_open(t) == ms(2026, 9, 22, 12, 0) and p.window_close(t) == ms(2026, 9, 22, 16, 0)


def test_intraday_anchors_on_the_utc_day():
    p = rp.PROFILES["intraday"]
    t = ms(2026, 9, 22, 23, 59)
    assert p.window_open(t) == ms(2026, 9, 22) and p.window_close(t) == ms(2026, 9, 23)


def test_weekly_anchors_on_monday_not_the_thursday_the_epoch_fell_on():
    p = rp.PROFILES["weekly"]
    tuesday = ms(2026, 9, 22, 10, 0)                      # 22 Sep 2026 is a Tuesday
    assert datetime.fromtimestamp(p.window_open(tuesday) / 1000, timezone.utc).weekday() == 0
    assert p.window_open(tuesday) == ms(2026, 9, 21) and p.window_close(tuesday) == ms(2026, 9, 28)
    assert p.window_open(ms(2026, 9, 21)) == ms(2026, 9, 21), "Monday 00:00 opens its own week"
    assert p.window_open(ms(2026, 9, 20, 23, 59)) == ms(2026, 9, 14), "Sunday night is last week"


def test_monthly_anchors_on_calendar_months_of_any_length():
    p = rp.PROFILES["monthly"]
    assert p.window_open(ms(2026, 2, 17)) == ms(2026, 2, 1) and p.window_close(ms(2026, 2, 17)) == ms(2026, 3, 1)
    assert p.window_close(ms(2026, 12, 31, 23)) == ms(2027, 1, 1), "December rolls the year"
    assert p.window_open(ms(2026, 10, 1)) == ms(2026, 10, 1)


@pytest.mark.parametrize("category", ["live", "intraday", "weekly", "monthly"])
def test_a_run_at_the_close_belongs_to_the_next_window(category):
    p = rp.PROFILES[category]
    t = ms(2026, 9, 22, 13, 47)
    close = p.window_close(t)
    assert p.window_open(close) == close
    assert p.window_open(close - 1) == p.window_open(t)


# ---------- the rubric, templated per category ----------
def test_live_questions_are_word_for_word_what_the_client_approved():
    live = rp.PROFILES["live"]
    bos = next(i for i in ALL_ITEMS if i.id == "smc_bos")
    assert bos.question(BULLISH, live) == "A confirmed 4h break of structure or change of character points UP."
    triple = next(i for i in ALL_ITEMS if i.id == "mtf_triple")
    assert triple.question(BEARISH, live) == "All three timeframes (15m, 1h, 4h) point DOWN."
    sent = next(i for i in ALL_ITEMS if i.id == "macro_sentiment")
    assert sent.question(BULLISH, live).endswith("over the next four hours.")
    assert bos.question(BULLISH) == bos.question(BULLISH, live), "live stays the default"


def test_other_categories_ask_about_their_own_timeframes_and_horizon():
    weekly = rp.PROFILES["weekly"]
    triple = next(i for i in ALL_ITEMS if i.id == "mtf_triple")
    q = triple.question(BULLISH, weekly)
    assert q == "All three timeframes (4h, 1d, 1w) point UP."
    sent = next(i for i in ALL_ITEMS if i.id == "macro_sentiment")
    assert sent.question(BEARISH, weekly).endswith("over the next seven days.")
    for d in DOMAINS:
        for step in d.steps(weekly):
            assert "15m" not in step, f"weekly checklist still mentions 15m: {step}"


def test_no_template_placeholder_survives_formatting():
    for p in rp.PROFILES.values():
        for i in ALL_ITEMS:
            for side in (BULLISH, BEARISH):
                assert "{" not in i.question(side, p), f"{p.category}/{i.id} left a placeholder"
        for d in DOMAINS:
            assert all("{" not in s for s in d.steps(p))
