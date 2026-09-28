"""The pinned Active Trade card as HTML: what a trader actually reads off it.

It goes out through st.markdown with HTML enabled, so every string from the database is escaped; and it is
a trade plan, so a number in the wrong column is worse than a blank one. The live block is the only part a
Refresh changes, which is the split the spec asks for.
"""
import pytest

from core.trades import LONG, SHORT, fmt_when, progress
from ui import panels

NOW = 1_760_000_000_000
HOUR = 3_600_000


def target(name="TP1", price=78_600.0, reward=1.5, due_ms=NOW + HOUR, due_from="Fibonacci time window",
           basis="the 1 sigma band on the 4h") -> dict:
    return {"name": name, "price": price, "reward_r": reward, "due_ms": due_ms, "due_from": due_from,
            "basis": basis}


def plan(**over) -> dict:
    base = {
        "asset": "BTC/USDT", "side": LONG, "entry": 78_000.0, "category": "live", "timeframe": "1h",
        "ok": True, "score": 72.0, "probability": 64.0, "for_points": 78.0, "against_points": 34.0,
        "stop_verdict": "sound", "stop_reasons": ["the stop sits beyond the last swing"],
        "targets": [target(), target("TP2", 79_200.0, 3.0, NOW + 2 * HOUR, "the second 4h close")],
        "pullback": {"available": True, "note": "", "ordinary": 400.0, "edge": 900.0, "breaking": 1800.0,
                     "horizon_bars": 12, "timeframe": "1h",
                     "levels": {"ordinary": 77_600.0, "edge": 77_100.0, "breaking": 76_200.0}},
        "schedule": [{"timeframe": "4h", "at_ms": NOW + HOUR // 2, "reason": "the 4h close"}],
        "alignment": [{"category": "live", "label": "Live Recon", "bias": "BULL", "verdict": "supports",
                       "note": ""}],
        "aligned": "with the higher timeframes", "traps": [],
        "for_domains": [{"key": "quant", "title": "Quant & Statistics", "points": 16.0}],
        "against_domains": [{"key": "quant", "title": "Quant & Statistics", "points": 7.0}],
        "head_call": "TAKE", "head_confidence": 0.62, "notes": [],
    }
    base.update(over)
    return base


def row(**over) -> dict:
    base = {"id": 1, "opened_ms": NOW - 3 * HOUR, "asset": "BTC/USDT", "side": LONG, "entry": 78_000.0,
            "stop": 77_600.0, "given_stop": None, "stop_verdict": "sound", "category": "live",
            "status": "open", "closed_ms": None, "closed_reason": None, "closed_price": None,
            "refreshed_ms": None, "plan": plan()}
    base.update(over)
    return base


def targets_as_objects(rows):
    from core.trades import TimedTarget
    return [TimedTarget(t["name"], t["price"], t["basis"], t["due_ms"], t["reward_r"], t["due_from"])
            for t in rows]


# ---------- the plan ----------
def test_the_card_names_the_position_its_score_and_its_estimate():
    h = panels.trade_card_html(row(), None, NOW)
    assert "BTC/USDT from $78,000" in h and "LONG" in h
    assert "score 72/100" in h and "64% confluence estimate" in h


def test_each_target_carries_its_price_reward_and_actual_due_date():
    """The date itself has to be on the card: asserting only on "Fibonacci time window" passed with the
    due-date column emptied."""
    h = panels.trade_card_html(row(), None, NOW)
    assert fmt_when(NOW + HOUR) in h, "TP1's date"
    assert fmt_when(NOW + 2 * HOUR) in h, "TP2's date"
    assert "1.50R" in h and "3.00R" in h
    assert "$78,600" in h and "$79,200" in h
    assert "the 1 sigma band on the 4h" in h and "the second 4h close" in h


def test_a_due_date_already_past_is_marked_as_such():
    h = panels.trade_card_html(row(plan=plan(targets=[target(due_ms=NOW - HOUR)])), None, NOW)
    assert "(passed)" in h


def test_a_first_target_nearer_than_the_stop_is_called_out():
    """A trade whose TP1 pays less than it risks is a losing proposition at TP1, and the card says so
    rather than leaving the trader to divide two numbers."""
    warned = panels.trade_card_html(row(plan=plan(targets=[target(reward=0.6)])), None, NOW)
    assert "reward below risk" in warned
    assert "risks more than it makes at TP1" in warned

    fine = panels.trade_card_html(row(plan=plan(targets=[target(reward=1.4)])), None, NOW)
    assert "reward below risk" not in fine


def test_a_corrected_stop_says_what_was_given_and_what_is_traded():
    h = panels.trade_card_html(row(given_stop=77_900.0,
                                   plan=plan(stop_verdict="would be hunted")), None, NOW)
    assert "stop would be hunted" in h and "trading at $77,600" in h
    assert "you gave $77,900" in h


def test_a_sound_stop_does_not_nag_about_the_given_one():
    h = panels.trade_card_html(row(given_stop=77_600.0, plan=plan(stop_verdict="sound")), None, NOW)
    assert "stop sound" in h and "you gave" not in h


def test_the_pullback_guardrail_reads_as_plain_english():
    h = panels.trade_card_html(row(), None, NOW)
    assert "a move to $77,600 is ordinary" in h
    assert "$77,100 is still normal - do not close there" in h
    assert "Past $76,200 this is no longer a pullback" in h
    assert "12 1h candles" in h


def test_an_unmeasurable_guardrail_says_why_rather_than_printing_zeros():
    h = panels.trade_card_html(row(plan=plan(pullback={"available": False, "note": "fewer than 32 1h candles",
                                                       "ordinary": 0.0, "edge": 0.0, "breaking": 0.0,
                                                       "horizon_bars": 12, "timeframe": "1h",
                                                       "levels": {}})), None, NOW)
    assert "No pullback guardrail: fewer than 32 1h candles" in h
    assert "is ordinary" not in h


def test_the_schedule_and_both_consultations_are_on_the_card():
    h = panels.trade_card_html(row(plan=plan(
        alignment=[{"category": "live", "label": "Live Recon", "bias": "BULL", "verdict": "supports",
                    "note": ""},
                   {"category": "weekly", "label": "Weekly", "bias": "BEAR", "verdict": "opposes",
                    "note": ""},
                   {"category": "monthly", "label": "Monthly", "bias": "", "verdict": "silent",
                    "note": "no anchored verdict yet"}],
        traps=[{"trap_id": 3, "side": "bull_trap", "timeframe": "4h", "level": 78_650.0, "near": "TP1",
                "distance": 50.0}])), None, NOW)
    assert "Chart monitoring schedule" in h and "the 4h close" in h
    assert "Live Recon: BULL - supports this trade" in h
    assert "Weekly: BEAR - against this trade" in h
    assert "Monthly: no verdict - no live verdict" in h and "no anchored verdict yet" in h
    assert "anchored bull trap on the 4h sits $50 from this trade's TP1" in h


def test_no_trap_on_the_levels_is_stated_rather_than_left_blank():
    h = panels.trade_card_html(row(), None, NOW)
    assert "no anchored trap sits on this trade's levels" in h


def test_the_head_and_both_domain_scores_are_shown():
    h = panels.trade_card_html(row(), None, NOW)
    assert "Head of the Active Trade desk: Take" in h and "62% confidence" in h
    assert "The case for (78/100)" in h and "The case against (34/100)" in h
    assert "Quant &amp; Statistics 16/20" in h


def test_the_estimate_is_never_presented_as_a_win_rate():
    h = panels.trade_card_html(row(), None, NOW)
    assert "not a backtested win rate" in h


# ---------- the live block, the only part Refresh changes ----------
def test_the_live_block_shows_pnl_in_r_and_which_targets_are_reached():
    prog = progress(LONG, 78_000.0, 77_600.0, targets_as_objects(plan()["targets"]), 78_650.0)
    h = panels.trade_card_html(row(), prog, NOW)
    assert "Now $78,650" in h and "+0.83%" in h and "+1.62R" in h
    assert "targets reached: TP1" in h


def test_a_position_with_nothing_reached_says_none_yet():
    prog = progress(LONG, 78_000.0, 77_600.0, targets_as_objects(plan()["targets"]), 78_100.0)
    assert "targets reached: none yet" in panels.trade_card_html(row(), prog, NOW)


def test_a_stopped_position_says_so_in_the_live_block():
    prog = progress(LONG, 78_000.0, 77_600.0, targets_as_objects(plan()["targets"]), 77_500.0)
    h = panels.trade_card_html(row(), prog, NOW)
    assert "the stop has been reached" in h


def test_no_price_says_the_numbers_are_the_plan_only():
    h = panels.trade_card_html(row(), None, NOW)
    assert "No live price this refresh" in h


def test_a_closed_card_says_when_and_why():
    h = panels.trade_card_html(row(status="closed", closed_ms=NOW - HOUR, closed_price=78_400.0,
                                   closed_reason="killed by the trader"), None, NOW)
    assert "Closed" in h and "$78,400" in h and "killed by the trader" in h
    assert "(closed)" in h


# ---------- unscored, and escaped ----------
def test_an_unscored_card_shows_its_levels_and_claims_no_score():
    h = panels.trade_card_html(row(plan=plan(ok=False, score=None, probability=None,
                                             notes=["No model provider is configured."])), None, NOW)
    assert "unscored: the sub-agents did not answer" in h
    assert "score 72/100" not in h and "confluence estimate" not in h
    assert "$78,600" in h, "the deterministic targets are still there"
    assert "No model provider is configured." in h


@pytest.mark.parametrize("field", ["asset", "side", "closed_reason", "stop_verdict"])
def test_every_stored_string_is_escaped(field):
    payload = "<script>alert(1)</script>"
    h = panels.trade_card_html(row(**{field: payload, "status": "closed",
                                      "plan": plan(stop_verdict=payload)}), None, NOW)
    assert "<script>" not in h, f"{field} reached the page as markup"


def test_the_plan_text_is_escaped_too():
    h = panels.trade_card_html(row(plan=plan(
        notes=["<img src=x onerror=alert(1)>"],
        stop_reasons=["<b>bold</b>"],
        alignment=[{"category": "live", "label": "<i>Live</i>", "bias": "<u>BULL</u>",
                    "verdict": "supports", "note": "<em>x</em>"}],
        targets=[target(basis="<script>x</script>", due_from="<script>y</script>")])), None, NOW)
    for markup in ("<img", "<b>bold", "<i>Live", "<u>BULL", "<em>", "<script>"):
        assert markup not in h, markup


def test_a_card_renders_with_an_empty_plan():
    assert panels.trade_card_html(row(plan={}), None, NOW)
    assert panels.trade_card_html({"id": 2, "side": SHORT, "asset": "ETH/USDT", "status": "open"},
                                  None, NOW)
