"""The head agent grades both teams, and names the protocols that failed.

The client's spec: gauge the accuracy of both teams, and when it is not optimal, produce a separate
accuracy report that pinpoints protocol failures, misread liquidity, false sentiment signals,
structural misalignment and missed traps, so the system can be corrected."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from core import live_accuracy as acc
from core import live_anchor as la
from core.agents.judge import ChoiceResult, JudgeResult
from core.agents.live_recon import LiveReconResult, resolve, score_team
from core.agents.recon_profiles import PROFILES
from core.agents.rubric import ALL_ITEMS, BEARISH, BULLISH
from core.store import Store

LIVE = PROFILES["live"]
H4 = 4 * 3_600_000
M15 = 900_000


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def result(bull_p, bear_p, trap="NO_TRAP", trap_p=0.8, items=None) -> LiveReconResult:
    tr = ChoiceResult(choice=trap, probabilities={trap: trap_p})
    bull_probs = {i.id: bull_p for i in ALL_ITEMS}
    bear_probs = {i.id: bear_p for i in ALL_ITEMS}
    for (side, item), p in (items or {}).items():
        (bull_probs if side == BULLISH else bear_probs)[item] = p
    bull = score_team(BULLISH, JudgeResult(probabilities=bull_probs, provider="typesafe"))
    bear = score_team(BEARISH, JudgeResult(probabilities=bear_probs, provider="typesafe"))
    bias, margin, conf, cons, ov, notes = resolve(bull, bear, tr)
    return LiveReconResult(generated_at=datetime.now(timezone.utc), bull=bull, bear=bear, trap=tr, bias=bias,
                           margin=margin, confidence=conf, consulted=cons, override=ov, notes=notes)


def frames_with(closes_at: dict[int, float]) -> dict[str, pd.DataFrame]:
    """15m candles whose close at each given boundary is the given price."""
    rows = []
    for close_ms, px in sorted(closes_at.items()):
        rows.append({"timestamp": close_ms - M15, "open": px, "high": px + 1, "low": px - 1, "close": px,
                     "volume": 1.0})
    return {"15m": pd.DataFrame(rows)}


# ---------- price at a candle boundary ----------
def test_price_is_read_at_the_exact_candle_boundary():
    fr = frames_with({H4 * 3: 81_500.0})
    assert acc.price_at(fr, H4 * 3) == 81_500.0


def test_no_completed_candle_at_the_boundary_means_no_price_not_a_neighbour():
    fr = frames_with({H4 * 3: 81_500.0})
    assert acc.price_at(fr, H4 * 3 + M15) is None
    assert acc.price_at({}, H4 * 3) is None


def test_a_coarser_frame_answers_when_the_fine_one_does_not_reach_back():
    daily = pd.DataFrame([{"timestamp": 0, "open": 1, "high": 2, "low": 0, "close": 70_000.0, "volume": 1}])
    fr = {"15m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), "1d": daily}
    assert acc.price_at(fr, 86_400_000) == 70_000.0


# ---------- scoring one window ----------
def test_a_bullish_call_is_right_when_price_rises_beyond_the_flat_band():
    s = acc.score_window("BULL", "NO_TRAP", 80.0, 30.0, 80_000.0, 81_000.0, LIVE)
    assert s.realised == "UP" and s.direction_hit is True
    assert s.move_pct == pytest.approx(1.25)
    assert s.bull_brier == pytest.approx((0.8 - 1) ** 2) and s.bear_brier == pytest.approx(0.3 ** 2)


def test_a_bearish_call_into_a_rally_is_a_miss():
    s = acc.score_window("BEAR", "NO_TRAP", 20.0, 80.0, 80_000.0, 81_000.0, LIVE)
    assert s.direction_hit is False and s.bear_brier == pytest.approx(0.64)


def test_balanced_is_right_when_price_stays_inside_the_band():
    s = acc.score_window("BALANCED", "NO_TRAP", 50.0, 50.0, 80_000.0, 80_100.0, LIVE)
    assert s.realised == "FLAT" and s.direction_hit is True


def test_the_flat_band_widens_with_the_horizon():
    move = (80_800.0 / 80_000.0 - 1) * 100          # +1.0%
    assert acc.score_window("BALANCED", None, 50, 50, 80_000.0, 80_800.0, LIVE).realised == "UP"
    assert acc.score_window("BALANCED", None, 50, 50, 80_000.0, 80_800.0, PROFILES["weekly"]).realised == "FLAT"
    assert move == pytest.approx(1.0)


def test_a_trap_risk_call_is_right_when_the_move_it_doubted_fails():
    assert acc.score_window("BULL TRAP RISK", "BULL_TRAP", 80, 30, 80_000.0, 79_000.0, LIVE).direction_hit
    assert acc.score_window("BULL TRAP RISK", "BULL_TRAP", 80, 30, 80_000.0, 80_050.0, LIVE).direction_hit
    assert not acc.score_window("BULL TRAP RISK", "BULL_TRAP", 80, 30, 80_000.0, 81_000.0, LIVE).direction_hit


def test_trap_calls_are_scored_on_their_own():
    assert acc.score_window("BULL", "BULL_TRAP", 80, 30, 80_000.0, 79_000.0, LIVE).trap_hit is True
    assert acc.score_window("BEAR", "BEAR_TRAP", 30, 80, 80_000.0, 81_000.0, LIVE).trap_hit is True
    assert acc.score_window("BULL", "NO_TRAP", 80, 30, 80_000.0, 80_050.0, LIVE).trap_hit is True
    assert acc.score_window("BULL", "GENUINE_MOVE", 80, 30, 80_000.0, 81_000.0, LIVE).trap_hit is True
    assert acc.score_window("BULL", "GENUINE_MOVE", 80, 30, 80_000.0, 79_000.0, LIVE).trap_hit is False
    assert acc.score_window("BULL", None, 80, 30, 80_000.0, 79_000.0, LIVE).trap_hit is None


# ---------- scoring the ledger ----------
def test_only_closed_windows_are_scored_and_each_only_once(store):
    la.publish(store, result(0.8, 0.3), H4 + 60_000, price=80_000.0)
    fr = frames_with({2 * H4: 81_000.0})
    assert acc.score_due(store, fr, now_ms=2 * H4 - 1) == [], "the candle has not closed"
    done = acc.score_due(store, fr, now_ms=2 * H4 + 60_000)
    assert len(done) == 1 and done[0].direction_hit is True
    assert acc.score_due(store, fr, now_ms=2 * H4 + 120_000) == [], "already scored"
    assert len(store.recon_scores()) == 1


def test_a_window_with_no_closing_price_waits_rather_than_guessing(store):
    la.publish(store, result(0.8, 0.3), H4 + 60_000, price=80_000.0)
    assert acc.score_due(store, {}, now_ms=3 * H4) == []
    assert store.recon_scores() == []


def test_scoring_keeps_categories_apart(store):
    la.publish(store, result(0.8, 0.3), H4 + 60_000, price=80_000.0)
    fr = frames_with({2 * H4: 81_000.0})
    acc.score_due(store, fr, now_ms=2 * H4 + 60_000, profile=PROFILES["intraday"])
    assert store.recon_scores() == [] and store.recon_scores(category="intraday") == []


# ---------- the report ----------
def seed(store, n, bull_p, bear_p, move_up: bool, items=None, trap="NO_TRAP"):
    """n consecutive live windows, each anchored at 80,000 and closing up or down 1.25%."""
    for k in range(n):
        start = (10 + k) * H4
        la.publish(store, result(bull_p, bear_p, trap=trap, items=items), start + 60_000, price=80_000.0)
        close_px = 81_000.0 if move_up else 79_000.0
        acc.score_due(store, frames_with({start + H4: close_px}), now_ms=start + H4 + 60_000)


def test_fewer_than_ten_scored_windows_is_still_collecting(store):
    seed(store, 4, 0.8, 0.3, move_up=False)
    rep = acc.diagnose(store)
    assert rep.status == "collecting" and rep.n == 4 and rep.issues == []
    assert "4 of 10" in rep.headline


def test_a_healthy_ledger_produces_no_calibration_report(store):
    seed(store, 12, 0.8, 0.2, move_up=True)
    rep = acc.diagnose(store)
    assert rep.status == "healthy" and rep.hit_rate == 1.0 and rep.issues == []


def test_poor_accuracy_names_the_protocols_that_failed(store):
    # the bullish team leans hard on sweeps and structure while price keeps falling
    items = {(BULLISH, "liq_sweep"): 0.95, (BULLISH, "smc_bos"): 0.9, (BULLISH, "macro_sentiment"): 0.9}
    seed(store, 12, 0.7, 0.3, move_up=False, items=items)
    rep = acc.diagnose(store)
    assert rep.status == "calibration needed" and rep.hit_rate == 0.0
    kinds = {i.kind for i in rep.issues}
    assert {"misread liquidity", "structural misalignment", "false sentiment"} <= kinds
    sweep = next(i for i in rep.issues if i.item == "liq_sweep")
    assert sweep.side == BULLISH and sweep.mean_probability == pytest.approx(0.95)
    assert "sweep" in sweep.finding.lower() and sweep.windows == 12
    assert rep.bull_brier > 0.25, "worse than a coin flip"


def test_missed_traps_are_reported(store):
    seed(store, 12, 0.8, 0.2, move_up=False, trap="GENUINE_MOVE")
    rep = acc.diagnose(store)
    trap = next(i for i in rep.issues if i.kind == "missed traps")
    assert "GENUINE_MOVE" in trap.finding and trap.windows == 12
    assert rep.trap_hit_rate == 0.0


def test_anchors_from_before_item_tracking_still_count_toward_direction(store):
    seed(store, 10, 0.8, 0.2, move_up=True)
    with store._lock:                                        # simulate an older payload without items
        rows = store._conn.execute("SELECT window_open_ms, payload FROM recon_anchors").fetchall()
        import json
        for r in rows:
            p = json.loads(r["payload"])
            p.pop("items", None)
            store._conn.execute("UPDATE recon_anchors SET payload=? WHERE window_open_ms=?",
                                (json.dumps(p), r["window_open_ms"]))
        store._conn.commit()
    rep = acc.diagnose(store)
    assert rep.n == 10 and rep.hit_rate == 1.0
