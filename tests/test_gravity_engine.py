from datetime import datetime, timezone
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from core import confluence
from core.gravity_engine import LONG, SHORT, PreemptiveSignal, evaluate
from core.layman_summary import render_dispatch_text
from core.master_card import master_trade_context


def compressed_frame():
    rng = np.random.default_rng(42)
    closes = [100_000.0]
    for _ in range(12):
        closes.append(closes[-1] * (1 + rng.normal(0, 0.002)))
    for _ in range(12):
        closes.append(closes[-1] * (1 + rng.normal(0, 0.00005)))
    close = np.asarray(closes)
    width = np.r_[np.full(13, 180.0), np.full(12, 8.0)]
    return pd.DataFrame({
        "open": close, "high": close + width / 2, "low": close - width / 2,
        "close": close, "volume": 1.0,
    })


def test_compression_leverage_absorption_and_nearby_liquidations_preempt_close_confirmation():
    frame = compressed_frame()
    price = float(frame["close"].iloc[-1])
    signal = evaluate(
        frame, price=price, atr=180.0, oi_change_pct=3.0, funding_rate=0.0005,
        absorption_drop=0.8, absorption_side="bids",
        liq_levels={"available": True, "longs": [{"price": price * 0.99, "notional_usd": 50_000_000}],
                    "shorts": []},
    )
    assert signal.defcon_1 and signal.direction == SHORT
    assert signal.structural_failure_probability >= 0.78
    assert signal.pull_velocity is not None and signal.pull_velocity < 0
    assert signal.target_low is not None and signal.safety_line > price
    assert "before candle-close confirmation" in signal.plain_english_reason


def test_missing_derivatives_and_book_data_stay_missing_and_cannot_trigger_defcon():
    signal = evaluate(compressed_frame(), price=100_000.0, atr=180.0)
    assert signal.available and signal.compression_score is not None
    assert signal.oi_buildup_score is None
    assert signal.funding_skew_score is None
    assert signal.absorption_drop is None
    assert not signal.defcon_1


def test_emergency_overrides_the_three_confluence_gate_without_faking_votes():
    signal = PreemptiveSignal(
        available=True, direction=LONG, defcon_1=True, target_high=102_000.0,
        safety_line=98_000.0, structural_failure_probability=0.9,
        plain_english_reason="A verified compression-and-leverage flush is imminent.",
    )
    family = NS(key="flow", side=SHORT)
    decision = confluence.solve_one_sided_puzzle(
        "intraday", [family], signal, "$100,000", [101_000.0], 98_000.0,
    )
    assert decision["verdict"] == "DEFCON_1_PREEMPTIVE"
    assert decision["action"] == "BUY" and decision["side"] == LONG
    assert decision["targets"] == [102_000.0]
    assert decision["aligned_count"] == 0


def test_preemptive_signal_surfaces_from_the_confluence_gate():
    signal = PreemptiveSignal(
        available=True, direction=SHORT, defcon_1=True, target_low=98_000.0,
        safety_line=102_000.0, structural_failure_probability=0.9,
        plain_english_reason="A downside flush is imminent.",
    )
    framework = NS(side=LONG, zone=(99_900.0, 100_100.0), targets=(101_000.0, 102_000.0))
    analysis = NS(chart_tf="1h", vol=NS(atr=100.0))
    gate = confluence.read(framework, analysis, confluence.Evidence(preemptive=signal), 1_790_035_200_000)
    assert gate.verdict == "DEFCON_1_PREEMPTIVE"
    assert gate.side == SHORT and gate.preemptive is signal


def test_normal_directives_need_three_aligned_major_families():
    families = [NS(key=key, side=LONG) for key in ("flow", "structure", "positioning")]
    decision = confluence.solve_one_sided_puzzle("intraday", families, None, "$100,000", [102_000.0], 98_000.0)
    assert decision["verdict"] == "TRADE" and decision["action"] == "BUY"
    thin = confluence.solve_one_sided_puzzle("intraday", families[:2], None, "$100,000", [], 98_000.0)
    assert thin["verdict"] == "DIRECTIONAL_BIAS" and thin["action"] == "BUY"
    balanced = confluence.solve_one_sided_puzzle(
        "intraday", [NS(key="flow", side=LONG), NS(key="structure", side=SHORT)],
        None, "$100,000", [], 98_000.0,
    )
    assert balanced["verdict"] == "BALANCED_CONFLICT"
    tied = confluence.solve_one_sided_puzzle(
        "intraday", [NS(key=key, side=side) for key, side in
                     (("f1", LONG), ("f2", LONG), ("f3", LONG), ("f4", SHORT), ("f5", SHORT), ("f6", SHORT))],
        None, "$100,000", [], 98_000.0, preferred_side=SHORT,
    )
    assert tied["verdict"] == "DIRECTIONAL_BIAS" and tied["action"] == "SELL"


def test_dispatch_has_both_clocks_and_distinct_horizon_contexts():
    now = datetime(2026, 11, 24, 17, 0, tzinfo=timezone.utc)
    decision = {"verdict": "TRADE", "action": "BUY", "entry": "$100,000",
                "targets": [102_000.0], "safety_line": 98_000.0, "reason": "Three families align."}
    text = render_dispatch_text("Intraday", 100_000.0, decision, desk_key="intraday", dt_utc=now)
    assert "17:00 UTC (22:00 PKT" in text
    assert "INTRADAY / SWING MASTER" in text and "$1,000-$3,000+" in text
    assert master_trade_context("monthly")["tier"] == "LONG / MONTHLY MASTER"
    assert "monthly structural range" in master_trade_context("monthly")["move_context"]
