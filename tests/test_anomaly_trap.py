"""His Section 5.3: named anomaly traps, positioning read against the candle. Extreme fear plus a crowded
short is a bullish squeeze trap; extreme greed plus over-leveraged longs is a liquidation flush trap."""
from types import SimpleNamespace as NS

from core.indicators import anomaly_trap as at

NOW = 1_790_060_400_000
DAY = 86_400_000
ABSORB_UP = {"cumulative_delta": {"state": "bullish absorption"}}
ABSORB_DOWN = {"cumulative_delta": {"state": "bearish absorption"}}
FLAT = {"cumulative_delta": {"state": "flat"}}


def test_extreme_fear_and_a_crowded_short_with_absorbing_flow_is_a_set_squeeze_trap():
    t = at.read(18, NS(long_account_pct=0.40, long_short_ratio=None), {"funding": {"latest_8h_rate": -0.0002}},
                ABSORB_UP, -2.3)
    assert t.name == "Bullish squeeze trap" and t.side == "LONG" and t.status == "set" and t.trapped == "shorts"
    assert t.headline == ("Bullish squeeze trap, set: extreme fear (18/100) + a crowded short (40% of accounts long) + "
                          "shorts paying to hold. BTC fell 2.3% in 24 hours; price was pushed down but buyers took "
                          "everything offered. The shorts are the crowd that gets forced out.")
    # taker buying into a falling tape absorbs too; a flat tape with nothing taken is only forming
    buying = at.read(18, NS(long_account_pct=0.40), {"taker_delta": {"delta_btc": 120.0}}, FLAT, -1.1)
    assert buying.status == "set" and "more buying than selling at market while price fell" in buying.flow
    forming = at.read(18, NS(long_account_pct=0.40), {"taker_delta": {"delta_btc": -40.0}}, FLAT, -1.1)
    assert forming.status == "forming" and forming.flow.startswith("no absorption yet")


def test_extreme_greed_and_leveraged_longs_is_a_flush_trap_and_greed_without_leverage_is_not():
    derivs = {"funding": {"latest_8h_rate": 0.0005}, "open_interest": [{"over": "4h", "change_pct": 3.1}]}
    t = at.read(82, NS(long_account_pct=0.66), derivs, ABSORB_DOWN, 1.8)
    assert t.name == "Liquidation flush trap" and t.side == "SHORT" and t.status == "set" and t.trapped == "longs"
    assert t.crowd == ("a crowded long (66% of accounts long)", "longs paying a high premium to hold",
                       "open positions growing (+3.1% in 4h)")
    assert "price was pushed up but sellers met every bid" in t.flow and t.candle == "BTC rose 1.8% in 24 hours"
    # leverage can also show as open interest at a 30-day high
    hist = [(NOW - d * DAY, 50_000.0 + (1_000.0 if d == 0 else 0.0)) for d in range(40, -1, -1)]
    high = at.read(82, NS(long_account_pct=0.66), {"funding": {"latest_8h_rate": 0.0001}}, FLAT, 0.4, hist, NOW)
    assert high is not None and "open positions at a 30-day high" in high.crowd and high.status == "forming"
    # a crowded long with open interest shrinking is greed, not a flush
    assert at.read(82, NS(long_account_pct=0.66), {"open_interest": [{"over": "4h", "change_pct": -0.8}]}, FLAT, 0.4) is None
    # a 30-day high is judged inside the window only
    old_high = [(NOW - 60 * DAY, 90_000.0)] + [(NOW - d * DAY, 50_000.0) for d in range(29, -1, -1)]
    assert at.oi_at_high(old_high, NOW) is True and at.oi_at_high(old_high + [(NOW + 1, 48_000.0)], NOW) is False


def test_nothing_extreme_or_nothing_crowded_is_no_trap():
    assert at.read(50, NS(long_account_pct=0.40), {"funding": {"latest_8h_rate": -0.0002}}, ABSORB_UP, -2.0) is None
    assert at.read(18, NS(long_account_pct=0.52), {"funding": {"latest_8h_rate": 0.0001}}, ABSORB_UP, -2.0) is None
    assert at.read(None, NS(long_account_pct=0.40), None, None, None) is None
    assert at.read(18, None, None, None, None) is None
    ratio_only = at.read(18, NS(long_account_pct=None, long_short_ratio=0.7), None, None, None)
    assert ratio_only is not None and "41% of accounts long" in ratio_only.crowd[0] and "not readable" in ratio_only.candle


def test_the_state_and_the_words_carry_no_indicator_names():
    t = at.read(18, NS(long_account_pct=0.40), {"funding": {"latest_8h_rate": -0.0002}}, ABSORB_UP, -2.3)
    s = t.to_state()
    assert set(s) == {"name", "side", "status", "trapped", "mood", "crowd", "candle", "flow", "headline"}
    text = s["headline"].lower()
    for word in ("funding", "cvd", "delta", "rsi", "ema", "squeeze score", "l/s"):
        assert word not in text, word


def test_from_evidence_reads_the_bundle_the_gate_already_has():
    ev = NS(fear_greed=18, futures=NS(long_account_pct=0.40), derivs={"funding": {"latest_8h_rate": -0.0002}},
            liquidity={"1h": ABSORB_UP, "4h": FLAT}, price_change_24h_pct=-2.3, oi_history=())
    assert at.from_evidence(ev, "1h").status == "set" and at.from_evidence(ev, "4h").status == "forming"
    assert at.from_evidence(NS(), "1h") is None
