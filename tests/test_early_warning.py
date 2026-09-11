from dataclasses import replace

from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.data.types import FuturesSnapshot
from core.indicators.early_warning import early_warnings
from core.indicators.levels import Level
from core.indicators.rsi import Divergence
from core.indicators.squeeze import SqueezeRisk


def _analysis():
    m = fixture_market()
    return m, __import__("core.indicators.category", fromlist=["analyze_category"]).analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)


def test_bull_trap_forming_requires_all_three_ingredients():
    m, a = _analysis()
    price, atr = a.price, a.vol.atr
    a.levels = [Level(price + 0.2 * atr, 3, "resistance", 0)]
    a.divergences = [Divergence("regular", "bearish", "forming", [(0, price), (1, price + 1)], [(0, 70.0), (1, 65.0)], (0, 1))]
    a.squeeze = SqueezeRisk(10, "NONE", [])
    a.reversal = None  # isolate trap logic from the Fibonacci reversal-window warning
    hot = FuturesSnapshot(source="x", funding_rate=0.0005, funding_7d_mean=0.0001, long_short_ratio=1.0)
    ws = early_warnings(a, hot, None)
    assert [w.kind for w in ws] == ["BULL_TRAP_FORMING"] and any("Funding" in d for d in ws[0].drivers)
    cold = FuturesSnapshot(source="x", funding_rate=0.0000, funding_7d_mean=0.0001, long_short_ratio=1.0)
    assert early_warnings(a, cold, None) == []
    a.divergences = []
    assert early_warnings(a, hot, None) == []


def test_bear_trap_forming_and_squeeze_states():
    m, a = _analysis()
    price, atr = a.price, a.vol.atr
    a.levels = [Level(price - 0.3 * atr, 2, "support", 0)]
    a.divergences = [Divergence("regular", "bullish", "confirmed", [(0, price), (1, price - 1)], [(0, 25.0), (1, 30.0)], (0, 1))]
    a.squeeze = SqueezeRisk(45, "SHORT_SQUEEZE", ["OI +5% in 24h"])
    a.reversal = None
    crowded_shorts = FuturesSnapshot(source="x", funding_rate=0.0001, funding_7d_mean=0.0001, long_short_ratio=0.7)
    ws = early_warnings(a, crowded_shorts, last_squeeze=40)
    kinds = [w.kind for w in ws]
    assert kinds == ["BEAR_TRAP_FORMING", "SQUEEZE_FORMING"]
    assert early_warnings(a, crowded_shorts, last_squeeze=50)[-1].kind == "BEAR_TRAP_FORMING"  # not rising → no squeeze forming
    a.squeeze = SqueezeRisk(72, "SHORT_SQUEEZE", ["OI +7% in 24h"])
    ws = early_warnings(a, FuturesSnapshot.unavailable("x", "y"), None)
    assert [w.kind for w in ws] == ["SQUEEZE_WARNING"] and ws[0].level == "warning"



def test_active_reversal_window_raises_warning():
    from core.indicators.reversal import ReversalWindow
    m, a = _analysis()
    a.levels, a.divergences = [], []
    a.squeeze = SqueezeRisk(10, "NONE", [])
    a.reversal = ReversalWindow("ACTIVE", "bearish", 5, 0, 0, 2, ["at resistance 78,600 (4 touches)"])
    ws = early_warnings(a, FuturesSnapshot.unavailable("x", "y"), None)
    assert [w.kind for w in ws] == ["REVERSAL_WINDOW"] and "at resistance" in ws[0].drivers[1]
