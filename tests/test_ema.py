import numpy as np
import pandas as pd

from core.indicators.ema import ema, ema_stack


def _df(closes):
    n = len(closes)
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0})


def test_ema_constant_series_is_constant():
    s = ema(pd.Series([50.0] * 100), 21)
    assert abs(s.iloc[-1] - 50.0) < 1e-9


def test_ema_stack_bull_on_uptrend():
    st = ema_stack(_df(np.linspace(100, 200, 260)))
    assert st.state == "BULL"
    assert st.values[21] > st.values[50] > st.values[100] > st.values[200]
    assert set(st.series) == {21, 50, 100, 200}


def test_ema_stack_bear_on_downtrend():
    assert ema_stack(_df(np.linspace(200, 100, 260))).state == "BEAR"


def test_ema_stack_missing_when_short_history():
    st = ema_stack(_df(np.linspace(100, 110, 60)))
    assert st.values[200] is None and st.state == "MIXED"
