import json
import pathlib

import numpy as np

from core.data.kraken_spot import parse_ohlc, parse_ticker

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_ohlc_shape_and_types():
    df = parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text()))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == np.int64
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] > 1_600_000_000_000  # milliseconds, not seconds
    assert (df["high"] >= df["low"]).all()
    assert len(df) >= 200


def test_parse_ticker_last_price():
    last = parse_ticker(json.loads((FX / "kraken_ticker.json").read_text()))
    assert 1_000 < last < 1_000_000
