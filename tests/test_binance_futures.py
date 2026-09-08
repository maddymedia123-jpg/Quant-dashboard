import json
import pathlib

import pytest

from core.data.binance_futures import parse_binance, parse_bybit

FX = pathlib.Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FX / name).read_text())


@pytest.mark.skipif(not (FX / "binance_funding.json").exists(), reason="binance fixtures not recorded")
def test_parse_binance_fields():
    s = parse_binance(_load("binance_funding.json"), _load("binance_oi.json"), _load("binance_oi_hist.json"),
                      _load("binance_ls.json"), _load("binance_taker.json"))
    assert s.available and s.source == "binance"
    assert -0.01 < s.funding_rate < 0.01
    assert s.open_interest > 1_000
    assert s.open_interest_usd > 1e8
    assert s.oi_change_24h_pct is not None
    assert 0.2 < s.long_short_ratio < 5
    assert 0 < s.long_account_pct < 1
    assert 0.2 < s.taker_buy_sell_ratio < 5
    assert len(s.oi_history) >= 20


def test_parse_bybit_fields():
    s = parse_bybit(_load("bybit_tickers.json"), _load("bybit_funding.json"), _load("bybit_oi.json"), _load("bybit_ratio.json"))
    assert s.available and s.source == "bybit"
    assert -0.01 < s.funding_rate < 0.01
    assert s.open_interest > 1_000
    assert s.oi_change_24h_pct is not None
    assert 0.2 < s.long_short_ratio < 5
    assert s.taker_buy_sell_ratio is None  # bybit has no public taker ratio
