"""Offline MarketSnapshot built from the recorded test fixtures.

Enabled with the environment variable TI_OFFLINE_FIXTURES=1. Intended for local
development and UI checks when the network is unavailable; the sidebar shows the
source as "fixture" so it can never be mistaken for live data."""
from __future__ import annotations

import json
import pathlib

from core.data.binance_futures import parse_binance
from core.data.deribit_options import parse_book_summary
from core.data.kraken_spot import parse_ohlc, parse_ticker
from core.data.sentiment import parse_fng
from core.data.types import MarketSnapshot, SpotSnapshot

FX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load(name: str):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def fixture_market() -> MarketSnapshot:
    frames = {}
    for tf, name in (("15m", "kraken_ohlc_15m.json"), ("1h", "kraken_ohlc_1h.json")):
        if (FX / name).exists():
            frames[tf] = parse_ohlc(_load(name))
    # higher timeframes are not recorded; reuse the 1h frame so every category renders
    for tf in ("4h", "1d", "1w"):
        frames.setdefault(tf, frames["1h"])
    spot = SpotSnapshot(source="fixture", last=parse_ticker(_load("kraken_ticker.json")), frames=frames)
    fut = parse_binance(_load("binance_funding.json"), _load("binance_oi.json"), _load("binance_oi_hist.json"),
                        _load("binance_ls.json"), _load("binance_taker.json"))
    fut.source = "fixture"
    opt = parse_book_summary(_load("deribit_book_summary.json"))
    opt.source = "fixture"
    sent = parse_fng(_load("fng.json"))
    sent.source = "fixture"
    return MarketSnapshot(spot=spot, futures=fut, options=opt, sentiment=sent)
