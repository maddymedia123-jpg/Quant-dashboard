"""Offline MarketSnapshot built from the recorded test fixtures.

Enabled with the environment variable TI_OFFLINE_FIXTURES=1. Intended for local
development and UI checks when the network is unavailable; every source is labelled
"fixture" so it can never be mistaken for live data."""
from __future__ import annotations

import json
import pathlib

from core.data.binance_futures import parse_binance
from core.data.coinlobster import parse_liquidations, parse_whales
from core.data.deribit_options import parse_book_summary
from core.data.hyperliquid import parse_meta
from core.data.kraken_spot import parse_ohlc, parse_ticker
from core.data.sentiment import parse_fng
from core.data.stablecoins import parse_stablecoins
from core.data.types import MarketSnapshot, SpotSnapshot

FX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load(name: str):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def _mark(snapshot):
    snapshot.source = "fixture"
    return snapshot


def fixture_spot() -> SpotSnapshot:
    frames = {}
    for tf, name in (("15m", "kraken_ohlc_15m.json"), ("1h", "kraken_ohlc_1h.json")):
        if (FX / name).exists():
            frames[tf] = parse_ohlc(_load(name))
    # higher timeframes are not recorded from Kraken; reuse the 1h frame so every category renders
    for tf in ("4h", "1d", "1w"):
        frames.setdefault(tf, frames["1h"])
    return SpotSnapshot(source="fixture", last=parse_ticker(_load("kraken_ticker.json")), frames=frames)


def fixture_context() -> dict:
    liqs = _mark(parse_liquidations(_load("coinlobster_liquidations.json")))
    whales = _mark(parse_whales(_load("coinlobster_whales.json"), _load("coinlobster_radar.json")))
    return {
        "futures": _mark(parse_binance(_load("binance_funding.json"), _load("binance_oi.json"), _load("binance_oi_hist.json"),
                                       _load("binance_ls.json"), _load("binance_taker.json"))),
        "options": _mark(parse_book_summary(_load("deribit_book_summary.json"))),
        "sentiment": _mark(parse_fng(_load("fng.json"))),
        "hyperliquid": _mark(parse_meta(_load("hyperliquid_meta.json"))),
        "liquidations": liqs,
        "whales": whales,
        "stablecoins": _mark(parse_stablecoins(_load("defillama_stablecoins.json"))),
    }


def fixture_market() -> MarketSnapshot:
    return MarketSnapshot(spot=fixture_spot(), **fixture_context())
