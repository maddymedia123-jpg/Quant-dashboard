"""USD-M perpetual futures context: funding, open interest, long/short, taker flow.
Binance first (richest public data); Bybit fallback (Binance geo-blocks US egress)."""
from __future__ import annotations

import asyncio
import logging
import statistics

import httpx

from core.config import SYMBOLS
from core.data.types import FuturesSnapshot

log = logging.getLogger(__name__)
BN = "https://fapi.binance.com"
BB = "https://api.bybit.com"


def _pct(new: float, old: float) -> float | None:
    return None if not old else (new - old) / old * 100.0


def parse_binance(funding: list, oi: dict, oi_hist: list, ls: list, taker: list) -> FuturesSnapshot:
    rates = [float(x["fundingRate"]) for x in funding]
    hist = [(int(x["timestamp"]), float(x["sumOpenInterest"])) for x in oi_hist]
    hist.sort()
    return FuturesSnapshot(
        source="binance",
        funding_rate=rates[-1] if rates else None,
        funding_7d_mean=statistics.fmean(rates) if rates else None,
        open_interest=float(oi["openInterest"]),
        open_interest_usd=float(oi_hist[-1]["sumOpenInterestValue"]) if oi_hist else None,
        oi_change_24h_pct=_pct(hist[-1][1], hist[0][1]) if len(hist) >= 2 else None,
        long_short_ratio=float(ls[-1]["longShortRatio"]) if ls else None,
        long_account_pct=float(ls[-1]["longAccount"]) if ls else None,
        taker_buy_sell_ratio=float(taker[-1]["buySellRatio"]) if taker else None,
        mark_price=float(funding[-1]["markPrice"]) if funding and funding[-1].get("markPrice") else None,
        oi_history=hist,
    )


def parse_bybit(tickers: dict, funding: dict, oi: dict, ratio: dict) -> FuturesSnapshot:
    t = tickers["result"]["list"][0]
    rates = [float(x["fundingRate"]) for x in funding["result"]["list"]]
    hist = [(int(x["timestamp"]), float(x["openInterest"])) for x in oi["result"]["list"]]
    hist.sort()
    rl = ratio["result"]["list"]
    buy = float(rl[0]["buyRatio"]) if rl else None
    sell = float(rl[0]["sellRatio"]) if rl else None
    return FuturesSnapshot(
        source="bybit",
        funding_rate=float(t["fundingRate"]),
        funding_7d_mean=statistics.fmean(rates) if rates else None,
        open_interest=float(t["openInterest"]),
        open_interest_usd=float(t["openInterestValue"]),
        oi_change_24h_pct=_pct(hist[-1][1], hist[0][1]) if len(hist) >= 2 else None,
        long_short_ratio=(buy / sell) if buy and sell else None,
        long_account_pct=buy,
        taker_buy_sell_ratio=None,
        mark_price=float(t["markPrice"]),
        oi_history=hist,
    )


async def _get(client: httpx.AsyncClient, url: str, **params):
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()


async def _binance(client: httpx.AsyncClient) -> FuturesSnapshot:
    s = SYMBOLS["binance"]
    funding, oi, oi_hist, ls, taker = await asyncio.gather(
        _get(client, f"{BN}/fapi/v1/fundingRate", symbol=s, limit=21),
        _get(client, f"{BN}/fapi/v1/openInterest", symbol=s),
        _get(client, f"{BN}/futures/data/openInterestHist", symbol=s, period="1h", limit=25),
        _get(client, f"{BN}/futures/data/globalLongShortAccountRatio", symbol=s, period="1h", limit=1),
        _get(client, f"{BN}/futures/data/takerlongshortRatio", symbol=s, period="1h", limit=1),
    )
    return parse_binance(funding, oi, oi_hist, ls, taker)


async def _bybit(client: httpx.AsyncClient) -> FuturesSnapshot:
    s = SYMBOLS["bybit"]
    tickers, funding, oi, ratio = await asyncio.gather(
        _get(client, f"{BB}/v5/market/tickers", category="linear", symbol=s),
        _get(client, f"{BB}/v5/market/funding/history", category="linear", symbol=s, limit=21),
        _get(client, f"{BB}/v5/market/open-interest", category="linear", symbol=s, intervalTime="1h", limit=25),
        _get(client, f"{BB}/v5/market/account-ratio", category="linear", symbol=s, period="1h", limit=1),
    )
    return parse_bybit(tickers, funding, oi, ratio)


async def fetch_futures(client: httpx.AsyncClient) -> FuturesSnapshot:
    errors = []
    for name, fn in (("binance", _binance), ("bybit", _bybit)):
        try:
            return await fn(client)
        except Exception as e:  # noqa: BLE001 - provider boundary
            log.warning("futures via %s failed: %s", name, e)
            errors.append(f"{name}: {e}")
    return FuturesSnapshot.unavailable("binance,bybit", " | ".join(errors))
