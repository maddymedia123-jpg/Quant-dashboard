"""USD-M perpetual futures context: funding, open interest, long/short, taker flow.
Binance first (richest public data); Bybit fallback (Binance geo-blocks US egress)."""
from __future__ import annotations

import asyncio
import logging
import statistics

import httpx

from core.config import SYMBOLS
from core.data.types import FuturesSnapshot, HyperliquidSnapshot, WhalesSnapshot

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
    """Binance → Bybit → Gate.io (all carry the full field set). Thinner fallbacks (CoinLobster relay,
    Kraken Futures) are applied in market.fetch_context."""
    from core.data.gate_futures import fetch_gate_futures  # local import keeps module load light
    errors = []
    for name, fn in (("binance", _binance), ("bybit", _bybit)):
        try:
            return await fn(client)
        except Exception as e:  # noqa: BLE001 - provider boundary
            log.warning("futures via %s failed: %s", name, e)
            errors.append(f"{name}: {e}")
    gate = await fetch_gate_futures(client)
    if gate.available:
        return gate
    errors.append(f"gate: {gate.error}")
    return FuturesSnapshot.unavailable("binance,bybit,gate", " | ".join(errors))


PREFERRED_VENUES = ("Binance Futures", "Bybit", "OKX", "Bitget Futures")


def compose_futures(whales: WhalesSnapshot | None, hl: HyperliquidSnapshot | None, errors: str = "") -> FuturesSnapshot:
    """US-safe fallback: build the futures context from CoinLobster's per-exchange market conditions and
    Hyperliquid. Funding and open interest come through; 7d funding mean, long/short and taker ratios and
    the 24h OI change are not available on this path and stay None."""
    w = whales if (whales is not None and whales.available) else None
    h = hl if (hl is not None and hl.available) else None
    if w is None and h is None:
        return FuturesSnapshot.unavailable("binance,bybit", errors or "direct exchanges blocked; no fallback data")
    funding = None
    venue = None
    if w:
        for v in PREFERRED_VENUES:
            if v in w.funding_by_exchange:
                funding, venue = float(w.funding_by_exchange[v]), v
                break
    if funding is None and h and h.funding_rate_1h is not None:
        funding, venue = float(h.funding_rate_1h) * 8.0, "hyperliquid 1h x8"
    oi_usd = None
    if w and w.oi_by_exchange_usd:
        oi_usd = next((float(w.oi_by_exchange_usd[v]) for v in PREFERRED_VENUES if v in w.oi_by_exchange_usd), None)
        if oi_usd is None:
            oi_usd = float(max(w.oi_by_exchange_usd.values()))
    mark = h.mark_price if h else None
    oi_contracts = (oi_usd / mark) if (oi_usd and mark) else (h.open_interest if h else None)
    if funding is None and oi_usd is None and oi_contracts is None:
        return FuturesSnapshot.unavailable("binance,bybit", errors or "direct exchanges blocked; fallback sources carried no funding/OI")
    base = "coinlobster+hyperliquid" if (w and h) else ("coinlobster" if w else "hyperliquid")
    src = f"{base} ({venue})" if venue else base
    return FuturesSnapshot(source=src, funding_rate=funding, funding_7d_mean=None, open_interest=oi_contracts,
                           open_interest_usd=oi_usd, oi_change_24h_pct=None, long_short_ratio=None, long_account_pct=None,
                           taker_buy_sell_ratio=None, mark_price=mark, oi_history=[])
