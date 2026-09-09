"""Kraken Futures public tickers: BTC perpetual funding, predicted funding, open interest, mark/index price.

Reachable from US egress, so it is a US-safe fallback for the futures context. Kraken publishes no
long/short or taker ratios. Funding is quoted as an absolute hourly amount; we normalise it to a relative
8h rate so it is comparable with Binance/Bybit. Kraken's BTC open interest is far smaller than Binance's,
so absolute OI from this path is not comparable across sources (percent changes still are)."""
from __future__ import annotations

import logging

import httpx

from core.data.types import FuturesSnapshot

log = logging.getLogger(__name__)
URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
HEADERS = {"User-Agent": "Mozilla/5.0 (trap-intel-dashboard)"}
PREFERRED = ("PF_XBTUSD", "PI_XBTUSD")


def parse_tickers(payload: dict) -> FuturesSnapshot:
    tickers = {t.get("symbol", "").upper(): t for t in payload.get("tickers", [])}
    t = next((tickers[s] for s in PREFERRED if s in tickers), None)
    if t is None:
        raise ValueError("no BTC perpetual in Kraken Futures tickers")
    mark = float(t["markPrice"]) if t.get("markPrice") is not None else None
    oi_contracts = float(t["openInterest"]) if t.get("openInterest") is not None else None
    # PF_ (flexible) open interest is in BTC; PI_ (inverse) is in USD contracts of $1
    symbol = t["symbol"].upper()
    if symbol.startswith("PI_"):
        oi_usd = oi_contracts
        oi_btc = (oi_usd / mark) if (oi_usd and mark) else None
    else:
        oi_btc = oi_contracts
        oi_usd = (oi_btc * mark) if (oi_btc and mark) else None
    fr_1h = float(t["fundingRate"]) if t.get("fundingRate") is not None else None
    # Kraken quotes absolute hourly funding: PF_ in USD per 1-BTC contract, PI_ in BTC per $1 contract.
    # Normalise to a relative 8h rate comparable with Binance/Bybit.
    if fr_1h is None or not mark:
        funding = None
    elif symbol.startswith("PI_"):
        funding = fr_1h * mark * 8.0
    else:
        funding = (fr_1h / mark) * 8.0
    return FuturesSnapshot(source=f"kraken-futures ({symbol})", funding_rate=funding, funding_7d_mean=None,
                           open_interest=oi_btc, open_interest_usd=oi_usd, oi_change_24h_pct=None, long_short_ratio=None,
                           long_account_pct=None, taker_buy_sell_ratio=None, mark_price=mark, oi_history=[])


async def fetch_kraken_futures(client: httpx.AsyncClient) -> FuturesSnapshot:
    try:
        r = await client.get(URL, headers=HEADERS)
        r.raise_for_status()
        return parse_tickers(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("kraken futures unavailable: %s", e)
        return FuturesSnapshot.unavailable("kraken-futures", str(e))
