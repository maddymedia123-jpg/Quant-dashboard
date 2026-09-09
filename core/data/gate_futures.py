"""Gate.io USDT perpetual public stats: funding (current + history), open interest history, account and
taker long/short ratios, liquidation sizes. Keyless; the richest non-Binance source we have found."""
from __future__ import annotations

import asyncio
import logging
import statistics

import httpx

from core.data.types import FuturesSnapshot

log = logging.getLogger(__name__)
BASE = "https://api.gateio.ws/api/v4/futures/usdt"
CONTRACT = "BTC_USDT"
HEADERS = {"User-Agent": "Mozilla/5.0 (trap-intel-dashboard)", "Accept": "application/json"}


def parse_gate(stats: list[dict], tickers: list[dict], funding: list[dict], contract: dict | None = None) -> FuturesSnapshot:
    if not stats:
        raise ValueError("gate contract_stats empty")
    rows = sorted(stats, key=lambda r: int(r["time"]))
    last = rows[-1]
    mult = float((contract or {}).get("quanto_multiplier") or 0.0001)  # BTC per contract
    mark = float(last.get("mark_price") or (tickers[0].get("mark_price") if tickers else 0) or 0) or None
    oi_btc = float(last["open_interest"]) * mult
    oi_usd = float(last["open_interest_usd"]) if last.get("open_interest_usd") else (oi_btc * mark if mark else None)
    first = rows[0]
    oi_change = ((float(last["open_interest"]) / float(first["open_interest"])) - 1.0) * 100.0 if float(first["open_interest"]) else None
    rates = sorted(((int(x["t"]), float(x["r"])) for x in funding), key=lambda x: x[0])
    t0 = tickers[0] if tickers else {}
    current = float(t0["funding_rate"]) if t0.get("funding_rate") not in (None, "") else (rates[-1][1] if rates else None)
    lsr_account = float(last["lsr_account"]) if last.get("lsr_account") is not None else None
    return FuturesSnapshot(
        source="gate",
        funding_rate=current,
        funding_7d_mean=statistics.fmean(r for _, r in rates) if rates else None,
        open_interest=oi_btc,
        open_interest_usd=oi_usd,
        oi_change_24h_pct=oi_change,
        long_short_ratio=lsr_account,
        long_account_pct=(lsr_account / (1.0 + lsr_account)) if lsr_account else None,
        taker_buy_sell_ratio=float(last["lsr_taker"]) if last.get("lsr_taker") is not None else None,
        mark_price=mark,
        oi_history=[(int(r["time"]) * 1000, float(r["open_interest"]) * mult) for r in rows],
    )


async def _get(client: httpx.AsyncClient, path: str, **params):
    r = await client.get(f"{BASE}{path}", params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()


async def fetch_gate_futures(client: httpx.AsyncClient) -> FuturesSnapshot:
    try:
        stats, tickers, funding, contract = await asyncio.gather(
            _get(client, "/contract_stats", contract=CONTRACT, interval="1h", limit=25),
            _get(client, "/tickers", contract=CONTRACT),
            _get(client, "/funding_rate", contract=CONTRACT, limit=21),
            _get(client, f"/contracts/{CONTRACT}"),
            return_exceptions=True,
        )
        if isinstance(stats, Exception):
            raise stats
        return parse_gate(stats, [] if isinstance(tickers, Exception) else tickers,
                          [] if isinstance(funding, Exception) else funding,
                          None if isinstance(contract, Exception) else contract)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("gate futures unavailable: %s", e)
        return FuturesSnapshot.unavailable("gate", str(e))
