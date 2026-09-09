"""CoinLobster public feeds: 24h perp liquidations, $100K+ whale trades, unusual-flow radar."""
from __future__ import annotations

import asyncio
import logging

import httpx

from core.data.types import LiquidationsSnapshot, WhalesSnapshot

log = logging.getLogger(__name__)
BASE = "https://coinlobster.com/api/public"


def parse_liquidations(d: dict) -> LiquidationsSnapshot:
    p = d["perp_liquidations"]
    btc = next((c for c in p.get("top_coins", []) if c.get("base") == "BTC"), None)
    return LiquidationsSnapshot(
        source="coinlobster", window=p.get("window", "24h"),
        total_usd=float(p["total_usd"]), long_usd=float(p["long_usd"]), short_usd=float(p["short_usd"]),
        long_count=int(p.get("long_count") or 0), short_count=int(p.get("short_count") or 0),
        last_hour_usd=float(p["last_hour_usd"]) if p.get("last_hour_usd") is not None else None,
        btc_usd=float(btc["usd"]) if btc else None,
        btc_long_usd=float(btc["longUsd"]) if btc else None,
        btc_short_usd=float(btc["shortUsd"]) if btc else None,
        hourly=[{"t": int(h["t"]), "long_usd": float(h["longUsd"]), "short_usd": float(h["shortUsd"]), "count": int(h["count"])}
                for h in p.get("hourly", [])],
        biggest=p.get("biggest"), venues=list(p.get("venues", [])),
    )


def parse_radar(d: dict) -> list[dict]:
    return [{"coin": r["coin"], "direction": r.get("direction"), "unusual": bool(r.get("unusual")), "count": r.get("count")}
            for r in d.get("rows", [])]


def parse_whales(d: dict, radar: dict | None = None) -> WhalesSnapshot:
    ws = d.get("whales", [])
    btc = [w for w in ws if str(w.get("pair", "")).upper().startswith("BTC/")]
    ts = [int(w["timestamp"]) for w in ws]
    mc = (btc[0].get("marketConditions") if btc else (ws[0].get("marketConditions") if ws else None)) or {}
    oi = {k: float(v["value"]) for k, v in (mc.get("openInterest") or {}).items() if isinstance(v, dict) and v.get("value") is not None}
    radar_rows = parse_radar(radar) if radar else []
    return WhalesSnapshot(
        source="coinlobster",
        window_minutes=((max(ts) - min(ts)) / 60_000) if len(ts) > 1 else None,
        btc_trades=len(btc),
        btc_buy_usd=sum(float(w["quantity_quote"]) for w in btc if w.get("isBuy")),
        btc_sell_usd=sum(float(w["quantity_quote"]) for w in btc if not w.get("isBuy")),
        btc_largest=max(btc, key=lambda w: float(w["quantity_quote"]), default=None),
        funding_by_exchange={k: float(v) for k, v in (mc.get("fundingRates") or {}).items()},
        oi_by_exchange_usd=oi,
        radar=radar_rows,
        btc_radar=next((r for r in radar_rows if r["coin"] == "BTC"), None),
    )


async def _get(client: httpx.AsyncClient, path: str):
    r = await client.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()


async def fetch_coinlobster(client: httpx.AsyncClient) -> tuple[LiquidationsSnapshot, WhalesSnapshot]:
    liq_p, wh_p, rad_p = await asyncio.gather(_get(client, "/liquidations"), _get(client, "/crypto-whales"),
                                              _get(client, "/whale-radar?window=4h"), return_exceptions=True)
    try:
        liqs = parse_liquidations(liq_p) if not isinstance(liq_p, Exception) else LiquidationsSnapshot.unavailable("coinlobster", str(liq_p))
    except Exception as e:  # noqa: BLE001
        liqs = LiquidationsSnapshot.unavailable("coinlobster", str(e))
    try:
        if isinstance(wh_p, Exception):
            whales = WhalesSnapshot.unavailable("coinlobster", str(wh_p))
        else:
            whales = parse_whales(wh_p, None if isinstance(rad_p, Exception) else rad_p)
    except Exception as e:  # noqa: BLE001
        whales = WhalesSnapshot.unavailable("coinlobster", str(e))
    return liqs, whales
