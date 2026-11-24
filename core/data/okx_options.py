"""OKX public BTC options fallback, normalized to the shared options snapshot schema."""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime

import httpx

from core.data.deribit_options import parse_book_summary
from core.data.types import OptionsSnapshot

log = logging.getLogger(__name__)
BASE = "https://www.okx.com/api/v5"
TIMEOUT = httpx.Timeout(4.0, connect=2.0)
UNDERLYING = "BTC-USD"


async def _get_data(client: httpx.AsyncClient, path: str, params: dict[str, str]) -> list[dict]:
    response = await client.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code") != "0":
        raise ValueError(f"{path} returned an invalid response")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} returned no data")
    return data


def parse_okx_options(
    summaries: list[dict], open_interest: list[dict], index_ticker: dict,
) -> OptionsSnapshot:
    """Build the existing options/gamma schema from OKX's real IV, OI, and index price."""
    try:
        spot = float(index_ticker["idxPx"])
        quote_ms = int(index_ticker["ts"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("OKX index ticker has no valid price or timestamp") from e
    if not math.isfinite(spot) or spot <= 0 or quote_ms <= 0:
        raise ValueError("OKX index ticker has no valid price or timestamp")

    oi_by_instrument = {}
    for row in open_interest:
        try:
            instrument = str(row["instId"])
            # oiCcy is the BTC-denominated amount; `oi` is venue contracts.
            amount = float(row["oiCcy"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(amount) and amount > 0:
            oi_by_instrument[instrument] = amount

    items = []
    for row in summaries:
        instrument = str(row.get("instId") or "")
        oi = oi_by_instrument.get(instrument)
        try:
            prefix, expiry_code, strike, kind = instrument.rsplit("-", 3)
            if prefix != "BTC-USD_UM" or kind not in ("C", "P") or oi is None:
                continue
            expiry = datetime.strptime(expiry_code, "%y%m%d").strftime("%d%b%y").upper()
            volatility = float(row["markVol"])
            strike_value = float(strike)
            if not math.isfinite(volatility) or volatility <= 0 or not math.isfinite(strike_value) or strike_value <= 0:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        items.append({
            "instrument_name": f"BTC-{expiry}-{strike_value:g}-{kind}",
            "open_interest": oi,
            "mark_iv": volatility * 100.0,
            "underlying_price": spot,
            "estimated_delivery_price": spot,
            "volume_usd": 0.0,
        })

    if not items:
        raise ValueError("OKX returned no options with nonzero open interest and mark volatility")

    payload = {
        "result": items,
        # The existing gamma engine expects Deribit's quote time in microseconds.
        "usOut": quote_ms * 1000,
    }
    snapshot = parse_book_summary(payload)
    snapshot.source = "okx"
    return snapshot


async def fetch_okx_options(client: httpx.AsyncClient) -> OptionsSnapshot:
    try:
        summary_data, oi_data, index_data = await asyncio.gather(
            _get_data(client, "public/opt-summary", {"uly": UNDERLYING}),
            _get_data(client, "public/open-interest", {"instType": "OPTION", "uly": UNDERLYING}),
            _get_data(client, "market/index-tickers", {"instId": UNDERLYING}),
        )
        return parse_okx_options(summary_data, oi_data, index_data[0])
    except Exception as e:  # noqa: BLE001 - provider boundary
        detail = str(e).strip()
        reason = f"{type(e).__name__}: {detail}" if detail else type(e).__name__
        log.warning("okx options unavailable: %s", reason)
        return OptionsSnapshot.unavailable("okx", reason)
