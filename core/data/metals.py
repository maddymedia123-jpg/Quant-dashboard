"""Gold spot from goldprice.dev (keyless free tier): the XAUUSD half of the multi-asset ask.

Two things this provider learned by being called rather than read about, both of which are guarded below:

* `symbols=` (plural) is not the parameter. Sending it returns a *different currency* - XAU quoted in AUD
  - with HTTP 200 and no warning. So the response's own `symbol` and `quote_currency` are checked against
  what was asked for, and a mismatch is treated as no data. A silently wrong gold price is worse than a
  missing one, because a dash cannot be traded on by mistake.
* Silver and copper are plan-gated on the free tier (`{"error":"plan_gated"}`), so only gold is fetched
  and nothing here implies the others are available.

The feed reports `is_stale` and `computed_at` itself, which is the same discipline the rest of this
dashboard runs on, so both are carried through rather than flattened into a price."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from core.data.types import MetalsSnapshot

log = logging.getLogger(__name__)

URL = "https://api.goldprice.dev/v1/prices"
SYMBOL = "XAU-USD-SPOT"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
# Past this the print is old enough that a trader should not be reading it as spot.
STALE_AFTER_MS = 30 * 60_000


def _ms(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int((d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def parse_gold(payload: dict, now_ms: int | None = None) -> MetalsSnapshot:
    """The gold row, or unavailable with the reason. Optional fields stay optional: the same endpoint
    returned bid and ask on one call and omitted them on the next, so neither is required."""
    if not isinstance(payload, dict):
        return MetalsSnapshot.unavailable("goldprice.dev", f"unexpected payload type {type(payload).__name__}")
    if payload.get("error"):
        return MetalsSnapshot.unavailable(
            "goldprice.dev", f"{payload['error']}: {payload.get('message') or payload.get('hint') or ''}".strip())

    rows = payload.get("symbols") or []
    row = next((r for r in rows if isinstance(r, dict) and str(r.get("symbol", "")).upper() == "XAU"), None)
    if row is None:
        return MetalsSnapshot.unavailable("goldprice.dev", "no XAU row in the response")

    quote = str(row.get("quote_currency") or "").upper()
    if quote != "USD":
        # the plural-parameter trap: HTTP 200, wrong currency, no warning
        return MetalsSnapshot.unavailable("goldprice.dev",
                                          f"XAU came back quoted in {quote or 'an unnamed currency'}, not USD")
    try:
        price = float(row["price"])
    except (KeyError, TypeError, ValueError):
        return MetalsSnapshot.unavailable("goldprice.dev", f"XAU price not a number: {row.get('price')!r}")
    if price <= 0:
        return MetalsSnapshot.unavailable("goldprice.dev", f"XAU price is {price}")

    def optional(key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    computed_ms = _ms(row.get("computed_at"))
    stale = bool(row.get("is_stale"))
    if not stale and computed_ms and now_ms and (now_ms - computed_ms) > STALE_AFTER_MS:
        # the feed says fresh, the clock disagrees; the clock wins
        stale = True
    return MetalsSnapshot(source="goldprice.dev", xau_usd=price, bid=optional("bid"), ask=optional("ask"),
                          unit=str(row.get("unit") or "troy_ounce"), is_stale=stale,
                          computed_ms=computed_ms)


async def fetch_gold(client: httpx.AsyncClient, now_ms: int | None = None) -> MetalsSnapshot:
    try:
        r = await client.get(URL, params={"symbol": SYMBOL}, headers=HEADERS)
        r.raise_for_status()
        return parse_gold(r.json(), now_ms)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("gold spot unavailable: %s", e)
        return MetalsSnapshot.unavailable("goldprice.dev", str(e))
