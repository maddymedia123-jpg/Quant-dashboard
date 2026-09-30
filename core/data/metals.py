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
# Gold has not traded under $1,000 an ounce since 2009. A $500 floor therefore keeps enormous headroom
# for any real regime while still catching the two provider bugs that matter: a one-place decimal shift
# (4,148 becomes 414) and a gram price sold as an ounce price (4,148 becomes 133). It does NOT catch a
# shift that lands inside the band, so this is a floor under the damage, not a proof of correctness.
PLAUSIBLE_USD = (500.0, 20_000.0)
# Anything else is a different instrument. The feed sends troy_ounce; a gram or kilogram price
# rendered under an ounce label would be wrong by three orders of magnitude.
EXPECTED_UNIT = "troy_ounce"


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


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

    unit = str(row.get("unit") or "").strip().lower()
    if not unit:
        # never fabricated: the response not saying "troy ounce" is not the same as it saying so, and a
        # gram price under an ounce label is wrong by a factor of 31
        return MetalsSnapshot.unavailable("goldprice.dev", "the XAU row does not say what unit it prices")
    if unit != EXPECTED_UNIT:
        return MetalsSnapshot.unavailable("goldprice.dev",
                                          f"XAU came back priced per {unit}, not per {EXPECTED_UNIT}")
    contract = str(row.get("contract_type") or "spot").strip().lower()
    if contract != "spot":
        return MetalsSnapshot.unavailable("goldprice.dev", f"XAU came back as {contract}, not spot")
    low, high = PLAUSIBLE_USD
    if not low <= price <= high:
        return MetalsSnapshot.unavailable(
            "goldprice.dev", f"XAU at {price:,.2f} is outside the plausible {low:,.0f}-{high:,.0f} band")

    computed_ms = _ms(row.get("computed_at"))
    stale = bool(row.get("is_stale"))
    # `now_ms or _now_ms()`: this guard was dead for its whole life because no caller passed now_ms and
    # `and now_ms` short-circuited every time. Gold does not trade at the weekend, so a Friday close
    # served on a Sunday with is_stale false is the ordinary case, and it rendered as live spot.
    reference = now_ms or _now_ms()
    if not stale and computed_ms and (reference - computed_ms) > STALE_AFTER_MS:
        # the feed says fresh, the clock disagrees; the clock wins
        stale = True
    return MetalsSnapshot(source="goldprice.dev", xau_usd=price, bid=optional("bid"), ask=optional("ask"),
                          unit=unit, is_stale=stale, computed_ms=computed_ms)


async def fetch_gold(client: httpx.AsyncClient, now_ms: int | None = None) -> MetalsSnapshot:
    try:
        r = await client.get(URL, params={"symbol": SYMBOL}, headers=HEADERS)
        r.raise_for_status()
        return parse_gold(r.json(), now_ms)
    except Exception as e:  # noqa: BLE001 - provider boundary
        # the type name matters: str(ReadTimeout()) is empty, which rendered as "No gold feed: ."
        log.error("gold spot unavailable: %s", e)
        return MetalsSnapshot.unavailable("goldprice.dev", f"{type(e).__name__}: {e}".strip(": "))
