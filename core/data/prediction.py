"""Polymarket odds via VoxOdds (keyless): what the crowd is actually pricing, and whether it is tradeable.

Polymarket's own Gamma API is the primary source and is keyless too, but it hands back everything - a
top-volume query returned a Dota 2 match - so it would need filtering and would still say nothing about
whether a quote can be filled. VoxOdds filters by category and adds an execution band per market
(`clean`, `watch`, `fragile`) with the flags behind it, which is the difference between a number and a
number you can act on. A 6% probability on a market flagged "extreme price" with thin depth is not a 6%
forecast, and showing it as one would be the same error as presenting a confluence score as a win rate.

The feed asks for attribution ("Data from Polymarket. Powered by VoxOdds."); it is carried through and
rendered, not dropped."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from core.data.types import PredictionSnapshot

log = logging.getLogger(__name__)

URL = "https://voxodds.com/api/v1/markets"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
CATEGORY = "Crypto"
# 60 requests a minute per IP, and this sits behind the 120-second context cache, so one call per refresh
# is well inside it. Kept small so the panel stays readable rather than because of the limit.
MAX_MARKETS = 40
BANDS = ("clean", "watch", "fragile")
ASSETS = (("BTC", ("bitcoin", "btc")), ("ETH", ("ethereum", "ether")), ("SOL", ("solana",)))


def _ms(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return int((d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None      # NaN is not a probability


def parse_markets(payload: dict) -> PredictionSnapshot:
    if not isinstance(payload, dict):
        return PredictionSnapshot.unavailable("voxodds", f"unexpected payload {type(payload).__name__}")
    rows = payload.get("markets")
    if not isinstance(rows, list):
        return PredictionSnapshot.unavailable("voxodds", "no markets array in the response")

    markets: list[dict] = []
    for row in rows[:MAX_MARKETS]:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question") or "").strip()
        prices = row.get("prices") or []
        yes = _float(prices[0]) if isinstance(prices, list) and prices else None
        if not question or yes is None or not 0.0 <= yes <= 1.0:
            continue                        # a market without a usable probability is not a market
        execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
        band = str(execution.get("band") or "").lower()
        low = question.lower()
        markets.append({
            "id": str(row.get("id") or ""), "question": question,
            "probability": yes,
            "outcome": (row.get("outcomes") or ["Yes"])[0] if isinstance(row.get("outcomes"), list) else "Yes",
            "volume_24h": _float(row.get("volume_24h")), "liquidity": _float(row.get("liquidity")),
            "end_ms": _ms(row.get("end_date")), "days_left": _float(row.get("days_left")),
            "uncertainty": _float(row.get("uncertainty")),
            # unknown bands are reported as unknown rather than quietly promoted to tradeable
            "band": band if band in BANDS else "unknown",
            "band_label": str(execution.get("label") or "").strip(),
            "band_score": _float(execution.get("score")),
            "flags": [str(f) for f in (execution.get("flags") or []) if f],
            "band_reason": str(execution.get("reason") or "").strip(),
            "url": str(row.get("polymarket_url") or row.get("voxodds_url") or ""),
            "assets": [code for code, words in ASSETS if any(w in low for w in words)],
        })
    if not markets:
        return PredictionSnapshot.unavailable("voxodds", "no usable crypto markets in the response")

    markets.sort(key=lambda m: -(m["volume_24h"] or 0.0))
    return PredictionSnapshot(source="voxodds", markets=markets,
                              attribution=str(payload.get("attribution") or "").strip(),
                              retrieved_ms=_ms(payload.get("retrieved_at")))


async def fetch_predictions(client: httpx.AsyncClient) -> PredictionSnapshot:
    try:
        r = await client.get(URL, params={"category": CATEGORY}, headers=HEADERS)
        r.raise_for_status()
        return parse_markets(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        # str(ReadTimeout()) is empty, which rendered as "No odds feed: ."
        log.error("prediction markets unavailable: %s", e)
        return PredictionSnapshot.unavailable("voxodds", f"{type(e).__name__}: {e}".strip(": "))
