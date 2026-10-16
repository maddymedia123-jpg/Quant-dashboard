"""BTC exchange flows from Coin Metrics' community API (keyless): coins moving onto and off exchanges.

Three daily series, free on the community tier:

* FlowInExNtv  - BTC sent to exchange addresses that day
* FlowOutExNtv - BTC withdrawn from exchange addresses that day
* SplyExNtv    - BTC held on exchanges at the day's close (the "exchange reserve")

These are ALL exchange flows, not only whales: Coin Metrics' per-size breakdowns and transfer counts are
paid (403 on the community key). They are daily, and the newest day is published at about 02:00 UTC the
next morning with status "flash" - preliminary, revised later - so the read is a day behind at best and is
labelled with its date.

Coin Metrics' exchange-address attribution is their own; it misses exchanges whose wallets they have not
tagged, so the reserve is a level for trend, not an audited total."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from core.data.types import ExchangeFlowSnapshot

log = logging.getLogger(__name__)

URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = "FlowInExNtv,FlowOutExNtv,SplyExNtv"
HISTORY_DAYS = 420          # a year of 7-day sums to measure "normal" against, plus room
MAX_BYTES = 2_000_000


def _ms(raw: str | None) -> int | None:
    try:
        return int(datetime.fromisoformat(str(raw)[:19]).replace(tzinfo=timezone.utc).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and x >= 0 else None


def parse_flows(payload: dict) -> ExchangeFlowSnapshot:
    """Days with all three series present, oldest first. A day missing one is dropped, not zero-filled:
    a missing inflow read as 0 would show as a record outflow day."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        err = payload.get("error") if isinstance(payload, dict) else None
        return ExchangeFlowSnapshot.unavailable("coinmetrics", f"no data array ({err or type(payload).__name__})")
    rows, statuses = [], set()
    for r in payload["data"]:
        if not isinstance(r, dict):
            continue
        t, i, o, s = _ms(r.get("time")), _num(r.get("FlowInExNtv")), _num(r.get("FlowOutExNtv")), _num(r.get("SplyExNtv"))
        if None in (t, i, o, s):
            continue
        rows.append({"time_ms": t, "inflow": i, "outflow": o, "net": i - o, "reserve": s})
        statuses.add(str(r.get("FlowInExNtv-status") or ""))
    if not rows:
        return ExchangeFlowSnapshot.unavailable("coinmetrics", "no complete days in the response")
    rows.sort(key=lambda x: x["time_ms"])
    dedup = {r["time_ms"]: r for r in rows}
    rows = [dedup[t] for t in sorted(dedup)]
    return ExchangeFlowSnapshot(source="coinmetrics", days=rows, as_of_ms=rows[-1]["time_ms"],
                                preliminary="flash" in statuses)


async def fetch_exchange_flows(client: httpx.AsyncClient, now_ms: int | None = None) -> ExchangeFlowSnapshot:
    now = now_ms or int(time.time() * 1000)
    start = datetime.fromtimestamp(now / 1000 - HISTORY_DAYS * 86_400, tz=timezone.utc).strftime("%Y-%m-%d")
    try:
        r = await client.get(URL, params={"assets": "btc", "metrics": METRICS, "frequency": "1d",
                                          "start_time": start, "page_size": 1000})
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise ValueError(f"response {len(r.content) // 1024} KB exceeds {MAX_BYTES // 1024} KB")
        return parse_flows(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("coinmetrics exchange flows unavailable: %s", e)
        return ExchangeFlowSnapshot.unavailable("coinmetrics", f"{type(e).__name__}: {e}".strip(": "))
