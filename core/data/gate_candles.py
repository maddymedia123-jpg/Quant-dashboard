"""Price candles from the Gate.io BTC_USDT perpetual - the futures market the dashboard reads first.

His framework reads futures price action, not spot. Binance and Bybit futures would be the obvious
venues, but both refuse US addresses and the dashboard is hosted on US servers; Gate.io already serves
this dashboard's funding, open interest and taker flow (CVD) from there, so price now comes from the
same contract the order flow is measured on. Kraken and Gemini spot stay behind it as fallbacks, and the
snapshot's source says which one answered.

Found by calling it:

* Up to 2,000 candles per request on 15m, 1h, 4h and 1d: three weeks of 15m, twelve weeks of 1h, eleven
  months of 4h, five and a half years of 1d. Daily history goes back to November 2019, so a second
  request fetches the years before the newest 2,000 days.
* Gate.io's own weekly candles ("7d") open on THURSDAY, not Monday, so weeks are built here from the
  daily candles on Monday 00:00 UTC, the same weeks every other desk and the 2W candles use.
* Volume `v` is in CONTRACTS; a BTC_USDT contract is 0.0001 BTC, so `v * 0.0001` is the BTC traded
  (checked against `sum`, the quote volume, divided by price).
* The newest candle is the one still in progress, exactly as on Kraken.
* The perp trades a few dollars off spot (funding basis). Levels from spot venues and from Deribit's
  index sit that far apart."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from core.data.gemini_spot import resample
from core.data.types import SpotSnapshot

log = logging.getLogger(__name__)

BASE = "https://api.gateio.ws/api/v4/futures/usdt"
CONTRACT = "BTC_USDT"
SOURCE = "gate.io BTC_USDT perp"
HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
FETCHED = ("15m", "1h", "4h", "1d")
LIMIT = 2000
CONTRACT_BTC = 0.0001        # BTC per contract, as Gate.io publishes it for BTC_USDT
HISTORY_FROM_S = 1_546_300_800   # 2019-01-01: older daily candles, before the newest LIMIT days
DAY_S = 86_400
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_candles(rows) -> pd.DataFrame:
    """Gate.io candle rows -> the dashboard's frame: ms timestamps, BTC volume, oldest first.

    A row with a missing or non-numeric price is dropped rather than read as zero."""
    if not isinstance(rows, list):
        raise ValueError(f"unexpected payload {type(rows).__name__}: {str(rows)[:120]}")
    out = []
    for r in rows:
        try:
            out.append([int(r["t"]) * 1000, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
                        float(r.get("v") or 0) * CONTRACT_BTC])
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise ValueError("no candles")
    df = pd.DataFrame(out, columns=COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


async def _get(client: httpx.AsyncClient, path: str, **params):
    r = await client.get(f"{BASE}/{path}", params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()


async def fetch_gate_candles(client: httpx.AsyncClient) -> SpotSnapshot:
    try:
        calls = [_get(client, "candlesticks", contract=CONTRACT, interval=tf, limit=LIMIT) for tf in FETCHED]
        calls.append(_get(client, "tickers", contract=CONTRACT))
        res = await asyncio.gather(*calls, return_exceptions=True)
        frames: dict[str, pd.DataFrame] = {}
        for tf, data in zip(FETCHED, res[:-1]):
            if isinstance(data, Exception):
                log.warning("gate candles %s failed: %s", tf, data)
                continue
            frames[tf] = parse_candles(data)
        if "1h" not in frames:
            raise RuntimeError("no 1h candles: " + str(res[1])[:120])
        if "1d" in frames:
            daily = frames["1d"]
            first_s = int(daily["timestamp"].iloc[0]) // 1000
            if first_s > HISTORY_FROM_S + DAY_S:
                try:
                    older = parse_candles(await _get(client, "candlesticks", contract=CONTRACT, interval="1d",
                                                     **{"from": HISTORY_FROM_S, "to": first_s - DAY_S}))
                    daily = (pd.concat([older, daily]).drop_duplicates("timestamp", keep="last")
                             .sort_values("timestamp").reset_index(drop=True))
                except Exception as e:  # noqa: BLE001 - older history is a bonus, not a requirement
                    log.warning("gate older daily candles failed: %s", e)
            frames["1d"] = daily
            frames["1w"] = resample(daily, 10080)
        tick = res[-1]
        last = None
        if not isinstance(tick, Exception) and isinstance(tick, list) and tick:
            try:
                last = float(tick[0]["last"])
            except (KeyError, TypeError, ValueError):
                last = None
        if last is None:
            last = float(frames["1h"]["close"].iloc[-1])
        return SpotSnapshot(source=SOURCE, symbol="BTC_USDT perp", last=last, frames=frames)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("gate candles unavailable: %s", e)
        return SpotSnapshot.unavailable(SOURCE, str(e))


def fetch_15m_since(start_ms: int, now_ms: int, client: httpx.Client | None = None) -> pd.DataFrame | None:
    """15-minute perp candles from `start_ms` to now, paged (the regular feed keeps only 2,000 - three
    weeks). For the experimental sine record, which must reach back to its go-live date. Gate serves the
    newest 10,000 points, about 104 days, so the record can grow for three months before it is capped.
    None on any failure: the caller falls back to the regular feed and says so."""
    step = 900
    own = client is None
    c = client or httpx.Client(timeout=20, headers={"User-Agent": "trap-intel-dashboard/1.0"})
    try:
        frames, to = [], now_ms // 1000 // step * step
        first = max(start_ms // 1000, to - 9_990 * step)
        while to > first:
            frm = max(first, to - 1_999 * step)
            r = c.get(f"{BASE}/candlesticks", params={"contract": CONTRACT, "interval": "15m", "from": frm, "to": to})
            r.raise_for_status()
            frames.append(parse_candles(r.json()))
            to = frm - step
        if not frames:
            return None
        df = pd.concat(frames)
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("Gate.io 15m history unavailable: %s", e)
        return None
    finally:
        if own:
            c.close()
