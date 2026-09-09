# Phase 1 — Data, Indicators, Charts, UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file, partly-fabricated Streamlit dashboard with a package that pulls real keyless market data, computes deterministic indicators per timeframe category, renders TradingView-style charts, and presents it in an overhauled UI. No LLM in this phase.

**Architecture:** `core/data` providers return typed pydantic snapshots with an `available` flag; `core/indicators` are pure functions over pandas DataFrames producing a `CategoryAnalysis`; `ui/charts.py` serialises an analysis into a self-contained lightweight-charts HTML document embedded with `components.html`; `app.py` is a thin shell. Everything network-free is unit-tested with pytest; providers are tested against recorded JSON fixtures.

**Tech Stack:** Python ≥3.12, Streamlit ≥1.40, pandas ≥2.2, numpy, pydantic 2, httpx (async), pytest, TradingView lightweight-charts 5.2.1 (standalone build from jsDelivr).

**Spec:** `docs/superpowers/specs/2026-09-09-trap-intelligence-design.md`

## Global Constraints

- Branch: `feature/trap-intelligence-phase1` (already created from `origin/main`). Never commit to `main`.
- No fabricated numbers anywhere. Any missing value renders as `—` and carries `available=False` upstream.
- No network calls in tests. Providers expose pure `parse_*` functions tested on fixtures in `tests/fixtures/`.
- No emoji as icons in headings. Neutral, institutional copy.
- Chart library pinned: `https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js` (verified 200, 197 922 B).
- Symbols: Kraken pair `XBTUSD`, Binance/Bybit `BTCUSDT`, Deribit currency `BTC`.
- Category map (name → chart TF / context TF): Live `15m/1h`, Intraday `1h/4h`, Weekly `4h/1d`, Monthly `1d/1w`.
- Timestamps are UTC. DataFrames use column `timestamp` in **milliseconds** (int64); chart payloads use **seconds**.
- Commit after every task with the attribution trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0186foiQoKUv1nKDTFvRPiJ5
  ```
- Run tests with: `.venv/Scripts/python -m pytest -q` from the repo root.

---

## File map

| Path | Responsibility |
|---|---|
| `requirements.txt` | runtime deps (drop `ccxt`, `plotly`; add `httpx`) |
| `requirements-dev.txt` | `pytest` |
| `pytest.ini` | `pythonpath = .`, quiet |
| `scripts/record_fixtures.py` | one-off: download live JSON into `tests/fixtures/` |
| `core/__init__.py`, `core/data/__init__.py`, `core/indicators/__init__.py`, `ui/__init__.py` | packages |
| `core/config.py` | category map, cache TTLs, symbols, macro events |
| `core/data/types.py` | `Availability`, `SpotSnapshot`, `FuturesSnapshot`, `OptionsSnapshot`, `SentimentSnapshot`, `MarketSnapshot` |
| `core/data/kraken_spot.py` | `parse_ohlc`, `parse_ticker`, `fetch_spot` |
| `core/data/binance_futures.py` | `parse_binance`, `parse_bybit`, `fetch_futures` |
| `core/data/deribit_options.py` | `parse_instrument`, `compute_max_pain`, `parse_book_summary`, `fetch_options` |
| `core/data/sentiment.py` | `parse_fng`, `fetch_sentiment` |
| `core/data/market.py` | `fetch_all()` concurrent + `load_market()` cached wrapper |
| `core/indicators/ema.py` | `ema`, `ema_stack` |
| `core/indicators/pivots.py` | `find_pivots` (shared) |
| `core/indicators/rsi.py` | `rsi`, `detect_divergences` |
| `core/indicators/levels.py` | `support_resistance` |
| `core/indicators/trendlines.py` | `fit_trendlines` |
| `core/indicators/fib_time.py` | `fib_time_zones` |
| `core/indicators/volatility.py` | `atr`, `volatility_profile` |
| `core/indicators/squeeze.py` | `squeeze_risk` |
| `core/indicators/direction.py` | `direction_call` |
| `core/indicators/trade_map.py` | `map_trade` |
| `core/indicators/category.py` | `analyze_category` → `CategoryAnalysis` |
| `ui/theme.py` | CSS, fonts, `kpi_strip`, `card`, `chip`, `unavailable_chip` |
| `ui/charts.py` | `build_chart_html`, `render_chart` |
| `ui/panels.py` | direction / divergence / volatility / agent-placeholder panels, active-trade result |
| `.streamlit/config.toml` | theme + server flags |
| `app.py` | shell |
| `tests/...` | one test module per source module |

---

### Task 0: Scaffolding, dependencies, fixtures

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`, `pytest.ini`, `core/__init__.py`, `core/data/__init__.py`, `core/indicators/__init__.py`, `ui/__init__.py`, `tests/__init__.py`, `scripts/record_fixtures.py`, `core/config.py`
- Create (generated): `tests/fixtures/*.json`

**Interfaces:**
- Produces: `core.config.CATEGORIES: dict[str, Category]`, `core.config.TTL`, `core.config.MACRO_EVENTS`.

- [ ] **Step 1: Write dependency files**

`requirements.txt`:
```
streamlit>=1.40
pandas>=2.2
numpy>=1.26
httpx>=0.27
pydantic>=2.7
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8
```

`pytest.ini`:
```
[pytest]
pythonpath = .
addopts = -q
testpaths = tests
```

- [ ] **Step 2: Install and create packages**

```bash
.venv/Scripts/python -m pip install -q -r requirements-dev.txt
mkdir -p core/data core/indicators ui tests/fixtures scripts
touch core/__init__.py core/data/__init__.py core/indicators/__init__.py ui/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `core/config.py`**

```python
"""Static configuration: categories, cache TTLs, symbols, curated macro events."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str            # machine key
    label: str          # UI label
    chart_tf: str       # timeframe drawn on the chart
    context_tf: str     # higher timeframe used for context
    horizon_bars: int   # bars of chart_tf used for expected-range horizon
    horizon_label: str  # human label for that horizon


CATEGORIES: dict[str, Category] = {
    "live": Category("live", "Live Recon", "15m", "1h", 4, "next hour"),
    "intraday": Category("intraday", "Intraday", "1h", "4h", 24, "next 24h"),
    "weekly": Category("weekly", "Weekly", "4h", "1d", 42, "next 7 days"),
    "monthly": Category("monthly", "Monthly", "1d", "1w", 30, "next 30 days"),
}

TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d", "1w")

TF_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}

# Cache TTLs (seconds)
TTL = {"spot": 25, "futures": 60, "options": 120, "sentiment": 1800}

SYMBOLS = {"kraken_pair": "XBTUSD", "binance": "BTCUSDT", "bybit": "BTCUSDT", "deribit": "BTC"}

CHART_BARS = 300  # bars drawn on each chart

LIGHTWEIGHT_CHARTS_URL = (
    "https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js"
)

# Manually curated until a news source is chosen (spec §11). Shown with a "curated" label.
MACRO_EVENTS = [
    {
        "title": "FOMC Rate Decision & Statement",
        "date_utc": "2026-09-17",
        "impact": "HIGH",
        "note": "No new entries into the release; flatten or hedge leveraged positions before 18:00 UTC.",
    }
]
```

- [ ] **Step 4: Write `scripts/record_fixtures.py`**

```python
"""Record live JSON responses into tests/fixtures so tests never touch the network.
Run once: .venv/Scripts/python scripts/record_fixtures.py
"""
from __future__ import annotations

import json
import pathlib

import httpx

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINTS = {
    "kraken_ohlc_15m.json": "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=15",
    "kraken_ohlc_1h.json": "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60",
    "kraken_ticker.json": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "binance_funding.json": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=21",
    "binance_oi.json": "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
    "binance_oi_hist.json": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=25",
    "binance_ls.json": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=1",
    "binance_taker.json": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=1h&limit=1",
    "bybit_tickers.json": "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
    "bybit_funding.json": "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=21",
    "bybit_oi.json": "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h&limit=25",
    "bybit_ratio.json": "https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=BTCUSDT&period=1h&limit=1",
    "deribit_book_summary.json": "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
    "fng.json": "https://api.alternative.me/fng/?limit=2",
}

with httpx.Client(timeout=20, headers={"User-Agent": "trap-intel-fixtures/1.0"}) as client:
    for name, url in ENDPOINTS.items():
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
        # trim very large payloads so the repo stays small
        if name.startswith("kraken_ohlc"):
            key = next(k for k in data["result"] if k != "last")
            data["result"][key] = data["result"][key][-400:]
        (OUT / name).write_text(json.dumps(data), encoding="utf-8")
        print(f"wrote {name} ({(OUT / name).stat().st_size} B)")
```

- [ ] **Step 5: Record fixtures and verify**

Run: `.venv/Scripts/python scripts/record_fixtures.py`
Expected: 14 lines `wrote ... B`. If Binance returns 451 from your network, delete the four `binance_*.json` lines from the fixture set for now and note it; the Bybit fixtures must succeed.

Run: `ls tests/fixtures | wc -l` → `14`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini core ui tests scripts
git commit -m "chore: scaffold core/ui packages, deps, config, recorded fixtures"
```

---

### Task 1: Snapshot types

**Files:**
- Create: `core/data/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Produces:
  - `Availability(available: bool=True, source: str="", fetched_at: datetime|None, error: str|None)`
  - `SpotSnapshot(Availability)`: `symbol: str`, `last: float|None`, `frames: dict[str, pd.DataFrame]`
  - `FuturesSnapshot(Availability)`: `funding_rate`, `funding_7d_mean`, `open_interest`, `open_interest_usd`, `oi_change_24h_pct`, `long_short_ratio`, `long_account_pct`, `taker_buy_sell_ratio`, `mark_price` (all `float|None`), `oi_history: list[tuple[int, float]]`
  - `OptionsSnapshot(Availability)`: `underlying_price`, `max_pain`, `max_pain_expiry: str|None`, `put_call_oi_ratio`, `put_call_volume_ratio`, `iv_atm`, `iv_skew`, `total_call_oi`, `total_put_oi`, `expiries: list[dict]`
  - `SentimentSnapshot(Availability)`: `fear_greed`, `classification: str|None`, `fear_greed_prev`
  - `MarketSnapshot(spot, futures, options, sentiment, generated_at)` with `.unavailable() -> list[str]`
  - `unavailable(cls, source, error) -> instance` classmethod on every snapshot

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from datetime import datetime, timezone

import pandas as pd

from core.data.types import (
    FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot,
)


def test_unavailable_factory_sets_flags():
    f = FuturesSnapshot.unavailable("binance", "451 blocked")
    assert f.available is False
    assert f.source == "binance"
    assert f.error == "451 blocked"
    assert f.funding_rate is None


def test_spot_holds_dataframes():
    df = pd.DataFrame({"timestamp": [1, 2], "open": [1.0, 2.0], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    s = SpotSnapshot(source="kraken", last=2.0, frames={"1h": df}, fetched_at=datetime.now(timezone.utc))
    assert s.frames["1h"].shape == (2, 6)


def test_market_lists_unavailable_sources():
    m = MarketSnapshot(
        spot=SpotSnapshot(source="kraken"),
        futures=FuturesSnapshot.unavailable("binance", "x"),
        options=OptionsSnapshot(source="deribit"),
        sentiment=SentimentSnapshot.unavailable("alternative.me", "timeout"),
    )
    assert m.unavailable() == ["futures", "sentiment"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.data.types'`

- [ ] **Step 3: Write `core/data/types.py`**

```python
"""Typed market snapshots. Every snapshot carries an availability flag so no
consumer can mistake a missing feed for a real value."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Availability(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    available: bool = True
    source: str = ""
    fetched_at: Optional[datetime] = Field(default_factory=_now)
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, source: str, error: str):
        return cls(available=False, source=source, error=error)


class SpotSnapshot(Availability):
    symbol: str = "BTC/USD"
    last: Optional[float] = None
    frames: dict[str, pd.DataFrame] = Field(default_factory=dict)


class FuturesSnapshot(Availability):
    funding_rate: Optional[float] = None          # latest 8h rate, e.g. 0.0001 = 0.01%
    funding_7d_mean: Optional[float] = None
    open_interest: Optional[float] = None         # contracts (BTC)
    open_interest_usd: Optional[float] = None
    oi_change_24h_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None      # accounts long / short
    long_account_pct: Optional[float] = None      # 0..1
    taker_buy_sell_ratio: Optional[float] = None
    mark_price: Optional[float] = None
    oi_history: list[tuple[int, float]] = Field(default_factory=list)  # (ms, contracts)


class OptionsSnapshot(Availability):
    underlying_price: Optional[float] = None
    max_pain: Optional[float] = None
    max_pain_expiry: Optional[str] = None
    put_call_oi_ratio: Optional[float] = None
    put_call_volume_ratio: Optional[float] = None
    iv_atm: Optional[float] = None                # percent
    iv_skew: Optional[float] = None               # put IV − call IV (percent points), OTM 5–15%
    total_call_oi: Optional[float] = None
    total_put_oi: Optional[float] = None
    expiries: list[dict[str, Any]] = Field(default_factory=list)


class SentimentSnapshot(Availability):
    fear_greed: Optional[int] = None
    classification: Optional[str] = None
    fear_greed_prev: Optional[int] = None


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spot: SpotSnapshot
    futures: FuturesSnapshot
    options: OptionsSnapshot
    sentiment: SentimentSnapshot
    generated_at: datetime = Field(default_factory=_now)

    def unavailable(self) -> list[str]:
        out = []
        for name in ("spot", "futures", "options", "sentiment"):
            if not getattr(self, name).available:
                out.append(name)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_types.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/data/types.py tests/test_types.py
git commit -m "feat(data): typed snapshots with availability flags"
```

---

### Task 2: Kraken spot provider

**Files:**
- Create: `core/data/kraken_spot.py`
- Test: `tests/test_kraken_spot.py`

**Interfaces:**
- Consumes: `SpotSnapshot`, `core.config.TIMEFRAMES`, `TF_MINUTES`, `SYMBOLS`.
- Produces:
  - `parse_ohlc(payload: dict) -> pd.DataFrame` columns `timestamp(int64 ms), open, high, low, close, volume` (float64), ascending.
  - `parse_ticker(payload: dict) -> float`
  - `async fetch_spot(client: httpx.AsyncClient) -> SpotSnapshot`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kraken_spot.py
import json
import pathlib

import numpy as np

from core.data.kraken_spot import parse_ohlc, parse_ticker

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_ohlc_shape_and_types():
    df = parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text()))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == np.int64
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] > 1_600_000_000_000  # milliseconds, not seconds
    assert (df["high"] >= df["low"]).all()
    assert len(df) >= 200


def test_parse_ticker_last_price():
    last = parse_ticker(json.loads((FX / "kraken_ticker.json").read_text()))
    assert 1_000 < last < 1_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_kraken_spot.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `core/data/kraken_spot.py`**

```python
"""Kraken public REST: OHLC for all category timeframes + last price."""
from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from core.config import SYMBOLS, TF_MINUTES, TIMEFRAMES
from core.data.types import SpotSnapshot

log = logging.getLogger(__name__)
BASE = "https://api.kraken.com/0/public"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _result_key(payload: dict) -> str:
    return next(k for k in payload["result"] if k != "last")


def parse_ohlc(payload: dict) -> pd.DataFrame:
    if payload.get("error"):
        raise ValueError(f"kraken error: {payload['error']}")
    rows = payload["result"][_result_key(payload)]
    # kraken row: [time(s), open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        [[int(r[0]) * 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows],
        columns=COLS,
    )
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").reset_index(drop=True)


def parse_ticker(payload: dict) -> float:
    if payload.get("error"):
        raise ValueError(f"kraken error: {payload['error']}")
    return float(payload["result"][_result_key(payload)]["c"][0])


async def _get(client: httpx.AsyncClient, path: str, **params) -> dict:
    r = await client.get(f"{BASE}/{path}", params=params)
    r.raise_for_status()
    return r.json()


async def fetch_spot(client: httpx.AsyncClient) -> SpotSnapshot:
    pair = SYMBOLS["kraken_pair"]
    try:
        tasks = [_get(client, "OHLC", pair=pair, interval=TF_MINUTES[tf]) for tf in TIMEFRAMES]
        tasks.append(_get(client, "Ticker", pair=pair))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        frames: dict[str, pd.DataFrame] = {}
        for tf, res in zip(TIMEFRAMES, results[:-1]):
            if isinstance(res, Exception):
                log.warning("kraken %s failed: %s", tf, res)
                continue
            frames[tf] = parse_ohlc(res)
        if not frames:
            raise RuntimeError("no OHLC frames returned")
        ticker = results[-1]
        last = parse_ticker(ticker) if not isinstance(ticker, Exception) else float(frames[next(iter(frames))]["close"].iloc[-1])
        return SpotSnapshot(source="kraken", last=last, frames=frames)
    except Exception as e:  # noqa: BLE001 — provider boundary
        log.error("kraken spot unavailable: %s", e)
        return SpotSnapshot.unavailable("kraken", str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_kraken_spot.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add core/data/kraken_spot.py tests/test_kraken_spot.py
git commit -m "feat(data): kraken spot provider with fixture-tested parser"
```

---

### Task 3: Futures provider (Binance → Bybit fallback)

**Files:**
- Create: `core/data/binance_futures.py`
- Test: `tests/test_binance_futures.py`

**Interfaces:**
- Produces:
  - `parse_binance(funding: list, oi: dict, oi_hist: list, ls: list, taker: list) -> FuturesSnapshot`
  - `parse_bybit(tickers: dict, funding: dict, oi: dict, ratio: dict) -> FuturesSnapshot`
  - `async fetch_futures(client) -> FuturesSnapshot`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_binance_futures.py
import json
import pathlib

import pytest

from core.data.binance_futures import parse_binance, parse_bybit

FX = pathlib.Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FX / name).read_text())


@pytest.mark.skipif(not (FX / "binance_funding.json").exists(), reason="binance fixtures not recorded")
def test_parse_binance_fields():
    s = parse_binance(_load("binance_funding.json"), _load("binance_oi.json"), _load("binance_oi_hist.json"),
                      _load("binance_ls.json"), _load("binance_taker.json"))
    assert s.available and s.source == "binance"
    assert -0.01 < s.funding_rate < 0.01
    assert s.open_interest > 1_000
    assert s.open_interest_usd > 1e8
    assert s.oi_change_24h_pct is not None
    assert 0.2 < s.long_short_ratio < 5
    assert 0 < s.long_account_pct < 1
    assert 0.2 < s.taker_buy_sell_ratio < 5
    assert len(s.oi_history) >= 20


def test_parse_bybit_fields():
    s = parse_bybit(_load("bybit_tickers.json"), _load("bybit_funding.json"), _load("bybit_oi.json"), _load("bybit_ratio.json"))
    assert s.available and s.source == "bybit"
    assert -0.01 < s.funding_rate < 0.01
    assert s.open_interest > 1_000
    assert s.oi_change_24h_pct is not None
    assert 0.2 < s.long_short_ratio < 5
    assert s.taker_buy_sell_ratio is None  # bybit has no public taker ratio
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_binance_futures.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `core/data/binance_futures.py`**

```python
"""USD-M perpetual futures context: funding, open interest, long/short, taker flow.
Binance first (richest public data); Bybit fallback (Binance geo-blocks US egress)."""
from __future__ import annotations

import asyncio
import logging
import statistics

import httpx

from core.config import SYMBOLS
from core.data.types import FuturesSnapshot

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
    errors = []
    for name, fn in (("binance", _binance), ("bybit", _bybit)):
        try:
            return await fn(client)
        except Exception as e:  # noqa: BLE001 — provider boundary
            log.warning("futures via %s failed: %s", name, e)
            errors.append(f"{name}: {e}")
    return FuturesSnapshot.unavailable("binance,bybit", " | ".join(errors))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_binance_futures.py -v`
Expected: 2 passed (or 1 passed + 1 skipped if Binance fixtures were not recordable)

- [ ] **Step 5: Commit**

```bash
git add core/data/binance_futures.py tests/test_binance_futures.py
git commit -m "feat(data): futures provider, binance with bybit fallback"
```

---

### Task 4: Deribit options provider (max pain, put/call, IV skew)

**Files:**
- Create: `core/data/deribit_options.py`
- Test: `tests/test_deribit_options.py`

**Interfaces:**
- Produces:
  - `parse_instrument(name: str) -> tuple[str expiry_iso, float strike, str kind('C'|'P')]`
  - `compute_max_pain(rows: list[dict]) -> float` where each row has `strike, kind, oi`
  - `parse_book_summary(payload: dict) -> OptionsSnapshot`
  - `async fetch_options(client) -> OptionsSnapshot`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deribit_options.py
import json
import pathlib

from core.data.deribit_options import compute_max_pain, parse_book_summary, parse_instrument

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_instrument():
    assert parse_instrument("BTC-25SEP26-105000-C") == ("2026-09-25", 105000.0, "C")
    assert parse_instrument("BTC-9SEP26-80000-P") == ("2026-09-09", 80000.0, "P")


def test_max_pain_simple_book():
    # Calls OI at 100 and 110, puts OI at 90 and 100. Pain is minimised at 100.
    rows = [
        {"strike": 100, "kind": "C", "oi": 10}, {"strike": 110, "kind": "C", "oi": 10},
        {"strike": 90, "kind": "P", "oi": 10}, {"strike": 100, "kind": "P", "oi": 10},
    ]
    assert compute_max_pain(rows) == 100


def test_parse_book_summary_fixture():
    s = parse_book_summary(json.loads((FX / "deribit_book_summary.json").read_text()))
    assert s.available and s.source == "deribit"
    assert 1_000 < s.underlying_price < 1_000_000
    assert s.max_pain is not None and s.max_pain_expiry
    assert 0.1 < s.put_call_oi_ratio < 5
    assert s.iv_atm is not None and 10 < s.iv_atm < 300
    assert s.iv_skew is not None
    assert len(s.expiries) >= 3
    assert all({"expiry", "max_pain", "call_oi", "put_oi"} <= set(e) for e in s.expiries)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_deribit_options.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `core/data/deribit_options.py`**

```python
"""Deribit public options book → max pain, put/call ratios, ATM IV, IV skew."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import httpx

from core.config import SYMBOLS
from core.data.types import OptionsSnapshot

log = logging.getLogger(__name__)
BASE = "https://www.deribit.com/api/v2/public"


def parse_instrument(name: str) -> tuple[str, float, str]:
    # BTC-25SEP26-105000-C
    _, exp, strike, kind = name.split("-")
    expiry = datetime.strptime(exp, "%d%b%y").date().isoformat()
    return expiry, float(strike), kind


def compute_max_pain(rows: list[dict]) -> float:
    strikes = sorted({r["strike"] for r in rows})
    best, best_pain = strikes[0], float("inf")
    for k in strikes:
        pain = 0.0
        for r in rows:
            if r["kind"] == "C":
                pain += r["oi"] * max(0.0, k - r["strike"])
            else:
                pain += r["oi"] * max(0.0, r["strike"] - k)
        if pain < best_pain:
            best, best_pain = k, pain
    return best


def parse_book_summary(payload: dict) -> OptionsSnapshot:
    items = payload["result"]
    by_exp: dict[str, list[dict]] = defaultdict(list)
    underlying = None
    call_oi = put_oi = call_vol = put_vol = 0.0
    for it in items:
        expiry, strike, kind = parse_instrument(it["instrument_name"])
        oi = float(it.get("open_interest") or 0.0)
        row = {"strike": strike, "kind": kind, "oi": oi, "iv": it.get("mark_iv"), "vol_usd": float(it.get("volume_usd") or 0.0)}
        by_exp[expiry].append(row)
        if underlying is None and it.get("underlying_price"):
            underlying = float(it["underlying_price"])
        if kind == "C":
            call_oi += oi; call_vol += row["vol_usd"]
        else:
            put_oi += oi; put_vol += row["vol_usd"]

    expiries = []
    for expiry, rows in sorted(by_exp.items()):
        c = sum(r["oi"] for r in rows if r["kind"] == "C")
        p = sum(r["oi"] for r in rows if r["kind"] == "P")
        if c + p < 50:  # ignore illiquid expiries
            continue
        expiries.append({"expiry": expiry, "max_pain": compute_max_pain(rows), "call_oi": c, "put_oi": p})

    # primary expiry = largest total OI (usually the quarterly)
    primary = max(expiries, key=lambda e: e["call_oi"] + e["put_oi"]) if expiries else None
    iv_atm = iv_skew = None
    if primary and underlying:
        rows = by_exp[primary["expiry"]]
        with_iv = [r for r in rows if r["iv"]]
        if with_iv:
            atm = min(with_iv, key=lambda r: abs(r["strike"] - underlying))
            atm_rows = [r for r in with_iv if r["strike"] == atm["strike"]]
            iv_atm = sum(r["iv"] for r in atm_rows) / len(atm_rows)
            otm_puts = [r["iv"] for r in with_iv if r["kind"] == "P" and 0.85 * underlying <= r["strike"] <= 0.95 * underlying]
            otm_calls = [r["iv"] for r in with_iv if r["kind"] == "C" and 1.05 * underlying <= r["strike"] <= 1.15 * underlying]
            if otm_puts and otm_calls:
                iv_skew = sum(otm_puts) / len(otm_puts) - sum(otm_calls) / len(otm_calls)

    return OptionsSnapshot(
        source="deribit",
        underlying_price=underlying,
        max_pain=primary["max_pain"] if primary else None,
        max_pain_expiry=primary["expiry"] if primary else None,
        put_call_oi_ratio=(put_oi / call_oi) if call_oi else None,
        put_call_volume_ratio=(put_vol / call_vol) if call_vol else None,
        iv_atm=iv_atm,
        iv_skew=iv_skew,
        total_call_oi=call_oi,
        total_put_oi=put_oi,
        expiries=expiries,
    )


async def fetch_options(client: httpx.AsyncClient) -> OptionsSnapshot:
    try:
        r = await client.get(f"{BASE}/get_book_summary_by_currency", params={"currency": SYMBOLS["deribit"], "kind": "option"})
        r.raise_for_status()
        return parse_book_summary(r.json())
    except Exception as e:  # noqa: BLE001 — provider boundary
        log.error("deribit unavailable: %s", e)
        return OptionsSnapshot.unavailable("deribit", str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_deribit_options.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/data/deribit_options.py tests/test_deribit_options.py
git commit -m "feat(data): deribit options provider with max pain, PCR, IV skew"
```

---

### Task 5: Sentiment provider

**Files:**
- Create: `core/data/sentiment.py`
- Test: `tests/test_sentiment.py`

**Interfaces:**
- Produces: `parse_fng(payload) -> SentimentSnapshot`, `async fetch_sentiment(client) -> SentimentSnapshot`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sentiment.py
import json
import pathlib

from core.data.sentiment import parse_fng

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_fng():
    s = parse_fng(json.loads((FX / "fng.json").read_text()))
    assert s.available and s.source == "alternative.me"
    assert 0 <= s.fear_greed <= 100
    assert s.classification
    assert 0 <= s.fear_greed_prev <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_sentiment.py -v` → FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `core/data/sentiment.py`**

```python
"""Crypto Fear & Greed index (alternative.me, keyless)."""
from __future__ import annotations

import logging

import httpx

from core.data.types import SentimentSnapshot

log = logging.getLogger(__name__)
URL = "https://api.alternative.me/fng/"


def parse_fng(payload: dict) -> SentimentSnapshot:
    data = payload["data"]
    cur = data[0]
    prev = data[1] if len(data) > 1 else None
    return SentimentSnapshot(
        source="alternative.me",
        fear_greed=int(cur["value"]),
        classification=cur["value_classification"],
        fear_greed_prev=int(prev["value"]) if prev else None,
    )


async def fetch_sentiment(client: httpx.AsyncClient) -> SentimentSnapshot:
    try:
        r = await client.get(URL, params={"limit": 2})
        r.raise_for_status()
        return parse_fng(r.json())
    except Exception as e:  # noqa: BLE001 — provider boundary
        log.error("fear&greed unavailable: %s", e)
        return SentimentSnapshot.unavailable("alternative.me", str(e))
```

- [ ] **Step 4: Run test** → 1 passed

- [ ] **Step 5: Commit**

```bash
git add core/data/sentiment.py tests/test_sentiment.py
git commit -m "feat(data): fear & greed sentiment provider"
```

---

### Task 6: Concurrent market loader

**Files:**
- Create: `core/data/market.py`
- Test: `tests/test_market.py`

**Interfaces:**
- Produces:
  - `async fetch_all(client: httpx.AsyncClient | None = None) -> MarketSnapshot`
  - `load_market() -> MarketSnapshot` (sync; Streamlit-cached in app via wrapper — see Task 20)
- Providers are injected through module-level names so tests can monkeypatch: `market.fetch_spot`, `market.fetch_futures`, `market.fetch_options`, `market.fetch_sentiment`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market.py
import asyncio

import pandas as pd

from core.data import market
from core.data.types import FuturesSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot


def test_fetch_all_isolates_provider_failures(monkeypatch):
    async def ok_spot(client):
        return SpotSnapshot(source="kraken", last=1.0, frames={"1h": pd.DataFrame()})

    async def boom(client):
        raise RuntimeError("network down")

    async def ok_opts(client):
        return OptionsSnapshot(source="deribit")

    async def ok_sent(client):
        return SentimentSnapshot(source="alternative.me", fear_greed=50)

    monkeypatch.setattr(market, "fetch_spot", ok_spot)
    monkeypatch.setattr(market, "fetch_futures", boom)
    monkeypatch.setattr(market, "fetch_options", ok_opts)
    monkeypatch.setattr(market, "fetch_sentiment", ok_sent)

    snap = asyncio.run(market.fetch_all())
    assert snap.spot.available
    assert snap.futures.available is False and "network down" in snap.futures.error
    assert snap.unavailable() == ["futures"]
```

- [ ] **Step 2: Run test to verify it fails** → FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `core/data/market.py`**

```python
"""Fetch every provider concurrently; a failing provider never takes the page down."""
from __future__ import annotations

import asyncio

import httpx

from core.data.binance_futures import fetch_futures
from core.data.deribit_options import fetch_options
from core.data.kraken_spot import fetch_spot
from core.data.sentiment import fetch_sentiment
from core.data.types import FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot

HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
TIMEOUT = httpx.Timeout(12.0, connect=6.0)


async def _guard(coro, fallback_cls, source: str):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001 — last line of defence
        return fallback_cls.unavailable(source, str(e))


async def fetch_all(client: httpx.AsyncClient | None = None) -> MarketSnapshot:
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS)
    try:
        spot, fut, opt, sent = await asyncio.gather(
            _guard(fetch_spot(client), SpotSnapshot, "kraken"),
            _guard(fetch_futures(client), FuturesSnapshot, "binance,bybit"),
            _guard(fetch_options(client), OptionsSnapshot, "deribit"),
            _guard(fetch_sentiment(client), SentimentSnapshot, "alternative.me"),
        )
    finally:
        if own:
            await client.aclose()
    return MarketSnapshot(spot=spot, futures=fut, options=opt, sentiment=sent)


def load_market() -> MarketSnapshot:
    """Synchronous entry point for Streamlit (wrapped with st.cache_data in app.py)."""
    return asyncio.run(fetch_all())
```

- [ ] **Step 4: Run test** → 1 passed

- [ ] **Step 5: Live smoke (one-off, not a test)**

Run: `.venv/Scripts/python -c "from core.data.market import load_market; m=load_market(); print(m.unavailable(), m.spot.last, m.futures.source, m.futures.funding_rate, m.options.max_pain, m.sentiment.fear_greed)"`
Expected: `[] <price> binance <rate> <max pain> <int>` (or `['futures']` if both futures sources fail from your network — then stop and investigate before continuing).

- [ ] **Step 6: Commit**

```bash
git add core/data/market.py tests/test_market.py
git commit -m "feat(data): concurrent market loader with per-provider isolation"
```

---

### Task 7: Pivots + EMA stack

**Files:**
- Create: `core/indicators/pivots.py`, `core/indicators/ema.py`
- Test: `tests/test_pivots.py`, `tests/test_ema.py`

**Interfaces:**
- Produces:
  - `find_pivots(values: np.ndarray, left: int, right: int, kind: Literal["high","low"]) -> list[int]`
  - `ema(close: pd.Series, n: int) -> pd.Series`
  - `EmaStack(values: dict[int, float|None], state: str, series: dict[int, pd.Series])` and `ema_stack(df, periods=(21,50,100,200)) -> EmaStack`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pivots.py
import numpy as np

from core.indicators.pivots import find_pivots


def test_find_pivot_highs_and_lows():
    v = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2], dtype=float)
    assert find_pivots(v, 2, 2, "high") == [2, 7]
    assert find_pivots(v, 2, 2, "low") == [4, 11]


def test_pivot_requires_full_window():
    v = np.array([5, 4, 3, 2, 1], dtype=float)
    assert find_pivots(v, 2, 2, "low") == []
```

```python
# tests/test_ema.py
import numpy as np
import pandas as pd

from core.indicators.ema import ema, ema_stack


def _df(closes):
    n = len(closes)
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0})


def test_ema_constant_series_is_constant():
    s = ema(pd.Series([50.0] * 100), 21)
    assert abs(s.iloc[-1] - 50.0) < 1e-9


def test_ema_stack_bull_on_uptrend():
    st = ema_stack(_df(np.linspace(100, 200, 260)))
    assert st.state == "BULL"
    assert st.values[21] > st.values[50] > st.values[100] > st.values[200]
    assert set(st.series) == {21, 50, 100, 200}


def test_ema_stack_bear_on_downtrend():
    assert ema_stack(_df(np.linspace(200, 100, 260))).state == "BEAR"


def test_ema_stack_missing_when_short_history():
    st = ema_stack(_df(np.linspace(100, 110, 60)))
    assert st.values[200] is None and st.state == "MIXED"
```

- [ ] **Step 2: Run tests to verify they fail** → `ModuleNotFoundError`

- [ ] **Step 3: Write the modules**

```python
# core/indicators/pivots.py
"""Swing pivot detection shared by RSI divergence, S/R, trendlines, fib time."""
from __future__ import annotations

from typing import Literal

import numpy as np


def find_pivots(values: np.ndarray, left: int, right: int, kind: Literal["high", "low"]) -> list[int]:
    out: list[int] = []
    n = len(values)
    for i in range(left, n - right):
        v = values[i]
        wl = values[i - left:i]
        wr = values[i + 1:i + 1 + right]
        if kind == "high":
            if v > wl.max() and v >= wr.max():
                out.append(i)
        else:
            if v < wl.min() and v <= wr.min():
                out.append(i)
    return out
```

```python
# core/indicators/ema.py
"""EMA 21/50/100/200 and stack state."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_PERIODS = (21, 50, 100, 200)


@dataclass
class EmaStack:
    values: dict[int, float | None]
    state: str  # BULL | BEAR | MIXED
    series: dict[int, pd.Series] = field(default_factory=dict)


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def ema_stack(df: pd.DataFrame, periods: tuple[int, ...] = DEFAULT_PERIODS) -> EmaStack:
    close = df["close"].astype(float)
    series: dict[int, pd.Series] = {}
    values: dict[int, float | None] = {}
    for n in periods:
        if len(close) >= n:
            s = ema(close, n)
            series[n] = s
            values[n] = float(s.iloc[-1])
        else:
            values[n] = None
    state = "MIXED"
    if all(v is not None for v in values.values()):
        seq = [float(close.iloc[-1])] + [values[n] for n in periods]
        if all(a > b for a, b in zip(seq, seq[1:])):
            state = "BULL"
        elif all(a < b for a, b in zip(seq, seq[1:])):
            state = "BEAR"
    return EmaStack(values=values, state=state, series=series)
```

- [ ] **Step 4: Run tests** → 6 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/pivots.py core/indicators/ema.py tests/test_pivots.py tests/test_ema.py
git commit -m "feat(indicators): pivots and EMA stack"
```

---

### Task 8: RSI + divergence (confirmed and forming)

**Files:**
- Create: `core/indicators/rsi.py`
- Test: `tests/test_rsi.py`

**Interfaces:**
- Produces:
  - `rsi(close: pd.Series, n: int = 14) -> pd.Series`
  - `Divergence(kind: "regular"|"hidden", direction: "bullish"|"bearish", status: "confirmed"|"forming", price_points: list[tuple[int,float]], rsi_points: list[tuple[int,float]], bars: tuple[int,int])` with `.label` property
  - `detect_divergences(df, rsi_series, left=3, right=3, lookback=60) -> list[Divergence]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rsi.py
import numpy as np
import pandas as pd

from core.indicators.rsi import detect_divergences, rsi


def _df(highs, lows=None):
    n = len(highs)
    highs = np.asarray(highs, dtype=float)
    lows = highs - 1.0 if lows is None else np.asarray(lows, dtype=float)
    close = (highs + lows) / 2
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": highs, "low": lows, "close": close, "volume": 1.0})


def test_rsi_extremes():
    up = rsi(pd.Series(np.linspace(100, 200, 60)))
    assert up.iloc[-1] > 95
    down = rsi(pd.Series(np.linspace(200, 100, 60)))
    assert down.iloc[-1] < 5
    assert np.isnan(up.iloc[5])  # warm-up


def test_regular_bearish_divergence_confirmed():
    highs = np.full(50, 100.0)
    highs[10] = 110.0   # first peak
    highs[30] = 115.0   # higher high
    r = pd.Series(np.full(50, 50.0)); r[10] = 80.0; r[30] = 70.0  # lower RSI high
    d = detect_divergences(_df(highs), r)
    assert any(x.kind == "regular" and x.direction == "bearish" and x.status == "confirmed" and x.bars == (10, 30) for x in d)


def test_regular_bullish_divergence_forming_on_last_bar():
    lows = np.full(40, 100.0)
    lows[10] = 90.0      # confirmed pivot low
    lows[39] = 85.0      # current bar making a lower low, not yet confirmed
    r = pd.Series(np.full(40, 50.0)); r[10] = 20.0; r[39] = 30.0  # RSI higher low
    d = detect_divergences(_df(lows + 1.0, lows), r)
    assert any(x.kind == "regular" and x.direction == "bullish" and x.status == "forming" for x in d)


def test_no_divergence_on_flat():
    d = detect_divergences(_df(np.full(50, 100.0)), pd.Series(np.full(50, 50.0)))
    assert d == []
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/rsi.py`**

```python
"""Wilder RSI and pivot-based divergence detection (regular + hidden, confirmed + forming)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Divergence:
    kind: str        # regular | hidden
    direction: str   # bullish | bearish
    status: str      # confirmed | forming
    price_points: list[tuple[int, float]]
    rsi_points: list[tuple[int, float]]
    bars: tuple[int, int]

    @property
    def label(self) -> str:
        return f"{self.kind.title()} {self.direction} ({self.status})"


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad.replace(0.0, np.nan))
    out = out.where(~((ad == 0) & au.notna()), 100.0)
    return out


def detect_divergences(df: pd.DataFrame, rsi_series: pd.Series, left: int = 3, right: int = 3, lookback: int = 60) -> list[Divergence]:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy()
    r = rsi_series.to_numpy(dtype=float)
    n = len(df)
    if n < left + right + 2:
        return []
    start = max(0, n - lookback)
    ph = [i for i in find_pivots(highs, left, right, "high") if i >= start]
    pl = [i for i in find_pivots(lows, left, right, "low") if i >= start]
    out: list[Divergence] = []

    def emit(kind, direction, status, p1, p2, arr):
        out.append(Divergence(kind, direction, status,
                              [(int(ts[p1]), float(arr[p1])), (int(ts[p2]), float(arr[p2]))],
                              [(int(ts[p1]), float(r[p1])), (int(ts[p2]), float(r[p2]))],
                              (p1, p2)))

    def check(p1, p2, arr, is_high, status):
        if np.isnan(r[p1]) or np.isnan(r[p2]) or arr[p1] == arr[p2] or r[p1] == r[p2]:
            return
        price_up = arr[p2] > arr[p1]
        rsi_up = r[p2] > r[p1]
        if is_high:
            if price_up and not rsi_up:
                emit("regular", "bearish", status, p1, p2, arr)
            elif not price_up and rsi_up:
                emit("hidden", "bearish", status, p1, p2, arr)
        else:
            if not price_up and rsi_up:
                emit("regular", "bullish", status, p1, p2, arr)
            elif price_up and not rsi_up:
                emit("hidden", "bullish", status, p1, p2, arr)

    if len(ph) >= 2:
        check(ph[-2], ph[-1], highs, True, "confirmed")
    if len(pl) >= 2:
        check(pl[-2], pl[-1], lows, False, "confirmed")

    last = n - 1
    if ph and last - ph[-1] > right and highs[last] >= highs[last - left:last].max():
        check(ph[-1], last, highs, True, "forming")
    if pl and last - pl[-1] > right and lows[last] <= lows[last - left:last].min():
        check(pl[-1], last, lows, False, "forming")
    return out
```

- [ ] **Step 4: Run tests** → 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/rsi.py tests/test_rsi.py
git commit -m "feat(indicators): RSI and regular/hidden divergence with forming state"
```

---

### Task 9: Support / resistance levels

**Files:**
- Create: `core/indicators/levels.py`
- Test: `tests/test_levels.py`

**Interfaces:**
- Produces: `Level(price: float, touches: int, kind: "support"|"resistance", last_touch_ms: int)`, `support_resistance(df, atr_value: float|None, left=5, right=5, lookback=200, tol_atr=0.35, max_levels=6) -> list[Level]` sorted high→low.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_levels.py
import numpy as np
import pandas as pd

from core.indicators.levels import support_resistance


def _zigzag(n=200, lo=100.0, hi=110.0, period=20):
    x = np.arange(n)
    tri = np.abs(((x % period) / (period / 2)) - 1.0)  # 1 → 0 → 1 triangle
    mid = lo + (hi - lo) * (1 - tri)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, n)
    close = mid + noise
    return pd.DataFrame({"timestamp": x * 60_000, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1.0})


def test_zigzag_yields_two_strong_levels():
    df = _zigzag()
    levels = support_resistance(df, atr_value=1.0)
    prices = [l.price for l in levels]
    assert any(abs(p - 110.2) < 1.0 for p in prices)
    assert any(abs(p - 99.8) < 1.0 for p in prices)
    strong = [l for l in levels if l.touches >= 3]
    assert len(strong) >= 2
    assert levels == sorted(levels, key=lambda l: l.price, reverse=True)
    assert all(l.kind in ("support", "resistance") for l in levels)
    assert len(levels) <= 6


def test_short_history_returns_empty():
    df = _zigzag(n=8)
    assert support_resistance(df, atr_value=1.0) == []
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/levels.py`**

```python
"""Swing-pivot support/resistance clustered by an ATR tolerance."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Level:
    price: float
    touches: int
    kind: str  # support | resistance (relative to current price)
    last_touch_ms: int


def support_resistance(df: pd.DataFrame, atr_value: float | None, left: int = 5, right: int = 5,
                       lookback: int = 200, tol_atr: float = 0.35, max_levels: int = 6) -> list[Level]:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < left + right + 2:
        return []
    highs = d["high"].to_numpy(dtype=float)
    lows = d["low"].to_numpy(dtype=float)
    ts = d["timestamp"].to_numpy()
    price = float(d["close"].iloc[-1])
    pts = [(highs[i], int(ts[i])) for i in find_pivots(highs, left, right, "high")]
    pts += [(lows[i], int(ts[i])) for i in find_pivots(lows, left, right, "low")]
    if not pts:
        return []
    pts.sort()
    tol = tol_atr * atr_value if atr_value else price * 0.002
    clusters: list[list[tuple[float, int]]] = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = []
    for c in clusters:
        mean = sum(p for p, _ in c) / len(c)
        levels.append(Level(price=mean, touches=len(c), kind="resistance" if mean > price else "support", last_touch_ms=max(t for _, t in c)))
    half = max(1, max_levels // 2)
    above = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: l.price - price)[:half]
    below = sorted([l for l in levels if l.kind == "support"], key=lambda l: price - l.price)[:half]
    return sorted(above + below, key=lambda l: l.price, reverse=True)
```

- [ ] **Step 4: Run tests** → 2 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/levels.py tests/test_levels.py
git commit -m "feat(indicators): clustered swing support/resistance"
```

---

### Task 10: Trendlines

**Files:**
- Create: `core/indicators/trendlines.py`
- Test: `tests/test_trendlines.py`

**Interfaces:**
- Produces: `Trendline(kind: "support"|"resistance", slope: float, r2: float, points: list[tuple[int,float]], value_now: float)`, `fit_trendlines(df, left=5, right=5, lookback=200, min_r2=0.8, n_points=3, extend_bars=10) -> list[Trendline]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trendlines.py
import numpy as np
import pandas as pd

from core.indicators.trendlines import fit_trendlines


def _rising_channel(n=120):
    x = np.arange(n, dtype=float)
    base = 100 + 0.5 * x                      # support line: 100 + 0.5x
    wave = 5 * np.abs(np.sin(x / 6.0))        # bounces off the line every ~19 bars
    low = base + wave
    high = low + 4.0
    close = (low + high) / 2
    return pd.DataFrame({"timestamp": (x * 60_000).astype("int64"), "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def test_support_trendline_matches_generating_line():
    tl = [t for t in fit_trendlines(_rising_channel()) if t.kind == "support"]
    assert tl, "expected a support trendline"
    t = tl[0]
    assert abs(t.slope - 0.5) < 0.05
    assert t.r2 > 0.95
    assert len(t.points) == 2 and t.points[1][0] > t.points[0][0]
    assert t.value_now > t.points[0][1]


def test_lines_below_r2_gate_are_dropped():
    rng = np.random.default_rng(1)
    n = 120
    close = np.full(n, 100.0)
    low = close - rng.uniform(0, 5, n)
    high = close + rng.uniform(0, 5, n)
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})
    lines = fit_trendlines(df, min_r2=0.99)
    assert all(t.r2 >= 0.99 for t in lines)
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/trendlines.py`**

```python
"""Least-squares trendlines through the last N swing lows (support) / highs (resistance)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicators.pivots import find_pivots


@dataclass
class Trendline:
    kind: str                        # support | resistance
    slope: float                     # price units per bar
    r2: float
    points: list[tuple[int, float]]  # (ms, price) start and projected end
    value_now: float


def fit_trendlines(df: pd.DataFrame, left: int = 5, right: int = 5, lookback: int = 200, min_r2: float = 0.8,
                   n_points: int = 3, extend_bars: int = 10) -> list[Trendline]:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < left + right + 2:
        return []
    ts = d["timestamp"].to_numpy(dtype="int64")
    interval = int(np.median(np.diff(ts))) if len(ts) > 1 else 60_000
    last = len(d) - 1
    out: list[Trendline] = []
    for kind, arr, pk in (("support", d["low"].to_numpy(dtype=float), "low"), ("resistance", d["high"].to_numpy(dtype=float), "high")):
        piv = find_pivots(arr, left, right, pk)[-n_points:]
        if len(piv) < n_points:
            continue
        x = np.array(piv, dtype=float)
        y = arr[piv]
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if r2 < min_r2:
            continue
        x0, x1 = piv[0], last + extend_bars
        out.append(Trendline(
            kind=kind, slope=float(slope), r2=float(r2),
            points=[(int(ts[x0]), float(slope * x0 + intercept)), (int(ts[last] + interval * extend_bars), float(slope * x1 + intercept))],
            value_now=float(slope * last + intercept),
        ))
    return out
```

- [ ] **Step 4: Run tests** → 2 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/trendlines.py tests/test_trendlines.py
git commit -m "feat(indicators): pivot-fitted trendlines with R2 gate"
```

---

### Task 11: Fibonacci time zones

**Files:**
- Create: `core/indicators/fib_time.py`
- Test: `tests/test_fib_time.py`

**Interfaces:**
- Produces: `FibZone(k: int, timestamp_ms: int, is_future: bool)`, `FibTime(anchor_ms: int, anchor_kind: "low"|"high", zones: list[FibZone], upcoming: list[FibZone])`, `fib_time_zones(df, lookback=100, max_future=3) -> FibTime|None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fib_time.py
import numpy as np
import pandas as pd

from core.indicators.fib_time import fib_time_zones


def _df(n=100, low_at=50, high_at=70):
    close = np.full(n, 100.0)
    low = close.copy(); high = close.copy()
    low[low_at] = 80.0; high[high_at] = 120.0
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def test_zones_from_anchor_with_future_extrapolation():
    ft = fib_time_zones(_df(), max_future=3)
    assert ft.anchor_kind == "low" and ft.anchor_ms == 50 * 60_000
    past = [z for z in ft.zones if not z.is_future]
    assert [z.k for z in past] == [1, 2, 3, 5, 8, 13, 21, 34]
    assert [z.k for z in ft.upcoming] == [55, 89, 144]
    assert ft.upcoming[0].timestamp_ms == (50 + 55) * 60_000  # extrapolated with bar interval


def test_anchor_is_earlier_extreme():
    ft = fib_time_zones(_df(low_at=70, high_at=40))
    assert ft.anchor_kind == "high" and ft.anchor_ms == 40 * 60_000


def test_too_short_returns_none():
    assert fib_time_zones(_df(n=3)) is None
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/fib_time.py`**

```python
"""Fibonacci time zones projected from the earliest major swing in the lookback window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIB = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)


@dataclass
class FibZone:
    k: int
    timestamp_ms: int
    is_future: bool


@dataclass
class FibTime:
    anchor_ms: int
    anchor_kind: str  # low | high
    zones: list[FibZone]
    upcoming: list[FibZone]


def fib_time_zones(df: pd.DataFrame, lookback: int = 100, max_future: int = 3) -> FibTime | None:
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < 5:
        return None
    ts = d["timestamp"].to_numpy(dtype="int64")
    interval = int(np.median(np.diff(ts)))
    i_lo = int(d["low"].to_numpy().argmin())
    i_hi = int(d["high"].to_numpy().argmax())
    anchor_i, kind = (i_lo, "low") if i_lo <= i_hi else (i_hi, "high")
    last_i = len(d) - 1
    zones: list[FibZone] = []
    future = 0
    for k in FIB:
        idx = anchor_i + k
        if idx <= last_i:
            zones.append(FibZone(k, int(ts[idx]), False))
        else:
            if future >= max_future:
                break
            zones.append(FibZone(k, int(ts[last_i] + (idx - last_i) * interval), True))
            future += 1
    return FibTime(anchor_ms=int(ts[anchor_i]), anchor_kind=kind, zones=zones, upcoming=[z for z in zones if z.is_future])
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/fib_time.py tests/test_fib_time.py
git commit -m "feat(indicators): fibonacci time zones with future projection"
```

---

### Task 12: Volatility profile

**Files:**
- Create: `core/indicators/volatility.py`
- Test: `tests/test_volatility.py`

**Interfaces:**
- Produces: `atr(df, n=14) -> pd.Series`, `VolProfile(atr, atr_pct, regime, sigma_pct, exp_low, exp_high, horizon_label)`, `volatility_profile(df, horizon_bars, horizon_label, ret_window=30) -> VolProfile`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_volatility.py
import numpy as np
import pandas as pd

from core.indicators.volatility import atr, volatility_profile


def _df(n=200, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, scale, n).cumsum()
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + scale, "low": close - scale, "close": close, "volume": 1.0})


def test_atr_positive_after_warmup():
    a = atr(_df())
    assert np.isnan(a.iloc[3]) and a.iloc[-1] > 0


def test_profile_brackets_price_and_has_regime():
    df = _df()
    vp = volatility_profile(df, horizon_bars=24, horizon_label="next 24h")
    price = float(df["close"].iloc[-1])
    assert vp.exp_low < price < vp.exp_high
    assert vp.regime in {"LOW", "NORMAL", "HIGH"}
    assert vp.atr_pct > 0 and vp.horizon_label == "next 24h"


def test_high_regime_when_recent_range_expands():
    df = _df()
    df.loc[df.index[-15:], "high"] += 15.0
    df.loc[df.index[-15:], "low"] -= 15.0
    assert volatility_profile(df, 4, "next hour").regime == "HIGH"
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/volatility.py`**

```python
"""ATR, relative volatility regime, and log-return expected range for a horizon."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolProfile:
    atr: float | None
    atr_pct: float | None
    regime: str            # LOW | NORMAL | HIGH | UNKNOWN
    sigma_pct: float | None
    exp_low: float | None
    exp_high: float | None
    horizon_label: str


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def volatility_profile(df: pd.DataFrame, horizon_bars: int, horizon_label: str, ret_window: int = 30) -> VolProfile:
    if len(df) < 20:
        return VolProfile(None, None, "UNKNOWN", None, None, None, horizon_label)
    close = df["close"].astype(float)
    price = float(close.iloc[-1])
    a = atr(df)
    atr_v = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else None
    atr_pct_series = (a / close * 100.0)
    atr_pct = float(atr_pct_series.iloc[-1]) if atr_v is not None else None
    med = float(atr_pct_series.tail(100).median()) if atr_pct is not None else None
    if atr_pct is None or med is None or np.isnan(med) or med == 0:
        regime = "UNKNOWN"
    elif atr_pct < 0.75 * med:
        regime = "LOW"
    elif atr_pct > 1.5 * med:
        regime = "HIGH"
    else:
        regime = "NORMAL"
    lr = np.log(close).diff().tail(ret_window)
    sigma = float(lr.std()) * math.sqrt(horizon_bars) if lr.notna().sum() >= 5 else None
    exp_low = price * math.exp(-sigma) if sigma is not None else None
    exp_high = price * math.exp(sigma) if sigma is not None else None
    return VolProfile(atr_v, atr_pct, regime, sigma * 100 if sigma is not None else None, exp_low, exp_high, horizon_label)
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/volatility.py tests/test_volatility.py
git commit -m "feat(indicators): ATR, volatility regime, expected range"
```

---

### Task 13: Squeeze risk

**Files:**
- Create: `core/indicators/squeeze.py`
- Test: `tests/test_squeeze.py`

**Interfaces:**
- Consumes: `FuturesSnapshot`
- Produces: `SqueezeRisk(score: int|None, direction: "LONG_SQUEEZE"|"SHORT_SQUEEZE"|"NONE"|"UNKNOWN", drivers: list[str])`, `squeeze_risk(fut, price_change_24h_pct) -> SqueezeRisk`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_squeeze.py
from core.data.types import FuturesSnapshot
from core.indicators.squeeze import squeeze_risk


def test_crowded_longs_flag_long_squeeze():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0005, funding_7d_mean=0.0001, oi_change_24h_pct=6.0, long_short_ratio=1.6)
    r = squeeze_risk(fut, price_change_24h_pct=0.2)
    assert r.direction == "LONG_SQUEEZE" and r.score >= 60
    assert any("funding" in d.lower() for d in r.drivers)


def test_crowded_shorts_flag_short_squeeze():
    fut = FuturesSnapshot(source="binance", funding_rate=-0.0004, funding_7d_mean=0.0, oi_change_24h_pct=5.0, long_short_ratio=0.6)
    r = squeeze_risk(fut, price_change_24h_pct=-0.3)
    assert r.direction == "SHORT_SQUEEZE" and r.score >= 60


def test_calm_market_none():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0001, funding_7d_mean=0.0001, oi_change_24h_pct=0.2, long_short_ratio=1.0)
    r = squeeze_risk(fut, price_change_24h_pct=3.0)
    assert r.direction == "NONE" and r.score < 40


def test_unavailable_futures():
    r = squeeze_risk(FuturesSnapshot.unavailable("binance,bybit", "451"), price_change_24h_pct=1.0)
    assert r.score is None and r.direction == "UNKNOWN"
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/squeeze.py`**

```python
"""Long/short squeeze risk from OI build, price stagnation, funding deviation, L/S skew."""
from __future__ import annotations

from dataclasses import dataclass

from core.data.types import FuturesSnapshot


@dataclass
class SqueezeRisk:
    score: int | None
    direction: str  # LONG_SQUEEZE | SHORT_SQUEEZE | NONE | UNKNOWN
    drivers: list[str]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def squeeze_risk(fut: FuturesSnapshot, price_change_24h_pct: float | None) -> SqueezeRisk:
    if not fut.available or fut.oi_change_24h_pct is None:
        return SqueezeRisk(None, "UNKNOWN", ["futures data unavailable"])
    oi_up = _clamp(fut.oi_change_24h_pct / 5.0, 0.0, 1.0)
    px = price_change_24h_pct or 0.0
    flat = 1.0 - _clamp(abs(px) / 2.0, 0.0, 1.0)
    fr = fut.funding_rate or 0.0
    mean = fut.funding_7d_mean or 0.0
    funding_hot = _clamp((fr - mean) / 0.0002, -1.0, 1.0)
    ls = fut.long_short_ratio
    ls_skew = 0.0 if ls is None else _clamp((ls - 1.0) / 0.5, -1.0, 1.0)

    long_crowd = max(0.0, funding_hot) * 0.4 + max(0.0, ls_skew) * 0.3 + oi_up * 0.2 + flat * 0.1
    short_crowd = max(0.0, -funding_hot) * 0.4 + max(0.0, -ls_skew) * 0.3 + oi_up * 0.2 + flat * 0.1

    drivers: list[str] = []
    if oi_up >= 0.5:
        drivers.append(f"Open interest +{fut.oi_change_24h_pct:.1f}% in 24h")
    if flat >= 0.5:
        drivers.append(f"Price {px:+.1f}% while positioning builds")
    if funding_hot >= 0.5:
        drivers.append(f"Funding {fr:+.4%} above 7d mean {mean:+.4%}")
    if funding_hot <= -0.5:
        drivers.append(f"Funding {fr:+.4%} below 7d mean {mean:+.4%}")
    if ls is not None and ls_skew >= 0.5:
        drivers.append(f"Long/short ratio {ls:.2f} long-crowded")
    if ls is not None and ls_skew <= -0.5:
        drivers.append(f"Long/short ratio {ls:.2f} short-crowded")

    if long_crowd >= short_crowd:
        score = round(long_crowd * 100)
        return SqueezeRisk(score, "LONG_SQUEEZE" if score >= 40 else "NONE", drivers)
    score = round(short_crowd * 100)
    return SqueezeRisk(score, "SHORT_SQUEEZE" if score >= 40 else "NONE", drivers)
```

- [ ] **Step 4: Run tests** → 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/squeeze.py tests/test_squeeze.py
git commit -m "feat(indicators): squeeze risk score from futures positioning"
```

---

### Task 14: Direction call

**Files:**
- Create: `core/indicators/direction.py`
- Test: `tests/test_direction.py`

**Interfaces:**
- Consumes: `EmaStack`, `Level`
- Produces: `cvd_slope(df, bars=5) -> float`, `DirectionCall(direction: "BULLISH"|"BEARISH"|"NEUTRAL", confidence: float, score: float, anchor: float|None, invalidation: float|None, drivers: list[str], flips_if: str)`, `direction_call(df, stack, rsi_series, levels, atr_value) -> DirectionCall`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_direction.py
import numpy as np
import pandas as pd

from core.indicators.direction import direction_call
from core.indicators.ema import ema_stack
from core.indicators.levels import Level
from core.indicators.rsi import rsi


def _trend(n=260, up=True):
    x = np.linspace(100, 160, n) if up else np.linspace(160, 100, n)
    rng = np.random.default_rng(0)
    close = x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 10.0})


def test_uptrend_is_bullish_with_anchor_on_support():
    df = _trend(up=True)
    price = float(df["close"].iloc[-1])
    levels = [Level(price - 2, 3, "support", 0), Level(price + 6, 2, "resistance", 0)]
    d = direction_call(df, ema_stack(df), rsi(df["close"]), levels, atr_value=1.0)
    assert d.direction == "BULLISH" and d.confidence > 0.3
    assert d.anchor == price - 2 and d.invalidation < d.anchor
    assert "below" in d.flips_if


def test_downtrend_is_bearish():
    df = _trend(up=False)
    d = direction_call(df, ema_stack(df), rsi(df["close"]), [], atr_value=1.0)
    assert d.direction == "BEARISH" and d.anchor is None


def test_flat_is_neutral():
    n = 260
    close = np.full(n, 100.0) + np.tile([0.1, -0.1], n // 2)
    df = pd.DataFrame({"timestamp": np.arange(n) * 60_000, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 10.0})
    d = direction_call(df, ema_stack(df), rsi(df["close"]), [], atr_value=0.3)
    assert d.direction == "NEUTRAL"
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/direction.py`**

```python
"""Composite per-category direction call: EMA stack, RSI, CVD slope, S/R position."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.indicators.ema import EmaStack
from core.indicators.levels import Level


@dataclass
class DirectionCall:
    direction: str      # BULLISH | BEARISH | NEUTRAL
    confidence: float   # 0..1
    score: float        # -1..1
    anchor: float | None
    invalidation: float | None
    drivers: list[str]
    flips_if: str


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def cvd_slope(df: pd.DataFrame, bars: int = 5) -> float:
    rng = (df["high"] - df["low"]).clip(lower=1e-9)
    buyer_ratio = (df["close"] - df["low"]) / rng
    delta = (buyer_ratio - 0.5) * 2.0 * df["volume"]
    cvd = delta.cumsum()
    if len(cvd) <= bars:
        return 0.0
    return float(cvd.iloc[-1] - cvd.iloc[-1 - bars])


def direction_call(df: pd.DataFrame, stack: EmaStack, rsi_series: pd.Series, levels: list[Level], atr_value: float | None) -> DirectionCall:
    price = float(df["close"].iloc[-1])
    atr_v = atr_value or price * 0.005
    drivers: list[str] = []

    if stack.state == "BULL":
        ema_vote = 1.0
        drivers.append("EMA stack bullish (price > 21 > 50 > 100 > 200)")
    elif stack.state == "BEAR":
        ema_vote = -1.0
        drivers.append("EMA stack bearish (price < 21 < 50 < 100 < 200)")
    else:
        e50 = stack.values.get(50)
        ema_vote = 0.0 if e50 is None else (0.3 if price > e50 else -0.3)
        if e50 is not None:
            drivers.append(f"Price {'above' if price > e50 else 'below'} EMA50, stack mixed")

    r = rsi_series.dropna()
    if len(r) >= 6:
        last, prev = float(r.iloc[-1]), float(r.iloc[-6])
        rsi_vote = _clamp((last - 50.0) / 25.0) * 0.7 + (0.3 if last > prev else -0.3 if last < prev else 0.0)
        drivers.append(f"RSI {last:.0f} and {'rising' if last > prev else 'falling'}")
    else:
        rsi_vote = 0.0

    slope = cvd_slope(df)
    vol_ref = float(df["volume"].tail(20).mean()) or 1.0
    cvd_vote = _clamp(slope / (vol_ref * 2.0))
    if abs(cvd_vote) >= 0.3:
        drivers.append(f"CVD {'rising' if cvd_vote > 0 else 'falling'} over last 5 bars")

    sr_vote = 0.0
    near_sup = [l for l in levels if l.kind == "support" and l.touches >= 2 and 0 <= price - l.price <= 0.5 * atr_v]
    near_res = [l for l in levels if l.kind == "resistance" and l.touches >= 2 and 0 <= l.price - price <= 0.5 * atr_v]
    if near_sup and not near_res:
        sr_vote = 0.5
        drivers.append(f"Holding support {near_sup[0].price:,.0f} ({near_sup[0].touches} touches)")
    elif near_res and not near_sup:
        sr_vote = -0.5
        drivers.append(f"Capped under resistance {near_res[0].price:,.0f} ({near_res[0].touches} touches)")

    score = 0.35 * ema_vote + 0.2 * rsi_vote + 0.2 * cvd_vote + 0.25 * sr_vote
    direction = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
    confidence = min(1.0, abs(score))

    supports = sorted([l for l in levels if l.kind == "support"], key=lambda l: price - l.price)
    resistances = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: l.price - price)
    anchor = invalidation = None
    if direction == "BULLISH" and supports:
        anchor = supports[0].price
        invalidation = anchor - 0.5 * atr_v
        flips = f"Close below {invalidation:,.0f} or EMA stack turns bearish"
    elif direction == "BEARISH" and resistances:
        anchor = resistances[0].price
        invalidation = anchor + 0.5 * atr_v
        flips = f"Close above {invalidation:,.0f} or EMA stack turns bullish"
    elif direction == "NEUTRAL":
        flips = "Break of nearest support/resistance with EMA confirmation"
    else:
        flips = "EMA stack reversal"
    return DirectionCall(direction, confidence, float(score), anchor, invalidation, drivers, flips)
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/direction.py tests/test_direction.py
git commit -m "feat(indicators): composite direction call with anchor and invalidation"
```

---

### Task 15: Category analysis assembly

**Files:**
- Create: `core/indicators/category.py`
- Test: `tests/test_category.py`

**Interfaces:**
- Consumes: everything from Tasks 7–14, `Category`, `FuturesSnapshot`, `CHART_BARS`.
- Produces:
  ```python
  @dataclass
  class CategoryAnalysis:
      key: str; label: str; chart_tf: str; context_tf: str
      price: float
      chart_df: pd.DataFrame           # last CHART_BARS rows of chart_tf frame
      ema: EmaStack                    # series aligned to full frame; slice with .tail(len(chart_df))
      rsi: pd.Series                   # aligned to chart_df
      divergences: list[Divergence]
      levels: list[Level]
      trendlines: list[Trendline]
      fib: FibTime | None
      vol: VolProfile
      direction: DirectionCall
      squeeze: SqueezeRisk
      price_change_24h_pct: float | None
      ctx_ema: EmaStack | None
  def analyze_category(cat: Category, frames: dict[str, pd.DataFrame], futures: FuturesSnapshot) -> CategoryAnalysis | None
  def price_change_24h(frames) -> float | None
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_category.py
import json
import pathlib

from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category, price_change_24h

FX = pathlib.Path(__file__).parent / "fixtures"


def _frames():
    return {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
            "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}


def test_live_category_on_fixtures():
    fut = FuturesSnapshot(source="binance", funding_rate=0.0001, funding_7d_mean=0.0001, oi_change_24h_pct=1.0, long_short_ratio=1.1)
    a = analyze_category(CATEGORIES["live"], _frames(), fut)
    assert a is not None and a.chart_tf == "15m" and a.context_tf == "1h"
    assert len(a.chart_df) <= 300 and len(a.rsi) == len(a.chart_df)
    assert a.direction.direction in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert a.vol.regime in {"LOW", "NORMAL", "HIGH", "UNKNOWN"}
    assert 0 <= len(a.levels) <= 6
    assert a.fib is not None and a.squeeze.score is not None
    assert a.ctx_ema is not None


def test_missing_frame_returns_none():
    assert analyze_category(CATEGORIES["weekly"], _frames(), FuturesSnapshot.unavailable("x", "y")) is None


def test_price_change_24h_uses_1h_frame():
    pc = price_change_24h(_frames())
    assert pc is not None and -50 < pc < 50
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/category.py`**

```python
"""Compose all indicators into one CategoryAnalysis per timeframe category."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.config import CHART_BARS, TF_MINUTES, Category
from core.data.types import FuturesSnapshot
from core.indicators.direction import DirectionCall, direction_call
from core.indicators.ema import EmaStack, ema_stack
from core.indicators.fib_time import FibTime, fib_time_zones
from core.indicators.levels import Level, support_resistance
from core.indicators.rsi import Divergence, detect_divergences, rsi
from core.indicators.squeeze import SqueezeRisk, squeeze_risk
from core.indicators.trendlines import Trendline, fit_trendlines
from core.indicators.volatility import VolProfile, volatility_profile


@dataclass
class CategoryAnalysis:
    key: str
    label: str
    chart_tf: str
    context_tf: str
    price: float
    chart_df: pd.DataFrame
    ema: EmaStack
    rsi: pd.Series
    divergences: list[Divergence]
    levels: list[Level]
    trendlines: list[Trendline]
    fib: FibTime | None
    vol: VolProfile
    direction: DirectionCall
    squeeze: SqueezeRisk
    price_change_24h_pct: float | None
    ctx_ema: EmaStack | None


def price_change_24h(frames: dict[str, pd.DataFrame]) -> float | None:
    for tf in ("1h", "15m", "4h"):
        df = frames.get(tf)
        if df is None:
            continue
        bars = 1440 // TF_MINUTES[tf]
        if len(df) > bars:
            return float(df["close"].iloc[-1] / df["close"].iloc[-1 - bars] - 1.0) * 100.0
    return None


def analyze_category(cat: Category, frames: dict[str, pd.DataFrame], futures: FuturesSnapshot) -> CategoryAnalysis | None:
    df = frames.get(cat.chart_tf)
    if df is None or len(df) < 30:
        return None
    ctx = frames.get(cat.context_tf)
    price = float(df["close"].iloc[-1])
    stack = ema_stack(df)
    r = rsi(df["close"])
    vol = volatility_profile(df, cat.horizon_bars, cat.horizon_label)
    levels = support_resistance(df, vol.atr)
    pc = price_change_24h(frames)
    return CategoryAnalysis(
        key=cat.key, label=cat.label, chart_tf=cat.chart_tf, context_tf=cat.context_tf, price=price,
        chart_df=df.tail(CHART_BARS).reset_index(drop=True),
        ema=stack,
        rsi=r.tail(CHART_BARS).reset_index(drop=True),
        divergences=detect_divergences(df, r),
        levels=levels,
        trendlines=fit_trendlines(df),
        fib=fib_time_zones(df),
        vol=vol,
        direction=direction_call(df, stack, r, levels, vol.atr),
        squeeze=squeeze_risk(futures, pc),
        price_change_24h_pct=pc,
        ctx_ema=ema_stack(ctx) if ctx is not None and len(ctx) >= 30 else None,
    )
```

- [ ] **Step 4: Run tests** → 3 passed. Then run the whole suite: `.venv/Scripts/python -m pytest` → all green.

- [ ] **Step 5: Commit**

```bash
git add core/indicators/category.py tests/test_category.py
git commit -m "feat(indicators): CategoryAnalysis assembly"
```

---

### Task 16: Trade map (Active Trade engine)

**Files:**
- Create: `core/indicators/trade_map.py`
- Test: `tests/test_trade_map.py`

**Interfaces:**
- Consumes: `CategoryAnalysis`, `FuturesSnapshot`
- Produces:
  ```python
  @dataclass
  class TradeMap:
      classification: str          # SCALP | SWING
      direction: str; entry: float; leverage: float; stop: float; target: float
      liquidation_price: float; dist_to_liq_pct: float; risk_label: str   # OPTIMAL | ELEVATED | EXTREME
      rr: float | None
      structures_above: list[str]; structures_below: list[str]
      stop_structural: bool
      warnings: list[str]
      anchor_triggers: list[str]
  def map_trade(form: dict, live: CategoryAnalysis | None, intraday: CategoryAnalysis | None, futures: FuturesSnapshot) -> TradeMap
  ```
  `form = {"dir": "LONG"|"SHORT", "entry": float, "lev": float, "sl": float, "tp": float}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trade_map.py
import json
import pathlib

from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade

FX = pathlib.Path(__file__).parent / "fixtures"


def _ctx():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    fut = FuturesSnapshot(source="binance", funding_rate=0.0002, funding_7d_mean=0.0001, oi_change_24h_pct=2.0, long_short_ratio=1.2)
    return analyze_category(CATEGORIES["live"], frames, fut), analyze_category(CATEGORIES["intraday"], frames, fut), fut


def test_long_trade_math_and_structures():
    live, intra, fut = _ctx()
    p = live.price
    tm = map_trade({"dir": "LONG", "entry": p, "lev": 10.0, "sl": p * 0.98, "tp": p * 1.04}, live, intra, fut)
    assert abs(tm.liquidation_price - p * (1 - 0.1 + 0.005)) < 1e-6
    assert abs(tm.rr - 2.0) < 1e-6
    assert tm.classification in {"SCALP", "SWING"}
    assert tm.risk_label in {"OPTIMAL", "ELEVATED", "EXTREME"}
    assert isinstance(tm.structures_above, list) and isinstance(tm.structures_below, list)
    assert any("Funding" in t for t in tm.anchor_triggers)


def test_short_liquidation_and_extreme_risk():
    live, intra, fut = _ctx()
    p = live.price
    tm = map_trade({"dir": "SHORT", "entry": p, "lev": 100.0, "sl": p * 1.02, "tp": p * 0.98}, live, intra, fut)
    assert tm.liquidation_price > p and tm.risk_label == "EXTREME"
    assert any("leverage" in w.lower() for w in tm.warnings)


def test_no_analysis_still_returns_math():
    fut = FuturesSnapshot.unavailable("binance,bybit", "451")
    tm = map_trade({"dir": "LONG", "entry": 100.0, "lev": 5.0, "sl": 95.0, "tp": 110.0}, None, None, fut)
    assert tm.rr == 2.0 and tm.structures_above == [] and any("unavailable" in t for t in tm.anchor_triggers)
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `core/indicators/trade_map.py`**

```python
"""Active-trade mapping: scalp vs swing, structure above/below, stop quality, liquidation, re-anchor triggers."""
from __future__ import annotations

from dataclasses import dataclass

from core.data.types import FuturesSnapshot
from core.indicators.category import CategoryAnalysis

MAINT_MARGIN = 0.005


@dataclass
class TradeMap:
    classification: str
    direction: str
    entry: float
    leverage: float
    stop: float
    target: float
    liquidation_price: float
    dist_to_liq_pct: float
    risk_label: str
    rr: float | None
    structures_above: list[str]
    structures_below: list[str]
    stop_structural: bool
    warnings: list[str]
    anchor_triggers: list[str]


def _structures(analysis: CategoryAnalysis | None, tag: str) -> list[tuple[float, str]]:
    if analysis is None:
        return []
    out = []
    for n, v in analysis.ema.values.items():
        if v is not None:
            out.append((v, f"EMA{n} {tag}"))
    for l in analysis.levels:
        out.append((l.price, f"{l.kind.title()} {tag} ({l.touches}x)"))
    return out


def _fmt(s: tuple[float, str]) -> str:
    return f"{s[1]} @ {s[0]:,.0f}"


def map_trade(form: dict, live: CategoryAnalysis | None, intraday: CategoryAnalysis | None, futures: FuturesSnapshot) -> TradeMap:
    d, entry, lev, sl, tp = form["dir"], float(form["entry"]), float(form["lev"]), float(form["sl"]), float(form["tp"])
    price = live.price if live else (intraday.price if intraday else entry)

    if d == "LONG":
        liq = entry * (1 - 1 / lev + MAINT_MARGIN)
        dist = (price - liq) / price * 100
    else:
        liq = entry * (1 + 1 / lev - MAINT_MARGIN)
        dist = (liq - price) / price * 100
    risk = "EXTREME" if dist < 2.0 else "ELEVATED" if dist < 5.0 else "OPTIMAL"
    risk_dist = abs(entry - sl)
    rr = (abs(tp - entry) / risk_dist) if risk_dist > 0 else None

    atr15 = live.vol.atr if live and live.vol.atr else None
    half_range_1h = (intraday.vol.exp_high - intraday.price) if intraday and intraday.vol.exp_high else None
    scalp = (atr15 is not None and abs(entry - price) < 3 * atr15) and (half_range_1h is not None and abs(tp - entry) <= half_range_1h)
    classification = "SCALP" if scalp else "SWING"

    structs = _structures(live, live.chart_tf if live else "") + _structures(intraday, intraday.chart_tf if intraday else "")
    above = sorted([s for s in structs if s[0] > entry], key=lambda s: s[0] - entry)[:4]
    below = sorted([s for s in structs if s[0] < entry], key=lambda s: entry - s[0])[:4]

    if d == "LONG":
        stop_structural = any(sl <= p < entry for p, _ in structs)
    else:
        stop_structural = any(entry < p <= sl for p, _ in structs)

    warnings: list[str] = []
    if atr15 is not None and risk_dist < atr15:
        warnings.append(f"Stop sits inside 15m noise (1xATR = {atr15:,.0f})")
    if lev > 20:
        warnings.append(f"Leverage {lev:.0f}x leaves {dist:.1f}% to liquidation")
    if not stop_structural:
        warnings.append("Stop is not protected by any EMA or swing level")
    if rr is not None and rr < 1.5:
        warnings.append(f"Reward:risk {rr:.2f} below 1.5")
    if (d == "LONG" and sl >= entry) or (d == "SHORT" and sl <= entry):
        warnings.append("Stop is on the wrong side of entry")

    if futures.available:
        fr = futures.funding_rate or 0.0
        oi = futures.oi_change_24h_pct
        ls = futures.long_short_ratio
        triggers = [
            f"Funding flips sign (now {fr:+.4%})",
            f"OI 24h change crosses +/-5% (now {oi:+.1f}%)" if oi is not None else "OI 24h change crosses +/-5%",
            f"Long/short ratio crosses 1.0 (now {ls:.2f})" if ls is not None else "Long/short ratio crosses 1.0",
        ]
    else:
        triggers = ["Funding / OI / long-short triggers inactive (futures data unavailable)"]
    sq = live.squeeze if live else None
    triggers.append(f"Squeeze score reaches 60 (now {sq.score})" if sq and sq.score is not None else "Squeeze score reaches 60")
    triggers.append("Trap classification changes (Director report, Phase 2)")
    triggers.append("Curated macro event inside 24h")

    return TradeMap(classification, d, entry, lev, sl, tp, liq, dist, risk, rr,
                    [_fmt(s) for s in above], [_fmt(s) for s in below], stop_structural, warnings, triggers)
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/indicators/trade_map.py tests/test_trade_map.py
git commit -m "feat(indicators): active trade map with structures and re-anchor triggers"
```

---

### Task 17: Theme + HTML building blocks

**Files:**
- Create: `ui/theme.py`, `.streamlit/config.toml`
- Test: `tests/test_theme.py`

**Interfaces:**
- Produces:
  - `palette(dark: bool) -> dict` keys: `bg, panel, border, grid, text, muted, accent, up, down, warn, fib, fibFuture, now, mono`
  - `inject_css(dark: bool) -> None` (Streamlit side effect)
  - `fmt_num(v, digits=0, prefix="", suffix="") -> str` (None → `—`), `fmt_pct(v, digits=2, signed=True) -> str`
  - `kpi_html(items: list[dict]) -> str`, `kpi_strip(items) -> None`; item = `{"label": str, "value": str, "delta": str|None, "tone": "up"|"down"|"warn"|None}`
  - `card_html(title: str, body_html: str, tone: str = "neutral") -> str`, `card(title, body_html, tone) -> None`
  - `chip(text: str, tone: str = "") -> str`
  - `tone_for_direction(direction: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_theme.py
from ui.theme import card_html, chip, fmt_num, fmt_pct, kpi_html, palette, tone_for_direction


def test_formatters_handle_none_and_numbers():
    assert fmt_num(None) == "—"
    assert fmt_num(79200.456, 0, "$") == "$79,200"
    assert fmt_num(0.1234, 2, suffix="%") == "0.12%"
    assert fmt_pct(1.234) == "+1.23%"
    assert fmt_pct(-0.5, 1) == "-0.5%"
    assert fmt_pct(None) == "—"


def test_kpi_html_escapes_and_renders_all_items():
    h = kpi_html([{"label": "Spot <b>", "value": "$1", "delta": "+1%", "tone": "up"}, {"label": "OI", "value": "—"}])
    assert "&lt;b&gt;" in h and h.count('class="ti-kpi"') == 2 and 'class="d up"' in h


def test_card_and_chip_and_palette():
    assert 'class="ti-card down"' in card_html("Bias", "<p>x</p>", "down")
    assert chip("Kraken", "up").startswith('<span class="ti-chip up">')
    assert palette(True)["bg"] != palette(False)["bg"]
    assert tone_for_direction("BULLISH") == "up" and tone_for_direction("NEUTRAL") == "neutral"
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `.streamlit/config.toml`**

```toml
[theme]
base = "light"
primaryColor = "#0F62FE"
backgroundColor = "#F6F8FB"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0B1220"
font = "sans serif"

[server]
headless = true
runOnSave = false

[browser]
gatherUsageStats = false
```

- [ ] **Step 4: Write `ui/theme.py`**

```python
"""Theme injection and small HTML building blocks. Institutional palette, no emoji icons."""
from __future__ import annotations

import html

import streamlit as st

_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

LIGHT = dict(bg="#F6F8FB", panel="#FFFFFF", border="#E3E8EF", grid="#EEF2F7", text="#0B1220", muted="#5B6472",
             accent="#0F62FE", up="#0E9F6E", down="#E02424", warn="#D97706", fib="#94A3B8", fibFuture="#0F62FE",
             now="#0B1220", mono=_MONO)
DARK = dict(bg="#0B0F17", panel="#111827", border="#1F2937", grid="#161E2E", text="#E5E7EB", muted="#9CA3AF",
            accent="#3B82F6", up="#10B981", down="#EF4444", warn="#F59E0B", fib="#4B5563", fibFuture="#60A5FA",
            now="#E5E7EB", mono=_MONO)


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


def fmt_num(v, digits: int = 0, prefix: str = "", suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{prefix}{v:,.{digits}f}{suffix}"


def fmt_pct(v, digits: int = 2, signed: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def tone_for_direction(direction: str) -> str:
    return {"BULLISH": "up", "BEARISH": "down"}.get(direction, "neutral")


def chip(text: str, tone: str = "") -> str:
    return f'<span class="ti-chip {tone}">{html.escape(text)}</span>'


def kpi_html(items: list[dict]) -> str:
    cells = []
    for i in items:
        delta = i.get("delta")
        tone = i.get("tone") or ""
        d = f'<div class="d {tone}">{html.escape(str(delta))}</div>' if delta else ""
        cells.append(f'<div class="ti-kpi"><div class="l">{html.escape(str(i["label"]))}</div>'
                     f'<div class="v">{html.escape(str(i["value"]))}</div>{d}</div>')
    return f'<div class="ti-kpis">{"".join(cells)}</div>'


def kpi_strip(items: list[dict]) -> None:
    st.markdown(kpi_html(items), unsafe_allow_html=True)


def card_html(title: str, body_html: str, tone: str = "neutral") -> str:
    return f'<div class="ti-card {tone}"><h4>{html.escape(title)}</h4>{body_html}</div>'


def card(title: str, body_html: str, tone: str = "neutral") -> None:
    st.markdown(card_html(title, body_html, tone), unsafe_allow_html=True)


def inject_css(dark: bool) -> None:
    p = palette(dark)
    st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {p['bg']}; color: {p['text']}; }}
header[data-testid="stHeader"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1440px; }}
section[data-testid="stSidebar"] {{ background: {p['panel']}; border-right: 1px solid {p['border']}; }}
section[data-testid="stSidebar"] * {{ color: {p['text']}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {p['border']}; }}
.stTabs [data-baseweb="tab"] {{ padding: 9px 14px; font-weight: 600; color: {p['muted']}; }}
.stTabs [aria-selected="true"] {{ color: {p['accent']}; border-bottom: 2px solid {p['accent']}; }}
h1, h2, h3, h4, p, li, label, .stMarkdown {{ color: {p['text']}; }}
.ti-title {{ font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
.ti-sub {{ color: {p['muted']}; font-size: 0.84rem; margin: 2px 0 10px 0; }}
.ti-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 0 0 12px 0; }}
.ti-kpi {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 10px; padding: 10px 12px; }}
.ti-kpi .l {{ font-size: 0.7rem; color: {p['muted']}; text-transform: uppercase; letter-spacing: 0.05em; }}
.ti-kpi .v {{ font-family: {p['mono']}; font-size: 1.02rem; font-weight: 600; margin-top: 2px; color: {p['text']}; }}
.ti-kpi .d {{ font-size: 0.74rem; margin-top: 2px; color: {p['muted']}; }}
.ti-kpi .d.up, .ti-chip.up {{ color: {p['up']}; }}
.ti-kpi .d.down, .ti-chip.down {{ color: {p['down']}; }}
.ti-kpi .d.warn, .ti-chip.warn {{ color: {p['warn']}; }}
.ti-card {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; border-left: 4px solid {p['muted']}; }}
.ti-card h4 {{ margin: 0 0 8px 0; font-size: 0.92rem; font-weight: 600; }}
.ti-card p, .ti-card li {{ font-size: 0.88rem; line-height: 1.45; margin: 0.15rem 0; color: {p['text']}; }}
.ti-card ul {{ padding-left: 1.1rem; margin: 0.2rem 0; }}
.ti-card .muted {{ color: {p['muted']}; }}
.ti-card.up {{ border-left-color: {p['up']}; }}
.ti-card.down {{ border-left-color: {p['down']}; }}
.ti-card.warn {{ border-left-color: {p['warn']}; }}
.ti-chip {{ display: inline-block; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; border: 1px solid {p['border']}; color: {p['muted']}; margin: 0 6px 6px 0; }}
.ti-chip.up {{ border-color: {p['up']}; }} .ti-chip.down {{ border-color: {p['down']}; }} .ti-chip.warn {{ border-color: {p['warn']}; }}
.ti-mono {{ font-family: {p['mono']}; }}
@media (max-width: 768px) {{ .block-container {{ padding-left: 0.6rem; padding-right: 0.6rem; }} .ti-kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
""", unsafe_allow_html=True)
```

- [ ] **Step 5: Run tests** → 3 passed

- [ ] **Step 6: Commit**

```bash
git add ui/theme.py .streamlit/config.toml tests/test_theme.py
git commit -m "feat(ui): theme, fonts, KPI/card/chip building blocks"
```

---

### Task 18: lightweight-charts renderer

**Files:**
- Create: `ui/charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: `CategoryAnalysis`, `palette`, `LIGHTWEIGHT_CHARTS_URL`, `TF_MINUTES`, `CATEGORIES`.
- Produces:
  - `chart_payload(a: CategoryAnalysis, dark: bool) -> dict` (pure, JSON-serialisable)
  - `build_chart_html(a, dark, height=560) -> str`
  - `render_chart(a, dark, height=560) -> None` (calls `streamlit.components.v1.html`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts.py
import json
import pathlib
import re

from core.config import CATEGORIES, LIGHTWEIGHT_CHARTS_URL
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from ui.charts import build_chart_html, chart_payload

FX = pathlib.Path(__file__).parent / "fixtures"


def _analysis():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    return analyze_category(CATEGORIES["live"], frames, FuturesSnapshot(source="binance", oi_change_24h_pct=1.0))


def test_payload_is_json_clean_and_consistent():
    a = _analysis()
    p = chart_payload(a, dark=False)
    json.dumps(p)  # no NaN / numpy types
    times = [c["time"] for c in p["candles"]]
    assert len(times) <= 300 and times == sorted(times) and times[-1] < 10_000_000_000  # seconds
    assert p["whitespace"] and p["whitespace"][0]["time"] > times[-1]
    assert {e["period"] for e in p["emas"]} <= {21, 50, 100, 200}
    assert all(0 <= r["value"] <= 100 for r in p["rsi"]["data"])
    assert p["now"] == times[-1]
    assert all(len(t["data"]) == 2 for t in p["trendlines"])
    assert all(len(p["projection"][k]) == 2 for k in ("upper", "lower", "mid"))
    assert all(f["time"] % 1 == 0 for f in p["fibs"])


def test_html_embeds_pinned_library_and_payload():
    h = build_chart_html(_analysis(), dark=True, height=500)
    assert LIGHTWEIGHT_CHARTS_URL in h
    m = re.search(r"const PAYLOAD = (\{.*?\});\n", h, re.S)
    assert m, "payload not embedded"
    payload = json.loads(m.group(1))
    assert payload["theme"]["panel"].startswith("#")
    assert "createChart" in h and "attachPrimitive" in h and "createSeriesMarkers" in h
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `ui/charts.py`**

```python
"""Serialise a CategoryAnalysis into a self-contained TradingView lightweight-charts document."""
from __future__ import annotations

import json
import math

import streamlit.components.v1 as components

from core.config import CATEGORIES, LIGHTWEIGHT_CHARTS_URL, TF_MINUTES
from core.indicators.category import CategoryAnalysis
from ui.theme import palette

EMA_COLORS = {21: "#F59E0B", 50: "#3B82F6", 100: "#8B5CF6", 200: "#EF4444"}


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def chart_payload(a: CategoryAnalysis, dark: bool) -> dict:
    p = palette(dark)
    df = a.chart_df
    n = len(df)
    t = (df["timestamp"] // 1000).astype("int64").tolist()
    last_t = t[-1]
    interval = TF_MINUTES[a.chart_tf] * 60
    horizon = CATEGORIES[a.key].horizon_bars

    candles = [{"time": t[i], "open": _f(df["open"].iloc[i]), "high": _f(df["high"].iloc[i]),
                "low": _f(df["low"].iloc[i]), "close": _f(df["close"].iloc[i])} for i in range(n)]

    # how far into the future the time axis must extend
    future_targets = [last_t + max(horizon, 12) * interval]
    if a.fib:
        future_targets += [z.timestamp_ms // 1000 for z in a.fib.upcoming]
    for tl in a.trendlines:
        future_targets.append(tl.points[1][0] // 1000)
    future_n = min(90, max(12, math.ceil((max(future_targets) - last_t) / interval)))
    whitespace = [{"time": last_t + k * interval} for k in range(1, future_n + 1)]

    emas = []
    for period, series in a.ema.series.items():
        vals = series.tail(n).tolist()
        data = [{"time": t[i], "value": _f(vals[i])} for i in range(n) if _f(vals[i]) is not None]
        emas.append({"period": period, "color": EMA_COLORS.get(period, p["muted"]), "data": data})

    levels = [{"price": _f(l.price), "title": f"{'R' if l.kind == 'resistance' else 'S'} {l.touches}x",
               "color": p["down"] if l.kind == "resistance" else p["up"]} for l in a.levels]

    trendlines = [{"kind": tl.kind, "color": p["down"] if tl.kind == "resistance" else p["up"],
                   "data": [{"time": tl.points[0][0] // 1000, "value": _f(tl.points[0][1])},
                            {"time": min(tl.points[1][0] // 1000, last_t + future_n * interval), "value": _f(tl.points[1][1])}]}
                  for tl in a.trendlines]

    fibs = []
    if a.fib:
        first_t = t[0]
        for z in a.fib.zones:
            zt = z.timestamp_ms // 1000
            if zt >= first_t:
                fibs.append({"time": int(zt), "label": f"F{z.k}", "future": z.is_future})

    price = a.price
    proj_end = last_t + min(future_n, max(horizon, 1)) * interval
    hi = _f(a.vol.exp_high) or price
    lo = _f(a.vol.exp_low) or price
    sign = {"BULLISH": 1.0, "BEARISH": -1.0}.get(a.direction.direction, 0.0)
    mid_end = price + sign * a.direction.confidence * ((hi - price) if sign >= 0 else (price - lo))
    projection = {
        "color": p["accent"],
        "upper": [{"time": last_t, "value": price}, {"time": proj_end, "value": hi}],
        "lower": [{"time": last_t, "value": price}, {"time": proj_end, "value": lo}],
        "mid": [{"time": last_t, "value": price}, {"time": proj_end, "value": mid_end}],
    }

    rvals = a.rsi.tolist()
    rsi_data = [{"time": t[i], "value": _f(rvals[i])} for i in range(n) if _f(rvals[i]) is not None]
    markers = []
    for d in a.divergences:
        color = p["up"] if d.direction == "bullish" else p["down"]
        for (ts_ms, _v) in d.rsi_points:
            markers.append({"time": ts_ms // 1000, "position": "belowBar" if d.direction == "bullish" else "aboveBar",
                            "color": color, "shape": "circle", "text": ("~" if d.status == "forming" else "") + d.kind[0].upper() + "D"})
    markers.sort(key=lambda m: m["time"])

    return {
        "theme": {k: p[k] for k in ("panel", "border", "grid", "text", "muted", "accent", "up", "down", "fib", "fibFuture", "now")},
        "title": f"{a.label} · {a.chart_tf}",
        "candles": candles, "whitespace": whitespace, "emas": emas, "levels": levels,
        "trendlines": trendlines, "fibs": fibs, "now": last_t, "projection": projection,
        "rsi": {"data": rsi_data, "markers": markers},
        "visibleBars": 140,
    }


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>html,body{margin:0;padding:0;background:__PANEL__;font-family:Inter,system-ui,sans-serif}#wrap{position:relative}#chart{width:100%;height:__HEIGHT__px}
#legend{position:absolute;left:10px;top:8px;z-index:5;font-size:11px;color:__TEXT__;display:flex;gap:10px;flex-wrap:wrap}
#legend span::before{content:"";display:inline-block;width:10px;height:2px;margin-right:4px;vertical-align:middle;background:var(--c)}</style>
<script src="__LIB__"></script></head><body><div id="wrap"><div id="legend"></div><div id="chart"></div></div>
<script>
const PAYLOAD = __PAYLOAD__;
(function(){
const P = PAYLOAD, T = P.theme, LW = LightweightCharts;
const el = document.getElementById('chart');
const chart = LW.createChart(el, {
  width: el.clientWidth, height: __HEIGHT__,
  layout: { background: { type: 'solid', color: T.panel }, textColor: T.text, fontFamily: 'Inter, system-ui, sans-serif',
            panes: { separatorColor: T.border, separatorHoverColor: T.border, enableResize: false } },
  grid: { vertLines: { color: T.grid }, horzLines: { color: T.grid } },
  rightPriceScale: { borderColor: T.border },
  timeScale: { borderColor: T.border, timeVisible: true, secondsVisible: false, rightOffset: 2 },
  crosshair: { mode: 0 },
});
const candles = chart.addSeries(LW.CandlestickSeries, { upColor: T.up, downColor: T.down, borderVisible: false, wickUpColor: T.up, wickDownColor: T.down });
candles.setData(P.candles.concat(P.whitespace));
const legend = document.getElementById('legend');
P.emas.forEach(e => {
  const s = chart.addSeries(LW.LineSeries, { color: e.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(e.data);
  const tag = document.createElement('span'); tag.style.setProperty('--c', e.color); tag.textContent = 'EMA' + e.period; legend.appendChild(tag);
});
P.levels.forEach(l => candles.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: LW.LineStyle.Dashed, axisLabelVisible: true, title: l.title }));
P.trendlines.forEach(t => {
  const s = chart.addSeries(LW.LineSeries, { color: t.color, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  s.setData(t.data);
});
['upper', 'lower', 'mid'].forEach(k => {
  const s = chart.addSeries(LW.LineSeries, { color: P.projection.color, lineWidth: k === 'mid' ? 2 : 1, lineStyle: LW.LineStyle.Dashed,
    priceLineVisible: false, lastValueVisible: k !== 'mid', crosshairMarkerVisible: false, title: k === 'mid' ? 'projection' : '' });
  s.setData(P.projection[k]);
});
const rsi = chart.addSeries(LW.LineSeries, { color: T.accent, lineWidth: 1, priceLineVisible: false, lastValueVisible: true, title: 'RSI 14' }, 1);
rsi.setData(P.rsi.data);
[70, 30].forEach(v => rsi.createPriceLine({ price: v, color: T.muted, lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: false }));
if (P.rsi.markers.length) LW.createSeriesMarkers(rsi, P.rsi.markers);
const panes = chart.panes(); if (panes[1]) panes[1].setHeight(Math.round(__HEIGHT__ * 0.24));

class VLineRenderer {
  constructor(x, color, dash, label) { this._x = x; this._color = color; this._dash = dash; this._label = label; }
  draw(target) {
    target.useBitmapCoordinateSpace(s => {
      if (this._x === null || this._x === undefined) return;
      const ctx = s.context, hr = s.horizontalPixelRatio, vr = s.verticalPixelRatio;
      const x = Math.round(this._x * hr);
      ctx.save();
      ctx.strokeStyle = this._color; ctx.lineWidth = Math.max(1, Math.floor(hr));
      if (this._dash) ctx.setLineDash([4 * hr, 4 * hr]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, s.bitmapSize.height); ctx.stroke();
      if (this._label) { ctx.setLineDash([]); ctx.fillStyle = this._color; ctx.font = `${11 * vr}px Inter, sans-serif`; ctx.fillText(this._label, x + 4 * hr, 12 * vr); }
      ctx.restore();
    });
  }
}
class VLineView {
  constructor(src) { this._src = src; this._x = null; }
  update() { this._x = this._src._chart.timeScale().timeToCoordinate(this._src._time); }
  renderer() { return new VLineRenderer(this._x, this._src._color, this._src._dash, this._src._label); }
}
class VLine {
  constructor(chart, time, color, dash, label) { this._chart = chart; this._time = time; this._color = color; this._dash = dash; this._label = label; this._views = [new VLineView(this)]; }
  updateAllViews() { this._views.forEach(v => v.update()); }
  paneViews() { return this._views; }
}
P.fibs.forEach(f => candles.attachPrimitive(new VLine(chart, f.time, f.future ? T.fibFuture : T.fib, true, f.label)));
candles.attachPrimitive(new VLine(chart, P.now, T.now, false, 'now'));

const total = P.candles.length + P.whitespace.length;
chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, P.candles.length - P.visibleBars), to: total });
new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
})();
</script></body></html>"""


def build_chart_html(a: CategoryAnalysis, dark: bool, height: int = 560) -> str:
    p = palette(dark)
    payload = json.dumps(chart_payload(a, dark), separators=(",", ":"))
    return (_TEMPLATE.replace("__LIB__", LIGHTWEIGHT_CHARTS_URL).replace("__HEIGHT__", str(height))
            .replace("__PANEL__", p["panel"]).replace("__TEXT__", p["text"]).replace("__PAYLOAD__", payload))


def render_chart(a: CategoryAnalysis, dark: bool, height: int = 560) -> None:
    components.html(build_chart_html(a, dark, height), height=height + 6, scrolling=False)
```

- [ ] **Step 4: Run tests** → 2 passed

- [ ] **Step 5: Visual check (one-off)**

```bash
.venv/Scripts/python - <<'EOF'
import json, pathlib
from core.config import CATEGORIES
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot
from core.indicators.category import analyze_category
from ui.charts import build_chart_html
FX = pathlib.Path("tests/fixtures")
frames = {"15m": parse_ohlc(json.loads((FX/"kraken_ohlc_15m.json").read_text())), "1h": parse_ohlc(json.loads((FX/"kraken_ohlc_1h.json").read_text()))}
a = analyze_category(CATEGORIES["live"], frames, FuturesSnapshot(source="binance", oi_change_24h_pct=1.0))
pathlib.Path("chart_preview.html").write_text(build_chart_html(a, False), encoding="utf-8")
print("wrote chart_preview.html")
EOF
```
Open `chart_preview.html` in a browser (or Playwright `file://` URL) and confirm: candles, four EMA lines with legend, dashed S/R price lines with labels, dotted trendlines, vertical fib lines with `F..` labels (future ones in accent colour), a solid `now` line, dashed projection cone, RSI pane with 70/30 and divergence markers. Delete `chart_preview.html` afterwards (do not commit it).

- [ ] **Step 6: Commit**

```bash
git add ui/charts.py tests/test_charts.py
git commit -m "feat(ui): lightweight-charts renderer with EMAs, S/R, trendlines, fib time, RSI pane"
```

---

### Task 19: Panels

**Files:**
- Create: `ui/panels.py`
- Test: `tests/test_panels.py`

**Interfaces:**
- Consumes: `CategoryAnalysis`, `MarketSnapshot`, `TradeMap`, theme helpers.
- Produces (all pure `*_html` builders return `str`; the `render_*` twins call `st.markdown`):
  - `kpis_for(a: CategoryAnalysis, m: MarketSnapshot) -> list[dict]`
  - `direction_html(a) -> str`, `divergence_html(a) -> str`, `volatility_html(a) -> str`, `agent_placeholder_html(a) -> str`
  - `squeeze_banner_html(a) -> str | None` (only when score ≥ 60)
  - `trade_result_html(tm: TradeMap) -> str`
  - `macro_html(events: list[dict]) -> str`
  - `data_status_html(m: MarketSnapshot) -> str`
  - `raw_metrics_rows(m: MarketSnapshot) -> list[tuple[str, str, str]]` (metric, value, source) — Section 1 of the future report
  - `render(html: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panels.py
import json
import pathlib

from core.config import CATEGORIES, MACRO_EVENTS
from core.data.kraken_spot import parse_ohlc
from core.data.types import FuturesSnapshot, MarketSnapshot, OptionsSnapshot, SentimentSnapshot, SpotSnapshot
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade
from ui import panels

FX = pathlib.Path(__file__).parent / "fixtures"


def _market():
    frames = {"15m": parse_ohlc(json.loads((FX / "kraken_ohlc_15m.json").read_text())),
              "1h": parse_ohlc(json.loads((FX / "kraken_ohlc_1h.json").read_text()))}
    return MarketSnapshot(
        spot=SpotSnapshot(source="kraken", last=float(frames["1h"]["close"].iloc[-1]), frames=frames),
        futures=FuturesSnapshot(source="binance", funding_rate=0.0006, funding_7d_mean=0.0001, open_interest=100000, open_interest_usd=8e9,
                                oi_change_24h_pct=7.0, long_short_ratio=1.7, taker_buy_sell_ratio=0.9),
        options=OptionsSnapshot(source="deribit", max_pain=80000, max_pain_expiry="2026-09-25", put_call_oi_ratio=0.6, iv_atm=55.0, iv_skew=3.0),
        sentiment=SentimentSnapshot.unavailable("alternative.me", "timeout"),
    )


def test_kpis_cover_required_fields_with_dash_for_missing():
    m = _market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    labels = [k["label"] for k in panels.kpis_for(a, m)]
    for req in ("Price", "Direction", "RSI 14", "ATR %", "Funding", "OI 24h", "Long/Short", "Fear & Greed"):
        assert req in labels
    fg = next(k for k in panels.kpis_for(a, m) if k["label"] == "Fear & Greed")
    assert fg["value"] == "—"


def test_panels_render_strings_and_squeeze_banner():
    m = _market()
    a = analyze_category(CATEGORIES["weekly"], {**m.spot.frames, "4h": m.spot.frames["1h"], "1d": m.spot.frames["1h"]}, m.futures)
    assert "ti-card" in panels.direction_html(a)
    assert "RSI" in panels.divergence_html(a)
    assert a.vol.horizon_label in panels.volatility_html(a)
    assert "Run Analysis" in panels.agent_placeholder_html(a)
    banner = panels.squeeze_banner_html(a)
    assert banner and "SQUEEZE" in banner


def test_trade_result_macro_and_status():
    m = _market()
    live = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    tm = map_trade({"dir": "LONG", "entry": live.price, "lev": 10.0, "sl": live.price * 0.98, "tp": live.price * 1.04}, live, None, m.futures)
    h = panels.trade_result_html(tm)
    assert tm.classification in h and "Liquidation" in h
    assert MACRO_EVENTS[0]["title"] in panels.macro_html(MACRO_EVENTS)
    status = panels.data_status_html(m)
    assert "unavailable" in status and "kraken" in status
    rows = panels.raw_metrics_rows(m)
    assert any(r[0] == "Max pain" and "80,000" in r[1] for r in rows)
```

- [ ] **Step 2: Run test to verify it fails** → `ModuleNotFoundError`

- [ ] **Step 3: Write `ui/panels.py`**

```python
"""Panel builders: pure HTML functions plus thin Streamlit renderers."""
from __future__ import annotations

import html

import streamlit as st

from core.data.types import MarketSnapshot
from core.indicators.category import CategoryAnalysis
from core.indicators.trade_map import TradeMap
from ui.theme import card_html, chip, fmt_num, fmt_pct, tone_for_direction


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _li(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>" if items else ""


def kpis_for(a: CategoryAnalysis, m: MarketSnapshot) -> list[dict]:
    f, s, o = m.futures, m.sentiment, m.options
    rsi_last = float(a.rsi.dropna().iloc[-1]) if a.rsi.notna().any() else None
    pc = a.price_change_24h_pct
    return [
        {"label": "Price", "value": fmt_num(a.price, 0, "$"), "delta": (fmt_pct(pc) + " 24h") if pc is not None else None,
         "tone": "up" if (pc or 0) >= 0 else "down"},
        {"label": "Direction", "value": a.direction.direction, "delta": f"confidence {a.direction.confidence:.0%}", "tone": tone_for_direction(a.direction.direction)},
        {"label": "RSI 14", "value": fmt_num(rsi_last, 0), "delta": a.ema.state + " EMA stack", "tone": {"BULL": "up", "BEAR": "down"}.get(a.ema.state)},
        {"label": "ATR %", "value": fmt_num(a.vol.atr_pct, 2, suffix="%"), "delta": a.vol.regime + " regime", "tone": "warn" if a.vol.regime == "HIGH" else None},
        {"label": "Funding", "value": fmt_pct(f.funding_rate * 100, 4) if f.funding_rate is not None else "—",
         "delta": (f"7d {fmt_pct(f.funding_7d_mean * 100, 4)}" if f.funding_7d_mean is not None else None), "tone": None},
        {"label": "OI 24h", "value": fmt_pct(f.oi_change_24h_pct, 1), "delta": (fmt_num(f.open_interest_usd / 1e9, 2, "$", "B") if f.open_interest_usd else None),
         "tone": "up" if (f.oi_change_24h_pct or 0) >= 0 else "down"},
        {"label": "Long/Short", "value": fmt_num(f.long_short_ratio, 2), "delta": (f"taker {fmt_num(f.taker_buy_sell_ratio, 2)}" if f.taker_buy_sell_ratio else None), "tone": None},
        {"label": "Max pain", "value": fmt_num(o.max_pain, 0, "$"), "delta": o.max_pain_expiry, "tone": None},
        {"label": "Fear & Greed", "value": fmt_num(s.fear_greed, 0), "delta": s.classification, "tone": None},
    ]


def direction_html(a: CategoryAnalysis) -> str:
    d = a.direction
    body = f"<p><strong>{d.direction}</strong> · confidence {d.confidence:.0%} · {a.chart_tf} chart, {a.context_tf} context"
    if a.ctx_ema:
        body += f" ({a.ctx_ema.state} on {a.context_tf})"
    body += "</p>"
    body += f"<p>Anchor {fmt_num(d.anchor, 0, '$')} · Invalidation {fmt_num(d.invalidation, 0, '$')}</p>"
    body += f"<p class='muted'>Flips if: {html.escape(d.flips_if)}</p>"
    body += _li(d.drivers)
    return card_html("Market direction", body, tone_for_direction(d.direction))


def divergence_html(a: CategoryAnalysis) -> str:
    if not a.divergences:
        return card_html("RSI divergence", "<p class='muted'>No RSI divergence in the last 60 bars.</p>")
    rows = []
    tone = "neutral"
    for d in a.divergences:
        p1, p2 = d.price_points
        rows.append(f"{d.label}: price {p1[1]:,.0f} → {p2[1]:,.0f}, RSI {d.rsi_points[0][1]:.0f} → {d.rsi_points[1][1]:.0f}")
        if d.status == "forming":
            tone = "warn"
        elif tone == "neutral":
            tone = "up" if d.direction == "bullish" else "down"
    return card_html("RSI divergence", _li(rows), tone)


def volatility_html(a: CategoryAnalysis) -> str:
    v, s = a.vol, a.squeeze
    body = f"<p>Regime <strong>{v.regime}</strong> · ATR {fmt_num(v.atr, 0, '$')} ({fmt_num(v.atr_pct, 2, suffix='%')})</p>"
    body += f"<p>Expected range {v.horizon_label}: {fmt_num(v.exp_low, 0, '$')} – {fmt_num(v.exp_high, 0, '$')} (±{fmt_num(v.sigma_pct, 2, suffix='%')})</p>"
    if s.score is None:
        body += "<p class='muted'>Squeeze risk: futures data unavailable.</p>"
    else:
        body += f"<p>Squeeze risk <strong>{s.score}/100</strong> · {s.direction.replace('_', ' ')}</p>" + _li(s.drivers)
    tone = "warn" if (s.score or 0) >= 60 or v.regime == "HIGH" else "neutral"
    return card_html("Volatility & positioning", body, tone)


def squeeze_banner_html(a: CategoryAnalysis) -> str | None:
    s = a.squeeze
    if s.score is None or s.score < 60:
        return None
    return card_html(f"Early warning: {s.direction.replace('_', ' ')} risk {s.score}/100", _li(s.drivers), "warn")


def agent_placeholder_html(a: CategoryAnalysis) -> str:
    return card_html("Head-agent summary", "<p class='muted'>Run Analysis (Phase 2) generates the Director's per-category summary here. "
                                            "Until then the deterministic direction, divergence and volatility panels above are the source of truth.</p>")


def trade_result_html(tm: TradeMap) -> str:
    tone = {"OPTIMAL": "up", "ELEVATED": "warn", "EXTREME": "down"}[tm.risk_label]
    body = (f"<p><strong>{tm.classification}</strong> {tm.direction} · entry {fmt_num(tm.entry, 0, '$')} · {tm.leverage:.0f}x · "
            f"stop {fmt_num(tm.stop, 0, '$')} · target {fmt_num(tm.target, 0, '$')} · R:R {fmt_num(tm.rr, 2)}</p>")
    body += f"<p>Liquidation {fmt_num(tm.liquidation_price, 0, '$')} ({tm.dist_to_liq_pct:.1f}% away) · risk <strong>{tm.risk_label}</strong> · "
    body += ("stop protected by structure" if tm.stop_structural else "stop not protected by structure") + "</p>"
    if tm.structures_above:
        body += "<p>Structure above</p>" + _li(tm.structures_above)
    if tm.structures_below:
        body += "<p>Structure below</p>" + _li(tm.structures_below)
    if tm.warnings:
        body += "<p>Warnings</p>" + _li(tm.warnings)
    body += "<p>Verdict stays anchored until</p>" + _li(tm.anchor_triggers)
    return card_html("Active trade stress-test", body, tone)


def macro_html(events: list[dict]) -> str:
    rows = [f"{e['date_utc']} · {e['title']} · {e['impact']} impact. {e['note']}" for e in events]
    return card_html("Macro calendar (curated)", _li(rows) + "<p class='muted'>Manually maintained until a news source is connected.</p>")


def data_status_html(m: MarketSnapshot) -> str:
    out = []
    for name in ("spot", "futures", "options", "sentiment"):
        s = getattr(m, name)
        label = f"{name}: {s.source}" + ("" if s.available else " unavailable")
        out.append(chip(label, "up" if s.available else "down"))
    return "".join(out)


def raw_metrics_rows(m: MarketSnapshot) -> list[tuple[str, str, str]]:
    f, o, s = m.futures, m.options, m.sentiment
    return [
        ("Spot", fmt_num(m.spot.last, 0, "$"), m.spot.source),
        ("Funding (8h)", fmt_pct(f.funding_rate * 100, 4) if f.funding_rate is not None else "—", f.source),
        ("Funding 7d mean", fmt_pct(f.funding_7d_mean * 100, 4) if f.funding_7d_mean is not None else "—", f.source),
        ("Open interest", fmt_num(f.open_interest, 0, suffix=" BTC"), f.source),
        ("OI 24h change", fmt_pct(f.oi_change_24h_pct, 1), f.source),
        ("Long/short ratio", fmt_num(f.long_short_ratio, 2), f.source),
        ("Taker buy/sell", fmt_num(f.taker_buy_sell_ratio, 2), f.source),
        ("Max pain", fmt_num(o.max_pain, 0, "$") + (f" ({o.max_pain_expiry})" if o.max_pain_expiry else ""), o.source),
        ("Put/call OI", fmt_num(o.put_call_oi_ratio, 2), o.source),
        ("ATM IV", fmt_num(o.iv_atm, 1, suffix="%"), o.source),
        ("IV skew (put − call)", fmt_num(o.iv_skew, 1, suffix=" pts"), o.source),
        ("Fear & Greed", (fmt_num(s.fear_greed, 0) + (f" {s.classification}" if s.classification else "")), s.source),
    ]
```

- [ ] **Step 4: Run tests** → 3 passed

- [ ] **Step 5: Commit**

```bash
git add ui/panels.py tests/test_panels.py
git commit -m "feat(ui): direction, divergence, volatility, trade, macro, status panels"
```

---

### Task 20: New `app.py` shell

**Files:**
- Modify: `app.py` (full replacement)

**Interfaces:**
- Consumes everything above. Session keys: `dark: bool`, `audit: list[dict]`, `last_dir: dict[str, str]`.

- [ ] **Step 1: Replace `app.py`**

```python
"""Trap Intelligence — BTC market-surveillance terminal (Phase 1: deterministic layer)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.config import CATEGORIES, MACRO_EVENTS, TTL
from core.data.market import load_market
from core.indicators.category import analyze_category
from core.indicators.trade_map import map_trade
from ui import panels
from ui.charts import render_chart
from ui.theme import inject_css, kpi_strip

st.set_page_config(page_title="Trap Intelligence | BTC", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _market():
    return load_market()


# ---------- sidebar ----------
st.session_state.setdefault("dark", False)
st.session_state.setdefault("audit", [])
st.session_state.setdefault("last_dir", {})

st.sidebar.markdown("<p class='ti-title'>Trap Intelligence</p><p class='ti-sub'>BTC market surveillance</p>", unsafe_allow_html=True)
dark = st.sidebar.toggle("Dark mode", value=st.session_state["dark"])
st.session_state["dark"] = dark
inject_css(dark)

if st.sidebar.button("Refresh data", use_container_width=True):
    _market.clear()
    st.rerun()
st.sidebar.button("Run Analysis", use_container_width=True, disabled=True,
                  help="The 13-agent Trap Intelligence report is delivered in Phase 2.")

with st.spinner("Loading market data"):
    m = _market()

st.sidebar.markdown("---")
st.sidebar.markdown(panels.data_status_html(m), unsafe_allow_html=True)
st.sidebar.caption(f"Updated {m.generated_at.astimezone(timezone.utc):%H:%M:%S} UTC · spot cache {TTL['spot']}s")

# ---------- analyses ----------
analyses = {}
if m.spot.available:
    for key, cat in CATEGORIES.items():
        a = analyze_category(cat, m.spot.frames, m.futures)
        if a is not None:
            analyses[key] = a
            prev = st.session_state["last_dir"].get(key)
            if prev and prev != a.direction.direction:
                st.session_state["audit"].append({
                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "category": cat.label, "from": prev, "to": a.direction.direction,
                    "price": round(a.price, 2), "reason": "; ".join(a.direction.drivers[:2]),
                })
            st.session_state["last_dir"][key] = a.direction.direction

# ---------- header ----------
st.markdown("<p class='ti-title'>BTC / USD · Trap Intelligence Terminal</p>"
            "<p class='ti-sub'>Deterministic market structure per timeframe. Director report and desk briefs arrive with Run Analysis (Phase 2).</p>",
            unsafe_allow_html=True)
if not m.spot.available:
    st.error(f"Spot data unavailable from {m.spot.source}: {m.spot.error}. Nothing to analyse.")

tab_labels = [c.label for c in CATEGORIES.values()] + ["Active Trade", "War Room"]
tabs = st.tabs(tab_labels)


def category_tab(tab, key):
    with tab:
        a = analyses.get(key)
        if a is None:
            st.info(f"Not enough {CATEGORIES[key].chart_tf} history to analyse this category.")
            return
        kpi_strip(panels.kpis_for(a, m))
        banner = panels.squeeze_banner_html(a) if key in ("weekly", "monthly") else None
        if banner:
            panels.render(banner)
        render_chart(a, dark)
        c1, c2 = st.columns(2)
        with c1:
            panels.render(panels.direction_html(a))
            panels.render(panels.volatility_html(a))
        with c2:
            panels.render(panels.divergence_html(a))
            panels.render(panels.agent_placeholder_html(a))
        if key in ("weekly", "monthly"):
            panels.render(panels.macro_html(MACRO_EVENTS))


for tab, key in zip(tabs[:4], CATEGORIES.keys()):
    category_tab(tab, key)

# ---------- active trade ----------
with tabs[4]:
    live, intraday = analyses.get("live"), analyses.get("intraday")
    spot = m.spot.last or (live.price if live else 0.0)
    with st.form("trade"):
        c1, c2, c3, c4, c5 = st.columns(5)
        t_dir = c1.selectbox("Direction", ["LONG", "SHORT"])
        t_entry = c2.number_input("Entry ($)", value=float(round(spot, 2)), step=10.0)
        t_lev = c3.number_input("Leverage (x)", min_value=1.0, max_value=125.0, value=10.0, step=1.0)
        t_sl = c4.number_input("Stop loss ($)", value=float(round(spot * 0.98, 2)), step=10.0)
        t_tp = c5.number_input("Take profit ($)", value=float(round(spot * 1.04, 2)), step=10.0)
        go = st.form_submit_button("Run stress-test", use_container_width=True)
    if go:
        tm = map_trade({"dir": t_dir, "entry": t_entry, "lev": t_lev, "sl": t_sl, "tp": t_tp}, live, intraday, m.futures)
        panels.render(panels.trade_result_html(tm))
    else:
        st.caption("Enter the position and run the stress-test. Structures come from the 15m and 1h EMAs and swing levels.")

# ---------- war room ----------
with tabs[5]:
    panels.render("<div class='ti-card'><h4>Trap Intelligence Report</h4><p class='muted'>Phase 2 wires the Director, the Bullish and Bearish desks, "
                  "the volatility agent and the accuracy agent to the Run Analysis button. Section 1 of that report, the raw metric snapshot, is live below.</p></div>")
    rows = panels.raw_metrics_rows(m)
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value", "Source"]), use_container_width=True, hide_index=True)
    with st.expander("Audit ledger — direction changes this session"):
        if st.session_state["audit"]:
            st.dataframe(pd.DataFrame(st.session_state["audit"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No direction changes recorded yet in this session.")
```

- [ ] **Step 2: Run the full test suite** → all green (`.venv/Scripts/python -m pytest`)

- [ ] **Step 3: Boot the app**

```bash
.venv/Scripts/python -m streamlit run app.py --server.headless true --server.port 8501 > streamlit.log 2>&1 &
sleep 15; curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8501; grep -iE "traceback|error" streamlit.log || echo "log clean"
```
Expected: `HTTP 200` and `log clean`. If `st.cache_data` cannot pickle `MarketSnapshot`, switch the decorator to `@st.cache_resource(ttl=TTL["spot"])` and note it in the commit message.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): new shell — category tabs with charts and panels, inline active trade, war room snapshot"
```

---

### Task 21: Browser verification, deprecation sweep, README

**Files:**
- Modify: `README.md`, any file flagged by the log

- [ ] **Step 1: Screenshot every tab with Playwright**

Navigate to `http://localhost:8501`, wait 20 s, then for each tab label (`Live Recon`, `Intraday`, `Weekly`, `Monthly`, `Active Trade`, `War Room`) click it and take a viewport screenshot. Check:
- KPI strip shows nine tiles, `—` where a source is unavailable.
- Chart renders candles, EMA legend, S/R labels, fib verticals, `now` line, projection cone, RSI pane.
- Direction / divergence / volatility cards populated; no raw `None` text anywhere.
- Active Trade: submit the default form → result card appears in the same tab.
- War Room: raw metric table with sources; audit expander present.
- Toggle dark mode → page and chart both switch.

- [ ] **Step 2: Sweep warnings**

`grep -n "use_container_width" app.py ui/*.py` — Streamlit ≥1.50 deprecates it; if the running version's log warns, change to `width="stretch"`. Re-check `streamlit.log` shows no warnings after a reload.

- [ ] **Step 3: README**

```markdown
# Trap Intelligence — BTC market-surveillance terminal

Streamlit dashboard with TradingView-style charts and deterministic market structure per timeframe (Live 15m, Intraday 1h, Weekly 4h, Monthly 1d): EMA 21/50/100/200, RSI with divergence detection, swing support/resistance, trendlines, Fibonacci time zones, volatility regime and expected range, squeeze risk, direction call with anchor and invalidation, and an active-trade stress-test. Phase 2 adds the 13-agent Director / Bullish desk / Bearish desk report via OpenRouter.

## Data (keyless public APIs)
Kraken spot OHLCV · Binance USD-M futures (Bybit fallback) for funding, open interest, long/short, taker flow · Deribit options for max pain, put/call, IV skew · alternative.me Fear & Greed. Every feed reports availability; missing data is shown as `—`, never invented.

## Run
    python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
    .venv/Scripts/python -m pytest
    .venv/Scripts/python -m streamlit run app.py

## Layout
`core/data` providers → `core/indicators` pure functions → `ui/` renderers → `app.py` shell. Specs and plans under `docs/superpowers/`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md app.py ui
git commit -m "docs: README for phase 1; fix streamlit deprecations"
```

---

### Task 22: Push and open the PR

- [ ] **Step 1: Final checks**

```bash
.venv/Scripts/python -m pytest
git status --short   # must be clean; chart_preview.html and streamlit.log must NOT be tracked
git log --oneline origin/main..HEAD
```

- [ ] **Step 2: Push**

```bash
git push -u origin feature/trap-intelligence-phase1
```

- [ ] **Step 3: Open PR** (with `gh` if authenticated, otherwise via the compare URL GitHub prints)

Title: `Phase 1: real data providers, indicators, TradingView charts, UI overhaul`

Body:
```
## What
- Splits the single-file app into core/data, core/indicators, ui packages.
- Real keyless feeds: Kraken spot, Binance futures (Bybit fallback), Deribit options, Fear & Greed. Every feed carries an availability flag; missing values render as "—".
- Deterministic per-category analysis: EMA 21/50/100/200, RSI + divergence (confirmed/forming), swing S/R, trendlines, fib time zones, volatility regime + expected range, squeeze risk, direction call with anchor/invalidation.
- TradingView lightweight-charts per category with EMAs, S/R, trendlines, fib verticals, now-line, projection cone, RSI pane.
- Active Trade stress-test inline with scalp/swing classification, structures above/below, liquidation, re-anchor triggers.
- War Room shows Section 1 raw metric snapshot; Run Analysis button present but disabled until Phase 2.

## Removed
- Hard-coded funding/open-interest constants, fixed liquidation clusters, random "predicted candles".

## Test
- `pytest` (unit tests on indicators + recorded API fixtures, no network).
- Manual: all six tabs screenshot-verified light and dark.

## Notes for Streamlit Cloud
- Python 3.12+ in app settings. No secrets needed for Phase 1.
- Binance is geo-blocked from US egress; the app falls back to Bybit automatically.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0186foiQoKUv1nKDTFvRPiJ5
```

- [ ] **Step 4: Report** the PR URL and the list of anything skipped or unverified.
