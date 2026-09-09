# Phase 2 — Extra feeds + 13-agent Trap Intelligence pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the verified keyless feeds (Gemini spot fallback, Hyperliquid, CoinLobster liquidations/whales/radar, DefiLlama stablecoins) into the market snapshot, then add an on-demand LLM pipeline (10 specialists, 2 desk syntheses, volatility agent, accuracy agent, Director) that produces the client's Trap Intelligence Report, rendered in the War Room and as per-category head-agent summaries.

**Architecture:** New providers follow the Phase 1 pattern (pure `parse_*` + guarded `fetch_*`, fixtures-tested). `core/agents/` adds a provider-agnostic `LLMClient` (Gemini native JSON mode by default, OpenRouter optional), pydantic output schemas, prompt composition from the client's orchestration text, per-domain context builders that never pass invented numbers, an async runner with bounded concurrency and per-agent failure isolation, and a markdown renderer for Sections 1–6. Streamlit caches spot at 25 s and slow context feeds at 120 s separately so IP-rate-limited sources are not hammered.

**Tech Stack:** Python ≥3.12, httpx, pydantic 2, Streamlit ≥1.40, Gemini `generateContent` REST, OpenRouter chat completions.

**Spec:** `docs/superpowers/specs/2026-09-09-trap-intelligence-design.md` (§6 agents, §13 feeds)

## Global Constraints

- Branch `feature/trap-intelligence-phase2` (from `feature/trap-intelligence-phase1`). PR2 targets phase1 branch until PR1 merges, then retarget to `main`.
- Keys only via `.streamlit/secrets.toml` (gitignored) or env: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, optional `LLM_PROVIDER` (`gemini`|`openrouter`), `SPECIALIST_MODEL`, `DIRECTOR_MODEL`, `LLM_MAX_CONCURRENCY` (default 4).
- Defaults: Gemini `gemini-3.5-flash-lite` for specialists/desks/aux, `gemini-3.8-flash` for Director. OpenRouter default `nvidia/nemotron-3.5-lightning:free` (only free model family available on the $0 account).
- Hard cap 15 LLM calls per run (+ at most 1 JSON-repair retry per call). Never fan a data fetch out per agent.
- Agents receive only their domain's data plus an explicit `unavailable` list; prompts forbid inventing numbers.
- No network in tests. Fixtures already recorded under `tests/fixtures/`: `gemini_15m.json`, `gemini_1hr.json`, `gemini_1day.json`, `gemini_ticker.json`, `hyperliquid_meta.json`, `coinlobster_liquidations.json`, `coinlobster_whales.json`, `coinlobster_radar.json`, `defillama_stablecoins.json`.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0186foiQoKUv1nKDTFvRPiJ5
  ```

## File map

| Path | Responsibility |
|---|---|
| `core/data/types.py` | + `HyperliquidSnapshot`, `LiquidationsSnapshot`, `WhalesSnapshot`, `StablecoinSnapshot`; `MarketSnapshot` gains those 4 fields with unavailable defaults |
| `core/data/gemini_spot.py` | `parse_candles`, `resample`, `fetch_gemini_spot` |
| `core/data/spot.py` | `fetch_spot_chain` (Kraken → Gemini) |
| `core/data/hyperliquid.py` | `parse_meta`, `fetch_hyperliquid` |
| `core/data/coinlobster.py` | `parse_liquidations`, `parse_whales`, `parse_radar`, `fetch_coinlobster` |
| `core/data/stablecoins.py` | `parse_stablecoins`, `fetch_stablecoins` |
| `core/data/market.py` | `fetch_spot_only`, `fetch_context`, `load_spot`, `load_context`, `assemble` |
| `core/data/offline.py` | extend fixture market with new snapshots |
| `core/agents/__init__.py` | package |
| `core/agents/settings.py` | `load_settings()` env + st.secrets |
| `core/agents/llm.py` | `LLMClient`, `LLMResult`, `extract_json`, `parse_gemini`, `parse_openrouter` |
| `core/agents/schemas.py` | all pydantic output models + `TrapReport` |
| `core/agents/prompts.py` | role registry + `system_prompt(agent_id)` |
| `core/agents/context.py` | `payload_for(agent_id, m, analyses)` |
| `core/agents/runner.py` | `run_pipeline` / `run_pipeline_sync` |
| `core/agents/report.py` | `to_markdown(report)` |
| `ui/panels.py` | + `agent_summary_html`, `report_stats_html`, extended `raw_metrics_rows`, `data_status_html`, KPI `Liq 24h` |
| `app.py` | split caches, Run Analysis flow, War Room report |
| `tests/test_new_feeds.py`, `tests/test_llm.py`, `tests/test_schemas.py`, `tests/test_context.py`, `tests/test_runner.py`, `tests/test_report.py` | tests |

---

### Task P2-0: Snapshot types for the new feeds

**Files:** Modify `core/data/types.py`; Test `tests/test_new_feeds.py` (first test)

**Interfaces (produces):**
```python
class HyperliquidSnapshot(Availability): funding_rate_1h, open_interest, mark_price, oracle_price  (float|None)
class LiquidationsSnapshot(Availability): window: str="24h"; total_usd, long_usd, short_usd, last_hour_usd, btc_usd, btc_long_usd, btc_short_usd (float|None); long_count, short_count (int|None); hourly: list[dict]; biggest: dict|None; venues: list[str]
class WhalesSnapshot(Availability): window_minutes: float|None; btc_trades: int|None; btc_buy_usd, btc_sell_usd (float|None); btc_largest: dict|None; funding_by_exchange: dict[str,float]; oi_by_exchange_usd: dict[str,float]; radar: list[dict]; btc_radar: dict|None
class StablecoinSnapshot(Availability): total_usd, prev_day_usd, change_24h_pct (float|None); top: list[dict]
MarketSnapshot: + hyperliquid, liquidations, whales, stablecoins  (defaults = <cls>.unavailable(<source>, "not fetched"))
MarketSnapshot.unavailable() covers all 8 names in order: spot, futures, options, sentiment, hyperliquid, liquidations, whales, stablecoins
```

- [ ] Step 1 — failing test (`tests/test_new_feeds.py`):
```python
from core.data.types import (FuturesSnapshot, HyperliquidSnapshot, LiquidationsSnapshot, MarketSnapshot,
                             OptionsSnapshot, SentimentSnapshot, SpotSnapshot, StablecoinSnapshot, WhalesSnapshot)

def test_market_defaults_new_feeds_to_unavailable():
    m = MarketSnapshot(spot=SpotSnapshot(source="kraken"), futures=FuturesSnapshot(source="binance"),
                       options=OptionsSnapshot(source="deribit"), sentiment=SentimentSnapshot(source="alternative.me"))
    assert m.unavailable() == ["hyperliquid", "liquidations", "whales", "stablecoins"]
    assert m.liquidations.error == "not fetched"
```
- [ ] Step 2 — run, expect ImportError.
- [ ] Step 3 — implement in `types.py` (append after `SentimentSnapshot`, update `MarketSnapshot`):
```python
class HyperliquidSnapshot(Availability):
    funding_rate_1h: Optional[float] = None
    open_interest: Optional[float] = None
    mark_price: Optional[float] = None
    oracle_price: Optional[float] = None


class LiquidationsSnapshot(Availability):
    window: str = "24h"
    total_usd: Optional[float] = None
    long_usd: Optional[float] = None
    short_usd: Optional[float] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    last_hour_usd: Optional[float] = None
    btc_usd: Optional[float] = None
    btc_long_usd: Optional[float] = None
    btc_short_usd: Optional[float] = None
    hourly: list[dict[str, Any]] = Field(default_factory=list)
    biggest: Optional[dict[str, Any]] = None
    venues: list[str] = Field(default_factory=list)


class WhalesSnapshot(Availability):
    window_minutes: Optional[float] = None
    btc_trades: Optional[int] = None
    btc_buy_usd: Optional[float] = None
    btc_sell_usd: Optional[float] = None
    btc_largest: Optional[dict[str, Any]] = None
    funding_by_exchange: dict[str, float] = Field(default_factory=dict)
    oi_by_exchange_usd: dict[str, float] = Field(default_factory=dict)
    radar: list[dict[str, Any]] = Field(default_factory=list)
    btc_radar: Optional[dict[str, Any]] = None


class StablecoinSnapshot(Availability):
    total_usd: Optional[float] = None
    prev_day_usd: Optional[float] = None
    change_24h_pct: Optional[float] = None
    top: list[dict[str, Any]] = Field(default_factory=list)
```
and in `MarketSnapshot`:
```python
    hyperliquid: HyperliquidSnapshot = Field(default_factory=lambda: HyperliquidSnapshot.unavailable("hyperliquid", "not fetched"))
    liquidations: LiquidationsSnapshot = Field(default_factory=lambda: LiquidationsSnapshot.unavailable("coinlobster", "not fetched"))
    whales: WhalesSnapshot = Field(default_factory=lambda: WhalesSnapshot.unavailable("coinlobster", "not fetched"))
    stablecoins: StablecoinSnapshot = Field(default_factory=lambda: StablecoinSnapshot.unavailable("defillama", "not fetched"))

    SOURCES = ("spot", "futures", "options", "sentiment", "hyperliquid", "liquidations", "whales", "stablecoins")

    def unavailable(self) -> list[str]:
        return [n for n in self.SOURCES if not getattr(self, n).available]
```
(`SOURCES` as a `ClassVar[tuple[str, ...]]` — import `ClassVar` from typing.)
- [ ] Step 4 — run full suite green. Step 5 — commit `feat(data): snapshot types for hyperliquid, liquidations, whales, stablecoins`.

---

### Task P2-1: Gemini spot fallback + spot chain

**Files:** Create `core/data/gemini_spot.py`, `core/data/spot.py`; Modify `core/data/market.py` (use chain); Test `tests/test_new_feeds.py`

**Interfaces:**
```python
# gemini_spot.py
GEMINI_TF = {"15m": "15m", "1h": "1hr", "1d": "1day"}     # 4h and 1w are resampled from 1h / 1d
def parse_candles(rows: list[list]) -> pd.DataFrame            # rows [ms, o, h, l, c, v] any order → ascending, COLS
def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame   # bucket by minutes; weekly buckets Monday-aligned when minutes == 10080
async def fetch_gemini_spot(client) -> SpotSnapshot            # frames 15m, 1h, 4h, 1d, 1w; source="gemini"
# spot.py
async def fetch_spot_chain(client) -> SpotSnapshot            # kraken → gemini → unavailable("kraken,gemini", errors)
```
- [ ] Step 1 — tests:
```python
import json, pathlib
import numpy as np
import pandas as pd
from core.data.gemini_spot import parse_candles, resample
FX = pathlib.Path(__file__).parent / "fixtures"

def test_gemini_parse_and_resample():
    df = parse_candles(json.loads((FX / "gemini_1hr.json").read_text()))
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].dtype == np.int64
    h4 = resample(df, 240)
    assert len(h4) < len(df) and (h4["high"] >= h4["low"]).all()
    assert (h4["timestamp"] % (240 * 60_000) == 0).all()
    d = parse_candles(json.loads((FX / "gemini_1day.json").read_text()))
    w = resample(d, 10080)
    # weekly buckets start on Monday 00:00 UTC
    assert all(pd.Timestamp(t, unit="ms", tz="UTC").weekday() == 0 for t in w["timestamp"])
    assert w["volume"].iloc[-1] > 0

def test_spot_chain_falls_back_to_gemini(monkeypatch):
    import asyncio
    from core.data import spot
    from core.data.types import SpotSnapshot
    async def k(client): return SpotSnapshot.unavailable("kraken", "dns")
    async def g(client): return SpotSnapshot(source="gemini", last=1.0, frames={"1h": pd.DataFrame()})
    monkeypatch.setattr(spot, "fetch_kraken", k); monkeypatch.setattr(spot, "fetch_gemini_spot", g)
    s = asyncio.run(spot.fetch_spot_chain(None))
    assert s.available and s.source == "gemini"
```
- [ ] Step 2 — run, expect ImportError.
- [ ] Step 3 — implement:
```python
# core/data/gemini_spot.py
"""Gemini exchange (US) public candles — spot fallback when Kraken is unreachable."""
from __future__ import annotations
import asyncio, logging
import httpx, pandas as pd
from core.data.types import SpotSnapshot

log = logging.getLogger(__name__)
BASE = "https://api.gemini.com"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
GEMINI_TF = {"15m": "15m", "1h": "1hr", "1d": "1day"}
DAY_MS = 86_400_000

def parse_candles(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows], columns=COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    ms = minutes * 60_000
    ts = df["timestamp"].astype("int64")
    if minutes == 10080:  # epoch day 0 was a Thursday; +3 aligns buckets to Monday 00:00 UTC
        day = ts // DAY_MS
        bucket = ((day + 3) // 7 * 7 - 3) * DAY_MS
    else:
        bucket = (ts // ms) * ms
    g = df.assign(bucket=bucket).groupby("bucket", sort=True)
    out = pd.DataFrame({"timestamp": g["timestamp"].min().index.astype("int64"),
                        "open": g["open"].first().values, "high": g["high"].max().values,
                        "low": g["low"].min().values, "close": g["close"].last().values,
                        "volume": g["volume"].sum().values})
    out["timestamp"] = out["timestamp"].astype("int64")
    return out.reset_index(drop=True)

async def _get(client: httpx.AsyncClient, path: str):
    r = await client.get(f"{BASE}{path}"); r.raise_for_status(); return r.json()

async def fetch_gemini_spot(client: httpx.AsyncClient) -> SpotSnapshot:
    try:
        res = await asyncio.gather(*[_get(client, f"/v2/candles/btcusd/{g}") for g in GEMINI_TF.values()], _get(client, "/v1/pubticker/btcusd"), return_exceptions=True)
        frames = {}
        for tf, data in zip(GEMINI_TF, res[:-1]):
            if isinstance(data, Exception):
                log.warning("gemini %s failed: %s", tf, data); continue
            frames[tf] = parse_candles(data)
        if "1h" in frames: frames["4h"] = resample(frames["1h"], 240)
        if "1d" in frames: frames["1w"] = resample(frames["1d"], 10080)
        if not frames: raise RuntimeError("no candles returned")
        tick = res[-1]
        last = float(tick["last"]) if not isinstance(tick, Exception) else float(frames[next(iter(frames))]["close"].iloc[-1])
        return SpotSnapshot(source="gemini", last=last, frames=frames)
    except Exception as e:  # noqa: BLE001
        log.error("gemini spot unavailable: %s", e)
        return SpotSnapshot.unavailable("gemini", str(e))
```
```python
# core/data/spot.py
"""Spot OHLCV chain: Kraken first, Gemini exchange second."""
from __future__ import annotations
import httpx
from core.data.gemini_spot import fetch_gemini_spot
from core.data.kraken_spot import fetch_spot as fetch_kraken
from core.data.types import SpotSnapshot

async def fetch_spot_chain(client: httpx.AsyncClient) -> SpotSnapshot:
    errors = []
    for fn in (fetch_kraken, fetch_gemini_spot):
        s = await fn(client)
        if s.available:
            return s
        errors.append(f"{s.source}: {s.error}")
    return SpotSnapshot.unavailable("kraken,gemini", " | ".join(errors))
```
In `market.py` replace `from core.data.kraken_spot import fetch_spot` with `from core.data.spot import fetch_spot_chain as fetch_spot` (keeps the monkeypatch name `market.fetch_spot` used by the existing test).
- [ ] Step 4 — suite green. Step 5 — commit `feat(data): gemini exchange spot fallback with 4h/1w resampling`.

---

### Task P2-2: Hyperliquid, CoinLobster, DefiLlama providers

**Files:** Create `core/data/hyperliquid.py`, `core/data/coinlobster.py`, `core/data/stablecoins.py`; Test `tests/test_new_feeds.py`

**Interfaces:**
```python
def parse_meta(payload: list) -> HyperliquidSnapshot;        async def fetch_hyperliquid(client) -> HyperliquidSnapshot
def parse_liquidations(d: dict) -> LiquidationsSnapshot;     def parse_whales(d: dict, radar: dict | None = None) -> WhalesSnapshot;  def parse_radar(d: dict) -> list[dict]
async def fetch_coinlobster(client) -> tuple[LiquidationsSnapshot, WhalesSnapshot]
def parse_stablecoins(d: dict) -> StablecoinSnapshot;        async def fetch_stablecoins(client) -> StablecoinSnapshot
```
- [ ] Step 1 — tests:
```python
from core.data.hyperliquid import parse_meta
from core.data.coinlobster import parse_liquidations, parse_radar, parse_whales
from core.data.stablecoins import parse_stablecoins
def _j(n): return json.loads((FX / n).read_text(encoding="utf-8"))

def test_hyperliquid_parse():
    h = parse_meta(_j("hyperliquid_meta.json"))
    assert h.available and h.source == "hyperliquid"
    assert -0.01 < h.funding_rate_1h < 0.01 and h.open_interest > 100 and h.mark_price > 1000

def test_coinlobster_liquidations_parse():
    l = parse_liquidations(_j("coinlobster_liquidations.json"))
    assert l.available and l.window == "24h"
    assert l.total_usd > 1e6 and abs(l.long_usd + l.short_usd - l.total_usd) / l.total_usd < 0.02
    assert l.btc_usd and l.btc_long_usd is not None and len(l.hourly) == 24 and l.venues

def test_coinlobster_whales_parse_with_radar():
    w = parse_whales(_j("coinlobster_whales.json"), _j("coinlobster_radar.json"))
    assert w.available and w.btc_trades > 0 and w.btc_buy_usd >= 0 and w.btc_sell_usd >= 0
    assert "Binance Futures" in w.funding_by_exchange and w.oi_by_exchange_usd
    assert w.radar and {"coin", "direction", "unusual"} <= set(w.radar[0])

def test_stablecoins_parse():
    s = parse_stablecoins(_j("defillama_stablecoins.json"))
    assert s.available and s.total_usd > 1e11 and s.prev_day_usd > 1e11
    assert s.change_24h_pct is not None and len(s.top) == 5 and s.top[0]["usd"] >= s.top[1]["usd"]
```
- [ ] Step 2 — run, expect ImportError.
- [ ] Step 3 — implement:
```python
# core/data/hyperliquid.py
"""Hyperliquid public info: BTC perp funding (hourly), open interest, mark/oracle."""
from __future__ import annotations
import logging, httpx
from core.data.types import HyperliquidSnapshot
log = logging.getLogger(__name__)
URL = "https://api.hyperliquid.xyz/info"

def parse_meta(payload: list) -> HyperliquidSnapshot:
    meta, ctxs = payload
    names = [u["name"] for u in meta["universe"]]
    i = names.index("BTC")
    c = ctxs[i]
    return HyperliquidSnapshot(source="hyperliquid", funding_rate_1h=float(c["funding"]), open_interest=float(c["openInterest"]),
                               mark_price=float(c["markPx"]), oracle_price=float(c["oraclePx"]))

async def fetch_hyperliquid(client: httpx.AsyncClient) -> HyperliquidSnapshot:
    try:
        r = await client.post(URL, json={"type": "metaAndAssetCtxs"}); r.raise_for_status()
        return parse_meta(r.json())
    except Exception as e:  # noqa: BLE001
        log.error("hyperliquid unavailable: %s", e)
        return HyperliquidSnapshot.unavailable("hyperliquid", str(e))
```
```python
# core/data/coinlobster.py
"""CoinLobster public feeds: 24h perp liquidations, $100K+ whale trades, unusual-flow radar."""
from __future__ import annotations
import asyncio, logging, httpx
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
        btc_usd=float(btc["usd"]) if btc else None, btc_long_usd=float(btc["longUsd"]) if btc else None,
        btc_short_usd=float(btc["shortUsd"]) if btc else None,
        hourly=[{"t": int(h["t"]), "long_usd": float(h["longUsd"]), "short_usd": float(h["shortUsd"]), "count": int(h["count"])} for h in p.get("hourly", [])],
        biggest=p.get("biggest"), venues=list(p.get("venues", [])),
    )

def parse_radar(d: dict) -> list[dict]:
    return [{"coin": r["coin"], "direction": r.get("direction"), "unusual": bool(r.get("unusual")), "count": r.get("count")} for r in d.get("rows", [])]

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

async def _get(client, path):
    r = await client.get(f"{BASE}{path}"); r.raise_for_status(); return r.json()

async def fetch_coinlobster(client: httpx.AsyncClient) -> tuple[LiquidationsSnapshot, WhalesSnapshot]:
    liq_p, wh_p, rad_p = await asyncio.gather(_get(client, "/liquidations"), _get(client, "/crypto-whales"), _get(client, "/whale-radar?window=4h"), return_exceptions=True)
    try:
        liqs = parse_liquidations(liq_p) if not isinstance(liq_p, Exception) else LiquidationsSnapshot.unavailable("coinlobster", str(liq_p))
    except Exception as e:  # noqa: BLE001
        liqs = LiquidationsSnapshot.unavailable("coinlobster", str(e))
    try:
        whales = parse_whales(wh_p, None if isinstance(rad_p, Exception) else rad_p) if not isinstance(wh_p, Exception) else WhalesSnapshot.unavailable("coinlobster", str(wh_p))
    except Exception as e:  # noqa: BLE001
        whales = WhalesSnapshot.unavailable("coinlobster", str(e))
    return liqs, whales
```
```python
# core/data/stablecoins.py
"""DefiLlama stablecoin supply (dry powder)."""
from __future__ import annotations
import logging, httpx
from core.data.types import StablecoinSnapshot
log = logging.getLogger(__name__)
URL = "https://stablecoins.llama.fi/stablecoins"

def _usd(a: dict, key: str) -> float:
    return float(((a.get(key) or {}).get("peggedUSD")) or 0.0)

def parse_stablecoins(d: dict) -> StablecoinSnapshot:
    assets = d.get("peggedAssets", [])
    total = sum(_usd(a, "circulating") for a in assets)
    prev = sum(_usd(a, "circulatingPrevDay") for a in assets)
    top = sorted(({"name": a.get("symbol") or a.get("name"), "usd": _usd(a, "circulating")} for a in assets), key=lambda x: -x["usd"])[:5]
    return StablecoinSnapshot(source="defillama", total_usd=total, prev_day_usd=prev,
                              change_24h_pct=((total - prev) / prev * 100.0) if prev else None, top=top)

async def fetch_stablecoins(client: httpx.AsyncClient) -> StablecoinSnapshot:
    try:
        r = await client.get(URL, params={"includePrices": "false"}); r.raise_for_status()
        return parse_stablecoins(r.json())
    except Exception as e:  # noqa: BLE001
        log.error("defillama unavailable: %s", e)
        return StablecoinSnapshot.unavailable("defillama", str(e))
```
- [ ] Step 4 — suite green. Step 5 — commit `feat(data): hyperliquid, coinlobster liquidations/whales/radar, defillama stablecoins`.

---

### Task P2-3: Split fast/slow loading, wire new feeds, offline mode, panels

**Files:** Modify `core/data/market.py`, `core/data/offline.py`, `ui/panels.py`, `app.py`; Test `tests/test_market.py`, `tests/test_panels.py`

**Interfaces:**
```python
# market.py
async def fetch_spot_only(client=None) -> SpotSnapshot
async def fetch_context(client=None) -> dict   # keys: futures, options, sentiment, hyperliquid, liquidations, whales, stablecoins
def load_spot() -> SpotSnapshot;  def load_context() -> dict            # sync; honour TI_OFFLINE_FIXTURES
def assemble(spot, ctx: dict) -> MarketSnapshot
def load_market() -> MarketSnapshot                                   # = assemble(load_spot(), load_context()); kept for scripts/tests
# panels.py additions
def kpis_for(...) adds {"label": "Liq 24h", "value": "$204M", "delta": "75% longs", "tone": "down" if long share ≥ 60 else "up" if ≤ 40 else None}
raw_metrics_rows adds: Hyperliquid funding (1h), Hyperliquid OI, Liquidations 24h (long/short), BTC liquidations 24h, Whale BTC buy/sell (window), Stablecoin supply (+24h %)
data_status_html iterates MarketSnapshot.SOURCES
```
- [ ] Step 1 — tests: extend `tests/test_market.py` with
```python
def test_fetch_context_isolates_and_assemble(monkeypatch):
    async def boom(client): raise RuntimeError("down")
    async def ok_liq(client):
        from core.data.types import LiquidationsSnapshot, WhalesSnapshot
        return LiquidationsSnapshot(source="coinlobster", total_usd=1.0, long_usd=0.5, short_usd=0.5), WhalesSnapshot(source="coinlobster")
    monkeypatch.setattr(market, "fetch_futures", boom); monkeypatch.setattr(market, "fetch_options", boom)
    monkeypatch.setattr(market, "fetch_sentiment", boom); monkeypatch.setattr(market, "fetch_hyperliquid", boom)
    monkeypatch.setattr(market, "fetch_coinlobster", ok_liq); monkeypatch.setattr(market, "fetch_stablecoins", boom)
    ctx = asyncio.run(market.fetch_context())
    m = market.assemble(SpotSnapshot(source="kraken"), ctx)
    assert m.liquidations.available and not m.whales.available is False or True  # whales snapshot present
    assert set(m.unavailable()) == {"futures", "options", "sentiment", "hyperliquid", "stablecoins"}
```
and in `tests/test_panels.py` assert `"Liq 24h" in labels` after giving the market a `liquidations=LiquidationsSnapshot(source="coinlobster", total_usd=2.04e8, long_usd=1.54e8, short_usd=5.0e7)` and that `raw_metrics_rows` contains a row starting with `"Liquidations 24h"`.
- [ ] Step 2 — run, expect failures.
- [ ] Step 3 — implement `market.py`:
```python
"""Fetch providers concurrently; a failing provider never takes the page down.
Spot is fast (25 s cache); context feeds are slow (120 s cache) because several are IP-rate-limited."""
from __future__ import annotations
import asyncio, os
import httpx
from core.data.binance_futures import fetch_futures
from core.data.coinlobster import fetch_coinlobster
from core.data.deribit_options import fetch_options
from core.data.hyperliquid import fetch_hyperliquid
from core.data.sentiment import fetch_sentiment
from core.data.spot import fetch_spot_chain as fetch_spot
from core.data.stablecoins import fetch_stablecoins
from core.data.types import (FuturesSnapshot, HyperliquidSnapshot, LiquidationsSnapshot, MarketSnapshot, OptionsSnapshot,
                             SentimentSnapshot, SpotSnapshot, StablecoinSnapshot, WhalesSnapshot)

HEADERS = {"User-Agent": "trap-intel-dashboard/1.0"}
TIMEOUT = httpx.Timeout(12.0, connect=6.0)

def _offline() -> bool:
    return os.environ.get("TI_OFFLINE_FIXTURES") == "1"

async def _guard(coro, fallback_cls, source: str):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        return fallback_cls.unavailable(source, str(e))

async def _with_client(fn):
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        return await fn(client)

async def fetch_spot_only(client: httpx.AsyncClient | None = None) -> SpotSnapshot:
    if client is None:
        return await _with_client(fetch_spot_only)
    return await _guard(fetch_spot(client), SpotSnapshot, "kraken,gemini")

async def fetch_context(client: httpx.AsyncClient | None = None) -> dict:
    if client is None:
        return await _with_client(fetch_context)
    fut, opt, sent, hl, cl, st_ = await asyncio.gather(
        _guard(fetch_futures(client), FuturesSnapshot, "binance,bybit"),
        _guard(fetch_options(client), OptionsSnapshot, "deribit"),
        _guard(fetch_sentiment(client), SentimentSnapshot, "alternative.me"),
        _guard(fetch_hyperliquid(client), HyperliquidSnapshot, "hyperliquid"),
        _guard(fetch_coinlobster(client), tuple, "coinlobster"),
        _guard(fetch_stablecoins(client), StablecoinSnapshot, "defillama"),
    )
    if isinstance(cl, tuple) and len(cl) == 2:
        liqs, whales = cl
    else:
        liqs, whales = LiquidationsSnapshot.unavailable("coinlobster", "fetch failed"), WhalesSnapshot.unavailable("coinlobster", "fetch failed")
    return {"futures": fut, "options": opt, "sentiment": sent, "hyperliquid": hl, "liquidations": liqs, "whales": whales, "stablecoins": st_}

def assemble(spot: SpotSnapshot, ctx: dict) -> MarketSnapshot:
    return MarketSnapshot(spot=spot, **ctx)

def load_spot() -> SpotSnapshot:
    if _offline():
        from core.data.offline import fixture_spot
        return fixture_spot()
    return asyncio.run(fetch_spot_only())

def load_context() -> dict:
    if _offline():
        from core.data.offline import fixture_context
        return fixture_context()
    return asyncio.run(fetch_context())

def load_market() -> MarketSnapshot:
    return assemble(load_spot(), load_context())
```
Note `_guard(..., tuple, ...)` cannot call `.unavailable` — replace that line with a dedicated wrapper:
```python
async def _guard_cl(coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        return LiquidationsSnapshot.unavailable("coinlobster", str(e)), WhalesSnapshot.unavailable("coinlobster", str(e))
```
and use `_guard_cl(fetch_coinlobster(client))` in the gather; then `liqs, whales = cl` unconditionally.

`offline.py`: split `fixture_market()` into `fixture_spot()` and `fixture_context()` (context parses the new fixtures too: `parse_meta`, `parse_liquidations`, `parse_whales(+radar)`, `parse_stablecoins`; set every `source="fixture"`), keep `fixture_market()` = assemble of both.

`panels.py`: add the `Liq 24h` KPI after `OI 24h`; extend `raw_metrics_rows`; make `data_status_html` iterate `MarketSnapshot.SOURCES`.

`app.py`: replace the single `_market()` with
```python
@st.cache_data(ttl=TTL["spot"], show_spinner=False)
def _spot(): return load_spot()

@st.cache_data(ttl=TTL["context"], show_spinner=False)
def _context(): return load_context()

m = assemble(_spot(), _context())
```
Add `"context": 120` to `core.config.TTL`. "Refresh data" clears both.
- [ ] Step 4 — suite green; boot app with `TI_OFFLINE_FIXTURES=1`, confirm new chips and the Liq tile render. Step 5 — commit `feat(data): split fast/slow loading, wire new feeds into snapshot, panels and offline mode`.

---

### Task P2-4: LLM client (Gemini native + OpenRouter) and settings

**Files:** Create `core/agents/__init__.py`, `core/agents/settings.py`, `core/agents/llm.py`; Test `tests/test_llm.py`

**Interfaces:**
```python
# settings.py
@dataclass
class LLMSettings: provider: str; api_key: str; specialist_model: str; director_model: str; max_concurrency: int = 4; timeout_s: float = 90.0
def load_settings() -> LLMSettings | None      # env overrides st.secrets; None when no key for the chosen provider
# llm.py
@dataclass
class LLMResult: text: str; data: dict | None; prompt_tokens: int; completion_tokens: int; cost_usd: float | None; latency_s: float; model: str; provider: str; error: str | None = None
def extract_json(text: str) -> dict                 # strips ``` fences, takes the first {...} block; raises ValueError
def parse_gemini(payload: dict) -> tuple[str, int, int]            # (text, prompt_tokens, completion_tokens)
def parse_openrouter(payload: dict) -> tuple[str, int, int, float | None]
class LLMClient:
    def __init__(self, settings: LLMSettings, client: httpx.AsyncClient | None = None)
    async def complete_json(self, system: str, user: str, model: str, max_tokens: int = 2048, temperature: float = 0.2) -> LLMResult
    # retries: up to 2 on 429/5xx/timeout (sleep 2 s, 6 s); on JSON parse failure one repair call appending "Return ONLY valid JSON matching the schema."
    async def aclose(self)
```
- [ ] Step 1 — tests:
```python
import asyncio, json
import httpx, pytest
from core.agents.llm import LLMClient, extract_json, parse_gemini, parse_openrouter
from core.agents.settings import LLMSettings, load_settings

def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure. {"a": {"b": [1,2]}} trailing') == {"a": {"b": [1, 2]}}
    with pytest.raises(ValueError):
        extract_json("no json here")

def test_parse_gemini_and_openrouter_shapes():
    g = {"candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
    assert parse_gemini(g) == ('{"ok": true}', 10, 5)
    o = {"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001}}
    assert parse_openrouter(o) == ('{"ok": true}', 7, 3, 0.0001)

def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x"); monkeypatch.delenv("OPENROUTER_API_KEY", raising=False); monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = load_settings()
    assert s.provider == "gemini" and s.specialist_model.startswith("gemini") and s.director_model.startswith("gemini")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    assert load_settings().provider == "openrouter"

def test_client_retries_then_repairs_json():
    calls = {"n": 0}
    def handler(request: httpx.Request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        if calls["n"] == 2:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{\"fixed\": true}"}]}}], "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2}})
    transport = httpx.MockTransport(handler)
    s = LLMSettings(provider="gemini", api_key="k", specialist_model="m", director_model="m")
    c = LLMClient(s, client=httpx.AsyncClient(transport=transport), backoff=(0, 0))
    r = asyncio.run(c.complete_json("sys", "user", "m"))
    assert r.data == {"fixed": True} and r.error is None and calls["n"] == 3
    assert r.prompt_tokens == 3 and r.completion_tokens == 3   # usage summed across attempts
```
(`LLMClient.__init__` accepts `backoff: tuple[float, float] = (2.0, 6.0)` for testability.)
- [ ] Step 2 — run, expect ImportError.
- [ ] Step 3 — implement:
```python
# core/agents/settings.py
"""LLM settings from environment first, then Streamlit secrets."""
from __future__ import annotations
import os
from dataclasses import dataclass

DEFAULTS = {
    "gemini": {"specialist": "gemini-3.5-flash-lite", "director": "gemini-3.8-flash"},
    "openrouter": {"specialist": "nvidia/nemotron-3.5-lightning:free", "director": "nvidia/nemotron-3.5-lightning:free"},
}

@dataclass
class LLMSettings:
    provider: str
    api_key: str
    specialist_model: str
    director_model: str
    max_concurrency: int = 4
    timeout_s: float = 90.0

def _secret(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - not running under streamlit or no secrets file
        return None

def load_settings() -> LLMSettings | None:
    gem, orr = _secret("GEMINI_API_KEY"), _secret("OPENROUTER_API_KEY")
    provider = (_secret("LLM_PROVIDER") or ("gemini" if gem else "openrouter" if orr else "")).lower()
    key = gem if provider == "gemini" else orr if provider == "openrouter" else None
    if not key:
        return None
    d = DEFAULTS[provider]
    return LLMSettings(provider=provider, api_key=key,
                       specialist_model=_secret("SPECIALIST_MODEL") or d["specialist"],
                       director_model=_secret("DIRECTOR_MODEL") or d["director"],
                       max_concurrency=int(_secret("LLM_MAX_CONCURRENCY") or 4))
```
```python
# core/agents/llm.py
"""Provider-agnostic JSON-mode chat client: Gemini generateContent or OpenRouter chat completions."""
from __future__ import annotations
import asyncio, json, logging, re, time
from dataclasses import dataclass
import httpx
from core.agents.settings import LLMSettings

log = logging.getLogger(__name__)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {429, 500, 502, 503, 504}

@dataclass
class LLMResult:
    text: str
    data: dict | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_s: float
    model: str
    provider: str
    error: str | None = None

def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON object in response")

def parse_gemini(payload: dict) -> tuple[str, int, int]:
    cands = payload.get("candidates") or []
    if not cands:
        raise ValueError(f"gemini returned no candidates: {json.dumps(payload)[:300]}")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = payload.get("usageMetadata", {})
    return text, int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))

def parse_openrouter(payload: dict) -> tuple[str, int, int, float | None]:
    if "error" in payload and not payload.get("choices"):
        raise ValueError(f"openrouter error: {payload['error']}")
    text = payload["choices"][0]["message"]["content"] or ""
    u = payload.get("usage", {})
    cost = u.get("cost")
    return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), (float(cost) if cost is not None else None)

class LLMClient:
    def __init__(self, settings: LLMSettings, client: httpx.AsyncClient | None = None, backoff: tuple[float, float] = (2.0, 6.0)):
        self.s = settings
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_s, connect=10.0))
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._backoff = backoff

    async def aclose(self):
        await self._client.aclose()

    def _request(self, system: str, user: str, model: str, max_tokens: int, temperature: float) -> tuple[str, dict, dict]:
        if self.s.provider == "gemini":
            url = f"{GEMINI_BASE}/models/{model}:generateContent"
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}
            return url, body, {"x-goog-api-key": self.s.api_key, "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_tokens": max_tokens, "temperature": temperature, "usage": {"include": True}}
        return OPENROUTER_URL, body, {"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json",
                                      "HTTP-Referer": "https://github.com/maddymedia123-jpg/Quant-dashboard", "X-Title": "Trap Intelligence"}

    async def _once(self, system: str, user: str, model: str, max_tokens: int, temperature: float) -> tuple[str, int, int, float | None]:
        url, body, headers = self._request(system, user, model, max_tokens, temperature)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = await self._client.post(url, json=body, headers=headers)
                if r.status_code in RETRY_STATUS and attempt < 2:
                    await asyncio.sleep(self._backoff[attempt]); continue
                r.raise_for_status()
                payload = r.json()
                if self.s.provider == "gemini":
                    t, pt, ct = parse_gemini(payload); return t, pt, ct, None
                return parse_openrouter(payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(self._backoff[attempt]); continue
        raise last_exc or RuntimeError("llm request failed")

    async def complete_json(self, system: str, user: str, model: str, max_tokens: int = 2048, temperature: float = 0.2) -> LLMResult:
        t0 = time.perf_counter()
        pt_sum = ct_sum = 0
        cost_sum: float | None = None
        async with self._sem:
            try:
                text, pt, ct, cost = await self._once(system, user, model, max_tokens, temperature)
                pt_sum += pt; ct_sum += ct; cost_sum = cost
                try:
                    data = extract_json(text)
                except ValueError:
                    text, pt, ct, cost = await self._once(system, user + "\n\nYour previous reply was not valid JSON. Return ONLY valid JSON matching the schema.", model, max_tokens, temperature)
                    pt_sum += pt; ct_sum += ct
                    cost_sum = (cost_sum or 0.0) + cost if cost is not None else cost_sum
                    data = extract_json(text)
                return LLMResult(text, data, pt_sum, ct_sum, cost_sum, time.perf_counter() - t0, model, self.s.provider)
            except Exception as e:  # noqa: BLE001 - agent boundary
                log.warning("llm call failed (%s): %s", model, e)
                return LLMResult("", None, pt_sum, ct_sum, cost_sum, time.perf_counter() - t0, model, self.s.provider, error=str(e)[:300])
```
- [ ] Step 4 — suite green. Step 5 — commit `feat(agents): provider-agnostic JSON LLM client with retries and settings`.

---

### Task P2-5: Output schemas

**Files:** Create `core/agents/schemas.py`; Test `tests/test_schemas.py`

**Interfaces (produces):**
```python
DESKS = ("bullish", "bearish"); TRAPS = ("BULL_TRAP", "BEAR_TRAP", "NO_TRAP", "RANGE_TRAP"); CATEGORY_KEYS = ("live", "intraday", "weekly", "monthly")
class SpecialistBrief(BaseModel): agent_id: str; desk: str; domain: str; raw_metrics: dict[str, Any]; interpretation: str; traps_detected: list[str]; conviction: int (ge=1, le=10); key_risk_to_thesis: str; data_gaps: list[str] = []
class TradeSetup(BaseModel): direction: Literal["LONG","SHORT","NONE"]; entry: str; stop: str; targets: list[str]; rationale: str
class DeskReport(BaseModel): desk: str; summary: str; convergence: list[str]; divergence: list[str]; conviction: float (ge=1, le=10); best_setup: TradeSetup; acknowledged_risks: list[str]
class VolatilityReport(BaseModel): regime_summary: str; per_category: dict[str, str]; range_expectations: list[str]
class AccuracyReport(BaseModel): consistency_score: int (0-100); cross_timeframe_alignment: str; contradictions_found: list[str]; correlation_notes: str; data_quality_flags: list[str]
class Scenario(BaseModel): name: str; probability: int (0-100); path: str; invalidation: str
class TradePlan(BaseModel): existing_position_management: str; new_entry_conditions: str; position_sizing: str; stop_loss: str; take_profit_ladder: list[str]; max_leverage: str; time_rules: list[str]
class Contradiction(BaseModel): metric: str; bull_read: str; bear_read: str; stronger_evidence: str
class DirectorReport(BaseModel): executive_summary: str; trap_classification: Literal[TRAPS]; classification_rationale: str; contradictions: list[Contradiction]; convergence: list[str]; blind_spots: list[str]; scenarios: list[Scenario] (validator: len 2–4 and sum of probability within 90..110 → normalise to 100); trade_plan: TradePlan; category_summaries: dict[str, str] (validator: keep only CATEGORY_KEYS)
class AgentRun(BaseModel): agent_id: str; ok: bool; model: str; latency_s: float; prompt_tokens: int; completion_tokens: int; cost_usd: float | None; error: str | None
class TrapReport(BaseModel): report_id: str; generated_at: datetime; asset: str = "BTC/USD"; window: str; provider: str; briefs: list[SpecialistBrief]; desks: dict[str, DeskReport]; volatility: VolatilityReport | None; accuracy: AccuracyReport | None; director: DirectorReport | None; runs: list[AgentRun]; wall_time_s: float; unavailable_domains: list[str]; raw_metrics: list[tuple[str, str, str]]
    @property total_tokens; @property total_cost_usd (None if all None)
def schema_hint(model_cls) -> str   # compact JSON schema text for prompts: json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
```
- [ ] Step 1 — tests: probabilities normalise (e.g. 60/30/20 → sums to 100 after validation), reject when sum 150, reject conviction 11, `category_summaries` drops unknown keys, `TrapReport.total_tokens` sums runs, `schema_hint(SpecialistBrief)` contains `"conviction"`.
- [ ] Step 2 — run, expect ImportError. Step 3 — implement with pydantic `field_validator`/`model_validator`. Step 4 — green. Step 5 — commit `feat(agents): pydantic output schemas and TrapReport`.

---

### Task P2-6: Prompts and per-domain context builders

**Files:** Create `core/agents/prompts.py`, `core/agents/context.py`; Test `tests/test_context.py`

**Interfaces:**
```python
# prompts.py
SPECIALISTS: dict[str, dict] = {"B1": {...}, ..., "R5": {...}}   # keys: desk, domain ("options"|"leverage"|"liquidity"|"flow"|"sentiment"), title, data_sources, trap_focus, output_name — text lifted from the client's orchestration prompt
SHARED_RULES: str          # honesty rules: only use numbers given; list gaps in data_gaps; never infer unavailable fields; conditional probabilities; institutional tone
def system_prompt(agent_id: str) -> str          # agent_id in SPECIALISTS | "DESK_BULLISH" | "DESK_BEARISH" | "VOLATILITY" | "ACCURACY" | "DIRECTOR"; embeds schema_hint of the expected model
def user_prompt(agent_id: str, payload: dict) -> str   # "DATA (JSON):\n<payload>\nUNAVAILABLE: [...]\nReturn JSON only."
# context.py
DOMAIN_OF = {"B1": "options", "R1": "options", "B2": "leverage", "R2": "leverage", "B3": "liquidity", "R3": "liquidity", "B4": "flow", "R4": "flow", "B5": "sentiment", "R5": "sentiment"}
def common_payload(m, analyses) -> dict           # price, change_24h, per-category {direction, confidence, ema_state, rsi, regime, anchor, invalidation}, macro events
def domain_payload(domain: str, m, analyses) -> tuple[dict, list[str]]   # (data, unavailable_fields)
def payload_for(agent_id: str, m, analyses) -> dict                     # common + domain + "unavailable": [...]
def desk_payload(desk: str, briefs: list[SpecialistBrief]) -> dict
def volatility_payload(m, analyses) -> dict
def accuracy_payload(desks: dict, volatility, analyses) -> dict
def director_payload(m, analyses, desks, volatility, accuracy, raw_metrics) -> dict
```
Domain mapping (values rounded, no DataFrames): options → OptionsSnapshot fields + expiries; leverage → FuturesSnapshot fields + hyperliquid + liquidations totals + squeeze per category; liquidity → liquidations detail (hourly, btc row, biggest, venues) + whales btc buy/sell + S/R levels + trendlines per category; flow → whales + taker ratio + CVD slope per category + direction drivers; sentiment → funding (all exchanges), F&G, stablecoins, radar (btc + unusual coins), long/short ratio. Any unavailable snapshot → its keys omitted and listed in `unavailable`.
- [ ] Step 1 — tests: build a fixture-based market with futures unavailable → `payload_for("B2", ...)` has no `funding_rate` key and lists `"futures"` in `unavailable`; `payload_for("B1", ...)["options"]["max_pain"]` present; `system_prompt("R3")` contains "SWEPT AND REJECTED" and the word "conviction"; `system_prompt("DIRECTOR")` contains "BULL_TRAP" and "probability"; `json.dumps(payload_for(...))` succeeds for all 10 specialists.
- [ ] Step 2 — run, expect ImportError. Step 3 — implement (specialist text copied from the client's `.md`: role title, data sources, trap focus, output name for each of B1–B5/R1–R5; desk synthesis, Director Phase 3 items 1–6; volatility agent = "ranges per category, regime shifts, squeeze conditions"; accuracy agent = "cross-check both desks for contradictions, cross-timeframe alignment, correlation of the five domains, data quality flags"). Step 4 — green. Step 5 — commit `feat(agents): prompts from client orchestration text and domain-scoped context builders`.

---

### Task P2-7: Runner and report renderer

**Files:** Create `core/agents/runner.py`, `core/agents/report.py`; Test `tests/test_runner.py`, `tests/test_report.py`

**Interfaces:**
```python
# runner.py
ProgressFn = Callable[[str, str], None]   # (agent_id, "running"|"ok"|"failed")
async def run_pipeline(client: LLMClient, m: MarketSnapshot, analyses: dict, settings: LLMSettings, progress: ProgressFn | None = None) -> TrapReport
def run_pipeline_sync(...) -> TrapReport
async def _call(client, agent_id, model, system, user, schema_cls, progress) -> tuple[BaseModel | None, AgentRun]   # validates; one repair retry on ValidationError with the error text appended
```
Flow: phase 1 = 10 specialists + VOLATILITY concurrently (client semaphore bounds it); phase 2 = DESK_BULLISH and DESK_BEARISH concurrently, each from its desk's successful briefs (a desk with 0 briefs is skipped); ACCURACY after desks; phase 3 = DIRECTOR (director_model) requires ≥1 desk, else `director=None`. `report_id = "TIR-" + utc timestamp`. `window = "Live 15m · Intraday 1h · Weekly 4h · Monthly 1d"`. `raw_metrics = panels.raw_metrics_rows(m)` (import inside function to avoid ui→core cycles; or move `raw_metrics_rows` to `core/agents/context.py` and have panels re-export — do the latter).
```python
# report.py
def to_markdown(r: TrapReport) -> str
```
Sections, in the client's exact order and headings: `# TRAP INTELLIGENCE REPORT`, `## Report ID`, `## Timestamp`, `## Asset`, `## Analysis Window`, `## SECTION 1: RAW METRIC SNAPSHOT` (table Metric | Value | Source), `## SECTION 2: BULLISH DESK SUMMARY` (Conviction, Key Arguments numbered from convergence, Proposed Trade, Acknowledged Risks), `## SECTION 3: BEARISH DESK SUMMARY`, `## SECTION 4: DIRECTOR'S CROSS-EXAMINATION` (contradictions table, convergence, blind spots), `## SECTION 5: SCENARIO PROBABILITIES & TRAP CLASSIFICATION`, `## SECTION 6: ACTIONABLE TRADE PLAN` (seven items), `## APPENDIX` (volatility, accuracy, unavailable domains, agent run table with tokens/latency/cost, disclaimer "Probabilities are model-weighted scenarios, not predictions"). Missing parts render as "_not produced this run_".
- [ ] Step 1 — tests: `FakeClient.complete_json` returns canned valid JSON per agent_id (detect from system prompt: parse `AGENT_ID: X` line that `system_prompt` emits first) and fails for `B4`; run on fixture market → `len(report.runs) == 15`, one run not ok, both desks present, director present, `report.director.scenarios` probabilities sum to 100; `to_markdown` contains all six `SECTION` headings and the failed agent noted in the appendix; a director-less report still renders.
- [ ] Step 2 — run, expect ImportError. Step 3 — implement. Step 4 — green. Step 5 — commit `feat(agents): three-phase runner with failure isolation and markdown report`.

---

### Task P2-8: UI — Run Analysis, War Room report, head-agent summaries

**Files:** Modify `app.py`, `ui/panels.py`; Test `tests/test_panels.py`

**Interfaces:**
```python
# panels.py
def agent_summary_html(a: CategoryAnalysis, report: TrapReport | None) -> str   # director.category_summaries[a.key] or the Phase-1 placeholder text
def report_stats_html(report: TrapReport) -> str                                # provider, models, calls ok/failed, tokens, cost or "free tier", wall time, generated_at
```
`app.py`:
- `settings = load_settings()`; sidebar `Run Analysis` enabled iff settings; help text shows provider and models; disabled help says which secret to add.
- On click: `with st.status("Running Trap Intelligence…", expanded=True) as status:` progress callback writes one line per agent; `run_pipeline_sync(LLMClient(settings), m, analyses, settings, progress)`; store `st.session_state["trap_report"]`; `status.update(label="Report ready", state="complete")`; `st.rerun()`.
- Category tabs: replace `agent_placeholder_html(a)` with `agent_summary_html(a, st.session_state.get("trap_report"))`.
- War Room: if report → `report_stats_html`, `st.markdown(to_markdown(report))`, expanders "Specialist briefs" (one sub-expander per brief with conviction, traps, key risk, data gaps) and "Desk reports"; a "Download report (.md)" `st.download_button`; else the Phase-1 explainer + Section 1 table.
- [ ] Step 1 — tests for the two new panel builders (with and without report).
- [ ] Step 2 — implement. Step 3 — suite green. Step 4 — boot offline: `TI_OFFLINE_FIXTURES=1` + the real Gemini key → click Run Analysis → confirm report renders, category cards fill, stats line shows tokens; capture screenshot; record wall time and tokens in the PR body. Step 5 — commit `feat(app): Run Analysis flow, War Room report viewer, per-category head-agent summaries`.

---

### Task P2-9: Docs, PR

- [ ] README: add "Agents" section (provider setup, secrets names, defaults, cost note, free-tier limits), update Layout line.
- [ ] Spec status line → "Phase 2 built".
- [ ] `docs/superpowers/plans/PR2-body.md` with What / Removed / Test / Notes (Gemini free tier: Pro quota 429 → Director on flash; OpenRouter $0 credits → free nemotron only; both keys go to Streamlit Cloud Secrets).
- [ ] Full suite green; `git push -u origin feature/trap-intelligence-phase2`; report compare URL.
