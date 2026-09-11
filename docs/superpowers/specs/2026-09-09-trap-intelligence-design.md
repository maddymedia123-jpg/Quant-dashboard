# Trap Intelligence Dashboard — Design Spec

Date: 2026-09-09
Status: Approved (design) · Phase 1 built (PR1) · Phase 2 built (PR2)
Repo: `maddymedia123-jpg/Quant-dashboard` · Branch: `feature/trap-intelligence-phase1`

## 1. Goal

Turn the existing single-file Streamlit "Mak Dashboard" into a BTC market-surveillance
terminal that (a) shows real, deterministic technical analysis per timeframe category on
TradingView-style charts, and (b) on demand, runs a 13-agent LLM pipeline (Director +
Bullish desk B1–B5 + Bearish desk R1–R5 + Volatility agent + Accuracy/Correlation agent)
that produces a structured Trap Intelligence Report.

Client inputs this spec is derived from:

- `You are the orchestration engine fo.md` (orchestration prompt; truncated after
  Section 3 of the output format — Sections 4–6 are authored here from its Phase 3 text).
- WhatsApp asks 2026-09-09: fib time zones + live "now" line, market direction per
  category, RSI with divergence-in-formation, S/R + trendlines + patterns per category on
  separate TradingView-style charts, EMA 21/50/100/200 for scalp (15m/1h) vs swing
  (intraday/week) mapping, verdicts anchored until CVD / OI / long-short / funding / trap /
  news changes, weekly+monthly trap and squeeze early warning, volatility-ranges agent,
  accuracy-checker + market-correlation agent.

## 2. Decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Front end | Hybrid: keep Streamlit, embed TradingView `lightweight-charts` via `components.html` | Keeps client's Streamlit Cloud deploy; gives TradingView look in days not weeks |
| LLM trigger | On-demand "Run Analysis" button, result cached in session with timestamp | Predictable OpenRouter cost (~15 calls/run) |
| Keys | OpenRouter only (`OPENROUTER_API_KEY`) | Only key the client has |
| Market data | Keyless public APIs: Kraken spot OHLCV (existing), Binance USD-M futures (funding, OI, OI history, long/short ratio, taker buy/sell) with Bybit fallback, Deribit options (OI/IV per strike → max pain, put/call, IV skew), alternative.me Fear & Greed | Verified reachable 2026-09-09 without keys |
| Honesty rule | No fabricated numbers. Every provider returns `available: bool`; unavailable domains are shown as "no data" and agents are told so | Current app hard-codes funding/OI, fixed liq clusters, and `np.random` "predicted" candles |
| Git flow | Feature branch → PR → `main` (`main` = Streamlit Cloud) | Never ship half-built code to the live URL |

## 3. Architecture

```
app.py                      thin Streamlit shell: theme, sidebar, tabs, wiring
core/
  data/
    types.py                pydantic snapshots (SpotSnapshot, FuturesSnapshot, OptionsSnapshot, SentimentSnapshot, MarketSnapshot)
    kraken_spot.py          OHLCV 15m/1h/4h/1d/1w (+ ticker)          → SpotSnapshot
    binance_futures.py      funding, OI, OI hist, L/S ratio, taker     → FuturesSnapshot (Bybit fallback inside)
    deribit_options.py      book summary → max pain, PCR, IV skew      → OptionsSnapshot
    sentiment.py            Fear & Greed                                → SentimentSnapshot
    market.py               fetch_all() concurrent, per-provider try/except, caches via st.cache_data(ttl)
  indicators/
    ema.py                  ema(series, n); stack state (bull/bear/mixed)
    rsi.py                  rsi(14); divergence(regular/hidden, confirmed/forming)
    levels.py               swing pivots → clustered S/R with touch counts
    trendlines.py           linear fit through last N pivots (upper/lower) → slope, current value
    fib_time.py             fib time zones from last major swing → future timestamps
    volatility.py           ATR, regime, expected range bands per category
    squeeze.py              long/short squeeze risk from OI Δ, price Δ, funding, L/S ratio
    trade_map.py            scalp vs swing classification vs EMAs + S/R; anchor triggers
    direction.py            per-category direction call (composite of EMA stack, RSI, CVD, S/R position)
  agents/                   (Phase 2)
    openrouter.py           httpx client, retries, cost/latency capture
    prompts/                director.md, desk_synth.md, b1..b5.md, r1..r5.md, volatility.md, accuracy.md
    schemas.py              pydantic: SpecialistBrief, DeskReport, DirectorReport
    runner.py               phase1 parallel → phase2 desks → phase3 director; returns TrapReport
    report.py               TrapReport → markdown (client's format, Sections 1–6)
  anchors.py                (Phase 3) anchored verdict store + invalidation triggers + accuracy log
ui/
  theme.py                  CSS injection, fonts, KPI strip, cards, badges
  charts.py                 lightweight-charts HTML builder (candles, EMAs, S/R, trendlines, fib verticals, now-line, RSI pane, projection zone)
  panels.py                 direction card, divergence panel, volatility panel, agent summary panel
.streamlit/config.toml      theme + server settings
tests/                      pytest for indicators + provider parsers (recorded fixtures)
```

### Data flow

1. `market.fetch_all()` runs the four providers concurrently (asyncio + httpx), each wrapped so a failure yields an unavailable snapshot, never an exception to the page.
2. `indicators.*` are pure functions over the spot DataFrames and the futures/options snapshots. They produce a `CategoryAnalysis` per category (Live=15m/1h, Intraday=1h/4h, Weekly=4h/1d, Monthly=1d/1w).
3. `ui.charts.render(category_analysis)` emits one HTML document per chart with the series and overlays serialised as JSON.
4. (Phase 2) `agents.runner.run(market_snapshot, analyses)` returns a `TrapReport`; the War Room tab renders it; per-category "head agent summary" panels read their section from it.
5. (Phase 3) `anchors` persists the last verdict per category in `st.session_state` and re-evaluates triggers on each refresh.

### Caching

- Spot OHLCV: 25 s. Futures: 60 s. Options: 120 s. Sentiment: 30 min.
- LLM report: session-scoped, keyed by market-snapshot hash; manual re-run always allowed.

## 4. Category model

| Category | Chart TF | Context TF | Use |
|---|---|---|---|
| Live Recon | 15m | 1h | scalp mapping, sweeps |
| Intraday | 1h | 4h | intraday swing |
| Weekly | 4h | 1d | swing, trap early-warning |
| Monthly | 1d | 1w | macro, squeeze early-warning |

Each category tab renders, top to bottom:

1. KPI strip: last price, category direction, RSI, ATR%, funding, OI Δ, L/S ratio, F&G (missing → "—").
2. Chart (lightweight-charts): candles + EMA 21/50/100/200 + S/R lines (labelled with touch count) + up to 2 trendlines + fib time verticals (with the next 3 projected) + "now" line + RSI sub-pane + shaded projected-direction zone (from `direction.py`, ATR-scaled, explicitly labelled *projection*, never fake candles).
3. Direction & anchor card: direction, confidence, anchor level, invalidation level, what would flip it.
4. Divergence + volatility panel: RSI divergences (confirmed / forming) and expected range for the category horizon.
5. Head-agent summary (Phase 2; placeholder text "Run Analysis to generate" in Phase 1).

## 5. Indicators — definitions

- **EMA stack**: 21/50/100/200 on the chart TF. State = BULL if price>21>50>100>200, BEAR if inverse, else MIXED.
- **RSI(14)** Wilder. **Divergence**: compare last two RSI pivots vs price pivots (pivot = local extreme with `left=right=3` bars). Regular bearish: price HH, RSI LH. Regular bullish: price LL, RSI HL. Hidden variants inverse. *Forming* = the second price pivot is the current bar (not yet confirmed by right-side bars).
- **S/R levels**: swing highs/lows (left=right=5) over last 200 bars, clustered within 0.35 × ATR; level strength = touches; keep top 6 nearest to price.
- **Trendlines**: least-squares through the last 3 swing lows (support) and last 3 swing highs (resistance) if R² ≥ 0.8; report current value + slope/bar.
- **Fib time zones**: anchor = last major swing (largest |Δ| in 100 bars); zones at 1,2,3,5,8,13,21,34 bars from anchor; project the next 3 not yet reached as timestamps.
- **Volatility**: ATR(14), ATR% of price, regime (LOW <0.6%, NORMAL, HIGH >1.5% on 1h ATR%), expected range = ±1σ of log-returns over the category horizon.
- **Squeeze risk**: score 0–100 from OI 24h Δ, price 24h Δ, funding vs 7-day mean, L/S ratio extreme; sign indicates long-squeeze (down) vs short-squeeze (up) risk.
- **Direction**: weighted vote of EMA stack (0.35), RSI slope/level (0.2), CVD slope (0.2), position vs nearest S/R (0.25) → BULLISH / BEARISH / NEUTRAL with 0–1 confidence.
- **Trade map**: scalp if |entry − price| < 1 ATR(15m)×3 and TP within 1h expected range; else swing. Reports nearest EMA/S/R above and below, whether SL sits beyond a structural level, and R:R.

## 6. Agents (Phase 2)

- Transport: OpenRouter chat completions via httpx, JSON mode, 2 retries with backoff, 60 s timeout per call.
- Models via config: `SPECIALIST_MODEL` (cheap, fast) and `DIRECTOR_MODEL` (strong). Defaults set in `config.toml`-readable secrets with env-var override.
- Input to every specialist = compact JSON of *only its domain* from the MarketSnapshot + indicator summary + explicit list of unavailable fields. System prompt = its section of the client's orchestration prompt. Output schema = `SpecialistBrief {raw_metrics, interpretation, traps_detected[], conviction 1-10, key_risk_to_thesis}`.
- Desk synthesis input = 5 briefs → `DeskReport {convergence[], divergence[], conviction, best_setup{entry,stop,targets[]}, acknowledged_risks[]}`.
- Director input = both desk reports + volatility + accuracy agent outputs → `DirectorReport` with contradiction analysis, convergence map, blind spots, scenario probabilities (primary/secondary/black-swan summing to 100), trap classification (BULL_TRAP / BEAR_TRAP / NO_TRAP / RANGE_TRAP), trade plan (position mgmt, entry conditions, sizing, structural SL, TP ladder, max leverage, time rules).
- Report markdown Sections 4–6 (authored here to complete the truncated spec):
  - **§4 Director's Cross-Examination**: contradictions table (metric · bull read · bear read · stronger evidence), convergence list, blind spots.
  - **§5 Scenario Probabilities & Trap Classification**: three scenarios with %, path description, invalidation level; final classification + one-line rationale.
  - **§6 Actionable Trade Plan**: the seven items from Phase 3 of the client prompt, each as a bullet with concrete numbers.
- Cost guard: hard cap of 16 calls per run; token budget per role; show tokens + $ estimate + wall time in the report header.

## 7. UI / theme

- Fonts: Inter (UI) + JetBrains Mono (numbers) via Google Fonts; `.streamlit/config.toml` light theme with slate palette; dark toggle in sidebar (CSS class swap + chart theme swap).
- Hide Streamlit default header/menu; single-line app title; 6 tabs max (Live, Intraday, Weekly, Monthly, Active Trade, War Room) with Audit Ledger moved into an expander under War Room.
- No emoji-as-icon in headings; inline SVG (lucide) where an icon is needed.
- Active Trade: form and result in the same tab; sidebar keeps only global controls (symbol fixed BTC/USD, refresh, dark toggle, Run Analysis).
- Mobile: columns stack below 768 px; charts full-width with reduced height.

## 8. Error handling

- Provider failure → snapshot `available=False, error=str`; UI shows a muted "source unavailable" chip; agents receive `"unavailable"` for those fields and are instructed not to infer them.
- Binance 451 (US geo-block) → automatic Bybit attempt → else unavailable.
- LLM failure per agent → that brief marked failed; desk synthesis proceeds with the rest; Director notes missing inputs. Whole-run failure → previous cached report remains visible with a banner.
- Indicator functions never raise on short data; they return `None` fields and the UI renders "insufficient history".

## 9. Testing

- `pytest tests/` — indicators on synthetic series with known answers (EMA vs pandas ewm, RSI vs reference values, divergence on constructed pivots, S/R clustering, fib zone timestamps, squeeze scoring edge cases).
- Provider parsers tested against recorded JSON fixtures (captured 2026-09-09) — no network in tests.
- Streamlit smoke: `python -m streamlit run app.py --server.headless true` + HTTP 200 + no ERROR lines in log; screenshot check of each tab via Playwright.
- Phase 2: runner tested with a fake OpenRouter transport returning canned JSON; schema validation failures covered.

## 10. Phases / PRs

1. **PR1 – Phase 1**: package split, providers (with fixtures), indicators + tests, lightweight-charts, theme + tab overhaul, Active Trade inline, KPI strip. No LLM. Deletes all fabricated data paths.
2. **PR2 – Phase 2**: OpenRouter client, prompts, runner, report renderer, War Room viewer, per-category head-agent summaries, cost display.
3. **PR3 – Phase 3**: anchored verdict store + invalidation triggers, accuracy ledger (verdict vs realised move), squeeze/trap early-warning banners on Weekly/Monthly.

## 11. Out of scope (for now)

Multi-asset support, persistent storage across sessions, background scheduling, real liquidation heatmaps (needs Coinglass key), news feed integration (macro event stays a manually-edited config entry until a source is chosen), user auth.

## 12. Open risks

- Streamlit Cloud egress is US → Binance likely blocked; Bybit fallback unverified from US.
- Deribit rate limits on 900-instrument summary; mitigated by 120 s cache.
- OpenRouter cost per run depends on model choice; defaults must be cheap until the client sets a budget.
- Python 3.14 locally vs Streamlit Cloud default (3.12/3.13); pin `runtime.txt`/`requirements.txt` accordingly.

## 13. Addendum 2026-09-09 (late) — additional keyless feeds verified for Phase 2

Probed from the public-apis list. All keyless, all returned 200 with usable shapes.

| Feed | Endpoint | Gives us | Fills |
|---|---|---|---|
| Gemini exchange (US) | `api.gemini.com/v2/candles/btcusd/{15m,30m,1hr,6hr,1day}` + `/v1/pubticker/btcusd` | ~1 300 bars of 15m, 364 daily; resample 1hr→4h, 1day→1w | **Spot fallback** when Kraken fails (US-safe, unlike Binance) |
| Hyperliquid | `POST api.hyperliquid.xyz/info {"type":"metaAndAssetCtxs"}` | BTC funding (hourly), open interest, mark/oracle price | Second futures source alongside Binance/Bybit |
| CoinLobster | `coinlobster.com/api/public/liquidations` | 24h perp liquidations: total/long/short USD + counts, **24 hourly buckets**, per-coin (BTC row), biggest, venues (Binance, OKX, BitMEX, Bybit, Gate, HTX, Bitfinex, Hyperliquid) | **Liquidation data** for B3/R3 without Coinglass (aggregate, not price-level heatmap) |
| CoinLobster | `coinlobster.com/api/public/crypto-whales` | last 100 whale trades ≥$100K across 15 exchanges with side, size, and `marketConditions` (funding per exchange, OI per exchange) | Whale flow + cross-exchange funding for B4/R4, B2/R2 |
| CoinLobster | `coinlobster.com/api/public/whale-radar?window={1h,4h,24h}` | per-coin unusual-flow flags vs baseline (magnitude blurred on free tier) | Sentiment / flow anomaly signal |
| DefiLlama | `stablecoins.llama.fi/stablecoins?includePrices=false` | total stablecoin supply (≈$311B) with prev-day → supply delta | Stablecoin "dry powder" for B5/R5 |
| CoinGecko | `/coins/bitcoin/ohlc?vs_currency=usd&days=1` | 30-min candles, coarse | last-resort spot |

Dead or keyed now: CryptoCompare (401 without key, includes its news feed), CoinCap v3 (key), btcnode.uk Reddit (403 upstream), CoinLobster hyperliquid-whales (auth). News source remains open; macro calendar stays curated.

Phase 2 plan impact: add `core/data/gemini_spot.py` (fallback in `fetch_spot`), `core/data/hyperliquid.py`, `core/data/coinlobster.py`, `core/data/stablecoins.py`; extend `MarketSnapshot` with `liquidations`, `whales`, `stablecoins`; feed those domains to B2/R2, B3/R3, B4/R4, B5/R5. All are IP-rate-limited: cache 60–120 s and never fan out per agent.

## 14. Phase 3 design — anchored verdicts, early warning, accuracy ledger (2026-09-10)

**Persistence.** A small SQLite store (`core/store.py`) at `TI_DATA_DIR` (default `./data`, gitignored). Standard SQL only so a Postgres backend can be added behind the same interface later. Caveat: Streamlit Cloud's filesystem is wiped on reboot/redeploy, so the ledger there is best-effort until a hosted database (`DATABASE_URL`) is configured; this is stated in the UI.

**Anchored verdicts (`core/anchors.py`).** Per category the store keeps the last *anchored* direction call with its context: direction, confidence, anchor and invalidation levels, price, time, funding sign, OI 24h %, long/short ratio, squeeze score, Director classification (if any), and the next curated macro event. On every refresh the fresh deterministic call is computed but the **displayed** verdict is the anchored one until at least one trigger fires:

| Trigger | Rule |
|---|---|
| funding_flip | sign(funding_rate) changed |
| oi_shift | abs(OI 24h % now − OI 24h % at anchor) ≥ 5 |
| long_short_cross | long/short ratio crossed 1.0 |
| squeeze | squeeze score ≥ 60 now and was < 60 at anchor |
| invalidation | close beyond the anchored invalidation level |
| trap_change | Director classification differs from the one at anchor (only when a new report exists) |
| macro_window | a curated macro event is within 24 h and was not at anchor time |
| stale | anchor older than 4× the category horizon |

When a trigger fires the verdict re-anchors to the fresh call and an `anchor_log` row records category, from, to, price, fired triggers. If the fresh direction differs but no trigger fired, the card shows "live read: X (unconfirmed)" under the anchored verdict.

**Early warning (`core/indicators/early_warning.py`).** Deterministic, per category, shown as banners on Weekly/Monthly (and as a chip elsewhere):
- BULL_TRAP_FORMING: price within 0.5 ATR below a resistance with ≥2 touches, regular bearish RSI divergence (confirmed or forming), funding above its 7-day mean or long/short > 1.2.
- BEAR_TRAP_FORMING: mirror (support, bullish divergence, funding below mean or long/short < 0.8).
- SQUEEZE_FORMING: squeeze score 40–59 and higher than the last stored score for that category; SQUEEZE_WARNING at ≥60.

**Accuracy ledger (`core/accuracy.py`).**
- Every (re)anchor logs a *call*: category, direction, price, horizon = category `horizon_bars` × timeframe. When the horizon has elapsed the call is scored against the current price: BULLISH hit if move > +0.25σ, BEARISH hit if move < −0.25σ, NEUTRAL hit if |move| ≤ 0.5σ (σ = expected-range sigma at call time).
- Every Director report logs classification and price; scored at 24 h and 7 d: BULL_TRAP hit if price lower, BEAR_TRAP hit if higher, RANGE_TRAP hit if |move| < 1σ(24h), NO_TRAP hit if the move agrees with the majority category direction at report time.
- War Room shows hit rate per category and per classification, sample counts, and the last 20 scored items. Small samples are labelled as such; nothing is claimed below 10 samples.

**Out of scope for Phase 3:** cross-viewer auth, Postgres backend (interface-ready only), news-driven triggers (no source yet).

## 15. News source decision (2026-09-10)

Client suggested Forex Factory. Its weekly calendar JSON (`nfs.faireconomy.media/ff_calendar_thisweek.json`, keyless, verified 200) is an economic calendar, not headlines, which matches the original News tab (single high-impact macro focus with consensus forecast and previous value). Adopted as the ninth source (`core/data/calendar.py`, `CalendarSnapshot`, `CalendarSnapshot.upcoming(now, window, min_impact, countries)`). Drives the Weekly/Monthly macro card, the `macro_window` anchor trigger and the agents' macro context. Headline news remains out of scope until a keyless provider exists.

## 16. Levels, Fibonacci and patterns rework (2026-09-12)

Audit on live Kraken candles showed the S/R "Nx" label counted confirmed swing pivots in the last 200 bars, not the reactions a trader sees (e.g. a 1h resistance labelled 1x had 9 visible reactions), never counted the last 5 bars, and could draw a line between its reactions. Fib time zones used a one-bar unit from a single anchor (differs from TradingView's two-point tool), fed agents only a count, and predicted nothing. Pattern detection from the client's ask ("trend line or patterns in making") was never built.

- **S/R** (`core/indicators/levels.py`): candidates from swing pivots over the same 300 bars the chart draws; groups capped at 2×tolerance wide; line at the median of in-band wicks; strength = distinct reactions (bars whose high or low is within ±0.35 ATR, consecutive touches ≤3 bars apart = one reaction), current bars included. Live re-audit: every label equals an independent recount.
- **Fib time** (`core/indicators/fib_time.py`): TradingView two-point geometry. Anchor 1 = earlier of the window's major high/low; anchor 2 = end of the following leg (running extreme until a 1.5 ATR retrace, leg ≥ 1 ATR); unit = bars between them (min 2); zones at 0,1,2,3,5,8,13… units. Retracements 0.236–0.786 of the major leg.
- **Reversal window** (`core/indicators/reversal.py`): ACTIVE when a fib zone is within ±min(3, unit/3) bars of now AND price has confluence (≥2-touch S/R within 0.5 ATR, recent regular RSI divergence, RSI ≥70/≤30, 0.5/0.618 retracement within 0.3 ATR, trendline within 0.5 ATR, forming reversal pattern); TIME_ONLY without confluence; else UPCOMING. Bias = majority of confluence votes. Raises a REVERSAL_WINDOW early warning.
- **Patterns** (`core/indicators/patterns.py`): double top/bottom (incl. second top/bottom being tested now), head & shoulders and inverse, ascending/descending/symmetrical triangles, rising/falling wedges, rising/falling channels, ranges; forming vs freshly confirmed (breakouts older than 5 bars dropped), breakout/target/invalidation, apex; duplicates folded (double inside a triangle or H&S; simultaneous double top + double bottom → one range).
- Surfaced on every category: chart overlays (retracement lines, dashed pattern lines with labels, two-point fib zones), a "Chart patterns" card, a "Fibonacci time & reversal window" card, summary sentences, and full fib/pattern/reversal data in the agents' category payload with a Director instruction to use them.
