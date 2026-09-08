# Trap Intelligence Dashboard — Design Spec

Date: 2026-09-09
Status: Approved (design), Phase 1 in progress
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
