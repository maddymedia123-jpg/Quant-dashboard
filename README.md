# Trap Intelligence — BTC market-surveillance terminal

Streamlit dashboard with TradingView-style charts and deterministic market structure per timeframe
(Live 15m, Intraday 1h, Weekly 4h, Monthly 1d): EMA 21/50/100/200, RSI with divergence detection
(confirmed and forming), swing support/resistance, trendlines, Fibonacci time zones, volatility regime and
expected range, squeeze risk, a direction call with anchor and invalidation, and an active-trade stress-test.
Phase 2 adds the 13-agent Director / Bullish desk / Bearish desk report via Gemini or OpenRouter.

## Data (keyless public APIs)

Kraken spot OHLCV (Gemini exchange fallback) · Binance USD-M futures (Bybit fallback) for funding, open
interest, long/short and taker flow · Hyperliquid funding and OI · CoinLobster 24h liquidations, $100K+
whale trades and unusual-flow radar · Deribit options for max pain, put/call and IV skew · DefiLlama
stablecoin supply · alternative.me Fear & Greed · Forex Factory weekly economic calendar (high-impact USD prints with forecast/previous; drives the macro-window trigger).
Every feed reports availability; missing data is shown as `—`, never invented.

## Run

    python -m venv .venv
    .venv/Scripts/pip install -r requirements-dev.txt
    .venv/Scripts/python -m pytest
    .venv/Scripts/python -m streamlit run app.py

Offline UI check without network: `TI_OFFLINE_FIXTURES=1 .venv/Scripts/python -m streamlit run app.py`
(serves the recorded fixtures under `tests/fixtures`; sources are labelled "fixture").

## Auto-refresh

The sidebar "Auto-refresh" selector (Off / 30s / 60s / 120s) reruns the whole page on a timer via a
Streamlit fragment, so the session (including the last Director report) is kept. Data caches still apply,
so the timer adds no API load; spot refreshes every 25s and context feeds every 120s at most.

## Agents (Phase 2)

`Run Analysis` in the sidebar runs the Trap Intelligence pipeline on demand: ten specialists
(Bullish B1-B5 and Bearish R1-R5 across options, leverage, liquidity, flow, sentiment) plus a
volatility agent run in parallel, each desk synthesises its briefs, an accuracy/correlation agent
cross-checks both desks, and the Director issues the final report: contradictions, convergence,
blind spots, probability-weighted scenarios, trap classification and an actionable trade plan,
plus a plain-English paragraph per timeframe shown on each tab. Results are cached for the session.

Secrets (`.streamlit/secrets.toml` locally, Streamlit Cloud Secrets in production):

    GEMINI_API_KEY = "..."          # default provider (Gemini generateContent, JSON mode)
    OPENROUTER_API_KEY = "..."      # optional alternative
    # optional overrides
    LLM_PROVIDER = "gemini"         # or "openrouter"
    SPECIALIST_MODEL = "gemini-3.5-flash-lite"
    DIRECTOR_MODEL = "gemini-3.7-flash"
    LLM_MAX_CONCURRENCY = "4"

One run is 15 LLM calls (plus at most one JSON-repair retry per call). Agents only ever see the
numbers in the data block and an explicit list of unavailable sources; they are instructed never to
invent a figure. On the Gemini free tier the Pro models return quota errors, so the Director runs on
Flash; on OpenRouter without credits only the free Nemotron models are available.

## Anchoring & accuracy (Phase 3)

Each timeframe's displayed direction is **anchored**: it holds until a trigger fires (funding flips
sign, open interest moves 5 points, long/short crosses 1.0, squeeze score reaches 60, price closes past
the invalidation level, the Director's trap classification changes, a curated macro event is within 24h,
or the anchor is stale). When the fresh read disagrees without a trigger, the card shows it as an
unconfirmed live read. Weekly and Monthly tabs raise early-warning banners for bull/bear traps and
squeezes as they form.

Every anchor change logs a direction call; every Director report logs its classification. Calls are
scored once their horizon elapses (Live 1h, Intraday 24h, Weekly 7d, Monthly 30d) and reports at 24h and
7d, using the rules in the spec (§14). The War Room shows hit rates per category and classification with
sample sizes; anything under 10 samples is labelled indicative.

Storage is SQLite at `TI_DATA_DIR` (default `./data`, gitignored). On Streamlit Cloud that file resets on
reboot or redeploy; point `TI_DATA_DIR` at a mounted volume or wait for the hosted-database option.

## Layout

`core/data` providers → `core/indicators` pure functions → `core/agents` (LLM client, prompts, context, runner, report) → `core/store`, `core/anchors`, `core/accuracy` (persistence, anchoring, scoring) → `ui/` renderers → `app.py` shell.
Design spec and implementation plans live under `docs/superpowers/`.
