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
stablecoin supply · alternative.me Fear & Greed.
Every feed reports availability; missing data is shown as `—`, never invented.

## Run

    python -m venv .venv
    .venv/Scripts/pip install -r requirements-dev.txt
    .venv/Scripts/python -m pytest
    .venv/Scripts/python -m streamlit run app.py

Offline UI check without network: `TI_OFFLINE_FIXTURES=1 .venv/Scripts/python -m streamlit run app.py`
(serves the recorded fixtures under `tests/fixtures`; sources are labelled "fixture").

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

## Layout

`core/data` providers → `core/indicators` pure functions → `core/agents` (LLM client, prompts, context, runner, report) → `ui/` renderers → `app.py` shell.
Design spec and implementation plans live under `docs/superpowers/`.
