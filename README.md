# Trap Intelligence — BTC market-surveillance terminal

Streamlit dashboard with TradingView-style charts and deterministic market structure per timeframe
(Live 15m, Intraday 1h, Weekly 4h, Monthly 1d): EMA 21/50/100/200, RSI with divergence detection
(confirmed and forming), swing support/resistance, trendlines, Fibonacci time zones, volatility regime and
expected range, squeeze risk, a direction call with anchor and invalidation, and an active-trade stress-test.
Phase 2 adds the 13-agent Director / Bullish desk / Bearish desk report via OpenRouter.

## Data (keyless public APIs)

Kraken spot OHLCV · Binance USD-M futures (Bybit fallback) for funding, open interest, long/short and taker
flow · Deribit options for max pain, put/call and IV skew · alternative.me Fear & Greed.
Every feed reports availability; missing data is shown as `—`, never invented.

## Run

    python -m venv .venv
    .venv/Scripts/pip install -r requirements-dev.txt
    .venv/Scripts/python -m pytest
    .venv/Scripts/python -m streamlit run app.py

Offline UI check without network: `TI_OFFLINE_FIXTURES=1 .venv/Scripts/python -m streamlit run app.py`
(serves the recorded fixtures under `tests/fixtures`; sources are labelled "fixture").

## Layout

`core/data` providers → `core/indicators` pure functions → `ui/` renderers → `app.py` shell.
Design spec and implementation plans live under `docs/superpowers/`.
