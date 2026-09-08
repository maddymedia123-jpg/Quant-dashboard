Title: Phase 1: real data providers, indicators, TradingView charts, UI overhaul

## What
- Splits the single-file app into `core/data`, `core/indicators`, `ui` packages; `app.py` is a thin shell.
- Real keyless feeds: Kraken spot, Binance USD-M futures (Bybit fallback), Deribit options, Fear & Greed. Every feed carries an availability flag; missing values render as "—".
- Deterministic per-category analysis (Live 15m, Intraday 1h, Weekly 4h, Monthly 1d): EMA 21/50/100/200, RSI + divergence (confirmed/forming), swing S/R, trendlines, fib time zones, volatility regime + expected range, squeeze risk, direction call with anchor/invalidation.
- TradingView lightweight-charts per category with EMAs, S/R, trendlines, fib verticals, now-line, projection cone, RSI pane with divergence markers.
- Active Trade stress-test inline with scalp/swing classification, structures above/below, liquidation distance, re-anchor triggers.
- War Room shows the Section 1 raw metric snapshot and the session audit ledger; Run Analysis button present but disabled until Phase 2.
- `TI_OFFLINE_FIXTURES=1` serves recorded fixtures for offline UI checks (sources labelled "fixture").

## Removed
- Hard-coded funding / open-interest constants, fixed liquidation clusters, random "predicted candles", ccxt and plotly dependencies.

## Test
- `pytest`: 53 tests (indicators on synthetic series; provider parsers on recorded fixtures; no network).
- Manual: Live Recon, Active Trade (form + result) and War Room tabs screenshot-verified on fixture data; chart component confirmed drawing candles, 4 EMAs, S/R labels, fib verticals, now-line, projection, RSI pane.
- Live loader verified against Binance, Deribit and alternative.me from the dev machine; Kraken could not be resolved by the dev machine's DNS at the time, so the live spot path is verified only via the parser fixtures and the graceful "spot unavailable" state. Please confirm on Streamlit Cloud.

## Notes for Streamlit Cloud
- Python 3.12+ in app settings. No secrets needed for Phase 1.
- Binance is geo-blocked from US egress; the app falls back to Bybit automatically.

Spec: `docs/superpowers/specs/2026-09-09-trap-intelligence-design.md` · Plan: `docs/superpowers/plans/2026-09-09-phase1-data-indicators-charts.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0186foiQoKUv1nKDTFvRPiJ5
