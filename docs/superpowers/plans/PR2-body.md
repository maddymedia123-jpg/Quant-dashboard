Title: Phase 2: extra keyless feeds + 13-agent Trap Intelligence pipeline (Gemini / OpenRouter)

Base: `feature/trap-intelligence-phase1` (retarget to `main` once PR1 merges).

## What
- **New feeds, all keyless and fixture-tested:** Gemini exchange candles as the US-safe spot fallback (Kraken → Gemini chain, 4h/1w resampled), Hyperliquid funding/OI, CoinLobster 24h liquidations (long/short, hourly, BTC row, venues), $100K+ whale trades with per-exchange funding/OI, unusual-flow radar, DefiLlama stablecoin supply. `MarketSnapshot` grows to eight sources; spot is cached 25 s and the slow context feeds 120 s so IP-rate-limited sources are never hammered.
- **Agent layer (`core/agents/`):** provider-agnostic JSON-mode `LLMClient` (Gemini `generateContent` default, OpenRouter optional; retries on 429/5xx/timeouts, one JSON-repair call), pydantic output schemas with tolerant coercion (`"60%"`, `"7/10"`) and scenario-probability normalisation, prompts composed from the client's orchestration text (B1–B5, R1–R5, desk synthesis, volatility agent, accuracy/correlation agent, Director), domain-scoped context builders that pass only each agent's data plus an explicit `unavailable` list, a three-phase async runner with per-agent failure isolation and a Director model fallback, and a markdown renderer for the client's Sections 1–6 plus an appendix (volatility, accuracy, agent run table, tokens, cost, wall time).
- **UI:** `Run Analysis` enabled when a key is present, live per-agent progress, War Room renders the full report with download, specialist briefs expander, and each category tab's head-agent card now shows the Director's plain-English paragraph for that timeframe. New `Liq 24h` KPI tile; raw metric table extended with the new sources.

## Verified end to end (real Gemini, fixture market data)
- Report `TIR-20260909-001811`: 15/16 agent calls ok, 75,303 tokens, 37 s wall, free tier (no cost reported). All six sections rendered; Director classification BULL TRAP with per-category summaries quoting chart levels.
- The one failure was `gemini-3.8-flash` returning 503 for the Director; the runner fell back to the specialist model and produced the report. Default Director model is now `gemini-3.7-flash` (200 on repeated probes).
- `pytest`: 80 tests, no network (recorded fixtures + fake LLM client).

## Notes for Streamlit Cloud
- Add `GEMINI_API_KEY` (and optionally `OPENROUTER_API_KEY`, `LLM_PROVIDER`, `SPECIALIST_MODEL`, `DIRECTOR_MODEL`) in the app's Secrets.
- Gemini key is free tier: Pro models return 429 quota; Flash models work. OpenRouter account has $0 credits: only free Nemotron models are reachable. Fund one of them for a stronger Director.
- One run ≈ 15 calls; on Gemini free-tier quotas that is comfortably within limits for on-demand use, not for auto-refresh.

Spec: `docs/superpowers/specs/2026-09-09-trap-intelligence-design.md` (§6, §13) · Plan: `docs/superpowers/plans/2026-09-09-phase2-agents.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0186foiQoKUv1nKDTFvRPiJ5
