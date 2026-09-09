Title: Phase 3: anchored verdicts, early-warning setups, persistent accuracy ledger

## What
- **Anchored verdicts.** The direction shown per timeframe is persisted and held until a defined trigger fires: funding flip, OI shift ≥ 5 pts, long/short crossing 1.0, squeeze ≥ 60, close past invalidation, Director trap-classification change, macro event within 24h, or a stale anchor (4× horizon). Re-anchors are logged with the triggers that fired; a disagreeing fresh read is shown as "live read, unconfirmed".
- **Early warning.** Deterministic BULL/BEAR TRAP FORMING (price at a ≥2-touch level + regular RSI divergence + funding/positioning skew) and SQUEEZE FORMING/WARNING banners on Weekly and Monthly.
- **Accuracy ledger.** Every anchor change logs a call scored after its horizon (±0.25σ rule, NEUTRAL within 0.5σ); every Director report logs its classification scored at 24h and 7d (BULL_TRAP down, BEAR_TRAP up, RANGE_TRAP inside σ, NO_TRAP with the majority direction). War Room shows hit rates by category and classification with sample sizes; <10 samples labelled indicative. The session-only audit list is replaced by a persistent anchor log.
- **Persistence.** `core/store.py`, SQLite via standard SQL at `TI_DATA_DIR` (default `./data`, gitignored), `:memory:` for tests. Interface is ready for a Postgres backend.

## Test
- `pytest`: 96 tests (16 new: store CRUD, each trigger in isolation, anchor hold/re-anchor flow, scoring rules at both horizons, early-warning ingredients, panel HTML).
- Manual: app boots in fixture mode, anchored cards render with "anchored … ago", audit ledger shows the initial anchors, accuracy panel shows the open-call count.

## Notes
- On Streamlit Cloud the SQLite file resets on reboot/redeploy. Set `TI_DATA_DIR` to persistent storage or add a hosted database in a follow-up.
- Spec §14 · Plan `docs/superpowers/plans/2026-09-10-phase3-anchors-accuracy.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ka9J6mCq2HunYCJTYxG878
