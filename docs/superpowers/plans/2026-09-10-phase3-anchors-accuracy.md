# Phase 3 — Anchored verdicts, early warning, accuracy ledger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Verdicts stay anchored until a defined trigger fires; Weekly/Monthly surface trap and squeeze setups as they form; every direction call and Director report is logged and scored against realised price so the War Room shows hit rates.

**Architecture:** `core/store.py` (SQLite, standard SQL, `TI_DATA_DIR`) → `core/anchors.py` (context, triggers, anchor step) → `core/accuracy.py` (scoring + summary) → `core/indicators/early_warning.py` (deterministic setups) → `ui/panels.py` + `app.py`. All pure logic unit-tested against a temp database; no network.

**Spec:** `docs/superpowers/specs/2026-09-09-trap-intelligence-design.md` §14

## Global Constraints
- Branch `feature/trap-intelligence-phase3` from `main` (PR1+PR2 merged). Commit trailer as before (session `01Ka9J6mCq2HunYCJTYxG878`).
- `data/` is gitignored. Store path from `TI_DATA_DIR` (default `./data/trap_intel.sqlite`); `":memory:"` allowed for tests.
- No claims of accuracy below 10 samples; UI labels small samples.

## Tasks

### P3-0 `core/store.py` + `tests/test_store.py`
```python
class Store:
    def __init__(self, path: str | None = None)            # creates schema; ":memory:" supported
    # anchors
    def get_anchor(self, category: str) -> dict | None      # {"payload": dict, "since_ms": int}
    def set_anchor(self, category: str, payload: dict, since_ms: int) -> None
    def log_anchor_change(self, ts_ms: int, category: str, from_dir: str | None, to_dir: str, price: float, triggers: list[str]) -> None
    def anchor_log(self, limit: int = 50) -> list[dict]
    # calls
    def add_call(self, ts_ms, category, direction, confidence, price, sigma_pct, horizon_ms) -> int
    def due_calls(self, now_ms: int) -> list[dict]           # unscored and ts_ms + horizon_ms <= now_ms
    def score_call(self, call_id: int, scored_ms: int, realised_pct: float, hit: bool) -> None
    def calls(self, limit: int = 200, scored_only: bool = False) -> list[dict]
    # reports
    def add_report(self, report_id, ts_ms, classification, price, sigma24_pct, majority_dir) -> bool   # False if duplicate
    def reports(self, limit: int = 50) -> list[dict]
    def score_report(self, row_id: int, horizon: str, move_pct: float, hit: bool) -> None   # horizon "24h" | "7d"
    # signals
    def add_signal(self, ts_ms, category, squeeze_score, price) -> None
    def last_signal(self, category: str, before_ms: int | None = None) -> dict | None
    def close(self) -> None
```
Tests: schema creates; anchor round-trip JSON; due_calls respects horizon; score_call marks; add_report dedups; last_signal returns most recent before a timestamp.

### P3-1 `core/anchors.py` + `tests/test_anchors.py`
```python
TRIGGER_OI_PCT = 5.0; SQUEEZE_LEVEL = 60; STALE_MULT = 4; MACRO_WINDOW_MS = 86_400_000
@dataclass
class AnchorContext: direction: str; confidence: float; anchor: float | None; invalidation: float | None; price: float; ts_ms: int;
    funding_sign: int | None; oi_change_24h_pct: float | None; long_short_ratio: float | None; squeeze_score: int | None;
    classification: str | None; macro_within_24h: bool; sigma_pct: float | None; drivers: list[str]; flips_if: str
def macro_within(now_ms: int, events=MACRO_EVENTS) -> bool
def build_context(a: CategoryAnalysis, m: MarketSnapshot, classification: str | None, now_ms: int) -> AnchorContext
def evaluate_triggers(prev: AnchorContext, cur: AnchorContext, horizon_ms: int) -> list[str]   # names from spec table
@dataclass
class AnchoredVerdict: category: str; anchored: AnchorContext; live: AnchorContext; since_ms: int; triggers_fired: list[str]; changed: bool
    @property unconfirmed -> bool   # live.direction != anchored.direction
def horizon_ms_for(key: str) -> int
def anchor_step(store, key, a, m, classification, now_ms) -> AnchoredVerdict   # persists anchor/log/call/signal
```
Tests: each trigger fires alone and not otherwise; first run anchors and logs; second run without triggers keeps anchor and marks unconfirmed when live differs; `macro_within` true the day before the curated event.

### P3-2 `core/accuracy.py` + `tests/test_accuracy.py`
```python
def score_due_calls(store, now_ms: int, price: float) -> int
def record_report(store, report, price: float, sigma24_pct: float | None, majority_dir: str, now_ms: int) -> bool
def score_reports(store, now_ms: int, price: float) -> int
@dataclass class HitRate: n: int; hits: int; @property rate -> float | None
def summary(store) -> dict   # {"by_category": {...}, "by_classification_24h": {...}, "recent_calls": [...], "recent_reports": [...], "total_scored": int}
def majority_direction(analyses) -> str
```
Rules exactly as spec §14. Tests: bullish call scored hit/miss around 0.25σ; neutral band; BULL_TRAP 24h hit when price lower; RANGE_TRAP 7d uses σ·√7; NO_TRAP follows majority; summary counts.

### P3-3 `core/indicators/early_warning.py` + `tests/test_early_warning.py`
```python
@dataclass class EarlyWarning: kind: str; level: str; drivers: list[str]     # kinds: BULL_TRAP_FORMING | BEAR_TRAP_FORMING | SQUEEZE_FORMING | SQUEEZE_WARNING; level forming|warning
def early_warnings(a: CategoryAnalysis, futures: FuturesSnapshot, last_squeeze: int | None) -> list[EarlyWarning]
```
Tests on constructed analyses (use real `analyze_category` on synthetic frames plus monkeypatched fields).

### P3-4 UI (`ui/panels.py`, `app.py`) + tests
- `anchored_direction_html(a, v: AnchoredVerdict) -> str` (anchored direction, since, price at anchor, anchor/invalidation, "re-anchored now: …" when changed, "live read X (unconfirmed)" when differs, watched triggers list).
- `early_warning_html(ws: list[EarlyWarning]) -> str | None`
- `accuracy_html(s: dict) -> str` (hit-rate table per category and classification with n; small-sample label; last scored items)
- `app.py`: `@st.cache_resource def _store()`; per category `anchor_step`; `score_due_calls`; when a new report is produced `record_report`; `score_reports`; audit expander reads `store.anchor_log()`; War Room gets Accuracy panel; Weekly/Monthly banners from `early_warnings`; sidebar chip about ledger persistence.

### P3-5 Docs + PR3
README section "Anchoring & accuracy", PR3-body.md, push, compare URL.
