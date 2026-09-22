"""SQLite persistence for anchored verdicts, the Live Recon 4h anchor and its side notes, the anchor log,
accuracy calls/reports and signal history.

Standard SQL only so a Postgres backend can sit behind the same interface later.
Path comes from TI_DATA_DIR (default ./data). ":memory:" is supported for tests."""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading

# Live Recon anchors were pinned before every category had a war room. They carry over as "live",
# and re-running this on every open is harmless: both statements skip rows already copied.
MIGRATE = """
INSERT OR IGNORE INTO recon_anchors (category, window_open_ms, window_close_ms, published_ms, bias, margin,
    confidence, trap, price, payload)
SELECT 'live', window_open_ms, window_close_ms, published_ms, bias, margin, confidence, trap, price, payload
FROM live_anchors;
INSERT INTO recon_side_notes (category, window_open_ms, ts_ms, kind, dedup_key, headline, detail, price)
SELECT 'live', l.window_open_ms, l.ts_ms, l.kind, l.dedup_key, l.headline, l.detail, l.price
FROM live_side_notes l
WHERE NOT EXISTS (SELECT 1 FROM recon_side_notes r WHERE r.category = 'live'
                  AND r.window_open_ms = l.window_open_ms AND r.dedup_key = l.dedup_key);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS anchors (
    category TEXT PRIMARY KEY, payload TEXT NOT NULL, since_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS anchor_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, category TEXT NOT NULL,
    from_dir TEXT, to_dir TEXT NOT NULL, price REAL, triggers TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, category TEXT NOT NULL, direction TEXT NOT NULL,
    confidence REAL, price REAL NOT NULL, sigma_pct REAL, horizon_ms INTEGER NOT NULL,
    scored_ms INTEGER, realised_pct REAL, hit INTEGER);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT UNIQUE NOT NULL, ts_ms INTEGER NOT NULL,
    classification TEXT NOT NULL, price REAL NOT NULL, sigma24_pct REAL, majority_dir TEXT,
    move_24h_pct REAL, hit_24h INTEGER, move_7d_pct REAL, hit_7d INTEGER);
CREATE TABLE IF NOT EXISTS live_anchors (
    window_open_ms INTEGER PRIMARY KEY, window_close_ms INTEGER NOT NULL, published_ms INTEGER NOT NULL,
    bias TEXT NOT NULL, margin REAL, confidence REAL, trap TEXT, price REAL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS live_side_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, window_open_ms INTEGER NOT NULL, ts_ms INTEGER NOT NULL,
    kind TEXT NOT NULL, dedup_key TEXT NOT NULL, headline TEXT NOT NULL, detail TEXT, price REAL);
CREATE INDEX IF NOT EXISTS idx_side_notes_window ON live_side_notes (window_open_ms, ts_ms);
CREATE TABLE IF NOT EXISTS recon_anchors (
    category TEXT NOT NULL, window_open_ms INTEGER NOT NULL, window_close_ms INTEGER NOT NULL,
    published_ms INTEGER NOT NULL, bias TEXT NOT NULL, margin REAL, confidence REAL, trap TEXT, price REAL,
    payload TEXT NOT NULL, PRIMARY KEY (category, window_open_ms));
CREATE TABLE IF NOT EXISTS recon_side_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, window_open_ms INTEGER NOT NULL,
    ts_ms INTEGER NOT NULL, kind TEXT NOT NULL, dedup_key TEXT NOT NULL, headline TEXT NOT NULL,
    detail TEXT, price REAL);
CREATE INDEX IF NOT EXISTS idx_recon_notes ON recon_side_notes (category, window_open_ms, ts_ms);
CREATE TABLE IF NOT EXISTS recon_scores (
    category TEXT NOT NULL, window_open_ms INTEGER NOT NULL, scored_ms INTEGER NOT NULL,
    open_price REAL NOT NULL, close_price REAL NOT NULL, move_pct REAL NOT NULL, realised TEXT NOT NULL,
    bias TEXT NOT NULL, direction_hit INTEGER NOT NULL, trap TEXT, trap_hit INTEGER,
    bull_brier REAL, bear_brier REAL, PRIMARY KEY (category, window_open_ms));
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, category TEXT NOT NULL,
    squeeze_score INTEGER, price REAL);
CREATE TABLE IF NOT EXISTS futures_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, funding_rate REAL, open_interest_usd REAL);
CREATE INDEX IF NOT EXISTS idx_futures_samples_ts ON futures_samples (ts_ms);
CREATE INDEX IF NOT EXISTS idx_calls_open ON calls (scored_ms, ts_ms);
CREATE INDEX IF NOT EXISTS idx_signals_cat ON signals (category, ts_ms);
"""


def default_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("TI_DATA_DIR", "data")) / "trap_intel.sqlite"


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


class Store:
    def __init__(self, path: str | os.PathLike | None = None):
        if path == ":memory:":
            self.path = ":memory:"
        else:
            p = pathlib.Path(path) if path else default_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.executescript(MIGRATE)
            self._conn.commit()

    # ---- anchors ----
    def get_anchor(self, category: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT payload, since_ms FROM anchors WHERE category=?", (category,)).fetchone()
        return {"payload": json.loads(r["payload"]), "since_ms": int(r["since_ms"])} if r else None

    def set_anchor(self, category: str, payload: dict, since_ms: int) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO anchors (category, payload, since_ms) VALUES (?,?,?) "
                               "ON CONFLICT(category) DO UPDATE SET payload=excluded.payload, since_ms=excluded.since_ms",
                               (category, json.dumps(payload, default=str), int(since_ms)))
            self._conn.commit()

    def log_anchor_change(self, ts_ms: int, category: str, from_dir: str | None, to_dir: str, price: float | None, triggers: list[str]) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO anchor_log (ts_ms, category, from_dir, to_dir, price, triggers) VALUES (?,?,?,?,?,?)",
                               (int(ts_ms), category, from_dir, to_dir, price, json.dumps(triggers)))
            self._conn.commit()

    def anchor_log(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = _rows(self._conn.execute("SELECT * FROM anchor_log ORDER BY ts_ms DESC, id DESC LIMIT ?", (int(limit),)))
        for r in rows:
            r["triggers"] = json.loads(r["triggers"])
        return rows

    # ---- calls ----
    def add_call(self, ts_ms: int, category: str, direction: str, confidence: float | None, price: float,
                 sigma_pct: float | None, horizon_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO calls (ts_ms, category, direction, confidence, price, sigma_pct, horizon_ms) VALUES (?,?,?,?,?,?,?)",
                (int(ts_ms), category, direction, confidence, float(price), sigma_pct, int(horizon_ms)))
            self._conn.commit()
            return int(cur.lastrowid)

    def due_calls(self, now_ms: int) -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute("SELECT * FROM calls WHERE scored_ms IS NULL AND ts_ms + horizon_ms <= ? ORDER BY ts_ms", (int(now_ms),)))

    def score_call(self, call_id: int, scored_ms: int, realised_pct: float, hit: bool) -> None:
        with self._lock:
            self._conn.execute("UPDATE calls SET scored_ms=?, realised_pct=?, hit=? WHERE id=?", (int(scored_ms), float(realised_pct), 1 if hit else 0, int(call_id)))
            self._conn.commit()

    def calls(self, limit: int = 200, scored_only: bool = False) -> list[dict]:
        q = "SELECT * FROM calls" + (" WHERE scored_ms IS NOT NULL" if scored_only else "") + " ORDER BY ts_ms DESC LIMIT ?"
        with self._lock:
            return _rows(self._conn.execute(q, (int(limit),)))

    # ---- reports ----
    def add_report(self, report_id: str, ts_ms: int, classification: str, price: float, sigma24_pct: float | None, majority_dir: str | None) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO reports (report_id, ts_ms, classification, price, sigma24_pct, majority_dir) VALUES (?,?,?,?,?,?)",
                (report_id, int(ts_ms), classification, float(price), sigma24_pct, majority_dir))
            self._conn.commit()
            return cur.rowcount == 1

    def reports(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute("SELECT * FROM reports ORDER BY ts_ms DESC LIMIT ?", (int(limit),)))

    def score_report(self, row_id: int, horizon: str, move_pct: float, hit: bool) -> None:
        col_move, col_hit = ("move_24h_pct", "hit_24h") if horizon == "24h" else ("move_7d_pct", "hit_7d")
        with self._lock:
            self._conn.execute(f"UPDATE reports SET {col_move}=?, {col_hit}=? WHERE id=?", (float(move_pct), 1 if hit else 0, int(row_id)))
            self._conn.commit()

    # ---- signals ----
    def add_signal(self, ts_ms: int, category: str, squeeze_score: int | None, price: float | None) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO signals (ts_ms, category, squeeze_score, price) VALUES (?,?,?,?)", (int(ts_ms), category, squeeze_score, price))
            self._conn.commit()

    def last_signal(self, category: str, before_ms: int | None = None) -> dict | None:
        q = "SELECT * FROM signals WHERE category=?" + (" AND ts_ms < ?" if before_ms is not None else "") + " ORDER BY ts_ms DESC, id DESC LIMIT 1"
        args = (category, int(before_ms)) if before_ms is not None else (category,)
        with self._lock:
            r = self._conn.execute(q, args).fetchone()
        return dict(r) if r else None

    # ---- futures samples (for derived OI change / funding mean) ----
    def add_futures_sample(self, ts_ms: int, funding_rate: float | None, open_interest_usd: float | None) -> None:
        with self._lock:
            last = self._conn.execute("SELECT ts_ms FROM futures_samples ORDER BY ts_ms DESC LIMIT 1").fetchone()
            if last and int(ts_ms) - int(last["ts_ms"]) < 60_000:
                return  # at most one sample per minute
            self._conn.execute("INSERT INTO futures_samples (ts_ms, funding_rate, open_interest_usd) VALUES (?,?,?)",
                               (int(ts_ms), funding_rate, open_interest_usd))
            self._conn.execute("DELETE FROM futures_samples WHERE ts_ms < ?", (int(ts_ms) - 8 * 86_400_000,))
            self._conn.commit()

    def futures_sample_at_or_before(self, ts_ms: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM futures_samples WHERE ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1", (int(ts_ms),)).fetchone()
        return dict(r) if r else None

    def oldest_futures_sample(self) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM futures_samples ORDER BY ts_ms ASC LIMIT 1").fetchone()
        return dict(r) if r else None

    def futures_samples_since(self, ts_ms: int) -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute("SELECT * FROM futures_samples WHERE ts_ms >= ? ORDER BY ts_ms ASC", (int(ts_ms),)))

    # ---- war rooms: one anchored summary per category per window, insert-once, never updated ----
    def put_live_anchor(self, window_open_ms: int, window_close_ms: int, published_ms: int, bias: str,
                        margin: float | None, confidence: float | None, trap: str | None, price: float | None,
                        payload: dict, category: str = "live") -> bool:
        """Publish the anchor for a window. Returns False when one already exists: the client's rule
        that a summary must never be overwritten mid-candle is enforced here, not in the caller."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO recon_anchors (category, window_open_ms, window_close_ms, published_ms, bias, "
                "margin, confidence, trap, price, payload) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(category, window_open_ms) DO NOTHING",
                (category, window_open_ms, window_close_ms, published_ms, bias, margin, confidence, trap,
                 price, json.dumps(payload)))
            self._conn.commit()
        return cur.rowcount > 0

    def get_live_anchor(self, window_open_ms: int | None = None, category: str = "live") -> dict | None:
        sql = "SELECT * FROM recon_anchors WHERE category=? "
        args: tuple = (category,)
        if window_open_ms is None:
            sql += "ORDER BY window_open_ms DESC LIMIT 1"
        else:
            sql += "AND window_open_ms=?"
            args = (category, window_open_ms)
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def live_anchors(self, limit: int = 20, category: str = "live") -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute(
                "SELECT category, window_open_ms, window_close_ms, published_ms, bias, margin, confidence, "
                "trap, price FROM recon_anchors WHERE category=? ORDER BY window_open_ms DESC LIMIT ?",
                (category, limit)))

    def add_side_note(self, window_open_ms: int, ts_ms: int, kind: str, dedup_key: str, headline: str,
                      detail: str | None = None, price: float | None = None, category: str = "live") -> bool:
        """Append-only. Returns False when the same change was already noted in this window."""
        with self._lock:
            seen = self._conn.execute(
                "SELECT 1 FROM recon_side_notes WHERE category=? AND window_open_ms=? AND dedup_key=?",
                (category, window_open_ms, dedup_key)).fetchone()
            if seen:
                return False
            self._conn.execute(
                "INSERT INTO recon_side_notes (category, window_open_ms, ts_ms, kind, dedup_key, headline, "
                "detail, price) VALUES (?,?,?,?,?,?,?,?)",
                (category, window_open_ms, ts_ms, kind, dedup_key, headline, detail, price))
            self._conn.commit()
        return True

    def side_notes(self, window_open_ms: int, category: str = "live") -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute(
                "SELECT * FROM recon_side_notes WHERE category=? AND window_open_ms=? ORDER BY ts_ms, id",
                (category, window_open_ms)))

    # ---- war-room accuracy: each closed window scored once against the price at its close ----
    def put_recon_score(self, category: str, window_open_ms: int, scored_ms: int, open_price: float,
                        close_price: float, move_pct: float, realised: str, bias: str, direction_hit: bool,
                        trap: str | None, trap_hit: bool | None, bull_brier: float | None,
                        bear_brier: float | None) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO recon_scores (category, window_open_ms, scored_ms, open_price, close_price, move_pct, "
                "realised, bias, direction_hit, trap, trap_hit, bull_brier, bear_brier) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(category, window_open_ms) DO NOTHING",
                (category, window_open_ms, scored_ms, open_price, close_price, move_pct, realised, bias,
                 int(direction_hit), trap, None if trap_hit is None else int(trap_hit), bull_brier, bear_brier))
            self._conn.commit()
        return cur.rowcount > 0

    def recon_scores(self, category: str = "live", limit: int = 50) -> list[dict]:
        with self._lock:
            return _rows(self._conn.execute(
                "SELECT * FROM recon_scores WHERE category=? ORDER BY window_open_ms DESC LIMIT ?",
                (category, limit)))

    def anchors_due(self, now_ms: int, category: str = "live") -> list[dict]:
        """Pinned anchors whose candle has closed and which have not been scored yet."""
        with self._lock:
            rows = _rows(self._conn.execute(
                "SELECT a.* FROM recon_anchors a LEFT JOIN recon_scores s "
                "ON s.category = a.category AND s.window_open_ms = a.window_open_ms "
                "WHERE a.category=? AND a.window_close_ms <= ? AND s.window_open_ms IS NULL "
                "ORDER BY a.window_open_ms", (category, now_ms)))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def scored_windows(self, category: str = "live", limit: int = 50) -> list[dict]:
        """Scores joined with the anchor they graded, newest first."""
        with self._lock:
            rows = _rows(self._conn.execute(
                "SELECT s.*, a.payload FROM recon_scores s JOIN recon_anchors a "
                "ON a.category = s.category AND a.window_open_ms = s.window_open_ms "
                "WHERE s.category=? ORDER BY s.window_open_ms DESC LIMIT ?", (category, limit)))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def close(self) -> None:
        with self._lock:
            self._conn.close()
