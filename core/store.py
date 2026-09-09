"""SQLite persistence for anchored verdicts, the anchor log, accuracy calls/reports and signal history.

Standard SQL only so a Postgres backend can sit behind the same interface later.
Path comes from TI_DATA_DIR (default ./data). ":memory:" is supported for tests."""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading

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
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, category TEXT NOT NULL,
    squeeze_score INTEGER, price REAL);
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

    def close(self) -> None:
        with self._lock:
            self._conn.close()
