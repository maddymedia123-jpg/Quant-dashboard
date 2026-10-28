"""Where the recorded order book lives: a plain SQLite file beside the main store, local to the machine
that records it (TI_DATA_DIR/order_book.sqlite).

Not Turso: a snapshot every few seconds is ~17,000 rows and ~17 MB a day, which has no business being
synced to the hosted store the dashboard reads. The collector writes here; the worker reads here and
writes only the small metrics report (core.book_metrics) to the main store, which is what the dashboard
shows. One writer, WAL mode, so the worker's reads never block the collector.

A row keeps the summary numbers as columns and the top KEEP_LEVELS levels per side as compact JSON
([[price, btc], ...]). `prune` keeps the file to RETAIN_DAYS."""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading

from core.data.order_book import DEPTH_BPS, KEEP_LEVELS, BookSnapshot, summary

RETAIN_DAYS = 35
_DEPTH_COLS = [f"{side}_{b:03d}" for b in DEPTH_BPS for side in ("bid", "ask")]
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS book_snapshots (
    ts_ms INTEGER PRIMARY KEY, book_id INTEGER, mid REAL NOT NULL, best_bid REAL NOT NULL, best_ask REAL NOT NULL,
    spread_bps REAL NOT NULL, {", ".join(f"{c} REAL NOT NULL" for c in _DEPTH_COLS)}, imbalance_025 REAL NOT NULL,
    levels TEXT NOT NULL);
"""


def default_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("TI_DATA_DIR", "data")) / "order_book.sqlite"


class BookStore:
    def __init__(self, path: str | os.PathLike | None = None, readonly: bool = False):
        p = pathlib.Path(path) if path else default_path()
        self.path = str(p)
        self.readonly = readonly
        if readonly and not p.exists():
            raise FileNotFoundError(f"no order book recorded at {p}")      # a reader never creates the file
        if p.name != ":memory:":
            p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        if not readonly:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- writing ----
    def put(self, snap: BookSnapshot) -> bool:
        """Store one snapshot; False when one with the same timestamp is already there."""
        s = summary(snap)
        if s.get("mid") is None:
            return False
        levels = json.dumps([[[round(p, 1), round(q, 4)] for p, q in snap.bids[:KEEP_LEVELS]],
                             [[round(p, 1), round(q, 4)] for p, q in snap.asks[:KEEP_LEVELS]]], separators=(",", ":"))
        cols = ["ts_ms", "book_id", "mid", "best_bid", "best_ask", "spread_bps", *_DEPTH_COLS, "imbalance_025", "levels"]
        vals = [int(snap.ts_ms), int(snap.book_id), s["mid"], s["best_bid"], s["best_ask"], s["spread_bps"],
                *[s[c] for c in _DEPTH_COLS], s["imbalance_025"], levels]
        with self._lock:
            cur = self._conn.execute(f"INSERT OR IGNORE INTO book_snapshots ({','.join(cols)}) VALUES "
                                     f"({','.join('?' * len(cols))})", vals)
            self._conn.commit()
        return cur.rowcount > 0

    def prune(self, before_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM book_snapshots WHERE ts_ms < ?", (int(before_ms),))
            self._conn.commit()
        return cur.rowcount

    # ---- reading ----
    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        bids, asks = json.loads(d.pop("levels"))
        d["bids"] = [(float(p), float(q)) for p, q in bids]
        d["asks"] = [(float(p), float(q)) for p, q in asks]
        return d

    def latest(self) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM book_snapshots ORDER BY ts_ms DESC LIMIT 1").fetchone()
        return self._row(r) if r else None

    def between(self, t0_ms: int, t1_ms: int, step_ms: int | None = None) -> list[dict]:
        """Rows with t0 <= ts < t1 in time order; with `step_ms`, the first row of each step (one a
        minute, say) so an hour is 60 rows, not 720."""
        with self._lock:
            if step_ms:
                rows = self._conn.execute(
                    "SELECT * FROM book_snapshots WHERE ts_ms IN (SELECT MIN(ts_ms) FROM book_snapshots "
                    "WHERE ts_ms >= ? AND ts_ms < ? GROUP BY ts_ms / ?) ORDER BY ts_ms",
                    (int(t0_ms), int(t1_ms), int(step_ms))).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM book_snapshots WHERE ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms",
                                          (int(t0_ms), int(t1_ms))).fetchall()
        return [self._row(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            r = self._conn.execute("SELECT COUNT(*) AS n, MIN(ts_ms) AS first_ms, MAX(ts_ms) AS last_ms "
                                   "FROM book_snapshots").fetchone()
            pages = self._conn.execute("PRAGMA page_count").fetchone()[0]
            size = self._conn.execute("PRAGMA page_size").fetchone()[0]
        return {"rows": int(r["n"] or 0), "first_ms": r["first_ms"], "last_ms": r["last_ms"],
                "bytes": int(pages) * int(size)}
