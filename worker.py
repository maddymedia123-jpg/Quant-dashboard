"""The 24/7 worker: what a page load does, every minute, whether or not anybody has the dashboard open.

Each pass fetches the market, builds the four desks, and runs core.pipeline - anchoring, scoring, trap
settlement, plan / cascade / TRAP alerts, the Market Pulse and every pinned trade's alerts. It makes no
model calls: reports are still run by a person from the dashboard, so the Gemini quota is untouched.

It shares the store with the dashboard (Turso, via TURSO_DATABASE_URL and TURSO_AUTH_TOKEN), and both may
send: each alert is claimed by one statement on the database, so whichever claims it first sends it.

Settings come from the environment (on the server, /opt/quant/.env through systemd): TURSO_*,
DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.

    python worker.py            # run forever, one pass a minute
    python worker.py --once     # one pass, then exit (for checking a deploy)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

from core import pipeline
from core.config import CATEGORIES
from core.data.market import _offline, fetch_all
from core.indicators.category import analyze_category
from core.store import Store

PASS_EVERY_S = 60
SECRET_KEYS = ("DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
log = logging.getLogger("worker")
_stop = False


def _secrets() -> dict:
    return {k: os.environ[k] for k in SECRET_KEYS if os.environ.get(k)}


def one_pass(store: Store, now_ms: int | None = None, market=None) -> pipeline.PassResult:
    """One pass: the same steps, in the same order, as a page load."""
    m = market if market is not None else asyncio.run(fetch_all())
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    if not m.spot.available:
        r = pipeline.PassResult()
        r.problems.append(f"no price from {m.spot.source}: {m.spot.error}")
        return r
    analyses = {}
    for key, cat in CATEGORIES.items():
        a = analyze_category(cat, m.spot.frames, m.futures)
        if a is not None:
            analyses[key] = a
    store.sync()
    latest = store.reports(limit=1)
    classification = latest[0].get("classification") if latest else None
    may_send = not _offline()        # fixture data is not the market: never sent anywhere
    r = pipeline.ledger(store, m, analyses, now, classification)
    pipeline.market_alerts(store, m, analyses, now, _secrets(), may_send, r)
    pipeline.all_trade_alerts(store, m.spot.last, now, _secrets(), may_send, r)
    book_report(store, now, r)
    return r


def book_report(store: Store, now_ms: int, r: pipeline.PassResult) -> None:
    """The last hour of the recorded order book (book_collector.py, this machine's file) as the scoop/sweep
    report on the shared store, for the dashboard. A machine with no recorder (the dashboard host, a dev
    box) writes nothing and says nothing: that is normal there. A recorder that has gone quiet is a
    problem, logged once per pass; a broken book file never fails the pass."""
    from core import book_metrics
    try:
        rep = book_metrics.report(now_ms)
    except Exception as e:  # noqa: BLE001 - a broken book file must not stop the alerts
        r.problems.append(f"order book report failed: {type(e).__name__}: {e}")
        return
    if rep.get("available"):
        store.put_book_report(now_ms, rep)
        log.info("order book: report written (%d min of book, %d sweeps, %d scoops, %.1f/%.1f BTC within 25 bps)",
                 rep.get("minutes_in_window", 0), len(rep.get("sweeps") or []), len(rep.get("scoops") or []),
                 (rep.get("depth") or {}).get("bid_025", 0.0), (rep.get("depth") or {}).get("ask_025", 0.0))
    elif rep.get("recorder") == "stale":
        r.problems.append(f"order book: {rep.get('reason')}")


def _on_signal(signum, _frame):
    global _stop
    log.info("signal %s: finishing the current pass, then stopping", signum)
    _stop = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="run one pass and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    store = Store()
    log.info("store: %s (%s); channels: %s", store.path, store.backend, ", ".join(_secrets()) or "none")
    while True:
        t0 = time.time()
        try:
            r = one_pass(store)
            log.info("pass %.1fs: %d desks anchored, %d traps settled, %d alerts sent%s",
                     time.time() - t0, len(r.verdicts), len(r.settled), r.sent,
                     ("; problems: " + " | ".join(r.problems)) if r.problems else "")
        except Exception:  # noqa: BLE001 - a bad pass is logged; the next minute tries again
            log.exception("pass failed")
        if args.once or _stop:
            return 0
        # start each pass on the minute: a 4h close is caught in its first minute, not its fifth
        time.sleep(max(1.0, PASS_EVERY_S - time.time() % PASS_EVERY_S))
        if _stop:
            return 0


if __name__ == "__main__":
    sys.exit(main())
