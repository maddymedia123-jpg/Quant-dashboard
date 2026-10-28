"""The order-book recorder: one Gate.io BTC_USDT perpetual book snapshot every POLL_S seconds, into the
local order_book.sqlite (core.book_store), around the clock.

Runs beside the worker as its own service (deploy/quant-book.service) because a book has to be recorded
as it happens - no exchange serves its history - and five seconds is too fine for the worker's minute
loop. It needs no secrets: the endpoint is public. It logs one line a minute, prunes rows older than
RETAIN_DAYS once an hour, and backs off on errors (5 s doubling to 60 s) rather than hammering a venue
that is refusing.

    python book_collector.py            # run forever
    python book_collector.py --once     # one snapshot, then exit (for checking a deploy)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

import httpx

from core.book_store import RETAIN_DAYS, BookStore
from core.data.order_book import fetch_multiplier, fetch_order_book

POLL_S = 5
LOG_EVERY_S = 60
PRUNE_EVERY_S = 3600
BACKOFF_MAX_S = 60
log = logging.getLogger("book")
_stop = False


def _on_signal(signum, _frame):
    global _stop
    log.info("signal %s: stopping after this snapshot", signum)
    _stop = True


async def run(store: BookStore, once: bool = False) -> int:
    async with httpx.AsyncClient(timeout=10.0) as client:
        mult = await fetch_multiplier(client)
        log.info("recording to %s; %s BTC per contract; every %ds", store.path, mult, POLL_S)
        stored = failed = 0
        backoff = POLL_S
        last_log = last_prune = time.time()
        while True:
            t0 = time.time()
            try:
                snap = await fetch_order_book(client, mult)
                if store.put(snap):
                    stored += 1
                backoff = POLL_S
            except Exception as e:  # noqa: BLE001 - provider boundary: log, back off, try again
                failed += 1
                log.warning("snapshot failed (%s: %s); next try in %ds", type(e).__name__, str(e)[:160], backoff)
                if once:
                    return 1
                await asyncio.sleep(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue
            if once:
                s = store.latest() or {}
                log.info("stored one snapshot: mid %s, %.1f/%.1f BTC within 25 bps", s.get("mid"), s.get("bid_025", 0), s.get("ask_025", 0))
                return 0
            now = time.time()
            if now - last_log >= LOG_EVERY_S:
                s = store.latest() or {}
                st = store.stats()
                log.info("%d stored, %d failed in the last minute; mid %s, %.1f/%.1f BTC within 25 bps, imbalance %+.2f; "
                         "%d rows, %.1f MB", stored, failed, s.get("mid"), s.get("bid_025", 0), s.get("ask_025", 0),
                         s.get("imbalance_025", 0), st["rows"], st["bytes"] / 1e6)
                stored = failed = 0
                last_log = now
            if now - last_prune >= PRUNE_EVERY_S:
                n = store.prune(int((now - RETAIN_DAYS * 86_400) * 1000))
                if n:
                    log.info("pruned %d rows older than %d days", n, RETAIN_DAYS)
                last_prune = now
            if _stop:
                return 0
            await asyncio.sleep(max(0.5, POLL_S - (time.time() - t0)))
            if _stop:
                return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="store one snapshot and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    store = BookStore()
    try:
        return asyncio.run(run(store, args.once))
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
