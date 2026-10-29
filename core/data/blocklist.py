"""Hosts that refuse this machine are left alone for a while.

The worker runs from a Google region in the US. Binance answers 451 there and Bybit 403, and the
cryptocurrency.cv aggregator answers 403 to it too; none of that changes from one minute to the next, yet
each pass asked all three again - three failed round trips and three warnings a minute, the same lines
filling the journal for days. A refusal (403 or 451) now parks the source for BLOCK_MS and says so once;
a 5xx or a timeout is not a refusal and is retried as before. The state lives in this process, so the
worker and a page each learn it for themselves and forget it on restart."""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

BLOCK_MS = 6 * 3_600_000
REFUSAL_CODES = (403, 451)
_until: dict[str, int] = {}


def _now() -> int:
    return int(time.time() * 1000)


def status_code(e: BaseException) -> int | None:
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return int(code) if code is not None else None


def blocked(name: str, now_ms: int | None = None) -> bool:
    until = _until.get(name)
    if until is None:
        return False
    if (now_ms if now_ms is not None else _now()) >= until:
        del _until[name]
        return False
    return True


def note_failure(name: str, e: BaseException, now_ms: int | None = None) -> bool:
    """Record a failure; True when it was a refusal and the source is now parked."""
    code = status_code(e)
    if code not in REFUSAL_CODES:
        return False
    now = now_ms if now_ms is not None else _now()
    _until[name] = now + BLOCK_MS
    log.warning("%s refused this host (HTTP %s): not asked again for %d hours", name, code, BLOCK_MS // 3_600_000)
    return True


def skipped(name: str) -> str:
    return f"{name}: skipped (refused this host earlier)"


def reset() -> None:
    _until.clear()
