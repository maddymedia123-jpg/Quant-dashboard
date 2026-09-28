"""What happened to a pinned trade, and pushing it to a channel.

The spec wants a push the moment a target is reached, a stop is hit, a TRAP develops near the position or
a monitoring checkpoint arrives. Two halves, deliberately separate:

* *Detection* is pure: it compares the plan against the current price and the trap ledger and returns the
  events that have occurred. It is idempotent by construction - every event carries a dedup key, and the
  store's unique index means the same target being reached is one event however many times the page
  reloads. Without that, a 30-second auto-refresh would alert on the same target 120 times an hour.
* *Dispatch* sends them. Telegram and Discord need only a token or a webhook, so both are implemented and
  work as soon as those exist in the secrets; email and SMS need an account and a sender identity that
  nobody has set up here, so they report themselves unconfigured rather than pretending to send. Nothing
  is marked as notified unless a channel confirmed delivery, so a failed send is retried on the next pass
  instead of being silently dropped.

There is no background worker on Streamlit Cloud: the page detects and dispatches while it is open, which
is honest about what this can do today. A cron host would close that gap and is not something the code can
supply for itself."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

TIMEOUT = 10.0
MAX_PER_PASS = 8          # a burst larger than this is a bug or a gap in the ledger, not news


@dataclass(frozen=True)
class Event:
    kind: str            # target | stop | trap | checkpoint
    dedup_key: str
    headline: str
    price: float | None = None

    @property
    def urgent(self) -> bool:
        return self.kind in ("stop", "trap")


def detect_events(trade: dict, price: float | None, progress=None, traps=(), checkpoints=(),
                  now_ms: int = 0) -> list[Event]:
    """Everything that has happened to this position which a trader would want to be told about.

    Pure, and stable across calls: the same state produces the same events with the same keys, so the
    store decides what is new rather than this function guessing."""
    out: list[Event] = []
    side = str(trade.get("side") or "")
    asset = str(trade.get("asset") or "the position")

    if progress is not None:
        for name in progress.hit:
            target = next((t for t in (trade.get("plan") or {}).get("targets", [])
                           if t.get("name") == name), {})
            level = target.get("price")
            out.append(Event("target", f"target:{name}",
                             f"{asset} {side}: {name} reached"
                             + (f" at {float(level):,.0f}" if level is not None else "")
                             + (f" ({progress.pnl_r:+.1f}R)" if progress.pnl_r else ""), price))
        if progress.stopped:
            out.append(Event("stop", "stop",
                             f"{asset} {side}: the stop at {float(trade.get('stop') or 0):,.0f} was "
                             f"reached ({progress.pnl_r:+.1f}R)", price))

    for t in traps:
        out.append(Event("trap", f"trap:{t.trap_id}:{t.near}",
                         f"{asset} {side}: an anchored {t.side.replace('_', ' ')} on the {t.timeframe} "
                         f"sits {t.distance:,.0f} from this trade's {t.near}", price))

    for c in checkpoints:
        if c.at_ms and c.at_ms <= now_ms:
            out.append(Event("checkpoint", f"checkpoint:{c.at_ms}",
                             f"{asset} {side}: {c.reason} has passed - review the position", price))

    return out[:MAX_PER_PASS]


# ---------- dispatch ----------
@dataclass(frozen=True)
class Channel:
    name: str
    configured: bool
    reason: str = ""


class Dispatcher:
    """Sends events to whichever channels are configured. Unconfigured ones are named, not hidden."""

    def __init__(self, secrets: dict | None = None, client: httpx.Client | None = None):
        s = secrets or {}
        self.telegram_token = str(s.get("TELEGRAM_BOT_TOKEN") or "")
        self.telegram_chat = str(s.get("TELEGRAM_CHAT_ID") or "")
        self.discord_webhook = str(s.get("DISCORD_WEBHOOK_URL") or "")
        self._client = client

    @property
    def channels(self) -> tuple[Channel, ...]:
        return (
            Channel("Telegram", bool(self.telegram_token and self.telegram_chat),
                    "needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"),
            Channel("Discord", bool(self.discord_webhook), "needs DISCORD_WEBHOOK_URL"),
            Channel("Email", False, "needs an SMTP account and a verified sender"),
            Channel("SMS", False, "needs a paid gateway account"),
        )

    @property
    def configured(self) -> bool:
        return any(c.configured for c in self.channels)

    def _post(self, url: str, payload: dict) -> bool:
        try:
            client = self._client or httpx.Client(timeout=TIMEOUT)
            response = client.post(url, json=payload)
            if response.status_code >= 400:
                log.warning("alert channel refused the send: %s %s", response.status_code, response.text[:200])
                return False
            return True
        except Exception as e:  # noqa: BLE001 - a channel being down must never break the page
            log.warning("alert channel unreachable: %s", e)
            return False
        finally:
            if self._client is None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass

    def send(self, events: list[Event]) -> tuple[list[Event], list[str]]:
        """Returns the events at least one channel accepted, and a note per channel that did not.

        An event only counts as delivered when a channel confirms it, so the caller marks exactly those
        as notified and the rest are tried again next pass."""
        if not events:
            return [], []
        text = "\n".join(("[!] " if e.urgent else "") + e.headline for e in events)
        delivered, problems = False, []

        if self.telegram_token and self.telegram_chat:
            ok = self._post(f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                            {"chat_id": self.telegram_chat, "text": text,
                             "disable_web_page_preview": True})
            delivered = delivered or ok
            if not ok:
                problems.append("Telegram did not accept the message")
        if self.discord_webhook:
            ok = self._post(self.discord_webhook, {"content": text})
            delivered = delivered or ok
            if not ok:
                problems.append("Discord did not accept the message")

        if not self.configured:
            problems.append("No alert channel is configured, so nothing was sent; the events are kept "
                            "and will go out once a channel exists.")
        return (list(events) if delivered else []), problems
