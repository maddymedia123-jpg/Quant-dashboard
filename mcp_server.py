"""The dashboard's MCP server: read-only tools so an AI assistant (Claude, ChatGPT, Cursor) can read the
day's plan, its status, the record, the desks and the alerts, and talk about them in Hammad's framework.

Thirteen tools, all reads of the shared store (core.mcp_tools), and the reader's guide (docs/mcp-guide.md),
which tells the assistant how to read every number; the guide is both a tool (how_to_read, because chat
connectors give the model tools and not resources) and a resource (guide://how-to-read). No tool writes,
fetches an exchange or runs a model: the assistant reads what the worker produced.

Two ways to run it:

    python mcp_server.py                      # stdio, for a desktop assistant on the same machine
    python mcp_server.py --http --port 8787   # streamable HTTP, for remote connectors; needs QUANT_MCP_TOKEN

Over HTTP every request must carry the token - `Authorization: Bearer <token>`, `X-Api-Key: <token>`, or
the path form `/k/<token>/mcp` for connectors that allow no header; the server refuses to start without a
token, so a bare port is never opened by accident. GET /healthz answers without a token so a check from
outside can see the service is up. Remote connectors in claude.ai and ChatGPT need an HTTPS
URL, so on the VM the port sits behind a TLS front (deploy/README.md).

The store: by default the usual one (TI_DATA_DIR, or Turso when TURSO_* is set), opened read-only -
no schema is created and nothing is written. QUANT_MCP_STORE points it at a file instead, which is how
the server on the VM reads the worker's embedded replica beside it without a second Turso connection."""
from __future__ import annotations

import argparse
import hmac
import logging
import os
import pathlib
import re
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from core import mcp_tools
from core.store import Store

HERE = pathlib.Path(__file__).resolve().parent
GUIDE = HERE / "docs" / "mcp-guide.md"
TOKEN_ENV = "QUANT_MCP_TOKEN"
STORE_ENV = "QUANT_MCP_STORE"
INSTRUCTIONS = ("Read-only access to a BTC Trap Intelligence dashboard: the day's Quant Forecast plan, its live status "
                "and alerts, the forecast record, the four desks' anchors, calls and reasoning, the Trap Intelligence "
                "reports, the declared traps, the trade journal, the order book and the week's Monday plan. Call how_to_read "
                "first: it explains every field and the house rules. Quote only numbers "
                "a tool returned; never extrapolate a forecast of your own. Times are UTC unless a field says PKT.")
log = logging.getLogger("mcp_server")


def open_store(path: str | None = None) -> Store:
    """The store, read-only. A path (or QUANT_MCP_STORE) names a file; otherwise the usual store settings."""
    p = path or os.environ.get(STORE_ENV, "").strip() or None
    return Store(p, readonly=True)


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8") if GUIDE.exists() else "The guide is missing from this build."


def build(store: Store) -> MCPServer:
    """The server with its store tools, the guide tool and the guide resource, bound to `store`."""
    mcp = MCPServer("Quant Dashboard", instructions=INSTRUCTIONS)

    @mcp.tool()
    def todays_plan(day_utc: str | None = None) -> dict:
        """The day's Quant Forecast plan as the dashboard's schema-v7 JSON: regime, the lean and its parts, the
        projected high and low with their hours, the whole-day invalidation, the 24 projected hourly candles,
        the timeline tree, the trade directive, the Monday plan, the self-audit and the raw-math drawer.
        `day_utc` ('YYYY-MM-DD') reads a past day; empty is today."""
        return mcp_tools.todays_plan(store, day_utc)

    @mcp.tool()
    def plan_status() -> dict:
        """Where today's plan stands right now: tier status (on track / caution / model held / invalidated /
        locked), the compact timeline, which window is live, the directive, each plan row of the day
        (version 0 the morning plan, 1 Scenario B) and every alert the bot sent today."""
        return mcp_tools.plan_status(store)

    @mcp.tool()
    def forecast_record() -> dict:
        """How the forecast has done: the 60-day backtest counts, the morning plans' live record over the
        last 10 and 30 scored days, each scored day in brief (projected against actual high and low), the
        liquidation-gravity shadow counts, and the desks' accuracy ledger with its chance baseline."""
        return mcp_tools.forecast_record(store)

    @mcp.tool()
    def desk_anchors() -> dict:
        """The four desks (Live Recon, Intraday, Weekly, Monthly): the direction each is anchored to, what it
        would say live, since when, and the last re-anchors with the triggers that fired."""
        return mcp_tools.desk_anchors(store)

    @mcp.tool()
    def recent_alerts(limit: int = 20) -> dict:
        """The last alerts the worker sent to the channel, newest first (Quant Forecast tiers, cascades, TRAP
        alerts, the Market Pulse, trade alerts). `limit` up to 100."""
        return mcp_tools.recent_alerts(store, limit)

    @mcp.tool()
    def monday_plan() -> dict:
        """This week's Sticky Monday Master Plan: weekly bias, the dealers' put and call walls, the baseline
        floor and ceiling and the weekly flip level, in plain words."""
        return mcp_tools.monday_plan(store)

    @mcp.tool()
    def trap_reports(limit: int = 20) -> dict:
        """The Trap Intelligence reports, newest first, without bodies: id, time, pipeline (1-agent or
        15-agent), classification, price, majority direction, plan line, 24h and 7d scores. `limit` up to 100."""
        return mcp_tools.trap_reports(store, limit)

    @mcp.tool()
    def trap_report(report_id: str | None = None, mode: str | None = None) -> dict:
        """One Trap Intelligence report in full: every desk brief (raw metrics, interpretation, traps
        detected), the bullish and bearish desk summaries, the director's classification, scenarios,
        contradictions, blind spots and trade plan, the per-desk summaries, the raw metric table and the
        report's plan. `report_id` names one (see trap_reports); empty is the newest, of `mode` if given."""
        return mcp_tools.trap_report(store, report_id, mode)

    @mcp.tool()
    def desk_calls(desk: str | None = None, limit: int = 50) -> dict:
        """The desks' direction calls, newest first, with each call's score once its horizon passed, and
        counts per desk. `desk`: live, intraday, weekly, monthly or empty for all. `limit` up to 500."""
        return mcp_tools.desk_calls(store, desk, limit)

    @mcp.tool()
    def traps(status: str | None = None, limit: int = 50) -> dict:
        """The TRAP desk's declared traps, newest first: side, timeframe, level, invalidation, where it
        plays out, score, head call, evidence and how it settled. `status`: active, played_out,
        invalidated, expired or empty for all."""
        return mcp_tools.traps(store, status, limit)

    @mcp.tool()
    def trade_journal(status: str | None = None, limit: int = 20) -> dict:
        """The pinned trades with their plan at open, stop verdict, status, close and every event logged
        since. `status`: open, closed or empty for all."""
        return mcp_tools.trade_journal(store, status, limit)

    @mcp.tool()
    def order_book() -> dict:
        """The latest order-book report the Market Pulse reads: depth around the mid, spread, resting size
        by band, imbalance, and the sweeps and scoops seen in the window, with the method."""
        return mcp_tools.order_book(store)

    @mcp.tool()
    def recon_windows(desk: str = "live", limit: int = 5) -> dict:
        """A desk's last anchored windows with the full reasoning: bias, margin, bull and bear points by
        domain and item, trap probabilities, the handshake, side notes and the window's score."""
        return mcp_tools.recon_windows(store, desk, limit)

    @mcp.tool()
    def how_to_read() -> str:
        """Read this first. The reader's guide: what every tool returns, how to read the plan, the record,
        the desks and the alerts, and the house rules for talking about them (markdown)."""
        return _guide()

    @mcp.resource("guide://how-to-read", name="How to read the dashboard", mime_type="text/markdown")
    def guide() -> str:
        """The same guide as a resource, for clients that read resources."""
        return _guide()

    return mcp


KEY_PREFIX = "/k/"


class BearerAuth:
    """ASGI middleware: every HTTP request must carry the token, except GET /healthz. Three ways, because
    connectors differ: `Authorization: Bearer <token>`; `X-Api-Key: <token>` (Claude's connector keeps the
    Authorization header for its own OAuth token); or the token in the path, `/k/<token>/mcp` (ChatGPT's
    connector allows no custom header at all), which is stripped before the app sees the request. The
    OAuth discovery paths answer 404, not 401, so a client that sees a 401 does not go looking for a
    login that does not exist."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token

    def _ok(self, given: str) -> bool:
        return bool(self.token) and bool(given) and hmac.compare_digest(given, self.token)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or "/"
        if path == "/healthz" and scope.get("method") == "GET":
            await _plain(send, 200, b"ok")
            return
        if path.startswith("/.well-known/"):
            await _plain(send, 404, b"not found")
            return
        headers = dict(scope.get("headers") or {})
        bearer = headers.get(b"authorization", b"").decode("latin-1")
        given = bearer[7:] if bearer.startswith("Bearer ") else headers.get(b"x-api-key", b"").decode("latin-1")
        if path.startswith(KEY_PREFIX):
            key, _, rest = path[len(KEY_PREFIX):].partition("/")
            if self._ok(key):
                scope = {**scope, "path": "/" + rest, "raw_path": ("/" + rest).encode()}
                await self.app(scope, receive, send)
                return
            given = ""
        if not self._ok(given):
            await _plain(send, 401, b"unauthorized", [(b"www-authenticate", b"Bearer")])
            return
        await self.app(scope, receive, send)


class RedactKey(logging.Filter):
    """The access log must not keep the path-form token: /k/<token>/ is logged as /k/REDACTED/."""
    _KEY = re.compile(r"/k/[^/\s\"]+")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(self._KEY.sub("/k/REDACTED", x) if isinstance(x, str) else x for x in record.args)
        if isinstance(record.msg, str):
            record.msg = self._KEY.sub("/k/REDACTED", record.msg)
        return True


async def _plain(send, status: int, body: bytes, extra: list | None = None):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())] + (extra or [])})
    await send({"type": "http.response.body", "body": body})


def http_app(mcp: MCPServer, token: str):
    """The streamable-HTTP ASGI app behind the bearer check (the MCP endpoint is /mcp). Stateless and plain
    JSON, so a TLS front can proxy it without sticky sessions; the Host check is off because the front
    rewrites it and the bearer token is the guard."""
    app = mcp.streamable_http_app(json_response=True, stateless_http=True,
                                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    return BearerAuth(app, token)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--store", default=None, help=f"store file to read (default: {STORE_ENV} or the usual store)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    store = open_store(a.store)
    log.info("store %s (%s), read-only", store.path, store.backend)
    mcp = build(store)
    if not a.http:
        mcp.run(transport="stdio")
        return 0
    token = os.environ.get(TOKEN_ENV, "").strip()
    if len(token) < 16:
        log.error("%s must be set (16+ characters) to serve HTTP; refusing to open the port", TOKEN_ENV)
        return 2
    import uvicorn
    logging.getLogger("uvicorn.access").addFilter(RedactKey())
    # a bounded graceful stop: behind a proxy, idle keep-alive connections otherwise hold a restart for
    # systemd's whole 90 s stop timeout (every tool call is one short POST; nothing long-lived is cut)
    uvicorn.run(http_app(mcp, token), host=a.host, port=a.port, log_level="info", timeout_graceful_shutdown=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
