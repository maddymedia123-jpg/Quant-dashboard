"""The MCP server: six read-only tools over the shared store, a bearer-guarded HTTP app, the guide resource."""
import asyncio
import json
import os
from types import SimpleNamespace as NS

import pytest

os.environ.setdefault("TI_OFFLINE_FIXTURES", "1")

from core import mcp_tools  # noqa: E402
from core import qf_alerts as qa  # noqa: E402
from core.store import Store  # noqa: E402
from tests.test_quant_forecast import DAY0, H, _M, hourly  # noqa: E402

NOW = DAY0 + 60_000
WRITE_PREFIXES = ("put_", "set_", "add_", "close_", "score_", "claim_", "mark_", "release_", "open_", "touch_",
                  "resolve_", "log_", "swap_", "track_", "rejudge_", "commit", "execute")


@pytest.fixture
def seeded(tmp_path):
    """A store the worker has written one morning into: a plan, the week's plan, a desk anchor, two alerts."""
    store = Store(tmp_path / "s.sqlite")
    hist = hourly(800, DAY0 - 800 * H)
    weekly = NS(price=float(hist["close"].iloc[-1]), direction=NS(direction="BULLISH", confidence=0.5, invalidation=1.0, drivers=[]),
                levels=[], vol=NS(exp_low=1.0, exp_high=2.0))
    ev = qa.run(store, _M(hist), NOW, weekly=weekly, weekly_gamma={"put_wall": 70_000.0, "call_wall": 90_000.0})
    assert [e.kind for e in ev] == ["qf_plan"]
    plan = store.qf_plans(DAY0)[0]["payload"]
    store.add_market_alert(NOW, "qf_plan", qa._key(plan, "plan"), ev[0].headline, plan["origin"])
    store.add_market_alert(NOW + 2 * H, "qf_caution", qa._key(plan, "caution", 2), "⚠️ CAUTION: test", plan["origin"])
    store.set_anchor("live", {"direction": "BULLISH", "confidence": 0.6, "anchor": 79_000.0}, NOW - H)
    store.log_anchor_change(NOW - H, "live", "BEARISH", "BULLISH", 80_000.0, ["funding_flip"])
    return store, plan


class _ReadsOnly:
    """A store that raises on anything but a read: the proof the tools never write."""

    def __init__(self, store):
        self._s = store
        self.called: set[str] = set()

    def __getattr__(self, name):
        if name.startswith(WRITE_PREFIXES):
            raise AssertionError(f"tool tried to write through store.{name}")
        self.called.add(name)
        return getattr(self._s, name)


def test_every_tool_reads_the_seeded_store_and_writes_nothing(seeded):
    store, plan = seeded
    ro = _ReadsOnly(store)
    out = {name: fn(ro, now_ms=NOW + 3 * H) for name, fn in mcp_tools.TOOLS.items()}
    for name, o in out.items():
        json.dumps(o)                                           # JSON-safe, nothing exotic inside
    assert ro.called <= set(mcp_tools.READS), ro.called - set(mcp_tools.READS)
    p = out["todays_plan"]
    assert set(p) >= {"audit_and_alert_status", "monday_master_plan_sticky", "layman_summary_and_timeline_tree",
                      "tradingview_chart_payload", "collapsible_raw_math_drawer", "store"}
    assert p["store"]["served_from"]["version"] == 0 and p["store"]["served_from"]["status"] == "active"
    assert p["audit_and_alert_status"]["tier_status"] == "TIER-1 CAUTION", "the last alert sent for the plan was a caution"
    assert p["audit_and_alert_status"]["bot_message"] == "⚠️ CAUTION: test"
    assert p["monday_master_plan_sticky"]["weekly_gex_walls"] == [70_000.0, 90_000.0]
    s = out["plan_status"]
    assert s["active"]["tier_status"] == "caution" and s["active"]["version"] == 0
    assert [a["kind"] for a in s["alerts_today"]] == ["plan", "caution"] and s["plans"][0]["regime"] == plan["regime"]
    assert "Trade" in s["active"]["directive"] or "No trade" in s["active"]["directive"] or s["active"]["directive"]
    r = out["forecast_record"]
    assert r["backtest_60d"]["days"] >= 1 and r["live_last_10_days"]["days"] == 0 and r["scored_days"] == []
    assert "desks" in r["desk_ledger"] and "shadow" in r["how_to_read"]
    d = out["desk_anchors"]
    assert d["desks"]["live"]["anchor"]["direction"] == "BULLISH" and d["desks"]["live"]["horizon_hours"] == 4
    assert d["desks"]["monthly"]["anchor"] is None and d["recent_reanchors"][0]["triggers"] == ["funding_flip"]
    a = out["recent_alerts"]
    assert [x["kind"] for x in a["alerts"]] == ["qf_caution", "qf_plan"], "newest first"
    m = out["monday_plan"]
    assert m["available"] and m["weekly_bias"] == "BULLISH" and m["weekly_gex_walls"] == [70_000.0, 90_000.0]


def test_a_day_without_a_plan_and_a_past_day_are_said_plainly(seeded):
    store, plan = seeded
    none = mcp_tools.todays_plan(store, "2026-01-05", now_ms=NOW)
    assert none["available"] is False and "no plan for this day" in none["reason"] and none["day_utc"] == "2026-01-05T00:00:00Z"
    assert mcp_tools.plan_status(store, now_ms=NOW + 3 * 86_400_000)["active"] is None
    assert mcp_tools.monday_plan(store, now_ms=NOW - 30 * 86_400_000)["available"] is False
    with pytest.raises(ValueError):
        mcp_tools.parse_day("yesterday", NOW)


def test_a_read_only_store_serves_the_tools_and_refuses_writes(seeded, tmp_path):
    store, plan = seeded
    ro = Store(tmp_path / "s.sqlite", readonly=True)
    assert ro.readonly and ro.backend == "sqlite"
    assert mcp_tools.plan_status(ro, now_ms=NOW + H)["active"]["version"] == 0
    assert mcp_tools.recent_alerts(ro, 1)["alerts"][0]["kind"] == "qf_caution"
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        ro.add_market_alert(NOW + 5 * H, "x", "x:1", "no", None)
    assert len(store.market_alerts_recent(10)) == 2, "nothing got through"


def test_the_server_lists_six_tools_the_guide_and_guards_http_with_a_bearer_token(seeded, monkeypatch):
    store, plan = seeded
    import mcp_server
    srv = mcp_server.build(store)
    tools = asyncio.run(srv.list_tools())
    assert sorted(t.name for t in tools) == sorted(mcp_tools.TOOLS)
    assert all(t.description for t in tools), "every tool explains itself"
    res = asyncio.run(srv.list_resources())
    assert [str(r.uri) for r in res] == ["guide://how-to-read"]
    text = asyncio.run(srv.read_resource("guide://how-to-read"))
    assert text and ("Quant Forecast" in text[0].content or "missing" in text[0].content)
    from starlette.testclient import TestClient
    app = mcp_server.http_app(srv, "0123456789abcdef0123")
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200 and c.get("/healthz").text == "ok"
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        heads = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        assert c.post("/mcp", json=body, headers=heads).status_code == 401
        assert c.post("/mcp", json=body, headers={**heads, "Authorization": "Bearer wrong"}).status_code == 401
        ok = c.post("/mcp", json=body, headers={**heads, "Authorization": "Bearer 0123456789abcdef0123"})
        assert ok.status_code != 401, ok.text
    # no token, no port
    monkeypatch.delenv(mcp_server.TOKEN_ENV, raising=False)
    monkeypatch.setenv(mcp_server.STORE_ENV, str(store.path))
    assert mcp_server.main(["--http", "--port", "1"]) == 2
