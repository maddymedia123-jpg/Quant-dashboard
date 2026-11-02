"""The MCP server: six read-only tools over the shared store plus the guide (a tool and a resource), a
bearer-guarded HTTP app."""
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
    # the rest of the dashboard: a report with a body, desk calls, a trap, a trade, the book, a recon window
    body = {"report_id": "TIR1-TEST", "mode": "1-agent", "briefs": [{"agent_id": "B1", "desk": "bullish", "domain": "options",
            "raw_metrics": {"call_wall": 90_000.0}, "interpretation": "calls stacked", "traps_detected": [], "conviction": 0.6}],
            "desks": {"bullish": {"summary": "up"}, "bearish": {"summary": "down"}},
            "director": {"trap_classification": "BULL_TRAP", "scenarios": [{"name": "flush", "probability": 55.0}]},
            "runs": [{"agent_id": "B1", "latency_s": 1.0, "prompt_tokens": 10}], "raw_metrics": [["Price", "$80,000", "gate"]],
            "plan": {"side": "SHORT", "zone_low": 80_100.0, "zone_high": 80_400.0, "stop": 80_700.0, "targets": [79_000.0]}}
    assert store.add_report("TIR1-TEST", NOW - 2 * H, "BULL_TRAP", 80_000.0, 1.2, "BEARISH", mode="1-agent",
                            plan="SHORT 80,100-80,400", payload=json.dumps(body))
    assert store.add_report("TIR-OLD", NOW - 30 * H, "NO_TRAP", 81_000.0, 1.0, "NEUTRAL", mode="15-agent")  # no body kept
    cid = store.add_call(NOW - 5 * H, "live", "BEARISH", 0.4, 80_500.0, 0.5, 4 * H)
    store.score_call(cid, NOW - H, -0.8, True)
    store.add_call(NOW - H, "weekly", "BULLISH", 0.3, 80_000.0, 2.0, 7 * 24 * H)
    tid_trap = store.put_trap(category="live", side="bull_trap", timeframe="15m", level=80_425.0, invalidation=80_522.0,
                              plays_out=79_941.0, declared_ms=NOW - 3 * H, score=67.8, head_call="ENGINEERED_TRAP",
                              evidence=["price holding above value while delta drains"], notes=[])
    store.resolve_trap(tid_trap, "invalidated", 80_530.0, NOW - 2 * H)
    tid = store.open_trade(opened_ms=NOW - 4 * H, asset="BTC/USDT", side="SHORT", entry=80_300.0, stop=80_700.0,
                           given_stop=80_600.0, stop_verdict="would be hunted", category="live",
                           plan={"thesis": "bull trap at 80.4k", "targets": [79_000.0]})
    store.add_trade_event(tid, NOW - 3 * H, "target", "target:TP1", "TP1 reached", 79_000.0)
    store.put_book_report(NOW - 10 * 60_000, {"available": True, "as_of_ms": NOW - 10 * 60_000,
                                              "depth": {"mid": 80_000.0, "spread_bps": 0.1, "imbalance_025": 0.2},
                                              "sweeps": [{"side": "asks", "direction": "up", "move_bps": 8.0}], "scoops": [],
                                              "method": "sweep: mid moved >= 5 bps within a minute"})
    w0 = NOW - 6 * H
    assert store.put_live_anchor(w0, w0 + 4 * H, w0 + 60_000, "BULL TRAP RISK", 26.3, 0.45, "BULL_TRAP", 80_100.0,
                                 {"bias": "BULL TRAP RISK", "domains": {"bullish": {"quant": 0.4}, "bearish": {"quant": 0.7}},
                                  "trap_probabilities": {"BULL_TRAP": 0.5}}, category="live")
    assert store.add_side_note(w0, w0 + 2 * H, "TRAP", "trap:1", "TRAP desk: bull trap anchored at 80,425.", category="live")
    assert store.put_recon_score("live", w0, w0 + 4 * H, 80_100.0, 79_900.0, -0.25, "FLAT", "BULL TRAP RISK", True,
                                 "BULL_TRAP", False, 0.3, 0.08)
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


def test_the_rest_of_the_dashboard_reads_back_reports_calls_traps_trades_book_and_recon(seeded):
    store, plan = seeded
    ro = _ReadsOnly(store)
    lst = mcp_tools.trap_reports(ro, now_ms=NOW)
    assert [r["report_id"] for r in lst["reports"]] == ["TIR1-TEST", "TIR-OLD"], "newest first, bodies or not"
    assert lst["reports"][0]["classification"] == "BULL_TRAP" and lst["reports"][0]["plan"] == "SHORT 80,100-80,400"
    assert lst["reports"][0]["scored"]["hit_24h"] is None, "not scored yet is None, not False"
    one = mcp_tools.trap_report(ro, now_ms=NOW)
    assert one["available"] and one["report_id"] == "TIR1-TEST" and one["report"]["director"]["trap_classification"] == "BULL_TRAP"
    assert one["report"]["briefs"][0]["raw_metrics"]["call_wall"] == 90_000.0 and one["report"]["plan"]["side"] == "SHORT"
    assert "runs" not in one["report"], "model latencies and token counts are not part of the analysis"
    assert mcp_tools.trap_report(ro, "TIR1-TEST", now_ms=NOW)["report_id"] == "TIR1-TEST"
    assert mcp_tools.trap_report(ro, mode="15-agent", now_ms=NOW)["available"] is False, "that report kept no body"
    assert mcp_tools.trap_report(ro, "NOPE", now_ms=NOW)["available"] is False
    calls = mcp_tools.desk_calls(ro, now_ms=NOW)
    assert [c["desk"] for c in calls["calls"]] == ["weekly", "live"] and calls["calls"][1]["hit"] is True
    assert calls["calls"][0]["hit"] is None and calls["calls"][0]["horizon_hours"] == 168.0
    assert calls["per_desk"] == {"weekly": {"calls": 1, "scored": 0, "hits": 0}, "live": {"calls": 1, "scored": 1, "hits": 1}}
    assert [c["desk"] for c in mcp_tools.desk_calls(ro, "live", now_ms=NOW)["calls"]] == ["live"]
    assert len(mcp_tools.desk_calls(ro, "live", limit=1, now_ms=NOW)["calls"]) == 1, \
        "the desk filter comes before the limit: the newest row overall is a weekly call"
    tr = mcp_tools.traps(ro, now_ms=NOW)["traps"]
    assert len(tr) == 1 and tr[0]["status"] == "invalidated" and tr[0]["resolved_price"] == 80_530.0
    assert tr[0]["evidence"] == ["price holding above value while delta drains"], "evidence comes back as a list"
    assert mcp_tools.traps(ro, "active", now_ms=NOW)["traps"] == []
    tj = mcp_tools.trade_journal(ro, now_ms=NOW)["trades"]
    assert len(tj) == 1 and tj[0]["side"] == "SHORT" and tj[0]["plan"]["thesis"] == "bull trap at 80.4k"
    assert [e["headline"] for e in tj[0]["events"]] == ["TP1 reached"] and tj[0]["stop_verdict"] == "would be hunted"
    assert mcp_tools.trade_journal(ro, "closed", now_ms=NOW)["trades"] == []
    ob = mcp_tools.order_book(ro, now_ms=NOW)
    assert ob["available"] and ob["depth"]["mid"] == 80_000.0 and ob["sweeps"][0]["move_bps"] == 8.0 and "ts_ms" not in ob
    rw = mcp_tools.recon_windows(ro, "live", now_ms=NOW)
    assert rw["desk"] == "live" and len(rw["windows"]) == 1
    w = rw["windows"][0]
    assert w["bias"] == "BULL TRAP RISK" and w["reasoning"]["trap_probabilities"] == {"BULL_TRAP": 0.5}
    assert w["side_notes"][0]["kind"] == "TRAP" and w["score"]["direction_hit"] is True and w["score"]["trap_hit"] is False
    assert mcp_tools.recon_windows(ro, "weekly", now_ms=NOW)["windows"] == []
    assert ro.called <= set(mcp_tools.READS), ro.called - set(mcp_tools.READS)
    for name in ("trap_reports", "trap_report", "desk_calls", "traps", "trade_journal", "order_book", "recon_windows"):
        json.dumps(mcp_tools.TOOLS[name](ro, now_ms=NOW))


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
    assert ro.report_body()["report_id"] == "TIR1-TEST" and mcp_tools.order_book(ro)["available"]
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        ro.add_market_alert(NOW + 5 * H, "x", "x:1", "no", None)
    assert len(store.market_alerts_recent(10)) == 2, "nothing got through"


def test_the_server_lists_the_tools_serves_the_guide_both_ways_and_guards_http(seeded, monkeypatch):
    store, plan = seeded
    import mcp_server
    srv = mcp_server.build(store)
    tools = asyncio.run(srv.list_tools())
    assert sorted(t.name for t in tools) == sorted([*mcp_tools.TOOLS, "how_to_read"])
    assert all(t.description for t in tools), "every tool explains itself"
    # chat connectors (Claude, ChatGPT) give the model tools and not resources: the guide is a tool as well
    got = asyncio.run(srv.call_tool("how_to_read", {}))
    body = got.content[0].text
    assert body.startswith("# How to read") and "House rules" in body and "## 2. The tools" in body, body[:120]
    assert "how_to_read" in mcp_server.INSTRUCTIONS, "the server tells a new client to read the guide first"
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
        # the two other ways in, for connectors that keep Authorization for themselves or allow no header
        assert c.post("/mcp", json=body, headers={**heads, "X-Api-Key": "0123456789abcdef0123"}).status_code == ok.status_code
        assert c.post("/mcp", json=body, headers={**heads, "X-Api-Key": "wrong"}).status_code == 401
        by_path = c.post("/k/0123456789abcdef0123/mcp", json=body, headers=heads)
        assert by_path.status_code == ok.status_code and by_path.text == ok.text, by_path.text
        assert c.post("/k/wrong/mcp", json=body, headers=heads).status_code == 401
        assert c.post("/k/wrong/mcp", json=body, headers={**heads, "X-Api-Key": "0123456789abcdef0123"}).status_code == 401, \
            "a wrong key in the path is not rescued by a header"
        assert c.get("/.well-known/oauth-protected-resource").status_code == 404, "no OAuth to discover"
        assert c.get("/.well-known/oauth-authorization-server", headers={"X-Api-Key": "0123456789abcdef0123"}).status_code == 404
    # no token, no port
    monkeypatch.delenv(mcp_server.TOKEN_ENV, raising=False)
    monkeypatch.setenv(mcp_server.STORE_ENV, str(store.path))
    assert mcp_server.main(["--http", "--port", "1"]) == 2


def test_the_access_log_never_keeps_a_path_token():
    import logging
    import mcp_server
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:0", "POST", "/k/0123456789abcdef0123/mcp", "1.1", 200), None)
    assert mcp_server.RedactKey().filter(rec) is True
    assert "0123456789abcdef0123" not in rec.getMessage() and "/k/REDACTED/mcp" in rec.getMessage()
