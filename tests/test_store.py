from core.store import Store


def test_anchor_round_trip_and_log(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    assert s.get_anchor("live") is None
    s.set_anchor("live", {"direction": "BULLISH", "levels": [1.5, 2]}, since_ms=1000)
    a = s.get_anchor("live")
    assert a["payload"]["direction"] == "BULLISH" and a["since_ms"] == 1000
    s.set_anchor("live", {"direction": "BEARISH"}, since_ms=2000)
    assert s.get_anchor("live")["payload"]["direction"] == "BEARISH"
    s.log_anchor_change(2000, "live", "BULLISH", "BEARISH", 100.0, ["funding_flip", "oi_shift"])
    log = s.anchor_log()
    assert len(log) == 1 and log[0]["triggers"] == ["funding_flip", "oi_shift"] and log[0]["to_dir"] == "BEARISH"


def test_calls_due_and_scoring():
    s = Store(":memory:")
    cid = s.add_call(ts_ms=0, category="live", direction="BULLISH", confidence=0.6, price=100.0, sigma_pct=1.0, horizon_ms=60_000)
    assert s.due_calls(now_ms=59_999) == []
    due = s.due_calls(now_ms=60_000)
    assert len(due) == 1 and due[0]["id"] == cid
    s.score_call(cid, scored_ms=60_000, realised_pct=0.8, hit=True)
    assert s.due_calls(now_ms=100_000) == []
    scored = s.calls(scored_only=True)
    assert scored[0]["hit"] == 1 and scored[0]["realised_pct"] == 0.8


def test_reports_dedup_and_score():
    s = Store(":memory:")
    assert s.add_report("TIR-1", 0, "BULL_TRAP", 100.0, 1.2, "BEARISH") is True
    assert s.add_report("TIR-1", 5, "NO_TRAP", 101.0, 1.2, "BEARISH") is False
    row = s.reports()[0]
    assert row["classification"] == "BULL_TRAP" and row["hit_24h"] is None
    s.score_report(row["id"], "24h", -1.5, True)
    s.score_report(row["id"], "7d", 2.0, False)
    row = s.reports()[0]
    assert row["hit_24h"] == 1 and row["move_24h_pct"] == -1.5 and row["hit_7d"] == 0


def test_signals_last_before():
    s = Store(":memory:")
    s.add_signal(100, "weekly", 45, 1.0)
    s.add_signal(200, "weekly", 55, 1.0)
    s.add_signal(150, "live", 10, 1.0)
    assert s.last_signal("weekly")["squeeze_score"] == 55
    assert s.last_signal("weekly", before_ms=200)["squeeze_score"] == 45
    assert s.last_signal("monthly") is None
