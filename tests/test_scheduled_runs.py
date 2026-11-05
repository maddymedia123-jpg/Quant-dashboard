"""Scheduled agent runs (election-day fix 6): when the worker wakes the Director and the Live war room, and
when it must not. The model runs are replaced; nothing here calls a model."""
import json
from types import SimpleNamespace as NS

from core import scheduled_runs as sr
from core.data.types import CalendarSnapshot
from core.store import Store
from core.trades import fmt_when

H = 3_600_000
MIN = 60_000
DAY = 1_793_750_400_000                       # 4 Nov 2026 00:00 UTC
PRINT = DAY + 13 * H + 30 * MIN               # a 13:30 UTC release


def ev(title, t_ms, impact="High", country="USD"):
    return {"title": title, "country": country, "impact": impact, "time_ms": t_ms, "date": "", "forecast": None, "previous": None}


def market(*events):
    return NS(calendar=CalendarSnapshot(source="forexfactory", events=list(events)), spot=NS(last=80_000.0),
              generated_at=None)


def fake_runs(monkeypatch, calls):
    monkeypatch.setattr(sr, "_director", lambda store, m, a, now, s: calls.append(("director", now)) or
                        {"report_id": "TIR1-x", "classification": "engineered trap", "side": "SHORT"})
    monkeypatch.setattr(sr, "_war_room", lambda store, m, a, now, s: calls.append(("war_room", now)) or "Live war room: BEARISH (anchored the window's summary)")


SETTINGS = NS(director_model="gemini-test", typesafe_key=None)
ANALYSES = {"live": NS(price=80_000.0)}


def test_a_high_impact_print_wakes_the_agents_once_after_it_has_settled(monkeypatch):
    calls, s = [], Store(":memory:")
    fake_runs(monkeypatch, calls)
    m = market(ev("CPI m/m", PRINT), ev("Core CPI m/m", PRINT), ev("ISM Services PMI", PRINT + H, impact="Medium"))
    r = NS(problems=[])
    assert sr.run(s, m, ANALYSES, PRINT + 10 * MIN, SETTINGS, r) is None and calls == [], "the first 15m candle has not closed"
    out = sr.run(s, m, ANALYSES, PRINT + 16 * MIN, SETTINGS, r)
    assert calls == [("director", PRINT + 16 * MIN), ("war_room", PRINT + 16 * MIN)]
    assert out["key"] == f"run:print:{PRINT}"
    assert out["headline"] == ("🤖 NEW REPORT · BTC $80,000\n"
                               "Triggered by the US CPI m/m + Core CPI m/m release at 13:30 UTC (18:30 PKT). "
                               "Report TIR1-x: engineered trap, plan sell. Details follow in the Pulse.")
    rows = s.market_alerts_keyed("run:")
    assert len(rows) == 1 and rows[0]["kind"] == "run" and rows[0]["headline"] == out["headline"]
    assert rows[0]["notified"] == 1, "wording batch 5: the report's own Pulse carries the news; the run line is the record, not sent"
    # the same print never runs twice, and the Medium print never runs at all
    assert sr.run(s, m, ANALYSES, PRINT + 17 * MIN, SETTINGS, r) is None and len(calls) == 2
    assert sr.run(s, m, ANALYSES, PRINT + H + 20 * MIN, SETTINGS, r) is None and len(calls) == 2
    assert r.problems == []
    s.close()


def test_a_day_plan_invalidation_wakes_them_and_a_stale_one_does_not(monkeypatch):
    calls, s = [], Store(":memory:")
    fake_runs(monkeypatch, calls)
    plan = {"day_ms": DAY, "version": 0, "created_ms": DAY + 5 * MIN, "end_ms": DAY + 24 * H, "regime": "neutral"}
    pid = s.put_qf_plan(plan)
    m = market()
    assert sr.run(s, m, ANALYSES, DAY + 16 * H, SETTINGS) is None and calls == []
    assert s.close_qf_plan(pid, "invalidated", "4h close above", DAY + 16 * H)
    out = sr.run(s, m, ANALYSES, DAY + 16 * H + MIN, SETTINGS)
    assert out["key"] == f"run:qf_invalid:{pid}" and len(calls) == 2
    assert "Triggered by the day plan being invalidated at 16:00 UTC (21:00 PKT)." in out["headline"]
    assert "forecast #" not in out["headline"]
    # yesterday's invalidation just after midnight is still fresh; one three hours old is not
    s2 = Store(":memory:")
    pid2 = s2.put_qf_plan(plan)
    s2.close_qf_plan(pid2, "invalidated", "x", DAY + 24 * H - 30 * MIN)
    assert sr.run(s2, m, ANALYSES, DAY + 24 * H + 10 * MIN, SETTINGS)["key"] == f"run:qf_invalid:{pid2}"
    s3 = Store(":memory:")
    pid3 = s3.put_qf_plan(plan)
    s3.close_qf_plan(pid3, "invalidated", "x", DAY + 13 * H)
    assert sr.run(s3, m, ANALYSES, DAY + 16 * H, SETTINGS) is None
    for x in (s, s2, s3):
        x.close()


def test_runs_are_half_an_hour_apart_six_a_day_and_a_throttled_trigger_is_skipped_for_good(monkeypatch):
    calls, s = [], Store(":memory:")
    fake_runs(monkeypatch, calls)
    first, second = PRINT, PRINT + 20 * MIN
    m = market(ev("CPI m/m", first), ev("Retail Sales m/m", second))
    assert sr.run(s, m, ANALYSES, first + 16 * MIN, SETTINGS)["key"] == f"run:print:{first}"
    assert sr.run(s, m, ANALYSES, second + 16 * MIN, SETTINGS) is None, "settled, but only 20 minutes after the last run: wait"
    assert sr.run(s, m, ANALYSES, first + 47 * MIN, SETTINGS)["key"] == f"run:print:{second}"
    assert len(calls) == 4
    # six runs today already: the seventh trigger is recorded as skipped and never retried
    full = Store(":memory:")
    for i in range(6):
        full.add_market_alert(PRINT - (i + 1) * H, "run", f"run:print:{PRINT - (i + 1) * H}", "🤖 Agents ran x.")
    out = sr.run(full, market(ev("NFP", PRINT)), ANALYSES, PRINT + 16 * MIN, SETTINGS)
    assert out["headline"] == ("🤖 NO NEW REPORT · BTC $80,000\n"
                               "Triggered by the US NFP release at 13:30 UTC (18:30 PKT), but 6 automatic runs already ran today. "
                               f"The next automatic run is allowed after {fmt_when(PRINT - 6 * H + 24 * H)}.")
    assert full.market_alerts_keyed("run:")[0]["notified"] == 0, "the cap line goes out: the trader learns why nothing new came"
    assert len(calls) == 4
    assert sr.run(full, market(ev("NFP", PRINT)), ANALYSES, PRINT + 17 * MIN, SETTINGS) is None
    # a skipped row is not a run: it neither counts toward the cap nor holds the gap (both the old "skipped …"
    # shape and the cap line since 6 Nov)
    # (a skipped row six minutes ago would hold the gap if it counted as a run: the trigger must still come back)
    fresh = ({"key": f"run:print:{PRINT}", "at_ms": PRINT, "reason": "after US CPI m/m (High) at 13:30 UTC (18:30 PKT)",
              "plain": "the US CPI m/m release at 13:30 UTC (18:30 PKT)"}, None)
    for skipped in ("skipped x", out["headline"]):
        assert sr.due([{"ts_ms": PRINT + 10 * MIN, "dedup_key": "run:print:1", "headline": skipped}], [ev("CPI m/m", PRINT)], [], PRINT + 16 * MIN) == fresh
    assert sr.due([{"ts_ms": PRINT + 10 * MIN, "dedup_key": "run:print:1", "headline": "🤖 NEW REPORT x"}], [ev("CPI m/m", PRINT)], [], PRINT + 16 * MIN) == (None, "waiting")
    s.close()
    full.close()


def test_a_stale_print_a_reboot_fixture_data_or_no_key_runs_nothing(monkeypatch):
    calls, s = [], Store(":memory:")
    fake_runs(monkeypatch, calls)
    old = market(ev("CPI m/m", PRINT))
    assert sr.run(s, old, ANALYSES, PRINT + 3 * H, SETTINGS) is None, "two hours on, the moment has passed"
    fresh = market(ev("CPI m/m", PRINT))
    assert sr.run(s, fresh, ANALYSES, PRINT + 16 * MIN, SETTINGS, may_run=False) is None, "fixture data"
    r = NS(problems=[])
    assert sr.run(s, fresh, ANALYSES, PRINT + 16 * MIN, None, r) is None and calls == []
    assert r.problems == ["scheduled run after US CPI m/m (High) at 13:30 UTC (18:30 PKT) skipped: no model key on this machine"]
    assert s.market_alerts_keyed("run:") == [], "no key: nothing claimed, so a machine with one still runs it"
    assert sr.run(s, fresh, {}, PRINT + 16 * MIN, SETTINGS) is None, "no desks, nothing to judge"
    s.close()


def test_a_failed_half_is_named_and_the_other_half_still_runs(monkeypatch):
    calls, s = [], Store(":memory:")

    def boom(store, m, a, now, settings):
        raise RuntimeError("quota")
    monkeypatch.setattr(sr, "_director", boom)
    monkeypatch.setattr(sr, "_war_room", lambda store, m, a, now, settings: calls.append("war_room") or "Live war room: BALANCED (2 side note(s))")
    r = NS(problems=[])
    out = sr.run(s, market(ev("CPI m/m", PRINT)), ANALYSES, PRINT + 16 * MIN, SETTINGS, r)
    assert calls == ["war_room"]
    assert out["headline"] == ("🤖 NO NEW REPORT · BTC $80,000\n"
                               "Triggered by the US CPI m/m release at 13:30 UTC (18:30 PKT), but the model was unavailable. "
                               "No plan is in force. The next automatic run comes with the next release or plan change, "
                               "not before 14:16 UTC (19:16 PKT).")
    assert r.problems == ["scheduled run: Director failed: RuntimeError: quota"], "the error goes to the pass, not the channel"
    assert s.market_alerts_keyed("run:")[0]["notified"] == 0, "a refused run is announced"
    # with a plan being watched, the line says which one still stands
    s.add_report("TIR1-20261105-153911", PRINT, "BEAR_TRAP", 80_000.0, 1.0, "NEUTRAL", mode="1-agent",
                 plan=json.dumps({"side": "LONG", "zone_low": 76000, "zone_high": 76500, "stop": 75000, "targets": [77000]}))
    out = sr.run(s, market(ev("CPI m/m", PRINT), ev("PPI m/m", PRINT + H)), ANALYSES, PRINT + H + 16 * MIN, SETTINGS, r)
    assert out["key"] == f"run:print:{PRINT + H}"
    assert "The last plan (report TIR1-20261105-153911, buy $76,000–$76,500) stays in force." in out["headline"]
    assert "quota" not in out["headline"] and "RuntimeError" not in out["headline"]
    s.close()


def test_the_store_rewrites_one_alert_text():
    s = Store(":memory:")
    i = s.add_market_alert(1, "run", "run:print:1", "before")
    assert s.set_market_alert_headline(i, "after") and s.market_alerts_keyed("run:")[0]["headline"] == "after"
    assert not s.set_market_alert_headline(999, "x")
    # a row marked sent can be reopened with its final text (a run that failed after claiming its row)
    s.mark_market_notified([i])
    assert s.market_alerts_keyed("run:")[0]["notified"] == 1
    assert s.set_market_alert_headline(i, "final", notified=0)
    row = s.market_alerts_keyed("run:")[0]
    assert row["headline"] == "final" and row["notified"] == 0
    s.close()
