"""Page-level smoke tests: the app must render, and the Live Recon ledger must be reachable.

These run the real app.py through Streamlit's AppTest against the offline fixtures, so a broken panel
or a store that cannot serve the Live Recon anchor fails here instead of on the deployed site."""
import pathlib

import time

import pytest

from core.store import Store

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")

# Every store method the page calls. A refactor that drops one must fail a test, not a page load.
REQUIRED_STORE_METHODS = (
    "get_anchor", "set_anchor", "log_anchor_change", "anchor_log", "add_call", "due_calls",
    "score_call", "calls", "add_report", "reports", "score_report", "add_signal", "last_signal",
    "add_futures_sample", "futures_sample_at_or_before", "oldest_futures_sample", "futures_samples_since",
    "put_live_anchor", "get_live_anchor", "live_anchors", "add_side_note", "side_notes",
    "put_recon_score", "recon_scores", "anchors_due", "scored_windows",
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180)
    return at.run()


def test_the_store_exposes_every_method_the_page_calls():
    missing = [n for n in REQUIRED_STORE_METHODS if not hasattr(Store, n)]
    assert missing == [], f"page would fail at runtime on: {missing}"


def test_the_page_renders_with_no_exception(app):
    assert not app.exception, [e.value for e in app.exception]
    assert "Run Live Recon war room" in [b.label for b in app.sidebar.button]


def test_the_live_recon_ledger_is_reachable(app):
    """Guards the deploy failure where a cached Store predated the live_anchors table."""
    warnings = [w.value for w in app.warning]
    assert not any("Anchor ledger unavailable" in w for w in warnings), warnings
    markdown = " ".join(m.value for m in app.markdown)
    assert "4-hour anchored summary" in markdown
    assert "Nothing anchored for the current 4h candle" in markdown


def test_the_live_recon_tab_shows_the_smc_and_liquidity_protocols(app):
    markdown = " ".join(m.value for m in app.markdown)
    assert "Market structure (SMC)" in markdown and "Liquidity &amp; order flow" in markdown
    assert "candle close-position proxy" in markdown, "delta must be labelled, not passed off as tape"
    assert not any("Protocols unavailable" in w.value for w in app.warning)


def test_every_category_tab_has_its_own_war_room(app):
    labels = [b.label for b in app.button]
    for name in ("Live Recon", "Intraday", "Weekly", "Monthly"):
        assert f"Run {name} war room" in labels, f"{name} tab has no war-room button"
    markdown = " ".join(m.value for m in app.markdown)
    for title in ("4-hour anchored summary", "Daily anchored summary", "Weekly anchored summary",
                  "Monthly anchored summary"):
        assert title in markdown, f"missing: {title}"
    for window in ("current 4h candle", "current daily candle", "current weekly candle",
                   "current monthly candle"):
        assert window in markdown


def test_every_category_tab_shows_protocols_for_its_own_timeframes(app):
    markdown = " ".join(m.value for m in app.markdown)
    assert markdown.count("Market structure (SMC)") == 4, "one SMC panel per category tab"
    assert "<td>1w</td>" in markdown, "weekly and monthly read the weekly candle"


def test_every_category_tab_shows_its_accuracy_report(app):
    markdown = " ".join(m.value for m in app.markdown)
    assert markdown.count("Accuracy report") >= 4
    for window in ("4h windows scored", "daily windows scored", "weekly windows scored", "monthly windows scored"):
        assert window in markdown
    assert not any("Accuracy audit unavailable" in w.value for w in app.warning)


def test_a_module_left_stale_by_a_redeploy_is_reloaded(tmp_path, monkeypatch):
    """What the live site hit on 9/22: Streamlit Cloud re-ran the new app.py against the previous revision's
    ui.theme still held in memory, so `from ui.theme import md_safe` raised ImportError until a reboot."""
    import sys
    import types

    import ui.theme

    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    monkeypatch.delattr(ui.theme, "md_safe")                     # the old module, as the server held it
    deploy = sys.modules.setdefault("_ti_deploy", types.ModuleType("_ti_deploy"))
    monkeypatch.setattr(deploy, "fingerprint", -1.0, raising=False)   # loaded by an earlier revision

    # the app will swap in fresh core/ui modules; put the originals back afterwards so the other tests keep
    # the class objects they imported (pydantic rejects an instance of a same-named class from a reload)
    project = lambda: [n for n in sys.modules if n.split(".")[0] in ("core", "ui")]  # noqa: E731
    saved = {n: sys.modules[n] for n in project()}
    try:
        at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run()
        assert not at.exception, [e.value for e in at.exception]
        assert hasattr(sys.modules["ui.theme"], "md_safe"), "the fresh module replaced the stale one"
    finally:
        for n in project():
            del sys.modules[n]
        sys.modules.update(saved)


def test_every_tab_shows_its_sub_agent_deck(app):
    """The deck reads the judges' own state, so it must appear once per category tab."""
    markdown = " ".join(m.value for m in app.markdown)
    assert markdown.count("Institutional quantitative matrix") == 4
    # labels that belong to the deck alone - "Hurst" also appears in the volatility matrix above it
    for reading in ("Point of control", "Options max pain", "Taker buy/sell", "Range position",
                    "Funding 7d mean", "Longs liquidated 24h"):
        assert markdown.count(reading) == 4, reading
    assert "A dash means no feed, not zero." in markdown


def test_the_trap_tab_is_present_with_its_console(app):
    assert not app.exception, [e.value for e in app.exception]
    assert "Sweep every timeframe for traps" in [b.label for b in app.button]
    markdown = " ".join(m.value for m in app.markdown)
    assert "TRAP intelligence · watchdog" in markdown
    # wording distinct from the desks' own "Nothing anchored for the current ... candle yet", which is
    # rendered once per desk: the old assertion matched that instead and passed with the console's
    # empty state deleted outright
    assert markdown.count("No trap is anchored.") == 1, "the empty console says so, once"
    assert "Nothing expires on a clock" in markdown
    assert not any("could not be read" in str(e.value) or "could not be drawn" in str(e.value)
                   for e in app.error)


def test_an_anchored_trap_warns_every_desk_that_reads_its_timeframe(tmp_path, monkeypatch):
    """The outbound half of the handshake, routed by timeframe.

    A 1d trap is one market event, and it belongs on every desk that reads the 1d chart - Intraday,
    Weekly and Monthly - but not on Live, which reads 15m/1h/4h and cannot see the level."""
    import streamlit as st

    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    from core.store import Store

    # the app caches its Store as a resource, and this process has already run the app for other tests
    st.cache_resource.clear()
    st.cache_data.clear()
    s = Store()
    # straddling the fixture price, so page load neither invalidates it nor plays it out
    s.put_trap(category="weekly", side="bull_trap", timeframe="1d", level=78_600.0, invalidation=78_900.0,
               plays_out=78_000.0, declared_ms=1, score=72.0, head_call="ENGINEERED_TRAP",
               evidence=["buyside liquidity swept and reclaimed"], notes=[])
    s.close()

    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)
    assert markdown.count("TRAP warning") == 3, "the three desks that read the 1d, and not Live"
    assert "1d at $78,600" in markdown and "invalidated at $78,900" in markdown.lower()
    assert "Anchored bull trap" in markdown, "and it appears in the console too"
    assert "buyside liquidity swept and reclaimed" in markdown, "the card carries its own evidence"
    assert "sub-agents 72/100" in markdown, "and the score that declared it"


def _fresh_store(tmp_path, monkeypatch):
    """A store the app will pick up, with its caches cleared - this process has already run the app."""
    import streamlit as st

    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    from core.store import Store

    st.cache_resource.clear()
    st.cache_data.clear()
    return Store()


def test_the_active_trade_tab_offers_the_form_and_says_nothing_is_tracked(app):
    assert not app.exception, [e.value for e in app.exception]
    labels = [b.label for b in app.button]
    assert "Open and track this trade" in labels
    assert "Run stress-test" in labels, "the leverage stress-test is still there"
    markdown = " ".join(m.value for m in app.markdown)
    assert "Active trade desk" in markdown
    captions = " ".join(c.value for c in app.caption)
    assert "No position is being tracked" in captions
    assert "No alert channel is configured" in captions, "and it says so rather than implying alerts work"


def test_a_pinned_trade_renders_its_whole_plan_and_survives_a_rerun(tmp_path, monkeypatch):
    """The spec's card: score, stop verdict, timed targets, the guardrail and the schedule, pinned."""
    store = _fresh_store(tmp_path, monkeypatch)
    now = int(time.time() * 1000)
    plan = {
        "asset": "BTC/USDT", "side": "LONG", "entry": 78_000.0, "category": "live", "timeframe": "1h",
        "ok": True, "score": 72.0, "probability": 64.0, "for_points": 78.0, "against_points": 34.0,
        "stop_verdict": "would be hunted",
        "stop_reasons": ["resting liquidity at 77,800 sits just 100 beyond the stop"],
        "targets": [{"name": "TP1", "price": 78_600.0, "basis": "the 1 sigma band on the 4h",
                     "due_ms": now + 3_600_000, "due_from": "Fibonacci time window", "reward_r": 1.5},
                    {"name": "TP2", "price": 79_200.0, "basis": "the 2 sigma band on the 4h",
                     "due_ms": now + 7_200_000, "due_from": "the second 4h close", "reward_r": 3.0}],
        "pullback": {"available": True, "note": "", "ordinary": 400.0, "edge": 900.0, "breaking": 1800.0,
                     "horizon_bars": 12, "timeframe": "1h",
                     "levels": {"ordinary": 77_600.0, "edge": 77_100.0, "breaking": 76_200.0}},
        "schedule": [{"timeframe": "4h", "at_ms": now + 1_800_000, "reason": "the 4h close"}],
        "alignment": [{"category": "live", "label": "Live Recon", "bias": "BULL", "verdict": "supports",
                       "note": ""},
                      {"category": "weekly", "label": "Weekly", "bias": "BEAR", "verdict": "opposes",
                       "note": ""}],
        "aligned": "the desks disagree",
        "traps": [{"trap_id": 3, "side": "bull_trap", "timeframe": "4h", "level": 78_650.0,
                   "near": "TP1", "distance": 50.0}],
        "for_domains": [{"key": "quant", "title": "Quant & Statistics", "points": 16.0}],
        "against_domains": [{"key": "quant", "title": "Quant & Statistics", "points": 7.0}],
        "head_call": "WAIT", "head_confidence": 0.62,
        "notes": ["The stop as given is would be hunted; the score above is for the corrected stop."],
    }
    tid = store.open_trade(opened_ms=now - 3_600_000, asset="BTC/USDT", side="LONG", entry=78_000.0,
                           stop=77_600.0, given_stop=77_900.0, stop_verdict="would be hunted",
                           category="live", plan=plan)
    store.close()

    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)

    assert "BTC/USDT from $78,000" in markdown and "score 72/100" in markdown
    assert "64% confluence estimate" in markdown
    assert "stop would be hunted" in markdown and "you gave $77,900" in markdown
    assert "resting liquidity at 77,800" in markdown
    for cell in ("TP1", "$78,600", "1.50R", "Fibonacci time window", "the 1 sigma band on the 4h"):
        assert cell in markdown, cell
    assert "a move to $77,600 is ordinary" in markdown and "$76,200 this is no longer a pullback" in markdown
    assert "the 4h close" in markdown
    assert "Live Recon: BULL - supports this trade" in markdown
    assert "Weekly: BEAR - against this trade" in markdown
    assert "anchored bull trap on the 4h sits $50" in markdown
    assert "Head of the Active Trade desk: Wait" in markdown
    assert "not a backtested win rate" in markdown, "the estimate is not presented as a win rate"

    assert f"tr_kill_{tid}" in [b.key for b in at.button]
    again = at.run()                                    # a rerun must not lose the card
    assert "BTC/USDT from $78,000" in " ".join(m.value for m in again.markdown)


def test_killing_a_trade_unpins_it(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    now = int(time.time() * 1000)
    tid = store.open_trade(opened_ms=now, asset="BTC/USDT", side="LONG", entry=78_000.0, stop=77_600.0,
                           given_stop=None, stop_verdict="supplied", category="live",
                           plan={"asset": "BTC/USDT", "side": "LONG", "ok": False, "targets": [],
                                 "notes": ["no provider configured"]})
    store.close()

    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "BTC/USDT from $78,000" in " ".join(m.value for m in at.markdown)

    kill = next(b for b in at.button if b.key == f"tr_kill_{tid}")
    after = kill.click().run()
    assert not after.exception, [e.value for e in after.exception]
    markdown = " ".join(m.value for m in after.markdown)
    captions = " ".join(c.value for c in after.caption)
    assert "Recently closed" in markdown or "killed by the trader" in markdown
    assert "No position is being tracked" in captions

    from core.store import Store
    reopened = Store()
    row = reopened.trade(tid)
    reopened.close()
    assert row["status"] == "closed" and row["closed_reason"] == "killed by the trader"


def test_an_unscored_card_still_shows_its_levels(tmp_path, monkeypatch):
    """When the provider is down the card carries its deterministic plan rather than an empty box."""
    store = _fresh_store(tmp_path, monkeypatch)
    now = int(time.time() * 1000)
    store.open_trade(opened_ms=now, asset="BTC/USDT", side="SHORT", entry=78_000.0, stop=78_400.0,
                     given_stop=None, stop_verdict="supplied", category="live",
                     plan={"asset": "BTC/USDT", "side": "SHORT", "ok": False,
                           "stop_verdict": "supplied",
                           "stop_reasons": ["the last swing sits at 78,300"],
                           "targets": [{"name": "TP1", "price": 77_400.0, "basis": "the 1 sigma band",
                                        "due_ms": now + 3_600_000, "due_from": "the next 1h close",
                                        "reward_r": 1.5}],
                           "notes": ["No model provider is configured, so the ten sub-agents were not run."]})
    store.close()

    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)
    assert "unscored: the sub-agents did not answer" in markdown
    assert "$77,400" in markdown and "the last swing sits at 78,300" in markdown
    assert "score" not in markdown.split("unscored")[1][:40], "no score is claimed"
