"""Page-level smoke tests: the app must render, and the Live Recon ledger must be reachable.

These run the real app.py through Streamlit's AppTest against the offline fixtures, so a broken panel
or a store that cannot serve the Live Recon anchor fails here instead of on the deployed site."""
import pathlib
import re

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


DESKS = ("live", "intraday", "weekly", "monthly")
SCREENS = DESKS + ("trap", "trade", "director", "macro")


def visit(at, screen):
    """Open one screen from the rail. Only the open screen renders, so a test reads the screen it means."""
    return at.radio(key="screen").set_value(screen).run()


PROTOCOLS = ("Volatility", "Structure", "Liquidity", "Quant matrix")


def protocol(at, desk, name):
    """Flip a desk's protocol switch. One reading shows at a time, so a test opens the one it means."""
    return at.button_group(key=f"proto_{desk}").set_value(name).run()


class Tour:
    """Everything the four desks render, gathered by opening each in turn and flipping each desk through
    its four protocols - what one run of the old tab strip used to show at once. Desk-wide content (the
    war room, the accuracy report) is read once per desk, on the first protocol."""

    def __init__(self, at, screens=DESKS, protocols=True):
        self.markdown, self.buttons, self.warnings, self.exceptions = [], [], [], []
        for screen in screens:
            at = visit(at, screen)
            self._take(at, everything=True)
            if protocols and screen in DESKS:
                for name in PROTOCOLS[1:]:
                    at = protocol(at, screen, name)
                    self._take(at, everything=False)
                at = protocol(at, screen, PROTOCOLS[0])
        self.text = " ".join(self.markdown)

    def _take(self, at, everything: bool):
        self.exceptions += [e.value for e in at.exception]
        self.warnings += [w.value for w in at.warning]
        if everything:
            self.markdown += [m.value for m in at.markdown]
            self.buttons += [b.label for b in at.button]
        else:
            # only the protocol body, so desk-wide panels are not counted four times
            self.markdown += [m.value for m in at.markdown
                              if any(k in m.value for k in ("Market structure (SMC)", "Liquidity &amp; order flow",
                                                            "Institutional quantitative matrix"))]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_OFFLINE_FIXTURES", "1")
    monkeypatch.setenv("TI_DATA_DIR", str(tmp_path))
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180)
    # explicit secrets, so the page never reads the developer's .streamlit/secrets.toml (which may hold a
    # live webhook and would make "no channel configured" depend on whose machine runs the suite)
    at.secrets["TI_TEST"] = "1"
    return at.run()


@pytest.fixture
def tour(app):
    t = Tour(app)
    assert not t.exceptions, t.exceptions
    return t


def test_the_store_exposes_every_method_the_page_calls():
    missing = [n for n in REQUIRED_STORE_METHODS if not hasattr(Store, n)]
    assert missing == [], f"page would fail at runtime on: {missing}"


def test_the_page_renders_with_no_exception(app):
    assert not app.exception, [e.value for e in app.exception]
    assert "Run Analysis" in [b.label for b in app.button]
    assert "Run Live Recon war room" in [b.label for b in app.button], "the default screen is the Live desk"


def test_the_rail_lists_four_desks_then_four_engines_and_opens_on_live(app):
    rail = app.radio(key="screen")
    assert list(rail.options) == ["Live Recon", "Intraday", "Weekly", "Monthly",
                                  "TRAP Intelligence", "Active Trade", "Director report", "Macro & News"]
    assert rail.value == "live"


def test_every_screen_renders_on_its_own(app):
    for screen in SCREENS:
        at = visit(app, screen)
        assert not at.exception, (screen, [e.value for e in at.exception])


def test_only_the_open_screen_renders(app):
    markdown = " ".join(m.value for m in app.markdown)
    assert "4-hour anchored summary" in markdown and "Daily anchored summary" not in markdown
    assert "· Engine</span>" not in markdown and "TRAP intelligence · watchdog" not in markdown,         "no engine renders under a desk"
    weekly = " ".join(m.value for m in visit(app, "weekly").markdown)
    assert "Weekly anchored summary" in weekly and "4-hour anchored summary" not in weekly


def test_the_live_recon_ledger_is_reachable(app):
    """Guards the deploy failure where a cached Store predated the live_anchors table."""
    warnings = [w.value for w in app.warning]
    assert not any("Anchor ledger unavailable" in w for w in warnings), warnings
    markdown = " ".join(m.value for m in app.markdown)
    assert "4-hour anchored summary" in markdown
    assert "Nothing anchored for the current 4h candle" in markdown


def test_the_live_recon_tab_shows_the_smc_and_liquidity_protocols(app):
    first = " ".join(m.value for m in app.markdown)
    assert "Volatility matrix" in first and "Market structure (SMC)" not in first, "one protocol at a time"
    smc = " ".join(m.value for m in protocol(app, "live", "Structure").markdown)
    assert "Market structure (SMC)" in smc
    liq = protocol(app, "live", "Liquidity")
    markdown = " ".join(m.value for m in liq.markdown)
    assert "Liquidity &amp; order flow" in markdown
    assert "candle close-position proxy" in markdown, "delta must be labelled, not passed off as tape"
    assert not any("Protocols unavailable" in w.value for w in liq.warning)


def test_every_category_tab_has_its_own_war_room(tour):
    labels = tour.buttons
    for name in ("Live Recon", "Intraday", "Weekly", "Monthly"):
        assert f"Run {name} war room" in labels, f"{name} tab has no war-room button"
    markdown = tour.text
    for title in ("4-hour anchored summary", "Daily anchored summary", "Weekly anchored summary",
                  "Monthly anchored summary"):
        assert title in markdown, f"missing: {title}"
    for window in ("current 4h candle", "current daily candle", "current weekly candle",
                   "current monthly candle"):
        assert window in markdown


def test_every_category_tab_shows_protocols_for_its_own_timeframes(tour):
    markdown = tour.text
    assert markdown.count("Market structure (SMC)") == 4, "one SMC panel per category tab"
    assert "<td>1w</td>" in markdown, "weekly and monthly read the weekly candle"


def test_every_category_tab_shows_its_accuracy_report(tour):
    markdown = tour.text
    assert markdown.count("Accuracy report") >= 4
    for window in ("4h windows scored", "daily windows scored", "weekly windows scored", "monthly windows scored"):
        assert window in markdown
    assert not any("Accuracy audit unavailable" in w for w in tour.warnings)


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


def test_every_tab_shows_its_sub_agent_deck(tour):
    """The deck reads the judges' own state, so it must appear once per category tab."""
    markdown = tour.text
    assert markdown.count("Institutional quantitative matrix") == 4
    # labels that belong to the deck alone - "Hurst" also appears in the volatility matrix above it
    for reading in ("Point of control", "Max pain 20", "Taker buy/sell", "Range position"):  # "Max pain <date>"
        assert markdown.count(reading) == 4, reading
    # each desk reads liquidations over its own window: one desk each
    for over in ("4h", "24h", "7d", "30d"):
        assert markdown.count(f"Longs liquidated {over}") == 1, over
    assert "A dash means no feed, not zero." in markdown


def test_the_trap_tab_is_present_with_its_console(app):
    app = visit(app, "trap")
    assert not app.exception, [e.value for e in app.exception]
    assert "Sweep every timeframe" in [b.label for b in app.button]
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
    markdown = Tour(at, DESKS + ("trap",)).text
    assert markdown.count("TRAP warning") == 3, "the three desks that read the 1d, and not Live"
    assert "1d at $78,600" in markdown and "invalidated above $78,900" in markdown.lower()
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
    app = visit(app, "trade")
    assert not app.exception, [e.value for e in app.exception]
    labels = [b.label for b in app.button]
    assert "Open and track this trade" in labels
    assert "Run stress-test" in labels, "the leverage stress-test is still there"
    markdown = " ".join(m.value for m in app.markdown)
    assert "Active Trade" in markdown
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

    at = visit(pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run(), "trade")
    assert not at.exception, [e.value for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)

    assert "BTC/USDT from $78,000" in markdown and "score 72/100" in markdown
    words = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markdown))
    assert "64% confluence estimate" in markdown
    assert "ti-stopv down'>would be hunted<" in markdown and "Given $77,900" in words
    assert "resting liquidity at 77,800" in markdown
    for cell in ("TP1", "$78,600", "1.50R", "Fibonacci time window", "the 1 sigma band on the 4h"):
        assert cell in markdown, cell
    assert "a move to $77,600 is ordinary" in markdown and "$76,200 this is no longer a pullback" in markdown
    assert "the 4h close" in markdown
    assert "Live Recon BULL supports this trade" in words
    assert "Weekly BEAR against this trade" in words
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

    at = visit(pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run(), "trade")
    assert not at.exception, [e.value for e in at.exception]
    assert "BTC/USDT from $78,000" in " ".join(m.value for m in at.markdown)

    kill = next(b for b in at.button if b.key == f"tr_kill_{tid}")
    asked = kill.click().run()
    assert "The card is unpinned and every alert for this position stops" in         " ".join(m.value for m in asked.markdown), "it asks first, in the dialog"
    from core.store import Store as _S
    _peek = _S()
    assert _peek.trade(tid)["status"] == "open", "nothing is killed until it is confirmed"
    _peek.close()
    confirm = next(b for b in asked.button if b.key == f"tr_kill_confirm_{tid}")
    after = confirm.click().run()
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

    at = visit(pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run(), "trade")
    assert not at.exception, [e.value for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)
    assert "unscored: the sub-agents did not answer" in markdown
    assert "$77,400" in markdown and "the last swing sits at 78,300" in markdown
    assert "score" not in markdown.split("unscored")[1][:40], "no score is claimed"


def test_the_phone_strip_and_the_rail_open_the_same_screen(app):
    """Two controls, one screen: the strip is what a phone sees, the rail what a desktop sees."""
    at = app.button_group(key="screen_nav").set_value("macro").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="screen").value == "macro"
    assert "08 · Engine</span><h1>Macro &amp; News" in " ".join(m.value for m in at.markdown)
    at = visit(at, "weekly")
    assert at.button_group(key="screen_nav").value == "weekly", "the rail moves the strip with it"


def test_clearing_the_strip_keeps_the_open_screen():
    # tested as a rule rather than through AppTest, whose set_value(None) on pills is a no-op - the
    # first version of this test passed with the guard deleted
    from ui.nav import pick_from_strip
    assert pick_from_strip("weekly", None) == ("weekly", "weekly")
    assert pick_from_strip("weekly", "") == ("weekly", "weekly")
    assert pick_from_strip("weekly", "macro") == ("macro", "macro")


def test_a_desk_warning_opens_the_trap_desk(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    store.put_trap(category="weekly", side="bull_trap", timeframe="1d", level=78_600.0, invalidation=78_900.0,
                   plays_out=78_000.0, declared_ms=1, score=72.0, head_call="ENGINEERED_TRAP",
                   evidence=["swept"], notes=[])
    store.close()
    at = visit(pytest.importorskip("streamlit.testing.v1").AppTest.from_file(APP, default_timeout=180).run(),
               "intraday")
    jump = next(b for b in at.button if b.key == "open_trap_intraday")
    at = jump.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="screen").value == "trap" and at.button_group(key="screen_nav").value == "trap"
    assert "05 · Engine" in " ".join(m.value for m in at.markdown)


def test_one_range_per_desk_and_the_strip_reads_the_desk_window(app):
    """The plain read, the volatility card and the forecast quote one 68% range; Live's strip says 4h."""
    import re
    md = " ".join(m.value for m in visit(app, "live").markdown)
    fc = re.search(r"68% range <strong>(\$[\d,]+) to (\$[\d,]+)</strong>", md)
    plain = re.search(r"Normal range for the next 4 hours \(where it ends two times in three\): (\$[\d,]+) to (\$[\d,]+)", md)
    assert fc and plain and fc.groups() == plain.groups(), (fc and fc.groups(), plain and plain.groups())
    assert "Liq 4h" in md and "Taker delta 4h" in md and "Liq 24h" not in md
