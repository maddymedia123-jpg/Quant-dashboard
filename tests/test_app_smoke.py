"""Page-level smoke tests: the app must render, and the Live Recon ledger must be reachable.

These run the real app.py through Streamlit's AppTest against the offline fixtures, so a broken panel
or a store that cannot serve the Live Recon anchor fails here instead of on the deployed site."""
import pathlib

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
