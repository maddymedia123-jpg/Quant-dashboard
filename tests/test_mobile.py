"""Mobile fixes from the phone review: each test pins one defect so it cannot quietly return."""
import pathlib
import re

from core.config import CATEGORIES
from core.data.offline import fixture_market
from core.indicators.category import analyze_category
from ui.charts import build_chart_html
from ui.theme import css_text, md_safe

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------- the sidebar must be reopenable ----------
def test_the_header_holding_the_sidebar_button_is_not_hidden():
    css = css_text(dark=False)
    assert not re.search(r'header\[data-testid="stHeader"\]\s*\{\s*display:\s*none', css), \
        "hiding the header removes the only way to reopen a closed sidebar on a phone"
    assert 'data-testid="stExpandSidebarButton"' in css and "visibility: visible" in css


# ---------- nothing wider than the screen ----------
def test_card_and_report_tables_scroll_sideways_instead_of_clipping():
    css = css_text(dark=False)
    assert re.search(r"\.ti-card table \{[^}]*overflow-x: auto", css)
    assert re.search(r'stMarkdownContainer"\] table \{[^}]*overflow-x: auto', css)


def test_the_tab_strip_scrolls_rather_than_overflowing():
    assert re.search(r'tab-list"\] \{[^}]*overflow-x: auto', css_text(dark=False))


# ---------- readable and tappable ----------
def test_no_text_is_set_below_twelve_pixels():
    sizes = [float(x) for x in re.findall(r"font-size:\s*([\d.]+)rem", css_text(dark=False))]
    assert sizes and min(sizes) >= 0.75, f"smallest font is {min(sizes)}rem"


def test_buttons_meet_the_44px_tap_target_on_phones():
    css = css_text(dark=False)
    phone = re.search(r"@media \(max-width: 768px\) \{(.*)\}\s*</style>", css, re.S)
    assert phone and "min-height: 44px" in phone.group(1)


# ---------- charts on a touch screen ----------
def _chart() -> str:
    m = fixture_market()
    a = analyze_category(CATEGORIES["live"], m.spot.frames, m.futures)
    return build_chart_html(a, dark=False)


def test_a_vertical_swipe_over_the_chart_scrolls_the_page():
    assert "vertTouchDrag: false" in _chart()


def test_price_line_titles_drop_on_narrow_screens_so_the_axis_stays_thin():
    h = _chart()
    assert "NARROW" in h and "NARROW ? '' :" in h


# ---------- dollar amounts must not become LaTeX ----------
def test_dollar_signs_are_escaped_for_markdown():
    assert md_safe("support $81,500 and resistance $82,300") == "support \\$81,500 and resistance \\$82,300"
    assert md_safe("already \\$5") == "already \\$5", "an escaped dollar is left alone"
    assert md_safe("no money here") == "no money here"


def test_every_markdown_render_of_model_text_goes_through_md_safe():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.markdown(md_safe(md))" in src
    assert "md_safe(b.interpretation)" in src and "md_safe(b.key_risk_to_thesis)" in src


# ---------- copy ----------
def test_no_stale_phase_labels_remain_on_screen():
    for path in ("app.py", "ui/panels.py"):
        assert "Phase 2" not in (ROOT / path).read_text(encoding="utf-8"), path


def test_phones_open_with_the_sidebar_closed():
    """Forced open, the 300px sidebar covered all but 90px of a 390px phone at load."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'initial_sidebar_state="auto"' in src


def test_the_sidebar_reopen_button_is_a_full_tap_target_on_phones():
    phone = re.search(r"@media \(max-width: 768px\) \{(.*)\}\s*</style>", css_text(dark=False), re.S).group(1)
    assert re.search(r'stExpandSidebarButton"\][^{]*\{[^}]*min-width: 44px[^}]*min-height: 44px', phone)
