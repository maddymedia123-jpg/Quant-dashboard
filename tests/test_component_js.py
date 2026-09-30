"""Syntax-check the JavaScript we ship inside components.

This exists because an unescaped apostrophe in a tooltip - `'...this desk's horizon...'` - was a
JavaScript syntax error that killed the entire script tag, so every chart on every tab rendered blank.
Nothing caught it: AppTest does not execute component JavaScript, and from Python a broken script is
just a string that looks like any other. The chart tests all passed while the charts were gone.

So the guard is to actually parse it. `node --check` is the cheapest real parser available, and these
tests skip rather than fail where node is not installed - a missing dev tool should not look like a
broken product.
"""
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed; cannot parse the JS")

INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def inline_js(html: str) -> str:
    """Every inline script in the document, concatenated. Scripts with a src are somebody else's."""
    return "\n".join(INLINE_SCRIPT.findall(html))


def parses(js: str) -> tuple[bool, str]:
    path = pathlib.Path(tempfile.mkdtemp()) / "component.mjs"
    path.write_text(js, encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def analyses():
    import os

    os.environ["TI_OFFLINE_FIXTURES"] = "1"
    os.environ.setdefault("TI_DATA_DIR", tempfile.mkdtemp())
    from core.config import CATEGORIES
    from core.data.offline import fixture_market
    from core.indicators.category import analyze_category

    m = fixture_market()
    for key, cfg in CATEGORIES.items():
        a = analyze_category(cfg, m.spot.frames, m.futures)
        if a is not None:
            yield key, a


@needs_node
@pytest.mark.parametrize("dark", [True, False])
def test_every_chart_script_parses(dark):
    from ui.charts import build_chart_html

    checked = 0
    for key, a in analyses():
        js = inline_js(build_chart_html(a, dark=dark))
        assert js.strip(), f"{key}: no inline script found to check"
        ok, err = parses(js)
        assert ok, f"{key} (dark={dark}) ships broken JavaScript:\n{err[:400]}"
        checked += 1
    assert checked == 4, "all four desks were checked"


@needs_node
@pytest.mark.parametrize("dark", [True, False])
def test_the_clock_script_parses(dark):
    from ui import panels

    js = inline_js(panels.clock_html(dark=dark))
    assert js.strip(), "no inline script in the clock"
    ok, err = parses(js)
    assert ok, f"the clock ships broken JavaScript:\n{err[:400]}"


@needs_node
def test_the_checker_can_actually_fail():
    """A verification that cannot fail verifies nothing - the lesson this whole file comes from."""
    ok, err = parses("const s = 'an unterminated string;")
    assert not ok and err, "node --check accepted a syntax error"
    ok, _ = parses("const s = 'fine';")
    assert ok, "node --check rejected valid JavaScript"


@needs_node
def test_the_exact_bug_that_blanked_the_charts_is_caught():
    """An apostrophe inside a single-quoted string, which is how the tooltip broke."""
    ok, _ = parses("el.title = 'this desk's horizon';")
    assert not ok, "the original failure mode must be detectable"
