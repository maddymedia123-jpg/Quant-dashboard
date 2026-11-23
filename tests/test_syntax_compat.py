"""The dashboard runs on Streamlit Cloud's Python, not this machine's. An f-string that reuses its own quote
inside the braces (f"{c['x']}" is fine; f'{c['x']}' is not) parses only from Python 3.12, and one such line
(core/indicators/horizons.py, 5 Nov) took the whole app down with a SyntaxError at import after Streamlit
rebuilt the environment on 23 Nov. ast.parse(feature_version=...) does not catch it, so this is a text check."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [ROOT / "app.py", ROOT / "worker.py", ROOT / "mcp_server.py", ROOT / "book_collector.py",
         *sorted((ROOT / "core").rglob("*.py"))]
# an f-string opened with one quote kind, then a brace, then that same quote before the brace closes
NESTED = re.compile(r"""(?<![A-Za-z0-9_])[fF][rR]?(?:'(?:[^'\\\n]|\\.)*\{[^}\n]*'|"(?:[^"\\\n]|\\.)*\{[^}\n]*")""")


def test_no_f_string_reuses_its_own_quote_inside_the_braces():
    hits = []
    for p in FILES:
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if NESTED.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "needs Python 3.12 or newer, which Streamlit Cloud does not run:\n" + "\n".join(hits)


def test_the_check_itself_fires():
    assert NESTED.search("""x = f'{c['price']:,.0f}'""")
    assert NESTED.search('''x = f"{d["k"]}"''')
    assert not NESTED.search("""x = f"{c['price']:,.0f}" """)
    assert not NESTED.search("""x = f'{d["k"]}' """)
    assert not NESTED.search("""x = 'plain {braces}' """)
