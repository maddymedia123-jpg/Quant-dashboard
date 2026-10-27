"""Theme injection and small HTML building blocks, in the Industry design system.

Industry is a wireframe: steel-blue on a light technical ground, Barlow Condensed headings over Barlow,
and cards drawn as blueprint objects - square-cornered, hairline-bordered, with "+" registration marks
at the corners and no surface fill. The tokens below are the design system's own (styles.css in the
Claude Design project); take colours from `palette()`, never hard-code a new one.

Up and down are monochrome by default, as the design specifies: up is the deep steel, down the near-black
ink, and candles tell them apart by fill (hollow up, solid down) rather than by hue. `signal=True`
switches to the design's optional green/red pair for readers who want the conventional colours."""
from __future__ import annotations

import html
import re

import streamlit as st

_BODY = "'Barlow', system-ui, -apple-system, sans-serif"
_HEADING = "'Barlow Condensed', 'Barlow', system-ui, sans-serif"

# The Industry tokens. `border` and `grid` are the divider (text at 16%) and the faint rule (text at 9%)
# flattened onto the ground, because the chart library takes solid colours only.
LIGHT = dict(bg="#F2F2F3", panel="#F2F2F3", surface="#E9E9EA", border="#CFD0D1", grid="#DFDFE0",
             text="#1D1F20", muted="#5D5E5F", accent="#5980A6", accent_soft="#EEF6FF", accent_deep="#416180",
             accent_ink="#2C455D", accent_night="#1D2D3D", up="#416180", down="#2B2B2D", warn="#2C455D",
             fib="#98989B", fibFuture="#5980A6", now="#1D1F20", mono=_BODY)
# The same system reversed onto a dark ink ground for the dark-mode toggle: steel lightens to its 400
# step so it keeps contrast, and "down" becomes the paper tone so hollow-versus-solid still reads.
DARK = dict(bg="#16181A", panel="#16181A", surface="#1E2124", border="#3A3D40", grid="#26292C",
            text="#E7E7EA", muted="#A3A5A8", accent="#94BCE3", accent_soft="#1F2D3A", accent_deep="#B5D9FD",
            accent_ink="#D6EBFF", accent_night="#2C455D", up="#94BCE3", down="#E7E7EA", warn="#D6EBFF",
            fib="#5D5D60", fibFuture="#94BCE3", now="#E7E7EA", mono=_BODY)
# The design's optional signal pair (oklch(0.5 0.12 150) and oklch(0.52 0.17 27), converted to sRGB).
SIGNAL = dict(up="#2E7A47", down="#B6372F", warn="#9A6300")
SIGNAL_DARK = dict(up="#4FB873", down="#F06A5F", warn="#E0A53A")


def palette(dark: bool, signal: bool = False) -> dict:
    p = dict(DARK if dark else LIGHT)
    if signal:
        p.update(SIGNAL_DARK if dark else SIGNAL)
    return p


def fmt_num(v, digits: int = 0, prefix: str = "", suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{prefix}{v:,.{digits}f}{suffix}"


def fmt_pct(v, digits: int = 2, signed: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def tone_for_direction(direction: str) -> str:
    return {"BULLISH": "up", "BEARISH": "down"}.get(direction, "neutral")


def chip(text: str, tone: str = "") -> str:
    return f'<span class="ti-chip {tone}">{html.escape(text)}</span>'


# The four registration marks every framed object wears (the design system's `.corner` children).
CORNERS = '<i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>'


def kpi_html(items: list[dict]) -> str:
    cells = []
    for i in items:
        delta = i.get("delta")
        tone = i.get("tone") or ""
        d = f'<div class="d {tone}">{html.escape(str(delta))}</div>' if delta else ""
        cells.append(f'<div class="ti-kpi"><div class="l">{html.escape(str(i["label"]))}</div>'
                     f'<div class="v">{html.escape(str(i["value"]))}</div>{d}</div>')
    return f'<div class="ti-kpis">{"".join(cells)}</div>'


def kpi_strip(items: list[dict]) -> None:
    st.markdown(kpi_html(items), unsafe_allow_html=True)


def card_html(title: str, body_html: str, tone: str = "neutral") -> str:
    return f'<div class="ti-card {tone}">{CORNERS}<h4>{html.escape(title)}</h4>{body_html}</div>'


def card(title: str, body_html: str, tone: str = "neutral") -> None:
    st.markdown(card_html(title, body_html, tone), unsafe_allow_html=True)


def md_safe(text: str) -> str:
    """Escape dollar signs for Markdown. Streamlit reads $...$ as LaTeX, so a sentence holding two prices
    ("support $81,500 and resistance $82,300") would otherwise render as a garbled math expression."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def inject_css(dark: bool, signal: bool = False) -> None:
    st.markdown(css_text(dark, signal), unsafe_allow_html=True)


def css_text(dark: bool, signal: bool = False) -> str:
    p = palette(dark, signal)
    label = (f"font-family: {_BODY}; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; "
             f"font-weight: 600;")
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: {_BODY}; font-variant-numeric: tabular-nums; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {p['bg']}; color: {p['text']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stExpandSidebarButton"] {{ visibility: visible !important; }}
#MainMenu, footer {{ visibility: hidden; }}
::selection {{ background: color-mix(in srgb, {p['accent']} 30%, transparent); }}
:focus-visible {{ outline: 2px solid {p['accent']} !important; outline-offset: 2px; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1440px; }}
h1, h2, h3, h4 {{ font-family: {_HEADING}; font-weight: 600; letter-spacing: 0.02em; }}
h1, h2, h3, h4, p, li, label, .stMarkdown {{ color: {p['text']}; }}
/* sidebar: the desk rail */
section[data-testid="stSidebar"] {{ background: {p['bg']}; border-right: 1px solid {p['border']}; }}
section[data-testid="stSidebar"] * {{ color: {p['text']}; }}
.ti-title {{ font-family: {_HEADING}; font-size: 1.5rem; font-weight: 600; line-height: 1; text-transform: uppercase; letter-spacing: 0.04em; margin: 0; }}
.ti-sub {{ color: {p['muted']}; font-size: 0.82rem; margin: 4px 0 12px 0; }}
/* buttons: square, hairline; the primary is the one solid object, in steel */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
  border-radius: 0; border: 1px solid {p['border']}; background: transparent; color: {p['text']};
  font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
.stButton button:hover:not(:disabled), [data-testid="stFormSubmitButton"] button:hover:not(:disabled) {{
  background: color-mix(in srgb, {p['text']} 7%, transparent); border-color: {p['border']}; color: {p['text']}; }}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {{
  background: {p['accent']}; border-color: {p['accent']}; color: {p['bg']}; }}
.stButton button[kind="primary"]:hover:not(:disabled) {{ background: {p['accent_deep']}; color: {p['bg']}; }}
.stButton button:disabled, .stButton button:disabled * {{ color: {p['muted']}; opacity: 0.45; }}
/* inputs and choices: square */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput div {{ border-radius: 0 !important; }}
[data-testid="stExpander"] details {{ border-radius: 0; border: 1px solid {p['border']}; }}
[data-testid="stExpander"] summary p {{ font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
/* tabs: condensed, uppercase, a steel underline on the open one */
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {p['border']}; overflow-x: auto; flex-wrap: nowrap; scrollbar-width: thin; }}
.stTabs [data-baseweb="tab"] {{ padding: 9px 14px; font-family: {_HEADING}; font-weight: 600; font-size: 1.02rem; text-transform: uppercase; letter-spacing: 0.04em; color: {p['muted']}; }}
.stTabs [data-baseweb="tab"] p {{ font-family: inherit; font-size: inherit; font-weight: inherit; text-transform: inherit; letter-spacing: inherit; }}
.stTabs [aria-selected="true"] {{ color: {p['text']}; border-bottom: 2px solid {p['accent']}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {p['accent']}; }}
/* KPIs: a ruled grid of equal cells, not floating tiles */
.ti-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: 0 0 16px 0;
  border-top: 1px solid {p['border']}; border-left: 1px solid {p['border']}; }}
.ti-kpi {{ padding: 14px 16px; border-right: 1px solid {p['border']}; border-bottom: 1px solid {p['border']}; min-width: 0; }}
.ti-kpi .l {{ {label} color: {p['muted']}; }}
.ti-kpi .v {{ font-family: {_HEADING}; font-size: 1.625rem; line-height: 1.875rem; font-weight: 600; margin-top: 2px; color: {p['text']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ti-kpi .d {{ font-size: 0.78rem; margin-top: 2px; color: {p['muted']}; }}
.ti-kpi .d.up, .ti-chip.up {{ color: {p['up']}; }}
.ti-kpi .d.down, .ti-chip.down {{ color: {p['down']}; }}
.ti-kpi .d.warn, .ti-chip.warn {{ color: {p['warn']}; }}
/* cards: blueprint objects - transparent, hairline, square, with corner registration marks */
.ti-card {{ position: relative; background: transparent; border: 1px solid {p['border']}; border-radius: 0; padding: 16px 18px; margin: 6px 0 22px 0; }}
.ti-card > .corner {{ position: absolute; width: 11px; height: 11px; color: color-mix(in srgb, {p['text']} 55%, transparent); }}
.ti-card > .corner::before, .ti-card > .corner::after {{ content: ""; position: absolute; background: currentColor; }}
.ti-card > .corner::before {{ left: 5px; top: 0; width: 1px; height: 100%; }}
.ti-card > .corner::after {{ top: 5px; left: 0; width: 100%; height: 1px; }}
.ti-card > .corner.tl {{ top: -6px; left: -6px; }} .ti-card > .corner.tr {{ top: -6px; right: -6px; }}
.ti-card > .corner.bl {{ bottom: -6px; left: -6px; }} .ti-card > .corner.br {{ bottom: -6px; right: -6px; }}
.ti-card h4 {{ {label} margin: 0 0 10px 0; color: {p['accent_deep']}; }}
.stApp [data-testid="stMarkdownContainer"] .ti-card p, .stApp [data-testid="stMarkdownContainer"] .ti-card li, .ti-card p, .ti-card li {{ font-size: 0.875rem; line-height: 1.43; margin: 0.2rem 0; color: {p['text']}; }}
.ti-card ul {{ padding-left: 1.1rem; margin: 0.2rem 0; }}
.ti-card .muted {{ color: {p['muted']}; }}
.ti-card strong {{ font-weight: 600; }}
/* a tone is a steel edge on the frame, never a fill */
.ti-card.up {{ border-color: {p['up']}; }}
.ti-card.down {{ border-color: {p['down']}; }}
.ti-card.warn {{ border-color: {p['warn']}; box-shadow: inset 3px 0 0 {p['warn']}; }}
/* chips: square tags */
.ti-chip {{ display: inline-block; font-size: 0.75rem; letter-spacing: 0.02em; padding: 2px 9px; border-radius: 0; border: 1px solid {p['border']}; color: {p['muted']}; margin: 0 6px 6px 0; }}
.ti-chip.up {{ border-color: {p['up']}; }} .ti-chip.down {{ border-color: {p['down']}; background: color-mix(in srgb, {p['down']} 8%, transparent); }}
.ti-chip.warn {{ border-color: {p['warn']}; background: {p['accent_soft']}; }}
.ti-mono {{ font-family: {_BODY}; font-variant-numeric: tabular-nums; }}
/* tables: themed header and row rules */
.ti-card table {{ display: block; overflow-x: auto; max-width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 4px 0 8px 0; }}
[data-testid="stMarkdownContainer"] table {{ display: block; overflow-x: auto; max-width: 100%; }}
.ti-card th, .ti-card td {{ padding: 6px 12px 6px 0; border-bottom: 1px solid {p['grid']}; text-align: left; color: {p['text']}; }}
.ti-card th {{ {label} color: {p['muted']}; border-bottom-color: {p['border']}; }}
.ti-card table, .ti-card th, .ti-card td {{ border-top: 0 !important; border-left: 0 !important; border-right: 0 !important; background: transparent; }}
/* labels and kickers */
.ti-label {{ display: block; {label} color: {p['muted']}; margin: 10px 0 2px 0; }}
.ti-kicker {{ {label} color: {p['accent_deep']}; }}
/* top bar: the price line every screen is read against */
.ti-top {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; padding: 4px 0 14px 0; margin-bottom: 18px; border-bottom: 1px solid {p['border']}; }}
.ti-top .ti-label {{ margin: 0; }}
.ti-top-price {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.875rem; line-height: 1; }}
.ti-top-chg {{ font-size: 0.9rem; }} .ti-top-chg.up {{ color: {p['up']}; }} .ti-top-chg.down {{ color: {p['down']}; }}
.ti-top-upd {{ font-size: 0.78rem; color: {p['muted']}; }}
.ti-gamma {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; padding: 8px 12px; margin: -8px 0 18px 0; border: 1px solid {p['border']}; border-left-width: 4px; font-size: 0.85rem; }}
.ti-gamma.up {{ border-left-color: {p['up']}; }} .ti-gamma.down {{ border-left-color: {p['down']}; }} .ti-gamma.neutral {{ border-left-color: {p['muted']}; }}
.ti-gamma-mark {{ display: inline-block; width: 10px; height: 10px; align-self: center; background: {p['muted']}; }}
.ti-gamma.up .ti-gamma-mark {{ background: {p['up']}; }} .ti-gamma.down .ti-gamma-mark {{ background: {p['down']}; }}
.ti-gamma-reg {{ font-family: {_HEADING}; font-weight: 600; font-size: 1rem; letter-spacing: 0.05em; }}
.ti-gamma.up .ti-gamma-reg {{ color: {p['up']}; }} .ti-gamma.down .ti-gamma-reg {{ color: {p['down']}; }}
.ti-gamma-net {{ color: {p['muted']}; }}
.ti-gamma-scope {{ color: {p['muted']}; font-size: 0.78rem; }}
.ti-gamma-lv {{ display: flex; flex-wrap: wrap; gap: 4px 14px; }} .ti-gamma-lv b {{ font-weight: 600; color: {p['muted']}; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
.ti-gamma-play {{ flex-basis: 100%; color: {p['muted']}; font-size: 0.8rem; }}
/* hero: the desk title and its candle countdown */
.ti-hero {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-end; justify-content: space-between; margin: 6px 0 26px 0; }}
.ti-hero-main {{ display: flex; flex-direction: column; gap: 10px; flex: 1 1 380px; min-width: 0; }}
.ti-hero h1 {{ font-family: {_HEADING}; font-weight: 600; font-size: clamp(2.5rem, 6vw, 4rem); line-height: 0.95; text-transform: uppercase; letter-spacing: 0.01em; margin: 0; padding: 0; }}
.ti-hero-meta {{ display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: 0.88rem; color: {p['muted']}; }}
.blueprint {{ position: relative; border: 1px solid {p['border']}; }}
.blueprint > .corner {{ position: absolute; width: 11px; height: 11px; color: color-mix(in srgb, {p['text']} 55%, transparent); }}
.blueprint > .corner::before, .blueprint > .corner::after {{ content: ""; position: absolute; background: currentColor; }}
.blueprint > .corner::before {{ left: 5px; top: 0; width: 1px; height: 100%; }}
.blueprint > .corner::after {{ top: 5px; left: 0; width: 100%; height: 1px; }}
.blueprint > .corner.tl {{ top: -6px; left: -6px; }} .blueprint > .corner.tr {{ top: -6px; right: -6px; }}
.blueprint > .corner.bl {{ bottom: -6px; left: -6px; }} .blueprint > .corner.br {{ bottom: -6px; right: -6px; }}
.ti-countdown {{ flex: 0 1 340px; padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; }}
.ti-countdown-top {{ display: flex; justify-content: space-between; gap: 12px; {label} color: {p['muted']}; }}
.ti-countdown-left {{ font-family: {_HEADING}; font-weight: 600; font-size: 2.1rem; line-height: 1; }}
.ti-countdown-at {{ font-size: 0.78rem; color: {p['muted']}; }}
.ti-bar {{ height: 4px; background: {p['grid']}; margin: 4px 0 8px 0; }}
.ti-bar > div {{ height: 100%; background: {p['accent']}; }}
.ti-bar.up > div {{ background: {p['up']}; }} .ti-bar.down > div {{ background: {p['down']}; }}
/* the anchored direction */
.ti-dir {{ font-family: {_HEADING}; font-weight: 600; font-size: 2.9rem; line-height: 1; text-transform: uppercase; margin: 2px 0 6px 0; }}
.ti-dir.up {{ color: {p['up']}; }} .ti-dir.down {{ color: {p['down']}; }}
.ti-pair {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid {p['border']}; border-bottom: 1px solid {p['border']}; margin: 10px 0 4px 0; }}
.ti-pair > div {{ padding: 8px 0; }} .ti-pair > div + div {{ padding-left: 14px; border-left: 1px solid {p['border']}; }}
.ti-pair .ti-label {{ margin: 0; }}
.ti-pair-v {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.5rem; }} .ti-pair-v.down {{ color: {p['down']}; }}
/* design primitives: dash lists, labelled figures, bars, sections */
.ti-dash {{ margin: 2px 0 !important; padding-left: 14px; text-indent: -14px; }}
.ti-kv {{ display: flex; flex-direction: column; gap: 1px; padding: 8px 0; min-width: 0; }}
.ti-kv .ti-label {{ margin: 0; }}
.ti-kv-v {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.2rem; line-height: 1.2; }}
.ti-kv-v.big {{ font-size: 1.9rem; }} .ti-kv-v.up {{ color: {p['up']}; }} .ti-kv-v.down {{ color: {p['down']}; }}
.ti-grid2, .ti-grid3, .ti-grid4 {{ display: grid; gap: 0 14px; border-top: 1px solid {p['border']}; margin: 6px 0; }}
.ti-grid2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .ti-grid3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.ti-grid4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.stApp h2.ti-h2, .ti-h2 {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.875rem !important; line-height: 2rem !important; text-transform: uppercase; letter-spacing: 0.01em; margin: 4px 0 14px 0; padding: 0; }}
.ti-h3 {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.35rem; line-height: 1.15; text-transform: uppercase; letter-spacing: 0.03em; margin: 2px 0 6px 0; }}
.ti-h3.up {{ color: {p['up']}; }} .ti-h3.down {{ color: {p['down']}; }}
.ti-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 6px 0; }} .ti-tags .ti-chip {{ margin: 0; }}
.ti-chip.accent {{ background: {p['accent_soft']}; border-color: transparent; color: {p['accent_ink']}; }}
.ti-chip.outline {{ border-color: {p['accent']}; color: {p['accent']}; }}
.ti-rows {{ display: flex; flex-direction: column; border-top: 1px solid {p['border']}; margin: 6px 0; }}
.ti-rows > * {{ padding: 6px 0; border-bottom: 1px solid {p['grid']}; }}
.ti-row3 {{ display: grid; grid-template-columns: 34px 1fr auto; gap: 8px; font-size: 0.86rem; }}
.ti-split {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.ti-foot {{ border-top: 1px solid {p['border']}; margin-top: 10px; }}
.ti-note {{ font-size: 0.8rem; margin-top: 8px !important; }}
.ti-wire-head {{ font-family: {_HEADING}; font-size: 1.35rem; letter-spacing: 0.03em; text-transform: uppercase; margin: 0 0 4px 0 !important; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
.ti-wire-line {{ font-size: 1.125rem !important; line-height: 1.6rem !important; margin: 0 0 10px 0 !important; max-width: 820px; }}
.ti-section {{ display: flex; align-items: center; gap: 14px; margin: 34px 0 4px 0; }}
.ti-rule {{ flex: 1; height: 1px; background: {p['border']}; }}
.ti-head-summary {{ display: flex; flex-direction: column; gap: 8px; max-width: 900px; margin: 30px 0 10px 0; }}
.ti-head-summary p {{ font-size: 1.06rem; line-height: 1.65; margin: 0; }}
.ti-head-summary .muted {{ color: {p['muted']}; font-size: 0.8rem; }}
.ti-bar.faint > div {{ background: {p['fib']}; }} .ti-bar.warn > div {{ background: {p['warn']}; }}
.ti-bar.tick60 {{ position: relative; height: 8px; }}
.ti-bar.tick60::after {{ content: ""; position: absolute; left: 60%; top: -3px; bottom: -3px; width: 1px; background: {p['text']}; }}
.ti-mx-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr)); gap: 22px; margin: 10px 0; }}
.ti-mx {{ padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; font-size: 0.86rem; }}
.ti-mx-row {{ border-bottom: 1px solid {p['grid']}; padding: 4px 0; }}
/* TRAP */
.ti-band {{ background: {p['accent_night']}; color: {p['bg']}; padding: 16px 20px; display: flex; flex-wrap: wrap; gap: 12px 28px; align-items: center; margin: 4px 0 18px 0; }}
.ti-band * {{ color: {p['bg']} !important; }}
.ti-band-title {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.ti-band-lines {{ flex: 1 1 360px; }} .ti-band-lines p {{ margin: 2px 0; font-size: 0.9rem; }}
.ti-band-lines strong {{ text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
.ti-band-note {{ opacity: 0.75; font-size: 0.8rem !important; }}
.ti-counts {{ display: flex; flex-wrap: wrap; gap: 10px 40px; margin: 10px 0; }}
.ti-count {{ display: flex; flex-direction: column; }} .ti-count .ti-label {{ margin: 0; }}
.ti-count .n {{ font-family: {_HEADING}; font-weight: 600; font-size: 3.25rem; line-height: 1; }}
.ti-range {{ margin: 8px 0; }} .ti-kv.right {{ text-align: right; }}
.ti-track {{ position: relative; height: 4px; background: {p['grid']}; margin: 4px 0 6px 0; }}
.ti-track-now {{ position: absolute; top: -5px; width: 2px; height: 14px; background: {p['text']}; }}
.ti-mini-doms {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 10px 0; font-size: 0.75rem; }}
.ti-mini-doms > div {{ display: flex; flex-direction: column; gap: 2px; min-width: 0; }}
.ti-mini-doms .muted {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ti-mini-doms .ti-bar {{ margin: 0; }}
.ti-watch {{ display: grid; grid-template-columns: 80px 44px 90px 1fr; gap: 10px; font-size: 0.86rem; align-items: baseline; }}
/* active trade */
.ti-trade-head {{ display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; margin-bottom: 8px; }}
.ti-trade-head .ti-chip {{ margin: 0; }} .ti-trade-head .ti-h3 {{ margin: 0; }}
.ti-ladder {{ position: relative; height: 92px; margin: 14px 8px 18px 8px; }}
.ti-ladder-track {{ position: absolute; left: 0; right: 0; top: 45px; height: 2px; background: {p['border']}; }}
.ti-ladder-seg {{ position: absolute; top: 42px; height: 8px; }}
.ti-ladder-seg.risk {{ background: color-mix(in srgb, {p['down']} 22%, transparent); }}
.ti-ladder-seg.reward {{ background: color-mix(in srgb, {p['up']} 22%, transparent); }}
.ti-ladder-mark {{ position: absolute; top: 0; bottom: 0; width: 0; }}
.ti-ladder-mark .tick {{ position: absolute; top: 38px; left: -1px; width: 2px; height: 16px; background: {p['text']}; }}
.ti-ladder-mark.down .tick {{ background: {p['down']}; }} .ti-ladder-mark.up .tick {{ background: {p['up']}; }}
.ti-ladder-mark.now .tick {{ background: {p['accent']}; height: 22px; top: 35px; }} .ti-ladder-mark.muted .tick {{ background: {p['muted']}; }}
.ti-ladder-mark .lab {{ position: absolute; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; font-size: 0.75rem; line-height: 1.15; white-space: nowrap; }}
.ti-ladder-mark .lab span:first-child {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {p['muted']}; }}
.ti-ladder-mark.top .lab {{ top: 0; }} .ti-ladder-mark.bottom .lab {{ top: 60px; }}
.ti-two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 18px 30px; margin: 10px 0; }}
.ti-three {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr)); gap: 18px 30px; margin: 14px 0 6px 0; border-top: 1px solid {p['border']}; padding-top: 6px; }}
.ti-verdict {{ font-family: {_HEADING}; font-weight: 600; font-size: 2.5rem; line-height: 1; text-transform: uppercase; margin: 4px 0; }}
.ti-fa {{ font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 8px 0 4px 0; }}
.ti-fa .up, .ti-row-rule .up {{ color: {p['up']}; }} .ti-fa .down, .ti-row-rule .down {{ color: {p['down']}; }}
.ti-row-rule .muted {{ color: {p['muted']}; }}
.ti-twin {{ display: grid; grid-template-columns: minmax(0, 1.4fr) 1fr auto; gap: 10px; align-items: center; font-size: 0.8rem; margin: 4px 0; }}
.ti-twin-bars {{ display: flex; flex-direction: column; gap: 2px; }}
.ti-twin-bars > div {{ height: 3px; background: {p['grid']}; }} .ti-twin-bars .f > div {{ height: 100%; background: {p['up']}; }}
.ti-twin-bars .a > div {{ height: 100%; background: {p['down']}; }}
.ti-targets {{ width: 100%; margin: 10px 0 !important; }}
.ti-handshake {{ border-left: 3px solid {p['warn']}; padding: 4px 0 4px 12px; margin: 10px 0; }}
.ti-row-rule {{ border-bottom: 1px solid {p['grid']}; padding: 6px 0; font-size: 0.86rem; }}
.ti-chan {{ grid-template-columns: 8px 110px 1fr; font-size: 0.86rem; }}
/* director */
.ti-chip.neutral-fill {{ background: {p['surface']}; border-color: transparent; color: {p['text']}; }}
.ti-scen {{ margin: 8px 0; }} .stApp .ti-scen p {{ font-size: 0.8125rem; margin: 0 !important; }} .ti-scen .ti-bar {{ margin: 3px 0; }}
.ti-desk-sum {{ border-top: 1px solid {p['border']}; padding-top: 8px; }}
.ti-brief {{ border-top: 1px solid {p['grid']}; padding: 10px 0; }} .ti-brief .ti-bar {{ margin: 4px 0; }}
.ti-agents {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 4px 14px; font-size: 0.8rem; }}
.ti-agents > div {{ display: flex; justify-content: space-between; gap: 6px; border-bottom: 1px solid {p['grid']}; padding: 2px 0; }}
/* macro and news */
.ti-big {{ font-family: {_HEADING}; font-weight: 600; font-size: 3rem; line-height: 2.875rem; margin: 4px 0; }}
.ti-news {{ display: grid; grid-template-columns: 64px 1fr; gap: 12px; font-size: 0.9rem; }}
.ti-news-title {{ font-weight: 500; }} .ti-news-title a {{ color: {p['text']}; text-decoration: none; }}
.ti-news-title a:hover {{ color: {p['accent_deep']}; text-decoration: underline; }}
.ti-news .ti-tags {{ margin: 4px 0 0 0; align-items: center; font-size: 0.78rem; }}
.ti-odds {{ display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(120px, 1fr) minmax(90px, 0.8fr); gap: 14px; align-items: center; font-size: 0.88rem; }}
.ti-odds-p {{ display: grid; grid-template-columns: 1fr 52px; gap: 8px; align-items: center; }} .ti-odds-p .ti-bar {{ margin: 0; }}
.ti-cal {{ display: flex; flex-direction: column; gap: 1px; font-size: 0.88rem; }} .ti-cal .ti-chip {{ margin: 0; }}
@media (max-width: 768px) {{ .ti-odds {{ grid-template-columns: 1fr; gap: 4px; }} }}
/* sizes the design sets explicitly */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li {{ font-size: 0.9375rem; line-height: 1.5; }}
.stApp .stButton button p, .stApp [data-testid="stFormSubmitButton"] button p, .stApp [data-testid="stDownloadButton"] button p {{ font-family: {_HEADING}; font-weight: 600; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.ti-dir.xl {{ font-size: 3.25rem; line-height: 3rem; }}
.ti-verdict.xl {{ font-size: 3.25rem; line-height: 3rem; }}
.ti-h3.lg {{ font-size: 1.5rem; }}
.ti-h4 {{ display: block; font-family: {_HEADING}; font-weight: 600; font-size: 1.25rem; text-transform: uppercase; letter-spacing: 0.03em; margin: 2px 0 6px 0; }}
.ti-card h4.big {{ font-family: {_HEADING}; font-size: 1.25rem; letter-spacing: 0.03em; color: {p['text']}; }}
.ti-grid4.live .ti-kv-v {{ font-size: 1.75rem; }} .ti-grid4.stress .ti-kv-v {{ font-size: 1.5rem; }} .ti-grid2.acc .ti-kv-v {{ font-size: 1.375rem; }}
.ti-trade-head .ti-h3 {{ font-size: 1.625rem; line-height: 1.75rem; }}
.ti-stopv {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.625rem; line-height: 1.75rem; text-transform: uppercase; margin: 4px 0; }}
.ti-stopv.up {{ color: {p['up']}; }} .ti-stopv.down {{ color: {p['down']}; }}
.ti-mx .ti-kicker {{ font-family: {_HEADING}; font-size: 1.06rem; letter-spacing: 0.03em; }}
.ti-team-row {{ grid-template-columns: 70px 1fr 36px !important; }}
.ti-team-row .n {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.375rem; text-align: right; }}
.ti-read {{ display: flex; flex-direction: column; gap: 10px; max-width: 900px; margin: 18px 0 26px 0; }}
.ti-read p {{ font-size: 1.125rem !important; line-height: 1.8rem !important; margin: 0; }}
.ti-hero-p {{ font-size: 0.9375rem !important; line-height: 1.44rem !important; color: {p['muted']}; max-width: 760px; margin: 0; }}
.stApp [data-testid="stMarkdownContainer"] .ti-card p.ti-sum-p, .stApp [data-testid="stMarkdownContainer"] .ti-desk-sum p {{ font-size: 1rem; line-height: 1.56rem; }}
.ti-early {{ border-color: {p['accent']} !important; box-shadow: none !important; padding: 16px 20px; }}
.ti-early-row {{ display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: baseline; }}
.ti-early-title {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.125rem; text-transform: uppercase; letter-spacing: 0.06em; color: {p['accent_ink']}; }}
.ti-early-drivers {{ display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: 0.875rem; margin-top: 6px; }}
.ti-triggers summary {{ cursor: pointer; color: {p['accent']}; font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.875rem; margin-top: 10px; }}
.ti-triggers .ti-tags {{ margin-top: 8px; }}
.st-key-ti_chart {{ position: relative; border: 1px solid {p['border']}; padding: 16px 16px 12px 16px; }}
.st-key-ti_chart::before, .st-key-ti_chart::after {{ content: "+"; position: absolute; font-size: 15px; line-height: 1; color: color-mix(in srgb, {p['text']} 55%, transparent); }}
.st-key-ti_chart::before {{ top: -8px; left: -5px; }} .st-key-ti_chart::after {{ bottom: -8px; right: -5px; }}
.ti-chart-head {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 16px; align-items: center; }}
.ti-chart-title {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.125rem; text-transform: uppercase; letter-spacing: 0.05em; }}
.ti-legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 0.75rem; color: {p['muted']}; }}
.ti-legend span {{ display: flex; align-items: center; gap: 6px; }}
.ti-legend .sw {{ display: inline-block; width: 14px; height: 2px; }}
.ti-legend .sw.ema21 {{ background: #416180; }} .ti-legend .sw.ema50 {{ background: #94BCE3; }}
.ti-legend .sw.up {{ width: 8px; height: 10px; border: 1px solid {p['up']}; }} .ti-legend .sw.down {{ width: 8px; height: 10px; background: {p['down']}; }}
.ti-legend .sw.env {{ width: 8px; height: 10px; border: 1px dashed {p['accent']}; }}
.ti-chart-cap {{ font-size: 0.75rem; line-height: 1.125rem; color: {p['muted']}; margin: 0; }}
/* dialogs: square, hairline, the design's condensed title */
[data-testid="stDialog"] [role="dialog"] {{ border-radius: 0 !important; border: 1px solid {p['border']}; }}
[data-testid="stDialog"] [role="dialog"] h2 {{ font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; font-size: 1.25rem !important; }}
.ti-cascade td.up {{ color: {p['up']}; font-weight: 600; }} .ti-cascade td.down {{ color: {p['down']}; font-weight: 600; }}
.ti-cascade td.muted {{ color: {p['muted']}; }}
/* war room */
.ti-card-meta {{ position: absolute; top: 16px; right: 18px; font-size: 0.78rem; color: {p['muted']}; }}
.ti-team-pts {{ font-family: {_HEADING}; font-weight: 600; font-size: 3rem; line-height: 1; margin: 6px 0; }}
.ti-team-pts span {{ font-size: 1rem; color: {p['muted']}; }} .ti-team-pts.up {{ color: {p['up']}; }} .ti-team-pts.down {{ color: {p['down']}; }}
.ti-dom {{ margin: 6px 0; font-size: 0.9375rem; line-height: 1.3rem; }} .ti-dom .ti-bar {{ margin: 3px 0 0 0; }}
.ti-prob {{ display: grid; grid-template-columns: 110px 1fr 44px; gap: 10px; align-items: center; font-size: 0.88rem; margin: 6px 0; }}
.ti-prob .ti-bar {{ margin: 0; }} .ti-prob .v {{ text-align: right; }}
.ti-sum {{ display: flex; flex-wrap: wrap; gap: 16px 40px; align-items: center; margin: 8px 0; }}
.ti-sum > div:first-child {{ flex: 1 1 300px; }}
.ti-teams {{ flex: 1 1 320px; display: flex; flex-direction: column; gap: 8px; font-size: 0.86rem; }}
.ti-team-row {{ display: grid; grid-template-columns: 130px 1fr; gap: 10px; align-items: center; }} .ti-team-row .ti-bar {{ margin: 0; }}
.ti-tag-solid {{ background: {p['accent_night']}; color: {p['bg']}; font-size: 0.75rem; padding: 2px 9px; letter-spacing: 0.04em; }}
.ti-note-item {{ border-top: 1px solid {p['grid']}; padding: 8px 0; }} .ti-note-item:first-of-type {{ border-top: 0; padding-top: 0; }}
/* feeds: square marks, filled when the feed answered, hollow when it did not */
.ti-feeds {{ display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }}
.ti-feed {{ display: grid; grid-template-columns: 8px 82px 1fr; gap: 8px; align-items: center; font-size: 0.78rem; line-height: 1.2; }}
.ti-feed .dot {{ width: 7px; height: 7px; background: {p['accent']}; border: 1px solid {p['accent']}; }}
.ti-feed.off .dot {{ background: transparent; border-color: {p['down']}; }}
.ti-feed.off {{ color: {p['down']}; }}
.ti-feed .src {{ color: {p['muted']}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {{ width: 100%; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {{ flex: 1; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{ display: flex; justify-content: space-between; align-items: baseline; }}
/* the rail: desks then engines, the open screen marked by a steel bar */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] {{ position: relative; padding: 7px 10px; border-left: 2px solid transparent; margin: 0; width: 100%; cursor: pointer; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div > div:first-child:not([data-testid]) {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] {{ border-left-color: {p['accent']}; background: color-mix(in srgb, {p['accent']} 12%, transparent); }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {{ background: color-mix(in srgb, {p['text']} 6%, transparent); }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.03em; margin: 0; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] [data-testid="stCaptionContainer"] p {{ font-family: {_BODY}; font-size: 0.78rem; font-weight: 400; text-transform: none; letter-spacing: 0; color: {p['muted']}; margin: 0; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > :nth-child(5) {{ margin-top: 34px; position: relative; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > :nth-child(5)::before {{ content: "Engines"; position: absolute; left: 10px; top: -26px; {label} color: {p['accent_deep']}; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label p {{ {label} color: {p['accent_deep']}; }}
/* segmented switches: square, the open option filled steel */
[data-testid="stButtonGroup"] button {{ border-radius: 0 !important; font-family: {_BODY}; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background: {p['accent']} !important; border-color: {p['accent']} !important; color: {p['bg']} !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {{ color: {p['bg']} !important; }}
/* the phone screen strip: hidden where the rail is visible */
@media (min-width: 769px) {{ .st-key-ti_mobile_nav, .st-key-ti_mobile_foot {{ display: none; }} }}
.st-key-ti_mobile_foot {{ border-top: 1px solid {p['border']}; margin-top: 40px; padding-top: 16px; }}
.st-key-ti_mobile_nav [data-testid="stButtonGroup"] {{ flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; border-top: 1px solid {p['border']}; border-bottom: 1px solid {p['border']}; }}
.st-key-ti_mobile_nav button {{ border-radius: 0 !important; border: 0 !important; border-bottom: 2px solid transparent !important; background: transparent !important; min-height: 44px; white-space: nowrap; }}
.st-key-ti_mobile_nav button p {{ font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
.st-key-ti_mobile_nav button[aria-checked="true"] {{ border-bottom-color: {p['accent']} !important; color: {p['text']} !important; background: transparent !important; }}
.st-key-ti_mobile_nav button[aria-checked="true"] p {{ color: {p['text']} !important; }}
@media (max-width: 768px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .ti-kpis {{ grid-template-columns: repeat(2, 1fr); }}
  .ti-kpi .v {{ font-size: 1.3rem; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 10px; white-space: nowrap; }}
  .ti-card {{ padding: 14px; }}
  .ti-card p, .ti-card li {{ font-size: 0.95rem; }}
  .stButton button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{ min-height: 44px; }}
  [data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button {{ min-width: 44px; min-height: 44px; }}
}}
</style>
"""
