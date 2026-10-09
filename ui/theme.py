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
.ti-kpi {{ padding: 12px 14px; border-right: 1px solid {p['border']}; border-bottom: 1px solid {p['border']}; min-width: 0; }}
.ti-kpi .l {{ {label} color: {p['muted']}; }}
.ti-kpi .v {{ font-family: {_HEADING}; font-size: 1.55rem; line-height: 1.2; font-weight: 600; margin-top: 2px; color: {p['text']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
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
.ti-card p, .ti-card li {{ font-size: 0.92rem; line-height: 1.5; margin: 0.2rem 0; color: {p['text']}; }}
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
.ti-top-price {{ font-family: {_HEADING}; font-weight: 600; font-size: 2rem; line-height: 1; }}
.ti-top-chg {{ font-size: 0.9rem; }} .ti-top-chg.up {{ color: {p['up']}; }} .ti-top-chg.down {{ color: {p['down']}; }}
.ti-top-upd {{ font-size: 0.78rem; color: {p['muted']}; }}
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
/* feeds: square marks, filled when the feed answered, hollow when it did not */
.ti-feeds {{ display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }}
.ti-feed {{ display: grid; grid-template-columns: 8px 82px 1fr; gap: 8px; align-items: center; font-size: 0.78rem; line-height: 1.2; }}
.ti-feed .dot {{ width: 7px; height: 7px; background: {p['accent']}; border: 1px solid {p['accent']}; }}
.ti-feed.off .dot {{ background: transparent; border-color: {p['down']}; }}
.ti-feed.off {{ color: {p['down']}; }}
.ti-feed .src {{ color: {p['muted']}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
/* the rail: desks then engines, the open screen marked by a steel bar */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] {{ position: relative; padding: 7px 10px; border-left: 2px solid transparent; margin: 0; width: 100%; cursor: pointer; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div > div:first-child:not([data-testid]) {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] {{ border-left-color: {p['accent']}; background: color-mix(in srgb, {p['accent']} 12%, transparent); }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {{ background: color-mix(in srgb, {p['text']} 6%, transparent); }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{ font-family: {_HEADING}; font-weight: 600; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.03em; margin: 0; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"] [data-testid="stCaptionContainer"] p {{ font-family: {_BODY}; font-size: 0.78rem; font-weight: 400; text-transform: none; letter-spacing: 0; color: {p['muted']}; margin: 0; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-of-type(5) {{ margin-top: 34px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-of-type(5)::before {{ content: "Engines"; position: absolute; left: 10px; top: -26px; {label} color: {p['accent_deep']}; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label p {{ {label} color: {p['accent_deep']}; }}
/* the phone screen strip: hidden where the rail is visible */
@media (min-width: 769px) {{ .st-key-ti_mobile_nav {{ display: none; }} }}
.st-key-ti_mobile_nav [data-testid="stButtonGroup"] {{ flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; border-top: 1px solid {p['border']}; border-bottom: 1px solid {p['border']}; }}
.st-key-ti_mobile_nav button {{ border-radius: 0 !important; border: 0 !important; border-bottom: 2px solid transparent !important; background: transparent !important; min-height: 44px; white-space: nowrap; }}
.st-key-ti_mobile_nav button p {{ font-family: {_HEADING}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
.st-key-ti_mobile_nav button[aria-checked="true"] {{ border-bottom-color: {p['accent']} !important; color: {p['text']} !important; }}
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
