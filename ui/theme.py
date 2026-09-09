"""Theme injection and small HTML building blocks. Institutional palette, no emoji icons."""
from __future__ import annotations

import html

import streamlit as st

_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

LIGHT = dict(bg="#F6F8FB", panel="#FFFFFF", border="#E3E8EF", grid="#EEF2F7", text="#0B1220", muted="#5B6472",
             accent="#0F62FE", up="#0E9F6E", down="#E02424", warn="#D97706", fib="#94A3B8", fibFuture="#0F62FE",
             now="#0B1220", mono=_MONO)
DARK = dict(bg="#0B0F17", panel="#111827", border="#1F2937", grid="#161E2E", text="#E5E7EB", muted="#9CA3AF",
            accent="#3B82F6", up="#10B981", down="#EF4444", warn="#F59E0B", fib="#4B5563", fibFuture="#60A5FA",
            now="#E5E7EB", mono=_MONO)


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


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
    return f'<div class="ti-card {tone}"><h4>{html.escape(title)}</h4>{body_html}</div>'


def card(title: str, body_html: str, tone: str = "neutral") -> None:
    st.markdown(card_html(title, body_html, tone), unsafe_allow_html=True)


def inject_css(dark: bool) -> None:
    p = palette(dark)
    st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {p['bg']}; color: {p['text']}; }}
header[data-testid="stHeader"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1440px; }}
section[data-testid="stSidebar"] {{ background: {p['panel']}; border-right: 1px solid {p['border']}; }}
section[data-testid="stSidebar"] * {{ color: {p['text']}; }}
section[data-testid="stSidebar"] button {{ background: {p['panel']}; color: {p['text']}; border: 1px solid {p['border']}; }}
section[data-testid="stSidebar"] button:hover:not(:disabled) {{ border-color: {p['accent']}; color: {p['accent']}; }}
section[data-testid="stSidebar"] button:disabled, section[data-testid="stSidebar"] button:disabled * {{ color: {p['muted']}; opacity: 0.8; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {p['border']}; }}
.stTabs [data-baseweb="tab"] {{ padding: 9px 14px; font-weight: 600; color: {p['muted']}; }}
.stTabs [aria-selected="true"] {{ color: {p['accent']}; border-bottom: 2px solid {p['accent']}; }}
h1, h2, h3, h4, p, li, label, .stMarkdown {{ color: {p['text']}; }}
.ti-title {{ font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
.ti-sub {{ color: {p['muted']}; font-size: 0.84rem; margin: 2px 0 10px 0; }}
.ti-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 0 0 12px 0; }}
.ti-kpi {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 10px; padding: 10px 12px; }}
.ti-kpi .l {{ font-size: 0.7rem; color: {p['muted']}; text-transform: uppercase; letter-spacing: 0.05em; }}
.ti-kpi .v {{ font-family: {p['mono']}; font-size: 1.02rem; font-weight: 600; margin-top: 2px; color: {p['text']}; }}
.ti-kpi .d {{ font-size: 0.74rem; margin-top: 2px; color: {p['muted']}; }}
.ti-kpi .d.up, .ti-chip.up {{ color: {p['up']}; }}
.ti-kpi .d.down, .ti-chip.down {{ color: {p['down']}; }}
.ti-kpi .d.warn, .ti-chip.warn {{ color: {p['warn']}; }}
.ti-card {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; border-left: 4px solid {p['muted']}; }}
.ti-card h4 {{ margin: 0 0 8px 0; font-size: 0.92rem; font-weight: 600; }}
.ti-card p, .ti-card li {{ font-size: 0.88rem; line-height: 1.45; margin: 0.15rem 0; color: {p['text']}; }}
.ti-card ul {{ padding-left: 1.1rem; margin: 0.2rem 0; }}
.ti-card .muted {{ color: {p['muted']}; }}
.ti-card.up {{ border-left-color: {p['up']}; }}
.ti-card.down {{ border-left-color: {p['down']}; }}
.ti-card.warn {{ border-left-color: {p['warn']}; }}
.ti-chip {{ display: inline-block; font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; border: 1px solid {p['border']}; color: {p['muted']}; margin: 0 6px 6px 0; }}
.ti-chip.up {{ border-color: {p['up']}; }} .ti-chip.down {{ border-color: {p['down']}; }} .ti-chip.warn {{ border-color: {p['warn']}; }}
.ti-mono {{ font-family: {p['mono']}; }}
.ti-card table {{ border-collapse: collapse; font-size: 0.84rem; margin: 4px 0 8px 0; }}
.ti-card th, .ti-card td {{ padding: 3px 10px 3px 0; border-bottom: 1px solid {p['border']}; text-align: left; color: {p['text']}; }}
.ti-card th {{ color: {p['muted']}; font-weight: 600; }}
@media (max-width: 768px) {{ .block-container {{ padding-left: 0.6rem; padding-right: 0.6rem; }} .ti-kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
""", unsafe_allow_html=True)
