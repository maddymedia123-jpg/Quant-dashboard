"""LLM settings from environment first, then Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULTS = {
    "gemini": {"specialist": "gemini-3.5-flash-lite", "director": "gemini-3.8-flash"},
    "openrouter": {"specialist": "nvidia/nemotron-3.5-lightning:free", "director": "nvidia/nemotron-3.5-lightning:free"},
}


@dataclass
class LLMSettings:
    provider: str
    api_key: str
    specialist_model: str
    director_model: str
    max_concurrency: int = 4
    timeout_s: float = 90.0


def _secret(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - not running under streamlit or no secrets file
        return None


def load_settings() -> LLMSettings | None:
    gem, orr = _secret("GEMINI_API_KEY"), _secret("OPENROUTER_API_KEY")
    provider = (_secret("LLM_PROVIDER") or ("gemini" if gem else "openrouter" if orr else "")).lower()
    key = gem if provider == "gemini" else orr if provider == "openrouter" else None
    if not key:
        return None
    d = DEFAULTS[provider]
    return LLMSettings(
        provider=provider, api_key=key,
        specialist_model=_secret("SPECIALIST_MODEL") or d["specialist"],
        director_model=_secret("DIRECTOR_MODEL") or d["director"],
        max_concurrency=int(_secret("LLM_MAX_CONCURRENCY") or 4),
    )
