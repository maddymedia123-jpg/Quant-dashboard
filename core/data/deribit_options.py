"""Deribit public options book -> max pain, put/call ratios, ATM IV, IV skew."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import httpx

from core.config import SYMBOLS
from core.data.types import OptionsSnapshot

log = logging.getLogger(__name__)
BASE = "https://www.deribit.com/api/v2/public"


def parse_instrument(name: str) -> tuple[str, float, str]:
    # BTC-25SEP26-105000-C
    _, exp, strike, kind = name.split("-")
    expiry = datetime.strptime(exp, "%d%b%y").date().isoformat()
    return expiry, float(strike), kind


def compute_max_pain(rows: list[dict]) -> float:
    strikes = sorted({r["strike"] for r in rows})
    best, best_pain = strikes[0], float("inf")
    for k in strikes:
        pain = 0.0
        for r in rows:
            if r["kind"] == "C":
                pain += r["oi"] * max(0.0, k - r["strike"])
            else:
                pain += r["oi"] * max(0.0, r["strike"] - k)
        if pain < best_pain:
            best, best_pain = k, pain
    return best


def parse_book_summary(payload: dict) -> OptionsSnapshot:
    items = payload["result"]
    by_exp: dict[str, list[dict]] = defaultdict(list)
    underlying = None
    call_oi = put_oi = call_vol = put_vol = 0.0
    for it in items:
        expiry, strike, kind = parse_instrument(it["instrument_name"])
        oi = float(it.get("open_interest") or 0.0)
        row = {"strike": strike, "kind": kind, "oi": oi, "iv": it.get("mark_iv"), "vol_usd": float(it.get("volume_usd") or 0.0)}
        by_exp[expiry].append(row)
        if underlying is None and it.get("underlying_price"):
            underlying = float(it["underlying_price"])
        if kind == "C":
            call_oi += oi
            call_vol += row["vol_usd"]
        else:
            put_oi += oi
            put_vol += row["vol_usd"]

    expiries = []
    for expiry, rows in sorted(by_exp.items()):
        c = sum(r["oi"] for r in rows if r["kind"] == "C")
        p = sum(r["oi"] for r in rows if r["kind"] == "P")
        if c + p < 50:  # ignore illiquid expiries
            continue
        expiries.append({"expiry": expiry, "max_pain": compute_max_pain(rows), "call_oi": c, "put_oi": p})

    # primary expiry = largest total OI (usually the quarterly)
    primary = max(expiries, key=lambda e: e["call_oi"] + e["put_oi"]) if expiries else None
    iv_atm = iv_skew = None
    if primary and underlying:
        rows = by_exp[primary["expiry"]]
        with_iv = [r for r in rows if r["iv"]]
        if with_iv:
            atm = min(with_iv, key=lambda r: abs(r["strike"] - underlying))
            atm_rows = [r for r in with_iv if r["strike"] == atm["strike"]]
            iv_atm = sum(r["iv"] for r in atm_rows) / len(atm_rows)
            otm_puts = [r["iv"] for r in with_iv if r["kind"] == "P" and 0.85 * underlying <= r["strike"] <= 0.95 * underlying]
            otm_calls = [r["iv"] for r in with_iv if r["kind"] == "C" and 1.05 * underlying <= r["strike"] <= 1.15 * underlying]
            if otm_puts and otm_calls:
                iv_skew = sum(otm_puts) / len(otm_puts) - sum(otm_calls) / len(otm_calls)

    return OptionsSnapshot(
        source="deribit",
        underlying_price=underlying,
        max_pain=primary["max_pain"] if primary else None,
        max_pain_expiry=primary["expiry"] if primary else None,
        put_call_oi_ratio=(put_oi / call_oi) if call_oi else None,
        put_call_volume_ratio=(put_vol / call_vol) if call_vol else None,
        iv_atm=iv_atm,
        iv_skew=iv_skew,
        total_call_oi=call_oi,
        total_put_oi=put_oi,
        expiries=expiries,
    )


async def fetch_options(client: httpx.AsyncClient) -> OptionsSnapshot:
    try:
        r = await client.get(f"{BASE}/get_book_summary_by_currency", params={"currency": SYMBOLS["deribit"], "kind": "option"})
        r.raise_for_status()
        return parse_book_summary(r.json())
    except Exception as e:  # noqa: BLE001 - provider boundary
        log.error("deribit unavailable: %s", e)
        return OptionsSnapshot.unavailable("deribit", str(e))
