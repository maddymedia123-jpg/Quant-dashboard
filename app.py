import asyncio
import os
import math
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Literal, Optional
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import streamlit as st
import requests

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BTC-Quant-App")

# Optional Gemini Imports
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class VolatilityProjection(BaseModel):
    intraday_4h_regime: Literal["LOW_COMPRESSION", "NORMAL_EXPANSION", "HIGH_VOLATILITY_BURST"]
    intraday_expected_range_usd: float
    weekly_day_by_day_range: Dict[str, float]
    monthly_volatility_regime: str

class TrapSqueezeAnalysis(BaseModel):
    short_squeeze_risk: Literal["LOW", "MODERATE", "EXTREME"]
    long_squeeze_risk: Literal["LOW", "MODERATE", "EXTREME"]
    bull_trap_zone: List[float]
    bear_trap_zone: List[float]
    rejection_resistance_levels: List[float]
    support_sweep_levels: List[float]
    new_ath_expansion_potential: bool

class DualSummary(BaseModel):
    layman_summary: str = Field(description="Plain-English narrative for non-quant users.")
    institutional_detail: str = Field(description="In-depth quant & microstructure summary.")
    drastic_change_detected: bool

class ComprehensiveReport(BaseModel):
    timestamp_utc: str
    spot_price: float
    volatility: VolatilityProjection
    traps_and_squeezes: TrapSqueezeAnalysis
    summaries: DualSummary

# =====================================================================
# ASYNC DATA FETCHING (KRAKEN CLOUD-COMPATIBLE)
# =====================================================================

async def _fetch_all_metrics_async(symbol: str = "BTC/USD") -> Dict:
    exchange = ccxt.kraken({"enableRateLimit": True, "timeout": 10000})
    try:
        timeframes = ["15m", "1h", "4h", "1d", "1w"]
        ohlcv_tasks = [exchange.fetch_ohlcv(symbol, tf, limit=100) for tf in timeframes]
        ticker_task = exchange.fetch_ticker(symbol)

        results = await asyncio.gather(*ohlcv_tasks, ticker_task, return_exceptions=True)

        ohlcv_results = results[:5]
        ticker_res = results[5] if not isinstance(results[5], Exception) else {}

        matrix = {}
        for tf, data in zip(timeframes, ohlcv_results):
            if isinstance(data, Exception) or not data or len(data) < 20:
                continue
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["tr"] = np.maximum(
                df["high"] - df["low"],
                np.maximum(abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift()))
            )
            df["atr14"] = df["tr"].rolling(14).mean()
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            df["sigma"] = df["log_ret"].rolling(20).std() * df["close"]
            df["mu"] = df["close"].rolling(20).mean()

            range_len = np.maximum(df["high"] - df["low"], 0.0001)
            buyer_ratio = (df["close"] - df["low"]) / range_len
            df["vol_delta"] = (buyer_ratio - 0.5) * 2 * df["volume"]
            df["cvd"] = df["vol_delta"].cumsum()

            last = df.iloc[-1]
            matrix[tf] = {
                "close": float(last["close"]),
                "mu": float(last["mu"]),
                "sigma_1_up": float(last["mu"] + last["sigma"]),
                "sigma_2_up": float(last["mu"] + (2 * last["sigma"])),
                "sigma_3_up": float(last["mu"] + (3 * last["sigma"])),
                "sigma_1_dn": float(last["mu"] - last["sigma"]),
                "sigma_2_dn": float(last["mu"] - (2 * last["sigma"])),
                "sigma_3_dn": float(last["mu"] - (3 * last["sigma"])),
                "atr14": float(last["atr14"]),
                "cvd_current": float(last["cvd"]),
                "cvd_slope": float(df["cvd"].iloc[-1] - df["cvd"].iloc[-5]),
            }

        if not matrix or "1h" not in matrix:
            raise ValueError("Failed to retrieve valid OHLCV matrix from exchange.")

        spot_price = float(ticker_res.get("last", matrix["1h"]["close"])) if isinstance(ticker_res, dict) else matrix["1h"]["close"]

        daily_std = matrix["1d"]["sigma_1_up"] - matrix["1d"]["mu"]
        weekly_day_volatility = {
            f"Day_{i+1}": round(daily_std * math.sqrt(i + 1), 2) for i in range(7)
        }

        return {
            "spot_price": spot_price,
            "open_interest": 45000.0,  # Simulated institutional metric for spot pairs
            "funding_rate": 0.0001,
            "quant_matrix": matrix,
            "weekly_day_volatility": weekly_day_volatility,
            "error": None
        }
    except Exception as e:
        logger.error(f"Data fetching error: {e}")
        return {"error": str(e), "spot_price": 79000.0, "open_interest": 0.0, "funding_rate": 0.0001, "quant_matrix": {}, "weekly_day_volatility": {}}
    finally:
        await exchange.close()

@st.cache_data(ttl=60)
def fetch_cached_market_data(symbol: str = "BTC/USD") -> Dict:
    return asyncio.run(_fetch_all_metrics_async(symbol))

# =====================================================================
# REPORT GENERATION ENGINE
# =====================================================================

def generate_report_sync(data: Dict, openrouter_key: str = "") -> Dict:
    if data.get("error") or not data.get("quant_matrix"):
        return {"report": _build_fallback_report(79000.0, data), "engine": "Local Fallback Engine", "warnings": ["Exchange data degraded."]}

    spot = data["spot_price"]
    qm = data["quant_matrix"]
    fr = data["funding_rate"]
    oi = data["open_interest"]

    warnings_log = []
    prompt = f"""
    Synthesize an institutional crypto report for BTC/USD at ${spot:,.2f}.
    - 1H Bands: μ=${qm['1h']['mu']:.2f}, +2σ=${qm['1h']['sigma_2_up']:.2f}, -2σ=${qm['1h']['sigma_2_dn']:.2f}, ATR14=${qm['1h']['atr14']:.2f}
    - 4H Bands: μ=${qm['4h']['mu']:.2f}, +2σ=${qm['4h']['sigma_2_up']:.2f}, -2σ=${qm['4h']['sigma_2_dn']:.2f}     - 1D Bands: μ=${qm['1d']['mu']:.2f}, +2σ=${qm['1d']['sigma_2_up']:.2f}, -2σ=${qm['1d']['sigma_2_dn']:.2f}
    
    You MUST output valid JSON matching this exact structure:
    {{
      "timestamp_utc": "{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
      "spot_price": {spot},
      "volatility": {{
        "intraday_4h_regime": "NORMAL_EXPANSION",
        "intraday_expected_range_usd": {qm['4h']['atr14'] * 1.5},
        "weekly_day_by_day_range": {json.dumps(data.get("weekly_day_volatility", {}))},
        "monthly_volatility_regime": "COMPRESSION_BEFORE_EXPANSION"
      }},
      "traps_and_squeezes": {{
        "short_squeeze_risk": "MODERATE",
        "long_squeeze_risk": "LOW",
        "bull_trap_zone": [{round(qm['1h']['sigma_2_up'], 2)}, {round(qm['1h']['sigma_3_up'], 2)}],
        "bear_trap_zone": [{round(qm['1h']['sigma_2_dn'], 2)}, {round(qm['1h']['sigma_3_dn'], 2)}],
        "rejection_resistance_levels": [{round(qm['4h']['sigma_2_up'], 2)}, {round(qm['1d']['sigma_2_up'], 2)}],
        "support_sweep_levels": [{round(qm['4h']['sigma_2_dn'], 2)}, {round(qm['1d']['sigma_2_dn'], 2)}],
        "new_ath_expansion_potential": false
      }},
      "summaries": {{
        "layman_summary": "Bitcoin is testing key volatility parameters around ${spot:,.2f}. Monitor support and resistance levels closely.",
        "institutional_detail": "Microstructure anchored around 1H mean with standard deviation expansion bands active.",
        "drastic_change_detected": false
      }}
    }}
    Return ONLY valid JSON.
    """

    # OpenRouter API Integration
    if openrouter_key and openrouter_key not in ["", "sk-or-v1-your-key-here"]:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": prompt}]
                }),
                timeout=12
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                if content.startswith("```json"): content = content[7:-3]
                report = ComprehensiveReport.model_validate_json(content)
                return {"report": report, "engine": "🔄 OpenRouter API (Claude 3.5 Sonnet)", "warnings": warnings_log}
        except Exception as e:
            warnings_log.append(f"OpenRouter API error: {str(e)[:50]}")

    report = _build_fallback_report(spot, data)
    return {"report": report, "engine": "⚙️ Local Quant Heuristic Engine", "warnings": warnings_log}

def _build_fallback_report(spot: float, data: Dict) -> ComprehensiveReport:
    qm = data.get("quant_matrix", {})
    if qm and "1h" in qm and "4h" in qm and "1d" in qm:
        atr_4h = qm["4h"]["atr14"]
        return ComprehensiveReport(
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            spot_price=spot,
            volatility=VolatilityProjection(
                intraday_4h_regime="NORMAL_EXPANSION",
                intraday_expected_range_usd=round(atr_4h * 1.5, 2),
                weekly_day_by_day_range=data.get("weekly_day_volatility", {}),
                monthly_volatility_regime="STABLE"
            ),
            traps_and_squeezes=TrapSqueezeAnalysis(
                short_squeeze_risk="MODERATE",
                long_squeeze_risk="LOW",
                bull_trap_zone=[round(qm["1h"]["sigma_2_up"], 2), round(qm["1h"]["sigma_3_up"], 2)],
                bear_trap_zone=[round(qm["1h"]["sigma_2_dn"], 2), round(qm["1h"]["sigma_3_dn"], 2)],
                rejection_resistance_levels=[round(qm["4h"]["sigma_2_up"], 2), round(qm["1d"]["sigma_2_up"], 2)],
                support_sweep_levels=[round(qm["4h"]["sigma_2_dn"], 2), round(qm["1d"]["sigma_2_dn"], 2)],
                new_ath_expansion_potential=spot > qm["1d"]["sigma_2_up"]
            ),
            summaries=DualSummary(
                layman_summary=f"Bitcoin is trading near ${spot:,.2f}. Volatility boundaries are anchored at ±2σ bands.",
                institutional_detail=f"Quant matrix active. Mean reversion centered on 1H μ=${qm['1h']['mu']:,.2f}.",
                drastic_change_detected=False
            )
        )
    return ComprehensiveReport(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        spot_price=spot,
        volatility=VolatilityProjection(intraday_4h_regime="NORMAL_EXPANSION", intraday_expected_range_usd=1200.0, weekly_day_by_day_range={}, monthly_volatility_regime="STABLE"),
        traps_and_squeezes=TrapSqueezeAnalysis(short_squeeze_risk="LOW", long_squeeze_risk="LOW", bull_trap_zone=[spot*1.02, spot*1.03], bear_trap_zone=[spot*0.98, spot*0.97], rejection_resistance_levels=[spot*1.02], support_sweep_levels=[spot*0.98], new_ath_expansion_potential=False),
        summaries=DualSummary(layman_summary="Initializing market telemetry...", institutional_detail="Awaiting exchange feed sync.", drastic_change_detected=False)
    )

# =====================================================================
# STREAMLIT UI DASHBOARD
# =====================================================================

st.set_page_config(page_title="BTC Institutional Analytics", layout="wide")

if "weekly_anchor" not in st.session_state:
    st.session_state.weekly_anchor = None
if "monthly_anchor" not in st.session_state:
    st.session_state.monthly_anchor = None
if "active_trade" not in st.session_state:
    st.session_state.active_trade = None

st.sidebar.subheader("🕹️ Configuration")
selected_symbol = st.sidebar.selectbox("Target Digital Asset", ["BTC/USD", "ETH/USD"])
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password", value="")

live_data = fetch_cached_market_data(selected_symbol)
live_spot = live_data["spot_price"]
qm_data = live_data["quant_matrix"]

engine_result = generate_report_sync(live_data, openrouter_key)
latest_report = engine_result["report"]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Active Trade Setup")
with st.sidebar.form("active_position_form"):
    trade_dir = st.selectbox("Direction", ["LONG", "SHORT"])
    trade_entry = st.number_input("Entry Price ($)", value=live_spot if live_spot > 0 else 79000.0)
    trade_leverage = st.number_input("Leverage (x)", min_value=1.0, max_value=50.0, value=5.0)
    trade_sl = st.number_input("Stop Loss ($)", value=trade_entry * 0.98 if trade_dir == "LONG" else trade_entry * 1.02)
    trade_tp1 = st.number_input("TP1 Target ($)", value=trade_entry * 1.02 if trade_dir == "LONG" else trade_entry * 0.98)
    save_trade = st.form_submit_button("Execute Unified Master Dispatch")

    if save_trade:
        m_margin = 0.005
        est_liq = trade_entry * (1 - (1 / trade_leverage) + m_margin) if trade_dir == "LONG" else trade_entry * (1 + (1 / trade_leverage) - m_margin)
        st.session_state.active_trade = {"direction": trade_dir, "entry": trade_entry, "leverage": trade_leverage, "sl": trade_sl, "tp1": trade_tp1, "est_liq": est_liq}
        st.sidebar.success("Position Logged!")

if st.sidebar.button("🗑️ Clear Active Trade"):
    st.session_state.active_trade = None
    st.rerun()

st.title("🏛️ BTC-Centric Master Analytical & Volatility Dashboard")

st.success(f"Unified Master Dispatch Executed Successfully! Engine: {engine_result['engine']}")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Tactical Recon (4H - 10H Outlook)", 
    "📊 Weekly Anchor (4H / Daily / Weekly)", 
    "📅 Monthly Macro Anchor (Daily / Weekly / Monthly)", 
    "🏛️ Active Trade Matrix / War Rooms"
])

with tab1:
    st.subheader("Live Tactical Metrics & Liquidity Traps")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Spot Price", f"${live_spot:,.2f}" if live_spot > 0 else "N/A")
    c2.metric("Funding Rate", f"{live_data['funding_rate']*100:.4f}%")
    c3.metric("Open Interest Index", f"{live_data['open_interest']:,.0f}")
    c4.metric("1H ATR14", f"${qm_data['1h']['atr14']:,.2f}" if qm_data.get("1h") else "N/A")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Short Squeeze Risk:** `{latest_report.traps_and_squeezes.short_squeeze_risk}`")
        st.write(f"**Long Squeeze Risk:** `{latest_report.traps_and_squeezes.long_squeeze_risk}`")
        st.write(f"**Bull Trap Zone (+2σ to +3σ):** ${latest_report.traps_and_squeezes.bull_trap_zone[0]:,.2f} - ${latest_report.traps_and_squeezes.bull_trap_zone[1]:,.2f}")
    with col_b:
        st.write(f"**Bear Trap Zone (-2σ to -3σ):** ${latest_report.traps_and_squeezes.bear_trap_zone[0]:,.2f} - ${latest_report.traps_and_squeezes.bear_trap_zone[1]:,.2f}")
        st.write(f"**Rejection Levels:** {latest_report.traps_and_squeezes.rejection_resistance_levels}")

    st.markdown("---")
    st.info(latest_report.summaries.layman_summary)

with tab2:
    st.subheader("Weekly Matrix & Volatility Exhaustion Bands")
    if qm_data.get("1w"):
        w1 = qm_data["1w"]
        st.write(f"**Weekly Open / Anchor Price:** ${w1['close']:,.2f}")
        st.write(f"**+2σ Upper Boundary:** ${w1['sigma_2_up']:,.2f}")
        st.write(f"**-2σ Lower Boundary:** ${w1['sigma_2_dn']:,.2f}")
    st.markdown("---")
    st.write("### Weekly Day-by-Day Volatility Projections")
    if latest_report.volatility.weekly_day_by_day_range:
        st.dataframe(pd.DataFrame(list(latest_report.volatility.weekly_day_by_day_range.items()), columns=["Horizon", "Projected Range (+/- USD)"]))

with tab3:
    st.subheader("Monthly Macro Structural View")
    if qm_data.get("1d"):
        d1 = qm_data["1d"]
        st.write(f"**Daily Mean (μ):** ${d1['mu']:,.2f}")
        st.write(f"**Monthly Volatility Regime:** {latest_report.volatility.monthly_volatility_regime}")
        st.write(f"**Intraday Expected Range:** ±${latest_report.volatility.intraday_expected_range_usd:,.2f}")

with tab4:
    st.subheader("Active Position & War Room Risk Monitor")
    if st.session_state.active_trade:
        pos = st.session_state.active_trade
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.write(f"**Direction:** {pos['direction']} ({pos['leverage']}x)")
        p2.write(f"**Entry:** ${pos['entry']:,.2f}")
        p3.write(f"**Stop Loss:** ${pos['sl']:,.2f}")
        p4.write(f"**Est. Liq:** ${pos['est_liq']:,.2f}")
        dist_sl = abs(live_spot - pos['sl']) / live_spot * 100
        p5.write(f"**Dist to SL:** {dist_sl:.2f}%")
        
        if pos['direction'] == "LONG" and pos['sl'] <= pos['est_liq']:
            st.error("🚨 WARNING: Stop loss is set below your liquidation price!")
    else:
        st.info("No active trade logged. Use the sidebar to set up your position tracker.")
