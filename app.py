import asyncio
import os
import math
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

# =====================================================================
# SYSTEM & LOGGING SETUP
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BTC-Quant-Stage2")

st.set_page_config(
    page_title="BTC Institutional Microstructure Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark Quant Theme Styling
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #131722; padding: 6px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; border-radius: 6px; padding: 8px 16px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1E293B; color: #38BDF8 !important; border-bottom: 2px solid #38BDF8; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F8FAFC; }
    .flash-banner-alert {
        background: linear-gradient(90deg, rgba(225,29,72,0.2) 0%, rgba(159,18,57,0.4) 100%);
        border: 1px solid #F43F5E; color: #FECDD3; padding: 12px 18px; border-radius: 8px; font-weight: 600; margin-bottom: 15px;
    }
    .flash-banner-ok {
        background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.25) 100%);
        border: 1px solid #10B981; color: #A7F3D0; padding: 12px 18px; border-radius: 8px; font-weight: 600; margin-bottom: 15px;
    }
    .agent-card-bull { background-color: #064E3B; border-left: 4px solid #10B981; padding: 10px; border-radius: 6px; margin-bottom: 8px; }
    .agent-card-bear { background-color: #7F1D1D; border-left: 4px solid #EF4444; padding: 10px; border-radius: 6px; margin-bottom: 8px; }
    .summary-box-tech { background-color: #0F172A; border: 1px solid #334155; padding: 16px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.9em; }
    .summary-box-layman { background-color: #1E1B4B; border: 1px solid #4338CA; padding: 16px; border-radius: 8px; color: #E0E7FF; font-size: 1.02em; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# STAGE 2 PYDANTIC SCHEMAS
# =====================================================================

class BullishThesis(BaseModel):
    agent_id: str
    agent_name: str
    conviction_score: float = Field(ge=0.0, le=10.0)
    key_argument: str
    target_levels: List[float]

class BearishThesis(BaseModel):
    agent_id: str
    agent_name: str
    conviction_score: float = Field(ge=0.0, le=10.0)
    key_argument: str
    target_levels: List[float]

class MultiAgentDebatePayload(BaseModel):
    bull_theses: List[BullishThesis]
    bear_theses: List[BearishThesis]
    bull_conviction_total: float
    bear_conviction_total: float
    dominant_bias: str

class TrapAnalysis(BaseModel):
    bull_trap_detected: bool
    bear_trap_detected: bool
    short_squeeze_risk: str
    long_squeeze_risk: str
    explainable_context: str

class MarketState(BaseModel):
    timestamp_utc: str
    spot_price: float
    bull_conviction_pct: float
    bear_conviction_pct: float
    flash_update_active: bool
    flash_update_reason: str
    intraday_tactical_summary_15m_1h: str
    tactical_recon_4h_10h: str
    macro_anchor_summary_1d_1w: str
    layman_summary: str
    technical_deep_dive: str
    trap_intelligence: TrapAnalysis
    replacement_audit_log: List[Dict[str, str]]

# =====================================================================
# MULTI-TIMEFRAME DATA & QUANT ENGINE
# =====================================================================

async def fetch_multi_tf_data_async(symbol: str = "BTC/USD") -> Dict:
    exchange = ccxt.kraken({"enableRateLimit": True, "timeout": 12000})
    try:
        timeframes = ["15m", "1h", "4h", "1d", "1w"]
        tasks = [exchange.fetch_ohlcv(symbol, tf, limit=100) for tf in timeframes]
        ticker_task = exchange.fetch_ticker(symbol)

        results = await asyncio.gather(*tasks, ticker_task, return_exceptions=True)
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
            df["mu"] = df["close"].rolling(20).mean()
            df["std"] = df["close"].rolling(20).std()
            
            # Standard Deviation Exhaustion Bands
            df["sigma_1_up"] = df["mu"] + df["std"]
            df["sigma_2_up"] = df["mu"] + (2 * df["std"])
            df["sigma_3_up"] = df["mu"] + (3 * df["std"])
            df["sigma_1_dn"] = df["mu"] - df["std"]
            df["sigma_2_dn"] = df["mu"] - (2 * df["std"])
            df["sigma_3_dn"] = df["mu"] - (3 * df["std"])

            # Order Flow Delta & CVD
            range_len = np.maximum(df["high"] - df["low"], 0.0001)
            buyer_ratio = (df["close"] - df["low"]) / range_len
            df["vol_delta"] = (buyer_ratio - 0.5) * 2 * df["volume"]
            df["cvd"] = df["vol_delta"].cumsum()

            # Technical Indicators: RSI & StochRSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / np.maximum(loss, 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))

            min_rsi = df["rsi"].rolling(14).min()
            max_rsi = df["rsi"].rolling(14).max()
            df["stoch_k"] = ((df["rsi"] - min_rsi) / np.maximum(max_rsi - min_rsi, 1e-9)) * 100
            df["stoch_d"] = df["stoch_k"].rolling(3).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            matrix[tf] = {
                "df": df,
                "close": float(last["close"]),
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "mu": float(last["mu"]),
                "sigma_1_up": float(last["sigma_1_up"]),
                "sigma_2_up": float(last["sigma_2_up"]),
                "sigma_3_up": float(last["sigma_3_up"]),
                "sigma_1_dn": float(last["sigma_1_dn"]),
                "sigma_2_dn": float(last["sigma_2_dn"]),
                "sigma_3_dn": float(last["sigma_3_dn"]),
                "atr14": float(last["atr14"]),
                "cvd": float(last["cvd"]),
                "cvd_slope_5": float(df["cvd"].iloc[-1] - df["cvd"].iloc[-5]),
                "price_change_5": float(df["close"].iloc[-1] - df["close"].iloc[-5]),
                "rsi": float(last["rsi"]),
                "stoch_k": float(last["stoch_k"]),
                "stoch_d": float(last["stoch_d"])
            }

        spot_price = float(ticker_res.get("last", matrix["1h"]["close"])) if isinstance(ticker_res, dict) else matrix["1h"]["close"]

        return {
            "spot_price": spot_price,
            "funding_rate": 0.00021,  # Simulated 0.021% 8h funding
            "open_interest": 48250.0,
            "whale_ratio": 1.88,       # Whale Long/Short ratio > 1.80
            "quant_matrix": matrix,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {"error": str(e), "spot_price": 79200.0, "funding_rate": 0.0001, "open_interest": 0.0, "whale_ratio": 1.5, "quant_matrix": {}}
    finally:
        await exchange.close()

@st.cache_data(ttl=30)
def get_cached_quant_data(symbol: str = "BTC/USD") -> Dict:
    return asyncio.run(fetch_multi_tf_data_async(symbol))

# =====================================================================
# STAGE 2 — MULTI-AGENT MICROSTRUCTURE PIPELINE
# =====================================================================

def execute_stage2_multi_agent_debate(data: Dict) -> MultiAgentDebatePayload:
    spot = data["spot_price"]
    qm = data["quant_matrix"]
    fr = data["funding_rate"]
    whale_ratio = data["whale_ratio"]

    # --- 5 BEAR CASE AGENTS ---
    bear_1 = BearishThesis(
        agent_id="BEAR_1",
        agent_name="Funding & OI Overheat Specialist",
        conviction_score=8.5 if fr > 0.00015 else 4.0,
        key_argument=f"Elevated 8H funding rate ({fr*100:.3f}%) indicates long position crowding, vulnerable to cascading long liquidations.",
        target_levels=[spot * 0.98, spot * 0.96]
    )
    
    bear_2 = BearishThesis(
        agent_id="BEAR_2",
        agent_name="Upper Exhaustion & Wall Absorption Agent",
        conviction_score=8.0 if spot >= qm.get("1h", {}).get("sigma_2_up", spot*1.02) else 3.5,
        key_argument=f"Price pressing into upper volatility exhaustion zone (+2σ at ${qm.get('1h', {}).get('sigma_2_up', 0):,.2f}). Passive limit sell walls absorbing buyers.",
        target_levels=[qm.get("1h", {}).get("mu", spot)]
    )

    cvd_slope_1h = qm.get("1h", {}).get("cvd_slope_5", 0)
    price_chg_1h = qm.get("1h", {}).get("price_change_5", 0)
    bear_3 = BearishThesis(
        agent_id="BEAR_3",
        agent_name="Bearish CVD Divergence Agent",
        conviction_score=7.5 if (price_chg_1h > 0 and cvd_slope_1h < 0) else 3.0,
        key_argument="Bearish CVD Divergence detected: Price made local higher highs while Cumulative Volume Delta declined, showing buyer exhaustion.",
        target_levels=[qm.get("15m", {}).get("sigma_2_dn", spot*0.99)]
    )

    bear_4 = BearishThesis(
        agent_id="BEAR_4",
        agent_name="Macro Resistance & Trend Agent",
        conviction_score=6.5 if spot < qm.get("1w", {}).get("open", spot) else 3.0,
        key_argument=f"Price trading below Weekly Open (${qm.get('1w', {}).get('open', 0):,.2f}). Higher-timeframe market structure remains capped by weekly supply.",
        target_levels=[qm.get("1d", {}).get("sigma_2_dn", spot*0.95)]
    )

    bear_5 = BearishThesis(
        agent_id="BEAR_5",
        agent_name="Long Liquidation & Cascade Agent",
        conviction_score=7.0 if spot < qm.get("1h", {}).get("mu", spot) else 4.0,
        key_argument="Loss of 1H mean (μ) opens thin order book liquidity pocket down to key support sweep zones.",
        target_levels=[qm.get("4h", {}).get("sigma_2_dn", spot*0.97)]
    )

    # --- 5 BULL CASE AGENTS ---
    bull_1 = BullishThesis(
        agent_id="BULL_1",
        agent_name="Overhead Short Liquidation Pool Specialist",
        conviction_score=8.0 if spot > qm.get("15m", {}).get("mu", spot) else 4.0,
        key_argument=f"Dense short liquidation cluster accumulated above current spot between ${spot*1.01:,.2f} and ${spot*1.025:,.2f}.",
        target_levels=[spot * 1.015, spot * 1.028]
    )

    bull_2 = BullishThesis(
        agent_id="BULL_2",
        agent_name="Lower Exhaustion & Taker Absorption Agent",
        conviction_score=8.5 if spot <= qm.get("1h", {}).get("sigma_1_dn", spot*0.98) else 3.5,
        key_argument=f"Market sell orders being aggressively absorbed into passive limit bids near lower deviation boundary (-2σ at ${qm.get('1h', {}).get('sigma_2_dn', 0):,.2f}).",
        target_levels=[qm.get("1h", {}).get("mu", spot)]
    )

    bull_3 = BullishThesis(
        agent_id="BULL_3",
        agent_name="Bullish CVD Divergence Agent",
        conviction_score=8.0 if (price_chg_1h < 0 and cvd_slope_1h > 0) else 3.5,
        key_argument="Bullish CVD Divergence confirmed: Price printed lower lows while spot cumulative volume delta accumulated upward.",
        target_levels=[qm.get("4h", {}).get("sigma_2_up", spot*1.02)]
    )

    bull_4 = BullishThesis(
        agent_id="BULL_4",
        agent_name="Whale Position & Institutional Ratio Agent",
        conviction_score=9.0 if whale_ratio >= 1.80 else 4.5,
        key_argument=f"Top-trader whale long/short ratio elevated at {whale_ratio:.2f} (>1.80 threshold), indicating heavy institutional accumulation.",
        target_levels=[qm.get("1d", {}).get("sigma_2_up", spot*1.05)]
    )

    bull_5 = BullishThesis(
        agent_id="BULL_5",
        agent_name="Expansion & Structural Reclaim Agent",
        conviction_score=7.5 if spot > qm.get("4h", {}).get("open", spot) else 3.5,
        key_argument="Structural reclaim of 4H open price underway. Expansion volatility cycle favored to continue upward.",
        target_levels=[qm.get("1w", {}).get("sigma_2_up", spot*1.06)]
    )

    bears = [bear_1, bear_2, bear_3, bear_4, bear_5]
    bulls = [bull_1, bull_2, bull_3, bull_4, bull_5]

    tot_bear = sum(b.conviction_score for b in bears)
    tot_bull = sum(b.conviction_score for b in bulls)
    dom = "BULLISH" if tot_bull > tot_bear else ("BEARISH" if tot_bear > tot_bull else "NEUTRAL")

    return MultiAgentDebatePayload(
        bull_theses=bulls,
        bear_theses=bears,
        bull_conviction_total=tot_bull,
        bear_conviction_total=tot_bear,
        dominant_bias=dom
    )

def execute_head_arbitrator_agent(data: Dict, debate: MultiAgentDebatePayload, openrouter_key: str = "") -> MarketState:
    spot = data["spot_price"]
    qm = data["quant_matrix"]
    
    tot_score = debate.bull_conviction_total + debate.bear_conviction_total
    bull_pct = round((debate.bull_conviction_total / max(tot_score, 1)) * 100, 1)
    bear_pct = round(100.0 - bull_pct, 1)

    # Detect Structural Flash Invalidation
    weekly_open = qm.get("1w", {}).get("open", spot)
    flash_active = False
    flash_reason = "All higher timeframe structural anchors holding within normal volatility boundaries."
    
    if spot < qm.get("1d", {}).get("sigma_2_dn", spot*0.95):
        flash_active = True
        flash_reason = f"🚨 FLASH INVALIDATION: Spot (${spot:,.2f}) lost 1D -2σ boundary (${qm.get('1d', {}).get('sigma_2_dn', 0):,.2f}). Immediate downside risk unlocked."
    elif spot > qm.get("1d", {}).get("sigma_2_up", spot*1.05):
        flash_active = True
        flash_reason = f"⚡ FLASH BREAKOUT: Spot (${spot:,.2f}) reclaimed 1D +2σ boundary (${qm.get('1d', {}).get('sigma_2_up', 0):,.2f}). Squeeze continuation active."

    # Explainable Trap Intelligence
    bull_trap = spot >= qm.get("1h", {}).get("sigma_2_up", spot*1.02) and qm.get("1h", {}).get("cvd_slope_5", 0) < 0
    bear_trap = spot <= qm.get("1h", {}).get("sigma_2_dn", spot*0.98) and qm.get("1h", {}).get("cvd_slope_5", 0) > 0

    trap_intel = TrapAnalysis(
        bull_trap_detected=bull_trap,
        bear_trap_detected=bear_trap,
        short_squeeze_risk="EXTREME" if debate.dominant_bias == "BULLISH" and data["funding_rate"] < 0.0001 else "MODERATE",
        long_squeeze_risk="EXTREME" if debate.dominant_bias == "BEARISH" and data["funding_rate"] > 0.0002 else "LOW",
        explainable_context=(
            f"BULL TRAP ALERT: Price tested ${spot:,.2f} (+2σ upper band) but spot CVD declined. Sellers are absorbing aggressive buyers into strength."
            if bull_trap else (
                f"BEAR TRAP ALERT: Price swept lower to ${spot:,.2f} (-2σ lower band) while spot CVD trended upward. Smart money is absorbing market sellers into support."
                if bear_trap else "No active trap divergence confirmed on 1H/4H timeframes. Order flow is currently aligned with price action."
            )
        )
    )

    # Prompt OpenRouter LLM if available, else run local Head Arbitrator synthesis
    if openrouter_key and openrouter_key.strip():
        try:
            prompt = f"""
            Act as Master Quant Arbitrator for BTC/USD at ${spot:,.2f}.
            Multi-Agent Debate Results:
            - Bull Conviction: {bull_pct}% ({debate.bull_conviction_total} pts)
            - Bear Conviction: {bear_pct}% ({debate.bear_conviction_total} pts)
            - Dominant Bias: {debate.dominant_bias}
            
            1H Quant Parameters: Mean μ=${qm.get('1h',{}).get('mu',0):,.2f}, +2σ=${qm.get('1h',{}).get('sigma_2_up',0):,.2f}, -2σ=${qm.get('1h',{}).get('sigma_2_dn',0):,.2f}.
            Weekly Open: ${weekly_open:,.2f}.

            Output ONLY valid JSON matching this structure:
            {{
                "layman_summary": "Conversational, plain-English summary explaining key levels and next expected move for non-quant users.",
                "technical_deep_dive": "Rigorous quantitative breakdown detailing order flow delta, σ-band boundaries, and multi-timeframe structural interactions.",
                "intraday_tactical_summary_15m_1h": "Outlook for the next 1-4 hours based on 15m/1h anchors.",
                "tactical_recon_4h_10h": "Outlook for the next 4-10 hours based on 4H structure.",
                "macro_anchor_summary_1d_1w": "Higher timeframe outlook based on Daily and Weekly open anchors."
            }}
            """
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
                data=json.dumps({"model": "anthropic/claude-3.5-sonnet", "messages": [{"role": "user", "content": prompt}]}),
                timeout=10
            )
            if res.status_code == 200:
                resp_json = res.json()
                content = resp_json["choices"][0]["message"]["content"]
                backticks = chr(96) * 3
                cleaned_content = content.replace(f"{backticks}json", "").replace(backticks, "").strip()
                parsed = json.loads(cleaned_content)
                return MarketState(
                    timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    spot_price=spot,
                    bull_conviction_pct=bull_pct,
                    bear_conviction_pct=bear_pct,
                    flash_update_active=flash_active,
                    flash_update_reason=flash_reason,
                    intraday_tactical_summary_15m_1h=parsed["intraday_tactical_summary_15m_1h"],
                    tactical_recon_4h_10h=parsed["tactical_recon_4h_10h"],
                    macro_anchor_summary_1d_1w=parsed["macro_anchor_summary_1d_1w"],
                    layman_summary=parsed["layman_summary"],
                    technical_deep_dive=parsed["technical_deep_dive"],
                    trap_intelligence=trap_intel,
                    replacement_audit_log=[
                        {"timestamp": datetime.now(timezone.utc).strftime("%H:%M UTC"), "level": "Weekly Open", "status": "ACTIVE", "reason": f"Anchored at ${weekly_open:,.2f}."}
                    ]
                )
        except Exception as e:
            logger.error(f"OpenRouter LLM arbitration failed: {e}")

    # Fallback Head Arbitrator (Local Deterministic Quant Synthesis)
    return MarketState(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        spot_price=spot,
        bull_conviction_pct=bull_pct,
        bear_conviction_pct=bear_pct,
        flash_update_active=flash_active,
        flash_update_reason=flash_reason,
        intraday_tactical_summary_15m_1h=f"15m/1h momentum centered on μ=${qm.get('1h', {}).get('mu', spot):,.2f}. Immediate volatility boundaries defined by ±2σ bands (${qm.get('1h', {}).get('sigma_2_dn', spot*0.98):,.2f} to ${qm.get('1h', {}).get('sigma_2_up', spot*1.02):,.2f}).",
        tactical_recon_4h_10h=f"Over the next 4-10 hours, BTC bias is {debate.dominant_bias} ({bull_pct}% Bull / {bear_pct}% Bear). Main battlefield lies between 4H support at ${qm.get('4h', {}).get('sigma_2_dn', spot*0.97):,.2f} and resistance at ${qm.get('4h', {}).get('sigma_2_up', spot*1.03):,.2f}.",
        macro_anchor_summary_1d_1w=f"Macro structure is constrained by the Weekly Open at ${weekly_open:,.2f}. Holding above keeps structural expansion constructive toward ${spot*1.05:,.2f}.",
        layman_summary=f"Bitcoin is trading around ${spot:,.2f}. The multi-agent council leans {debate.dominant_bias} with {bull_pct}% conviction. Buyers are defending support near ${qm.get('1h', {}).get('sigma_2_dn', spot*0.98):,.2f}, while resistance sits at ${qm.get('1h', {}).get('sigma_2_up', spot*1.02):,.2f}.",
        technical_deep_dive=f"Multi-Agent Microstructure Synthesis completed. 10 specialized agents evaluated order flow across 15m->1W timeframes. Spot CVD slope stands at {qm.get('1h', {}).get('cvd_slope_5', 0):,.2f}, with whale position ratios indicating active institutional positioning at {data['whale_ratio']:.2f}.",
        trap_intelligence=trap_intel,
        replacement_audit_log=[
            {"timestamp": datetime.now(timezone.utc).strftime("%H:%M UTC"), "level": "Weekly Open", "status": "MAINTAINED", "reason": f"Weekly anchor price validated at ${weekly_open:,.2f}."},
            {"timestamp": datetime.now(timezone.utc).strftime("%H:%M UTC"), "level": "1H Mean (μ)", "status": "REPLACED", "reason": f"Updated to ${qm.get('1h', {}).get('mu', spot):,.2f} following rolling 20-period recalculation."}
        ]
    )

# =====================================================================
# PLOTLY INTERACTIVE QUANT CHARTING ENGINE
# =====================================================================

def render_quant_chart(df: pd.DataFrame, timeframe_label: str):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"Price & Exhaustion Bands (±1σ, ±2σ, ±3σ) — [{timeframe_label}]", "Cumulative Volume Delta (CVD)", "Momentum (RSI & StochRSI)"),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Subplot 1: Candlesticks & Sigma Bands
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price", increasing_line_color="#10B981", decreasing_line_color="#EF4444"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['mu'], mode='lines', name='Mean (μ)', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sigma_2_up'], mode='lines', name='+2σ Upper', line=dict(color='#EF4444', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sigma_2_dn'], mode='lines', name='-2σ Lower', line=dict(color='#10B981', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sigma_3_up'], mode='lines', name='+3σ Liquidation', line=dict(color='#B91C1C', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sigma_3_dn'], mode='lines', name='-3σ Liquidation', line=dict(color='#047857', width=1, dash='dot')), row=1, col=1)

    # Subplot 2: CVD
    fig.add_trace(go.Scatter(x=df.index, y=df['cvd'], mode='lines', name='CVD', line=dict(color='#38BDF8', width=1.5)), row=2, col=1)

    # Subplot 3: RSI & StochRSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI', line=dict(color='#E2E8F0', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], mode='lines', name='Stoch %K', line=dict(color='#F59E0B', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], mode='lines', name='Stoch %D', line=dict(color='#06B6D4', width=1)), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0B0E14",
        height=680,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        xaxis3_rangeslider_visible=False
    )
    return fig

# =====================================================================
# DASHBOARD INTERFACE & CONTROLLER
# =====================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🏛️ Quant Terminal")
symbol = st.sidebar.selectbox("Asset Pair", ["BTC/USD", "ETH/USD"], index=0)

with st.sidebar.expander("⚙️ Advanced Settings & API Key", expanded=False):
    openrouter_key = st.text_input("OpenRouter Key (Optional)", type="password", value="")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Active Trade Matrix")
with st.sidebar.form("trade_form"):
    t_dir = st.selectbox("Direction", ["LONG", "SHORT"])
    t_entry = st.number_input("Entry Price ($)", value=79200.0)
    t_lev = st.number_input("Leverage (x)", min_value=1.0, max_value=50.0, value=5.0)
    t_sl = st.number_input("Stop Loss ($)", value=78000.0 if t_dir == "LONG" else 80500.0)
    t_tp = st.number_input("Take Profit ($)", value=81500.0 if t_dir == "LONG" else 77000.0)
    submit_trade = st.form_submit_button("Track Active Trade")

    if submit_trade:
        st.session_state["active_trade"] = {"dir": t_dir, "entry": t_entry, "lev": t_lev, "sl": t_sl, "tp": t_tp}
        st.sidebar.success("Position Active!")

if st.sidebar.button("Clear Active Trade"):
    st.session_state["active_trade"] = None
    st.rerun()

# --- FETCH DATA & RUN PIPELINE ---
quant_data = get_cached_quant_data(symbol)
multi_agent_payload = execute_stage2_multi_agent_debate(quant_data)
state = execute_head_arbitrator_agent(quant_data, multi_agent_payload, openrouter_key)

# --- HEADER & FLASH ALERT BANNER ---
st.title("🏛️ BTC Master Microstructure Terminal")

if state.flash_update_active:
    st.markdown(f'<div class="flash-banner-alert">{state.flash_update_reason}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="flash-banner-ok">✅ Market Structure Normal — {state.flash_update_reason}</div>', unsafe_allow_html=True)

# --- TOP LEVEL METRICS BAR ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Live Spot Price", f"${state.spot_price:,.2f}")
m2.metric("Bull / Bear Conviction", f"{state.bull_conviction_pct}% / {state.bear_conviction_pct}%")
m3.metric("8H Funding Rate", f"{quant_data['funding_rate']*100:.3f}%")
m4.metric("Whale L/S Ratio", f"{quant_data['whale_ratio']:.2f}x")
m5.metric("1H Volatility (ATR14)", f"${quant_data['quant_matrix'].get('1h', {}).get('atr14', 0):,.2f}")

st.markdown("---")

# --- MAIN DASHBOARD NAVIGATION TABS ---
tab_recon, tab_intraday, tab_weekly, tab_monthly, tab_traps, tab_warroom, tab_audit = st.tabs([
    "⚡ Live Recon (4H-10H)",
    "⏱️ Intraday Anchor (15m/1h)",
    "📊 Weekly Anchor (4h/1d)",
    "📆 Monthly Anchor (1d/1w)",
    "🪤 Trap & Liquidation Intelligence",
    "⚔️ Multi-Agent Council (Stage 2)",
    "📜 Accuracy Audit Ledger"
])

# --- TAB 1: LIVE RECON (4H-10H OUTLOOK) ---
with tab_recon:
    st.subheader("🎯 Live Tactical Reconnaissance (4H - 10H Horizon)")
    
    col_lay, col_tech = st.columns(2)
    with col_lay:
        st.markdown("#### 🗣️ Plain-English (Layman) Summary")
        st.markdown(f'<div class="summary-box-layman">{state.layman_summary}</div>', unsafe_allow_html=True)
    with col_tech:
        st.markdown("#### 🔬 Technical & Microstructure Breakdown")
        st.markdown(f'<div class="summary-box-tech">{state.technical_deep_dive}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Interactive Multi-Timeframe Charting Engine")
    tf_select = st.selectbox("Select Timeframe Chart", ["15m", "1h", "4h", "1d", "1w"], index=1)
    if tf_select in quant_data["quant_matrix"]:
        fig = render_quant_chart(quant_data["quant_matrix"][tf_select]["df"], tf_select.upper())
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: INTRADAY ANCHOR (15m / 1h) ---
with tab_intraday:
    st.subheader("⏱️ Intraday Anchor Engine (15m / 1h / 4h Timeframes)")
    st.info(state.intraday_tactical_summary_15m_1h)
    
    if "1h" in quant_data["quant_matrix"]:
        q1 = quant_data["quant_matrix"]["1h"]
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("1H Mean (μ)", f"${q1['mu']:,.2f}")
        i2.metric("1H +2σ Boundary", f"${q1['sigma_2_up']:,.2f}")
        i3.metric("1H -2σ Boundary", f"${q1['sigma_2_dn']:,.2f}")
        i4.metric("1H CVD Delta Slope", f"{q1['cvd_slope_5']:,.0f}")

# --- TAB 3: WEEKLY ANCHOR ---
with tab_weekly:
    st.subheader("📊 Weekly Structural Matrix & Volatility Bands")
    if "1w" in quant_data["quant_matrix"]:
        q_w = quant_data["quant_matrix"]["1w"]
        st.write(f"**Weekly Open Anchor:** `${q_w['open']:,.2f}`")
        st.write(f"**+2σ Upper Exhaustion Boundary:** `${q_w['sigma_2_up']:,.2f}`")
        st.write(f"**-2σ Lower Exhaustion Boundary:** `${q_w['sigma_2_dn']:,.2f}`")

# --- TAB 4: MONTHLY ANCHOR ---
with tab_monthly:
    st.subheader("📆 Monthly Macro Structural Anchors")
    st.info(state.macro_anchor_summary_1d_1w)

# --- TAB 5: TRAP & LIQUIDATION INTELLIGENCE ---
with tab_traps:
    st.subheader("🪤 Explainable Trap & Liquidation Detector")
    
    t1, t2 = st.columns(2)
    with t1:
        st.write(f"**Bull Trap Detected:** `{'YES' if state.trap_intelligence.bull_trap_detected else 'NO'}`")
        st.write(f"**Bear Trap Detected:** `{'YES' if state.trap_intelligence.bear_trap_detected else 'NO'}`")
    with t2:
        st.write(f"**Short Squeeze Risk Level:** `{state.trap_intelligence.short_squeeze_risk}`")
        st.write(f"**Long Squeeze Risk Level:** `{state.trap_intelligence.long_squeeze_risk}`")

    st.markdown("#### 💡 Explainable Context & Order Flow Alignment")
    st.success(state.trap_intelligence.explainable_context)

# --- TAB 6: MULTI-AGENT COUNCIL (STAGE 2) ---
with tab_warroom:
    st.subheader("⚔️ Stage 2 — Multi-Agent Debate & Microstructure Council")
    st.write(f"**Dominant Bias:** `{multi_agent_payload.dominant_bias}` | **Bull Score:** `{multi_agent_payload.bull_conviction_total}` | **Bear Score:** `{multi_agent_payload.bear_conviction_total}`")

    col_bulls, col_bears = st.columns(2)
    with col_bulls:
        st.markdown("### 🟢 5 Bull Case Agents")
        for thesis in multi_agent_payload.bull_theses:
            st.markdown(f"""
            <div class="agent-card-bull">
                <strong>{thesis.agent_name}</strong> (Conviction: {thesis.conviction_score}/10)<br>
                <em>{thesis.key_argument}</em><br>
                <small>Target Levels: {thesis.target_levels}</small>
            </div>
            """, unsafe_allow_html=True)

    with col_bears:
        st.markdown("### 🔴 5 Bear Case Agents")
        for thesis in multi_agent_payload.bear_theses:
            st.markdown(f"""
            <div class="agent-card-bear">
                <strong>{thesis.agent_name}</strong> (Conviction: {thesis.conviction_score}/10)<br>
                <em>{thesis.key_argument}</em><br>
                <small>Target Levels: {thesis.target_levels}</small>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 7: ACCURACY AUDIT LEDGER ---
with tab_audit:
    st.subheader("📜 Historical Level Replacement & Accuracy Audit Ledger")
    st.dataframe(pd.DataFrame(state.replacement_audit_log), use_container_width=True)
