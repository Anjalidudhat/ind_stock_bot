"""
=============================================================
  streamlit_app.py — માત્ર Streamlit UI Code
  આ file માં:
    - Page layout, sidebar
    - Charts (Candlestick, RSI, MACD, Sentiment, etc.)
    - Tabs: Price, Technical, Sentiment, Global, Sector, Fundamentals
    - Final Verdict display

  Run: streamlit run streamlit_app.py
=============================================================
"""

import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# આ project ની બીજી files import
from analysis import (
    fetch_stock_data,
    add_features,
    get_fundamentals,
    analyze_sentiment,
    get_global_signals,
    get_sector_rotation,
    get_tech_reasons,
    train_ml,
    forecast_price,
    compute_verdict,
)


# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
def setup_page():
    """Streamlit page configuration અને custom CSS"""
    st.set_page_config(
        page_title="Stock ML Analyzer Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Dark theme CSS
    st.markdown("""
    <style>
    .stApp { background-color: #0a0f1e; }

    /* Verdict box */
    .verdict-box {
        background: linear-gradient(135deg, #0d2a1a, #1a4530);
        border: 2px solid #00e5a0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #e0f0ff;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar() -> tuple[str, bool]:
    """
    Sidebar render કરો
    Returns: (ticker, analyze_button_clicked)
    """
    with st.sidebar:
        st.title("📈 Stock ML Analyzer Pro")
        st.markdown("**10-Day Prediction Engine**")
        st.divider()

        # Ticker input
        ticker = st.text_input(
            "Enter Stock Ticker",
            value="RELIANCE.NS",
            # placeholder="e.g. TSLA, AAPL, RELIANCE.NS"
        ).strip().upper()

        # # Quick select buttons
        # st.markdown("**Quick Select:**")
        # c1, c2 = st.columns(2)
        # with c1:
        #     if st.button("RELIANCE.NS", use_container_width=True): ticker = "RELIANCE.NS"
        #     if st.button("INFY.NS",     use_container_width=True): ticker = "INFY.NS"
        #     if st.button("HDFCBANK.NS", use_container_width=True): ticker = "HDFCBANK.NS"
        # with c2:
        #     if st.button("TSLA", use_container_width=True): ticker = "TSLA"
        #     if st.button("AAPL", use_container_width=True): ticker = "AAPL"
        #     if st.button("NVDA", use_container_width=True): ticker = "NVDA"

        st.divider()
        analyze_btn = st.button("🚀 Analyze Now", type="primary", use_container_width=True)

        # st.divider()
        # st.caption("⚠️ Educational purposes only.")
        # st.caption("Not financial advice.")

    return ticker, analyze_btn


# ─────────────────────────────────────────────
#  WELCOME SCREEN (before analysis)
# ─────────────────────────────────────────────
def render_welcome():
    """Analyze button press ના પહેલા welcome screen"""
    st.markdown("""
    <div style='text-align:center; padding: 60px 0; color: #e0f0ff;'>
        <h1>📈 Stock ML Analyzer Pro</h1>
        <h3 style='color:#6a90b0;'>8-Factor 10-Day Prediction Engine</h3>
        <br>
        <p>Sidebar માં ticker enter કરો અને <b>Analyze Now</b> click કરો</p>
        <br>
        <table style='margin:auto; border-collapse:collapse; color:#c8d8e8;'>
            <tr><td style='padding:8px;'>✅ Technical Indicators (20/50/200 DMA, RSI, MACD, BB)</td></tr>
            <tr><td style='padding:8px;'>✅ News Sentiment (TextBlob + VADER)</td></tr>
            <tr><td style='padding:8px;'>✅ Fundamental Data (P/E, Revenue, Analyst Rating)</td></tr>
            <tr><td style='padding:8px;'>✅ Global Signals (S&P500, Oil, Gold, VIX, USD/INR)</td></tr>
            <tr><td style='padding:8px;'>✅ Sector Rotation (India + US sectors)</td></tr>
            <tr><td style='padding:8px;'>✅ 3-Model ML Ensemble (RF + XGBoost + GBM)</td></tr>
            <tr><td style='padding:8px;'>✅ 10-Day Day-by-Day Price Forecast</td></tr>
            <tr><td style='padding:8px;'>✅ Multi-Signal Verdict Engine</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER METRICS (top row)
# ─────────────────────────────────────────────
def render_header_metrics(ticker, df, forecast, ml, sentiment):
    """Top 5 metric cards"""
    st.title(f"📊 {ticker} — Analysis Report")

    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2])
    day_change    = (current_price - prev_price) / prev_price * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",    f"${current_price:.2f}",              f"{day_change:+.2f}%")
    c2.metric("10-Day Forecast",  f"${forecast['forecast_price']:.2f}", f"{forecast['change_pct']:+.1f}%")
    c3.metric("ML Confidence",    f"{ml['next_day_proba']*100:.0f}%",   ml["direction"])
    c4.metric("News Sentiment",   sentiment["overall"],                 f"score: {sentiment['avg_score']:+.3f}")
    c5.metric("Model Accuracy",   f"{ml['ens_accuracy']*100:.1f}%",    "Ensemble avg")

    return current_price


# ─────────────────────────────────────────────
#  VERDICT SECTION
# ─────────────────────────────────────────────
def render_verdict(verdict):
    """Final verdict + gauge chart + signal table"""
    st.subheader("🎯 Final Verdict")
    v_col1, v_col2 = st.columns([2, 1])

    # Verdict box
    with v_col1:
        st.markdown(f"""
        <div class='verdict-box'>
            {verdict['verdict']}
            <br>
            <small style='font-size:0.6em; color:#aaa;'>
                Based on {len(verdict['scores'])} factors &nbsp;|&nbsp;
                Confidence: {verdict['confidence']}%
            </small>
        </div>
        """, unsafe_allow_html=True)

    # Gauge chart
    with v_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=verdict["total"],
            title={"text": "Signal Score", "font": {"color": "#e0f0ff"}},
            gauge={
                "axis":  {"range": [-6, 6], "tickcolor": "#6a90b0"},
                "bar":   {"color": "#00e5a0" if verdict["total"] > 0 else "#ff4d6d"},
                "steps": [
                    {"range": [-6, -3], "color": "#2d0a0a"},
                    {"range": [-3,  0], "color": "#2d1a0a"},
                    {"range": [ 0,  3], "color": "#0a2d1a"},
                    {"range": [ 3,  6], "color": "#0a1a2d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": verdict["total"]}
            },
            number={"font": {"color": "#e0f0ff"}}
        ))
        fig_gauge.update_layout(
            height=200,
            paper_bgcolor="#0a0f1e",
            font_color="#e0f0ff",
            margin=dict(t=30, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Signal breakdown table
    sig_data = [
        {
            "Factor": factor,
            "Signal": "🟢 BUY" if s == 1 else ("🔴 SELL" if s == -1 else "⚪ NEUTRAL")
        }
        for factor, s in verdict["scores"].items()
    ]
    st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  TAB 1: PRICE + FORECAST CHART
# ─────────────────────────────────────────────
def render_price_tab(df, forecast, ticker):
    """Candlestick + Moving Averages + Bollinger + Forecast + Volume"""
    close_vals = df["Close"].squeeze()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{ticker} — Price + 10-Day Forecast", "Volume"]
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index[-100:],
        open=df["Open"].iloc[-100:].squeeze(),
        high=df["High"].iloc[-100:].squeeze(),
        low=df["Low"].iloc[-100:].squeeze(),
        close=close_vals.iloc[-100:],
        name="OHLC",
        increasing_line_color="#00e5a0",
        decreasing_line_color="#ff4d6d"
    ), row=1, col=1)

    # Moving Average lines
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["SMA_20"].iloc[-100:],
        name="20 DMA",  line=dict(color="#f0c040", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["SMA_50"].iloc[-100:],
        name="50 DMA",  line=dict(color="#00a8ff", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["SMA_200"].iloc[-100:],
        name="200 DMA", line=dict(color="#ff6b9d", width=1.5, dash="dot")), row=1, col=1)

    # Bollinger Bands (shaded area)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["BB_upper"].iloc[-100:],
        name="BB Upper", line=dict(color="#6a90b0", width=1, dash="dash"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["BB_lower"].iloc[-100:],
        name="BB Lower", line=dict(color="#6a90b0", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(106,144,176,0.05)"), row=1, col=1)

    # Forecast line (dashed green)
    fig.add_trace(go.Scatter(
        x=forecast["future_dates"], y=forecast["future_prices"],
        name="10-Day Forecast",
        line=dict(color="#00e5a0", width=2.5, dash="dash"),
        mode="lines+markers"
    ), row=1, col=1)

    # Volume bars (green=up day, red=down day)
    vol_colors = [
        "#00e5a0" if c >= o else "#ff4d6d"
        for c, o in zip(
            df["Close"].iloc[-100:].squeeze(),
            df["Open"].iloc[-100:].squeeze()
        )
    ]
    fig.add_trace(go.Bar(
        x=df.index[-100:], y=df["Volume"].iloc[-100:].squeeze(),
        name="Volume", marker_color=vol_colors, opacity=0.7
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=600,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#1e3a5f"),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(gridcolor="#1e3a5f")
    fig.update_yaxes(gridcolor="#1e3a5f")
    st.plotly_chart(fig, use_container_width=True)

    # Day-by-day forecast table
    st.subheader("📅 10-Day Day-by-Day Forecast Table")
    fc_df = forecast["daily_table"].copy()
    fc_df["Direction"] = fc_df["Change %"].apply(lambda x: "📈 UP" if x > 0 else "📉 DOWN")
    st.dataframe(fc_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  TAB 2: TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def render_technical_tab(df, tech, ml):
    """RSI + MACD + Stochastic charts + Bullish/Bearish signals"""

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["RSI (14)", "MACD", "Stochastic Oscillator"],
        row_heights=[0.33, 0.34, 0.33]
    )

    # RSI (overbought/oversold zones)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["RSI"].iloc[-100:],
        name="RSI", line=dict(color="#f0c040", width=1.5)), row=1, col=1)
    fig.add_hline(y=70, line_color="#ff4d6d", line_dash="dash", annotation_text="Overbought", row=1, col=1)
    fig.add_hline(y=30, line_color="#00e5a0", line_dash="dash", annotation_text="Oversold",   row=1, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", row=1, col=1)

    # MACD histogram + lines
    macd_hist   = df["MACD_hist"].iloc[-100:]
    macd_colors = ["#00e5a0" if v >= 0 else "#ff4d6d" for v in macd_hist]
    fig.add_trace(go.Bar(x=df.index[-100:], y=macd_hist,
        name="MACD Hist", marker_color=macd_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["MACD"].iloc[-100:],
        name="MACD",   line=dict(color="#00a8ff", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["MACD_sig"].iloc[-100:],
        name="Signal", line=dict(color="#ff6b9d", width=1.5)), row=2, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Stoch_K"].iloc[-100:],
        name="Stoch K", line=dict(color="#c77dff", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df["Stoch_D"].iloc[-100:],
        name="Stoch D", line=dict(color="#f0c040", width=1.5)), row=3, col=1)
    fig.add_hline(y=80, line_color="#ff4d6d", line_dash="dash", row=3, col=1)
    fig.add_hline(y=20, line_color="#00e5a0", line_dash="dash", row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=600
    )
    fig.update_xaxes(gridcolor="#1e3a5f")
    fig.update_yaxes(gridcolor="#1e3a5f")
    st.plotly_chart(fig, use_container_width=True)

    # Bullish / Bearish signal cards
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("### 📈 Bullish Signals")
        for r in tech["up"]:
            st.success(f"▲ {r}")
        if not tech["up"]:
            st.info("No bullish signals currently")

    with b_col2:
        st.markdown("### 📉 Bearish Signals")
        for r in tech["dn"]:
            st.error(f"▼ {r}")
        if not tech["dn"]:
            st.info("No bearish signals currently")

    # ML Feature Importance
    st.subheader("🔍 Top ML Features (by Importance)")
    top_feat = ml["feature_importance"].head(10)
    fig_feat = px.bar(
        x=top_feat.values, y=top_feat.index,
        orientation="h",
        color=top_feat.values,
        color_continuous_scale="teal",
        labels={"x": "Importance", "y": "Feature"}
    )
    fig_feat.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=350,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_feat, use_container_width=True)


# ─────────────────────────────────────────────
#  TAB 3: NEWS SENTIMENT
# ─────────────────────────────────────────────
def render_sentiment_tab(sentiment):
    """Sentiment bar chart + news table"""
    st.subheader(f"🗞️ News Sentiment — {sentiment['overall']}")
    sent_df = sentiment["dataframe"]

    # Summary metrics
    pos = len(sent_df[sent_df["label"] == "Positive"])
    neg = len(sent_df[sent_df["label"] == "Negative"])
    neu = len(sent_df[sent_df["label"] == "Neutral"])
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positive News", pos, "Bullish")
    c2.metric("🔴 Negative News", neg, "Bearish")
    c3.metric("🟡 Neutral News",  neu, "")

    # Horizontal bar chart
    fig_s = px.bar(
        sent_df, x="score", y="headline",
        orientation="h", color="score",
        color_continuous_scale=["#ff4d6d", "#f0c040", "#00e5a0"],
        range_color=[-1, 1],
        labels={"score": "Sentiment Score", "headline": ""}
    )
    fig_s.add_vline(x=0, line_color="#4a7090")
    fig_s.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=450,
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # Detailed table
    st.dataframe(
        sent_df[["label", "score", "headline"]].rename(columns={
            "label": "Sentiment", "score": "Score", "headline": "Headline"
        }),
        use_container_width=True, hide_index=True
    )


# ─────────────────────────────────────────────
#  TAB 4: GLOBAL MARKETS
# ─────────────────────────────────────────────
def render_global_tab(global_sig):
    """Global market metrics + bar chart"""
    st.subheader("🌍 Global Market Signals")

    # Metric cards (4 per row)
    cols = st.columns(4)
    for i, (name, data) in enumerate(global_sig.items()):
        with cols[i % 4]:
            chg   = data.get("change", 0)
            price = data.get("price", "N/A")
            try:
                st.metric(name, f"{float(price):.2f}", f"{float(chg):+.2f}%")
            except Exception:
                st.metric(name, str(price))

    st.divider()

    # Bar chart: % change for each global index
    g_names   = list(global_sig.keys())
    g_changes = [float(global_sig[n].get("change", 0) or 0) for n in g_names]
    g_colors  = ["#00e5a0" if c > 0 else "#ff4d6d" for c in g_changes]

    fig_g = go.Figure(go.Bar(
        x=g_names, y=g_changes,
        marker_color=g_colors,
        text=[f"{c:+.2f}%" for c in g_changes],
        textposition="outside"
    ))
    fig_g.add_hline(y=0, line_color="#4a7090")
    fig_g.update_layout(
        title="Global Market % Change (Latest Day)",
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=350
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # Interpretation guide
    st.info("""
    **Global Signal Interpretation:**
    - 📈 **S&P500 UP** → Indian & global markets often follow next day
    - 📉 **VIX HIGH** → Fear in market, expect volatility
    - 🛢️ **Crude Oil UP** → Inflationary pressure, may hurt oil-importing markets
    - 💰 **Gold UP** → Risk-off sentiment (investors moving to safety)
    - 📊 **Bond Yield UP** → Growth/tech stocks may face pressure
    - 💵 **USD/INR UP** → Rupee weakening, FII outflow risk
    """)


# ─────────────────────────────────────────────
#  TAB 5: SECTOR ROTATION
# ─────────────────────────────────────────────
def render_sector_tab(sectors):
    """Sector rotation bar chart + table"""
    st.subheader("🔄 Sector Rotation Analysis")

    if not sectors:
        st.info("Sector data loading...")
        return

    sec_df = pd.DataFrame(sectors).T.reset_index()
    sec_df.columns = ["Sector", "1W Return %", "1M Return %", "Momentum"]
    sec_df = sec_df.sort_values("1W Return %", ascending=False)

    # Bar chart
    fig_sec = px.bar(
        sec_df, x="Sector", y="1W Return %",
        color="1W Return %",
        color_continuous_scale=["#ff4d6d", "#f0c040", "#00e5a0"],
        text="1W Return %"
    )
    fig_sec.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_sec.add_hline(y=0, line_color="#4a7090")
    fig_sec.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a1520",
        font=dict(color="#e0f0ff"), height=400,
        showlegend=False, coloraxis_showscale=False
    )
    st.plotly_chart(fig_sec, use_container_width=True)

    # Table
    st.dataframe(sec_df, use_container_width=True, hide_index=True)

    st.info("""
    **Sector Rotation Tips:**
    - 🟢 **Strong sector** = money flow ત્યાં છે, sector stocks ઉપર જઈ શકે
    - 🔴 **Weak sector** = institutional selling, avoid or short
    - Analyze stock ના sector ની performance ચકાસો
    """)


# ─────────────────────────────────────────────
#  TAB 6: FUNDAMENTALS
# ─────────────────────────────────────────────
def render_fundamentals_tab(fundamentals, current_price):
    """P/E, Revenue, Analyst data + 52-week range"""
    st.subheader("📋 Fundamental Data")

    if not fundamentals:
        st.warning("Fundamental data not available for this ticker.")
        return

    def fmt(v):
        """Format large numbers nicely"""
        if isinstance(v, float):
            if v > 1e9:   return f"${v/1e9:.1f}B"
            if v > 1e6:   return f"${v/1e6:.1f}M"
            if abs(v) < 1: return f"{v*100:.1f}%"
            return f"{v:.2f}"
        if v is None:   return "N/A"
        return str(v)

    # Next earnings date warning
    next_earn = fundamentals.get("Next Earnings")
    if next_earn and isinstance(next_earn, (int, float)):
        earn_date = datetime.fromtimestamp(next_earn).strftime("%d %b %Y")
        st.warning(f"📅 **Next Earnings Date: {earn_date}** — Expect high volatility around this date!")

    # Key metric cards (3 columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("P/E Ratio",      fmt(fundamentals.get("PE Ratio")))
        st.metric("Forward P/E",    fmt(fundamentals.get("Forward PE")))
        st.metric("P/B Ratio",      fmt(fundamentals.get("PB Ratio")))
    with c2:
        st.metric("Revenue Growth", fmt(fundamentals.get("Revenue Growth")))
        st.metric("Earnings Growth",fmt(fundamentals.get("Earnings Growth")))
        st.metric("Profit Margin",  fmt(fundamentals.get("Profit Margin")))
    with c3:
        st.metric("Debt/Equity",    fmt(fundamentals.get("Debt/Equity")))
        st.metric("ROE",            fmt(fundamentals.get("ROE")))
        st.metric("Analyst Target", fmt(fundamentals.get("Analyst Target")))

    st.divider()

    # Analyst recommendation
    rec       = fundamentals.get("Recommendation", "N/A")
    rec_lower = str(rec).lower()
    if "buy" in rec_lower or "outperform" in rec_lower:
        st.success(f"**Analyst Recommendation: {str(rec).upper()}**")
    elif "sell" in rec_lower or "underperform" in rec_lower:
        st.error(f"**Analyst Recommendation: {str(rec).upper()}**")
    else:
        st.info(f"**Analyst Recommendation: {str(rec).upper()}**")

    # Sector & Industry
    c1, c2 = st.columns(2)
    c1.info(f"**Sector:** {fundamentals.get('Sector', 'N/A')}")
    c2.info(f"**Industry:** {fundamentals.get('Industry', 'N/A')}")

    # 52-Week range progress bar
    h52 = fundamentals.get("52W High")
    l52 = fundamentals.get("52W Low")
    if h52 and l52 and h52 != "N/A" and isinstance(h52, float):
        range_pct = (current_price - l52) / (h52 - l52) * 100 if (h52 - l52) > 0 else 0
        st.subheader("📊 52-Week Price Range")
        st.progress(int(min(max(range_pct, 0), 100)))
        st.caption(
            f"52W Low: ${l52:.2f}  |  "
            f"Current: **${current_price:.2f}** ({range_pct:.0f}% of range)  |  "
            f"52W High: ${h52:.2f}"
        )


# ─────────────────────────────────────────────
#  ML MODEL DETAILS EXPANDER
# ─────────────────────────────────────────────
def render_ml_details(ml):
    """Expandable section with individual model details"""
    with st.expander("🤖 ML Model Details — Click to Expand"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Random Forest Accuracy",  f"{ml['rf_accuracy']*100:.1f}%",  f"P(UP)={ml['rf_p']*100:.0f}%")
        c2.metric("XGBoost Accuracy",        f"{ml['xgb_accuracy']*100:.1f}%", f"P(UP)={ml['xgb_p']*100:.0f}%")
        c3.metric("Gradient Boost Accuracy", f"{ml['gbm_accuracy']*100:.1f}%", f"P(UP)={ml['gbm_p']*100:.0f}%")

        st.info(
            f"**Weighted Ensemble:**  "
            f"RF × 0.35 + XGB × 0.40 + GBM × 0.25  =  "
            f"**{ml['next_day_proba']*100:.1f}% UP probability**"
        )


# ─────────────────────────────────────────────
#  MAIN FUNCTION — App Entry Point
# ─────────────────────────────────────────────
def main():
    # 1. Page setup
    setup_page()

    # 2. Sidebar → get ticker + button
    ticker, analyze_btn = render_sidebar()

    # 3. Welcome screen if not analyzed yet
    if not analyze_btn:
        render_welcome()
        return

    # 4. Fetch & process data
    with st.spinner(f"📥 {ticker} નો data fetch કરીએ છીએ..."):
        df_raw = fetch_stock_data(ticker)

    if df_raw is None or df_raw.empty:
        st.error(f"❌ **{ticker}** નો data મળ્યો નહીં. finance.yahoo.com પર ticker check કરો.")
        return

    with st.spinner("⚙️ Indicators compute + Models train કરીએ છીએ..."):
        df           = add_features(df_raw.copy())
        sentiment    = analyze_sentiment(ticker)
        fundamentals = get_fundamentals(ticker)
        global_sig   = get_global_signals()
        sectors      = get_sector_rotation()
        tech         = get_tech_reasons(df)
        ml           = train_ml(df, sentiment["avg_score"], global_sig)
        forecast     = forecast_price(df)
        verdict      = compute_verdict(ml, forecast, sentiment, global_sig, fundamentals, tech)

    # 5. Header metrics
    current_price = render_header_metrics(ticker, df, forecast, ml, sentiment)
    st.divider()

    # 6. Verdict
    render_verdict(verdict)
    st.divider()

    # 7. Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price & Forecast",
        "📊 Technical Indicators",
        "🗞️ News Sentiment",
        "🌍 Global Markets",
        "🔄 Sector Rotation",
        "📋 Fundamentals"
    ])

    with tab1: render_price_tab(df, forecast, ticker)
    with tab2: render_technical_tab(df, tech, ml)
    with tab3: render_sentiment_tab(sentiment)
    with tab4: render_global_tab(global_sig)
    with tab5: render_sector_tab(sectors)
    with tab6: render_fundamentals_tab(fundamentals, current_price)

    # 8. ML details expander
    st.divider()
    render_ml_details(ml)

    # 9. Footer
    st.caption("⚠️ Educational purposes only. Not financial advice. Past performance ≠ future results.")


# Entry point
if __name__ == "__main__":
    main()