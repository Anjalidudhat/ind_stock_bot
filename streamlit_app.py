"""
=============================================================
  STOCK ML ANALYZER — Streamlit Web UI
  Run: streamlit run streamlit_app.py
  
  NOTE: stock_bot.py must be in the SAME folder as this file
=============================================================
"""

import streamlit as st
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Import all ML logic from stock_bot.py ──
from stock_bot import (
    fetch_stock_data,
    add_features,
    analyze_sentiment,
    train_ml,
    forecast_price,
    get_tech_reasons,
    safe_float,
    PERIOD,
    FORECAST_DAYS,
    FEATURES,
)

# ── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="Stock ML Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #050b14; color: #c8d8e8; }
    .main .block-container { padding-top: 1.5rem; }

    [data-testid="metric-container"] {
        background: #0a1520;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label { color: #4a7090 !important; font-size: 12px; letter-spacing: 1px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0f0ff !important; font-size: 22px; }

    .verdict-box {
        border-radius: 10px; padding: 18px 24px; margin: 12px 0;
        font-size: 18px; font-weight: 700; letter-spacing: 1px; text-align: center;
    }
    .buy  { background: #00e5a015; border: 2px solid #00e5a0; color: #00e5a0; }
    .sell { background: #ff4d6d15; border: 2px solid #ff4d6d; color: #ff4d6d; }
    .hold { background: #f0c04015; border: 2px solid #f0c040; color: #f0c040; }

    .news-card { background: #0a1520; border-radius: 8px; padding: 12px 16px; margin: 6px 0; border-left: 4px solid; }
    .news-pos { border-left-color: #00e5a0; }
    .news-neg { border-left-color: #ff4d6d; }
    .news-neu { border-left-color: #f0c040; }

    .section-title {
        font-size: 13px; letter-spacing: 2px; color: #4a7090;
        text-transform: uppercase; margin-bottom: 8px;
        padding-bottom: 4px; border-bottom: 1px solid #1e3a5f;
    }
    .reason-up   { background:#00e5a015; border:1px solid #00e5a040; border-radius:6px; padding:6px 12px; margin:4px 0; color:#00e5a0; font-size:13px; }
    .reason-down { background:#ff4d6d15; border:1px solid #ff4d6d40; border-radius:6px; padding:6px 12px; margin:4px 0; color:#ff4d6d; font-size:13px; }

    [data-testid="stSidebar"] { background: #0a1520; border-right: 1px solid #1e3a5f; }
    [data-testid="stSidebar"] * { color: #c8d8e8 !important; }

    .stTextInput input {
        background: #0d1b2a !important; border: 1px solid #1e3a5f !important;
        color: #e0f0ff !important; font-size: 18px !important;
        font-weight: 700 !important; letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }
    .stButton > button {
        background: #00a8ff !important; color: #050b14 !important;
        font-weight: 700 !important; letter-spacing: 2px !important;
        border: none !important; padding: 10px 30px !important;
        border-radius: 6px !important; width: 100% !important;
    }
    .stButton > button:hover { background: #33b8ff !important; }
    .stTabs [data-baseweb="tab"] { background: transparent; border: 1px solid #1e3a5f; color: #6a90b0; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background: #0d1b2a !important; border-color: #00a8ff !important; color: #00a8ff !important; }
    .stProgress > div > div { background: #00a8ff !important; }
    hr { border-color: #1e3a5f; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── CHART FUNCTIONS ──────────────────────────
def make_price_chart(df, forecast, ticker):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#050b14")
    ax.set_facecolor("#0a1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="#6a90b0")
    close = df["Close"].squeeze()
    ax.plot(df.index[-120:], close.iloc[-120:], color="#00a8ff", lw=1.8, label="Historical Price")
    ax.plot(forecast["future_dates"], forecast["future_prices"],
            color="#00e5a0", lw=2, ls="--", label=f"Forecast ({FORECAST_DAYS}d)")
    ax.axvline(df.index[-1], color="#f0c040", lw=1, ls=":", alpha=0.8, label="Today")
    ax.fill_between(df.index[-120:], close.iloc[-120:], alpha=0.05, color="#00a8ff")
    ax.set_title(f"{ticker} — Price + 30-Day Forecast", color="#e0f0ff", fontsize=12, fontweight="bold", pad=10)
    ax.legend(framealpha=0.15, labelcolor="#c8d8e8", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.tight_layout()
    return fig

def make_rsi_chart(df):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#050b14")
    ax.set_facecolor("#0a1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="#6a90b0")
    ax.plot(df.index[-120:], df["RSI"].iloc[-120:], color="#f0c040", lw=1.5)
    ax.axhline(70, color="#ff4d6d", lw=1, ls="--", alpha=0.7, label="Overbought (70)")
    ax.axhline(30, color="#00e5a0", lw=1, ls="--", alpha=0.7, label="Oversold (30)")
    ax.fill_between(df.index[-120:], df["RSI"].iloc[-120:], 70,
                    where=df["RSI"].iloc[-120:] > 70, alpha=0.15, color="#ff4d6d")
    ax.fill_between(df.index[-120:], df["RSI"].iloc[-120:], 30,
                    where=df["RSI"].iloc[-120:] < 30, alpha=0.15, color="#00e5a0")
    ax.set_title("RSI (14)", color="#e0f0ff", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 100)
    ax.legend(framealpha=0.15, labelcolor="#c8d8e8", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.tight_layout()
    return fig

def make_bb_chart(df):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#050b14")
    ax.set_facecolor("#0a1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="#6a90b0")
    close = df["Close"].squeeze()
    ax.plot(df.index[-120:], close.iloc[-120:], color="#00a8ff", lw=1.5, label="Close")
    ax.plot(df.index[-120:], df["BB_upper"].iloc[-120:], color="#ff4d6d", lw=1, ls="--", label="Upper Band")
    ax.plot(df.index[-120:], df["BB_lower"].iloc[-120:], color="#00e5a0", lw=1, ls="--", label="Lower Band")
    ax.fill_between(df.index[-120:], df["BB_upper"].iloc[-120:], df["BB_lower"].iloc[-120:],
                    alpha=0.04, color="#00a8ff")
    ax.set_title("Bollinger Bands", color="#e0f0ff", fontsize=11, fontweight="bold", pad=8)
    ax.legend(framealpha=0.15, labelcolor="#c8d8e8", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.tight_layout()
    return fig

def make_feature_chart(ml):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#050b14")
    ax.set_facecolor("#0a1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="#6a90b0")
    top = ml["feature_importance"].head(8)
    ax.barh(top.index[::-1], top.values[::-1], color="#00a8ff", edgecolor="#1e3a5f", alpha=0.85)
    ax.set_title("Top ML Feature Importances", color="#e0f0ff", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="y", labelsize=9, colors="#c8d8e8")
    plt.tight_layout()
    return fig

def make_sentiment_chart(sentiment):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#050b14")
    ax.set_facecolor("#0a1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
    ax.tick_params(colors="#6a90b0")
    sent_df = sentiment["dataframe"]
    colors  = ["#00e5a0" if v > 0.05 else ("#ff4d6d" if v < -0.05 else "#f0c040") for v in sent_df["score"]]
    labels  = [h[:30] + "..." if len(h) > 30 else h for h in sent_df["headline"]]
    ax.barh(labels[::-1], sent_df["score"].values[::-1], color=colors[::-1], edgecolor="#1e3a5f")
    ax.axvline(0, color="#4a7090", lw=1)
    ax.set_title("News Sentiment Scores", color="#e0f0ff", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="y", labelsize=7, colors="#c8d8e8")
    plt.tight_layout()
    return fig


# ── MAIN UI ──────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 Stock ML Analyzer")
        st.markdown("---")
        ticker_input = st.text_input(
            "Enter Ticker Symbol",
            placeholder="TSLA, AAPL, INFY.NS...",
            max_chars=10
        ).strip().upper()

        analyze_btn = st.button("🔍 ANALYZE", use_container_width=True)

        st.markdown("---")
        st.markdown("**Quick picks:**")
        cols = st.columns(2)
        quick = ["TSLA", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"]
        for i, q in enumerate(quick):
            if cols[i % 2].button(q, key=f"q_{q}", use_container_width=True):
                ticker_input = q
                analyze_btn  = True

        st.markdown("---")
        st.caption("⚠️ Educational use only.\nNot financial advice.")

    # Header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>
        <div style='width:10px;height:10px;border-radius:50%;background:#00e5a0;box-shadow:0 0 8px #00e5a0'></div>
        <span style='font-size:11px;letter-spacing:3px;color:#00e5a0;font-weight:600'>LIVE ANALYSIS</span>
    </div>
    <h1 style='color:#e0f0ff;font-size:28px;margin:0;letter-spacing:2px'>
        STOCK INTEL <span style='color:#00a8ff'>ML</span>
    </h1>
    <p style='color:#4a7090;font-size:13px;letter-spacing:1px;margin-top:4px'>
        Enter any ticker → ML prediction, news impact & price forecast
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if analyze_btn and ticker_input:
        with st.spinner(f"⏳ Analyzing {ticker_input}..."):
            progress = st.progress(0, text="Fetching stock data...")
            df_raw = fetch_stock_data(ticker_input)

            if df_raw is None or df_raw.empty:
                st.error(f"❌ Could not fetch data for **{ticker_input}**. Check ticker or try again in a minute.")
                st.info("💡 Verify at: https://finance.yahoo.com | Run: `pip install --upgrade yfinance curl_cffi`")
                return

            progress.progress(20, text="Computing technical indicators...")
            df = add_features(df_raw)

            progress.progress(40, text="Analyzing news sentiment...")
            sentiment = analyze_sentiment(ticker_input)

            progress.progress(60, text="Training ML models...")
            ml = train_ml(df, sentiment["avg_score"])

            progress.progress(80, text="Generating price forecast...")
            forecast  = forecast_price(df)
            tech      = get_tech_reasons(df)

            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

        # Stock Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {ticker_input}")
            st.caption(f"Data: {df.index[0].date()} → {df.index[-1].date()}  |  {len(df)} trading days")
        with col2:
            chg = forecast["change_pct"]
            color = "#00e5a0" if chg >= 0 else "#ff4d6d"
            arrow = "▲" if chg >= 0 else "▼"
            st.markdown(f"""
            <div style='text-align:right'>
                <div style='font-size:28px;font-weight:700;color:#e0f0ff'>${forecast['current_price']:.2f}</div>
                <div style='color:{color};font-size:14px;font-weight:600'>{arrow} {abs(chg):.1f}% (30d forecast)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Key Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        rsi_val = safe_float(df["RSI"].iloc[-1])
        m1.metric("RSI (14)",       f"{rsi_val:.1f}", delta="Oversold" if rsi_val < 35 else ("Overbought" if rsi_val > 65 else "Normal"))
        m2.metric("ML Confidence",  f"{ml['next_day_proba']*100:.0f}%", delta=ml["direction"])
        m3.metric("ML Accuracy",    f"{ml['ens_accuracy']*100:.1f}%")
        m4.metric("Sentiment",      sentiment["overall"].split()[1])
        m5.metric("30d Forecast",   f"${forecast['forecast_price']:.2f}", delta=f"{forecast['change_pct']:+.1f}%")

        st.markdown("---")

        # Verdict
        s_ml   = +1 if ml["next_day_proba"] >= 0.55 else (-1 if ml["next_day_proba"] <= 0.45 else 0)
        s_fc   = +1 if forecast["change_pct"] > 2    else (-1 if forecast["change_pct"] < -2 else 0)
        s_sent = +1 if sentiment["avg_score"] > 0.05 else (-1 if sentiment["avg_score"] < -0.05 else 0)
        total  = s_ml + s_fc + s_sent

        if total >= 2:    vt, vs, vc = "🟢  STRONG BUY",    "Multiple signals confirm upside",      "buy"
        elif total == 1:  vt, vs, vc = "🟡  WEAK BUY",      "Slight upside bias",                   "hold"
        elif total == 0:  vt, vs, vc = "⚪  HOLD / NEUTRAL", "Signals are mixed",                   "hold"
        elif total == -1: vt, vs, vc = "🟠  WEAK SELL",     "Slight downside risk",                 "sell"
        else:             vt, vs, vc = "🔴  STRONG SELL",   "Multiple signals confirm downside",    "sell"

        st.markdown(f"<div class='verdict-box {vc}'>{vt}<br><span style='font-size:13px;font-weight:400;opacity:0.8'>{vs}</span></div>", unsafe_allow_html=True)

        # Signal breakdown
        sig = lambda s: "🟢 BUY" if s==1 else ("🔴 SELL" if s==-1 else "⚪ NEUTRAL")
        sc1, sc2, sc3 = st.columns(3)
        for col, label, sig_val, detail in [
            (sc1, "ML MODEL",       s_ml,   f"{ml['next_day_proba']*100:.0f}% confidence"),
            (sc2, "PRICE FORECAST", s_fc,   f"{forecast['change_pct']:+.1f}% in 30d"),
            (sc3, "SENTIMENT",      s_sent, f"score: {sentiment['avg_score']:+.3f}"),
        ]:
            col.markdown(f"""<div style='text-align:center;padding:10px;background:#0a1520;border-radius:8px;border:1px solid #1e3a5f'>
                <div style='font-size:11px;color:#4a7090;letter-spacing:2px'>{label}</div>
                <div style='font-size:18px;font-weight:700;margin-top:4px'>{sig(sig_val)}</div>
                <div style='font-size:12px;color:#4a7090'>{detail}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 CHARTS", "📰 NEWS", "🔧 TECHNICAL", "🤖 ML"])

        with tab1:
            st.pyplot(make_price_chart(df, forecast, ticker_input))
            c1, c2 = st.columns(2)
            with c1: st.pyplot(make_rsi_chart(df))
            with c2: st.pyplot(make_bb_chart(df))

        with tab2:
            sent_df = sentiment["dataframe"]
            up_news = sent_df[sent_df["score"] > 0.05]
            dn_news = sent_df[sent_df["score"] < -0.05]
            nu_news = sent_df[sent_df["score"].between(-0.05, 0.05)]

            st.markdown("<div class='section-title'>📈 News Pushing Stock UP</div>", unsafe_allow_html=True)
            if up_news.empty:
                st.info("No significantly positive news found.")
            for _, row in up_news.iterrows():
                st.markdown(f"<div class='news-card news-pos'><b style='color:#e0f0ff'>{row['headline']}</b><br><span style='color:#00e5a0;font-size:12px'>Score: {row['score']:+.3f} | {row['label']}</span></div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-top:16px'>📉 News Pushing Stock DOWN</div>", unsafe_allow_html=True)
            if dn_news.empty:
                st.info("No significantly negative news found.")
            for _, row in dn_news.iterrows():
                st.markdown(f"<div class='news-card news-neg'><b style='color:#e0f0ff'>{row['headline']}</b><br><span style='color:#ff4d6d;font-size:12px'>Score: {row['score']:+.3f} | {row['label']}</span></div>", unsafe_allow_html=True)

            if not nu_news.empty:
                st.markdown("<div class='section-title' style='margin-top:16px'>🟡 Neutral News</div>", unsafe_allow_html=True)
                for _, row in nu_news.iterrows():
                    st.markdown(f"<div class='news-card news-neu'><b style='color:#e0f0ff'>{row['headline']}</b><br><span style='color:#f0c040;font-size:12px'>Score: {row['score']:+.3f}</span></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(make_sentiment_chart(sentiment))

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>▲ Bullish Signals</div>", unsafe_allow_html=True)
                if tech["up"]:
                    for r in tech["up"]: st.markdown(f"<div class='reason-up'>▲ {r}</div>", unsafe_allow_html=True)
                else:
                    st.info("No strong bullish signals.")
            with c2:
                st.markdown("<div class='section-title'>▼ Bearish Signals</div>", unsafe_allow_html=True)
                if tech["dn"]:
                    for r in tech["dn"]: st.markdown(f"<div class='reason-down'>▼ {r}</div>", unsafe_allow_html=True)
                else:
                    st.info("No strong bearish signals.")

            st.markdown("---")
            st.markdown("<div class='section-title'>Key Indicator Values</div>", unsafe_allow_html=True)
            row = df.iloc[-1]
            ti1,ti2,ti3,ti4 = st.columns(4)
            ti1.metric("RSI",       f"{safe_float(row.get('RSI',0)):.1f}")
            ti2.metric("MACD",      f"{safe_float(row.get('MACD',0)):.3f}")
            ti3.metric("SMA 20",    f"${safe_float(row.get('SMA_20',0)):.2f}")
            ti4.metric("SMA 50",    f"${safe_float(row.get('SMA_50',0)):.2f}")
            ti5,ti6,ti7,ti8 = st.columns(4)
            ti5.metric("BB Upper",  f"${safe_float(row.get('BB_upper',0)):.2f}")
            ti6.metric("BB Lower",  f"${safe_float(row.get('BB_lower',0)):.2f}")
            ti7.metric("ATR",       f"{safe_float(row.get('ATR',0)):.2f}")
            ti8.metric("1d Return", f"{safe_float(row.get('Return_1d',0))*100:.2f}%")

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Model Accuracies</div>", unsafe_allow_html=True)
                st.metric("Random Forest",  f"{ml['rf_accuracy']*100:.1f}%")
                st.metric("XGBoost",        f"{ml['xgb_accuracy']*100:.1f}%")
                st.metric("Ensemble (Avg)", f"{ml['ens_accuracy']*100:.1f}%")
                st.markdown("<br>", unsafe_allow_html=True)
                conf = ml["next_day_proba"]
                st.markdown(f"""<div style='background:#0a1520;border-radius:8px;padding:16px;border:1px solid #1e3a5f;text-align:center'>
                    <div style='font-size:28px;font-weight:700;color:{"#00e5a0" if conf>=0.5 else "#ff4d6d"}'>{ml["direction"]}</div>
                    <div style='font-size:13px;color:#4a7090;margin-top:4px'>Confidence: {conf*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                st.progress(float(conf))
            with c2:
                st.markdown("<div class='section-title'>Feature Importance (RF)</div>", unsafe_allow_html=True)
                st.pyplot(make_feature_chart(ml))

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#4a7090'>
            <div style='font-size:48px;margin-bottom:16px'>📊</div>
            <div style='font-size:18px;color:#6a90b0;margin-bottom:8px'>Enter a stock ticker to get started</div>
            <div style='font-size:13px'>Try: TSLA, AAPL, NVDA, INFY.NS, RELIANCE.NS</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()




