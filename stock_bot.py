





import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs if installed

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI popup — saves PNG instead
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

import yfinance as yf
import ta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ── CONFIG ──────────────────────────────────
PERIOD        = "1y"   # 1 year data (faster than 2y)
FORECAST_DAYS = 30


# ── 1. FETCH DATA ────────────────────────────
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    print(f"\n📥 Fetching data for {ticker}...")
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)
        except Exception as e:
            print(f"   ⚠️  Error (attempt {attempt}/3): {e}")
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df.dropna(inplace=True)
            print(f"   ✅ {len(df)} days loaded ({df.index[0].date()} → {df.index[-1].date()})")
            return df
        else:
            wait = attempt * 8
            if attempt < 3:
                print(f"   ⏳ No data (attempt {attempt}/3). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Failed after 3 attempts.")
                print(f"      • Check ticker at https://finance.yahoo.com")
                print(f"      • Run: pip install --upgrade yfinance curl_cffi --break-system-packages")
                return pd.DataFrame()
    return pd.DataFrame()


# ── 2. TECHNICAL FEATURES ────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️  Computing technical indicators...")
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    df["SMA_20"]    = ta.trend.sma_indicator(close, 20)
    df["SMA_50"]    = ta.trend.sma_indicator(close, 50)
    df["EMA_12"]    = ta.trend.ema_indicator(close, 12)
    df["MACD"]      = ta.trend.macd(close)
    df["MACD_sig"]  = ta.trend.macd_signal(close)
    df["RSI"]       = ta.momentum.rsi(close, 14)
    df["BB_upper"]  = ta.volatility.bollinger_hband(close)
    df["BB_lower"]  = ta.volatility.bollinger_lband(close)
    df["ATR"]       = ta.volatility.average_true_range(high, low, close)
    df["OBV"]       = ta.volume.on_balance_volume(close, vol)
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Volatility"]= df["Return_1d"].rolling(20).std() * np.sqrt(252)
    df["Target"]    = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    print(f"   ✅ {len(df)} rows, {df.shape[1]} features")
    return df


# ── 3. SENTIMENT ANALYSIS ────────────────────
def analyze_sentiment(ticker: str) -> dict:
    print(f"\n🗞️  Running sentiment analysis...")
    stock = yf.Ticker(ticker)
    news_items = stock.news[:10] if stock.news else []

    headlines = [item.get("content", {}).get("title", "") or item.get("title", "")
                 for item in news_items if item]
    headlines = [h for h in headlines if h]

    if not headlines:
        headlines = [
            f"{ticker} reports quarterly earnings",
            f"Analysts update price target for {ticker}",
            f"{ticker} announces new product strategy",
            f"Market outlook for {ticker} sector",
            f"{ticker} trading volume increases",
        ]

    vader = SentimentIntensityAnalyzer()
    results = []
    for h in headlines:
        tb    = TextBlob(h).sentiment.polarity
        vd    = vader.polarity_scores(h)["compound"]
        score = (tb + vd) / 2
        label = "🟢 Positive" if score > 0.05 else ("🔴 Negative" if score < -0.05 else "🟡 Neutral")
        results.append({"headline": h[:65], "score": round(score, 3), "label": label})

    df_s = pd.DataFrame(results)
    avg  = df_s["score"].mean()
    overall = "🟢 BULLISH" if avg > 0.05 else ("🔴 BEARISH" if avg < -0.05 else "🟡 NEUTRAL")

    print(f"   Overall: {overall}  (avg={avg:.3f})")
    for _, r in df_s.iterrows():
        print(f"   {r['label']}  {r['score']:+.3f}  {r['headline']}")

    return {"dataframe": df_s, "avg_score": avg, "overall": overall}


# ── 4. ML MODELS (RF + XGBoost) ──────────────
FEATURES = ["SMA_20","SMA_50","EMA_12","MACD","MACD_sig",
            "RSI","BB_upper","BB_lower","ATR","OBV",
            "Return_1d","Return_5d","Volatility"]

def train_ml(df: pd.DataFrame, sentiment_score: float) -> dict:
    print("\n🤖 Training ML models...")
    df = df.copy()
    df["Sentiment"] = sentiment_score
    feats = FEATURES + ["Sentiment"]

    X = df[feats].values
    y = df["Target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))

    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                eval_metric="logloss", random_state=42, verbosity=0)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_acc = accuracy_score(y_te, xgb_m.predict(X_te))

    latest   = np.array(df[feats].iloc[-1]).reshape(1, -1)
    rf_p     = rf.predict_proba(latest)[0][1]
    xgb_p    = xgb_m.predict_proba(latest)[0][1]
    ens_p    = (rf_p + xgb_p) / 2
    ens_acc  = (rf_acc + xgb_acc) / 2
    direction = "📈 UP" if ens_p >= 0.5 else "📉 DOWN"

    print(f"   RF Accuracy  : {rf_acc*100:.1f}%")
    print(f"   XGB Accuracy : {xgb_acc*100:.1f}%")
    print(f"   Next-Day     : {direction}  ({ens_p*100:.1f}% confidence)")

    feat_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    return {"next_day_proba": ens_p, "direction": direction,
            "rf_accuracy": rf_acc, "xgb_accuracy": xgb_acc,
            "ens_accuracy": ens_acc, "feature_importance": feat_imp}


# ── 5. LINEAR REGRESSION FORECAST (replaces LSTM) ──
def forecast_price(df: pd.DataFrame) -> dict:
    print(f"\n📈 Forecasting next {FORECAST_DAYS} days (Linear Regression)...")
    close = df["Close"].squeeze().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close).flatten()

    X = np.arange(len(scaled)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, scaled)

    future_X = np.arange(len(scaled), len(scaled) + FORECAST_DAYS).reshape(-1, 1)
    future_scaled = model.predict(future_X).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled).flatten()
    future_prices = np.clip(future_prices, 0, None)

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=FORECAST_DAYS)
    current      = float(df["Close"].iloc[-1])
    forecast     = float(future_prices[-1])
    change_pct   = (forecast - current) / current * 100
    trend        = "📈 UPTREND" if change_pct > 0 else "📉 DOWNTREND"

    print(f"   Current Price : ${current:.2f}")
    print(f"   Forecast ({FORECAST_DAYS}d): ${forecast:.2f}  ({change_pct:+.1f}%)")
    print(f"   Trend         : {trend}")

    return {"future_dates": future_dates, "future_prices": future_prices,
            "current_price": current, "forecast_price": forecast,
            "change_pct": change_pct, "trend": trend}


# ── 6. TECHNICAL REASONS ─────────────────────
def safe_float(val, default=0.0) -> float:
    try:
        return float(val.iloc[0] if isinstance(val, pd.Series) else val)
    except:
        return default

def get_tech_reasons(df: pd.DataFrame) -> dict:
    row   = df.iloc[-1]
    close = safe_float(df["Close"].iloc[-1])
    rsi   = safe_float(row.get("RSI", 50))
    macd  = safe_float(row.get("MACD", 0))
    msig  = safe_float(row.get("MACD_sig", 0))
    sma20 = safe_float(row.get("SMA_20", 0))
    sma50 = safe_float(row.get("SMA_50", 0))
    bbu   = safe_float(row.get("BB_upper", 9e9))
    bbl   = safe_float(row.get("BB_lower", 0))
    ret1  = safe_float(row.get("Return_1d", 0))

    up, dn = [], []

    if rsi < 35:   up.append(f"RSI={rsi:.1f} → Oversold, bounce likely UP")
    elif rsi > 65: dn.append(f"RSI={rsi:.1f} → Overbought, pullback likely")

    if macd > msig: up.append("MACD above signal → Bullish momentum")
    else:           dn.append("MACD below signal → Bearish momentum")

    if close > sma20 > sma50:  up.append("Price > SMA20 > SMA50 → Strong uptrend")
    elif close < sma20 < sma50: dn.append("Price < SMA20 < SMA50 → Strong downtrend")

    if close >= bbu * 0.98:    dn.append("Near Bollinger Upper Band → Resistance")
    elif close <= bbl * 1.02:  up.append("Near Bollinger Lower Band → Support")

    if ret1 > 0.03:    up.append(f"1-day return {ret1*100:.1f}% → Momentum UP")
    elif ret1 < -0.03: dn.append(f"1-day return {ret1*100:.1f}% → Selling pressure")

    return {"up": up, "dn": dn}


# ── 7. PRINT REPORT ──────────────────────────
def print_report(ticker, sentiment, ml, forecast, tech):
    sent_df = sentiment["dataframe"]
    up_news = sent_df[sent_df["score"] > 0.05]
    dn_news = sent_df[sent_df["score"] < -0.05]
    nu_news = sent_df[sent_df["score"].between(-0.05, 0.05)]

    W = 62
    print("\n" + "╔" + "═"*W + "╗")
    print(f"║  📰  NEWS IMPACT REPORT — {ticker:<{W-27}}║")
    print("╠" + "═"*W + "╣")

    def row(icon, text, width=W):
        print(f"║  {icon}  {text:<{width-5}}║")

    print(f"║  📈  NEWS PUSHING STOCK UP:{' '*(W-27)}║")
    if up_news.empty:
        row("  ", "No significantly positive news found.")
    else:
        for _, r in up_news.iterrows():
            row("✅", f"{r['headline'][:52]}  score:{r['score']:+.2f}")

    print("╠" + "═"*W + "╣")
    print(f"║  📉  NEWS PUSHING STOCK DOWN:{' '*(W-29)}║")
    if dn_news.empty:
        row("  ", "No significantly negative news found.")
    else:
        for _, r in dn_news.iterrows():
            row("❌", f"{r['headline'][:52]}  score:{r['score']:+.2f}")

    if not nu_news.empty:
        print("╠" + "═"*W + "╣")
        print(f"║  🟡  NEUTRAL NEWS:{' '*(W-18)}║")
        for _, r in nu_news.iterrows():
            row("◈ ", r["headline"][:56])

    print("╠" + "═"*W + "╣")
    print(f"║  🔧  TECHNICAL REASONS:{' '*(W-23)}║")
    for r in tech["up"]: row("▲ ", r[:56])
    for r in tech["dn"]: row("▼ ", r[:56])
    if not tech["up"] and not tech["dn"]: row("  ", "Mixed technical signals.")

    print("╠" + "═"*W + "╣")
    chg = forecast["change_pct"]
    row("📈" if chg > 0 else "📉",
        f"30-DAY FORECAST: ${forecast['current_price']:.2f} → ${forecast['forecast_price']:.2f}  ({chg:+.1f}%)")

    # Signals
    s_ml   = +1 if ml["next_day_proba"] >= 0.55 else (-1 if ml["next_day_proba"] <= 0.45 else 0)
    s_fc   = +1 if forecast["change_pct"] > 2 else (-1 if forecast["change_pct"] < -2 else 0)
    s_sent = +1 if sentiment["avg_score"] > 0.05 else (-1 if sentiment["avg_score"] < -0.05 else 0)
    total  = s_ml + s_fc + s_sent

    if total >= 2:    verdict = "🟢  STRONG BUY  — Multiple signals confirm upside"
    elif total == 1:  verdict = "🟡  WEAK BUY    — Slight upside bias"
    elif total == 0:  verdict = "⚪  HOLD/NEUTRAL — Signals are mixed"
    elif total == -1: verdict = "🟠  WEAK SELL   — Slight downside risk"
    else:             verdict = "🔴  STRONG SELL — Multiple signals confirm downside"

    sig_label = lambda s: "BUY ▲" if s==1 else ("SELL ▼" if s==-1 else "NEUTRAL")
    print("╠" + "═"*W + "╣")
    print(f"║  📊  SIGNAL BREAKDOWN:{' '*(W-22)}║")
    row("  ", f"ML Model (RF+XGB) : {sig_label(s_ml):<20} conf:{ml['next_day_proba']*100:.0f}%")
    row("  ", f"Price Forecast    : {sig_label(s_fc):<20} ({chg:+.1f}%)")
    row("  ", f"News Sentiment    : {sig_label(s_sent):<20} score:{sentiment['avg_score']:+.3f}")
    print("╠" + "═"*W + "╣")
    print(f"║  ➤  FINAL VERDICT: {verdict:<{W-18}}║")
    print("╚" + "═"*W + "╝")
    print("\n⚠️  Educational purposes only. Not financial advice.\n")


# ── 8. CHARTS ────────────────────────────────
def plot_results(df, forecast, ml, sentiment, ticker):
    print("📊 Saving chart...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#050b14")

    for ax in axes.flatten():
        ax.set_facecolor("#0a1520")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.tick_params(colors="#6a90b0")

    close = df["Close"].squeeze()

    # Plot 1: Price + Forecast
    ax = axes[0, 0]
    ax.plot(df.index[-100:], close.iloc[-100:], color="#00a8ff", lw=1.5, label="Historical")
    ax.plot(forecast["future_dates"], forecast["future_prices"],
            color="#00e5a0", lw=2, ls="--", label="Forecast")
    ax.axvline(df.index[-1], color="#f0c040", lw=1, ls=":", alpha=0.7)
    ax.set_title(f"{ticker} — Price Forecast", color="#e0f0ff", fontweight="bold")
    ax.legend(framealpha=0.2, labelcolor="#c8d8e8", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # Plot 2: RSI
    ax = axes[0, 1]
    ax.plot(df.index[-100:], df["RSI"].iloc[-100:], color="#f0c040", lw=1.5)
    ax.axhline(70, color="#ff4d6d", lw=1, ls="--", alpha=0.6)
    ax.axhline(30, color="#00e5a0", lw=1, ls="--", alpha=0.6)
    ax.set_title("RSI (14)", color="#e0f0ff", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # Plot 3: Feature Importance
    ax = axes[1, 0]
    top = ml["feature_importance"].head(8)
    ax.barh(top.index[::-1], top.values[::-1], color="#00a8ff", edgecolor="#1e3a5f")
    ax.set_title("Top ML Features", color="#e0f0ff", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8, colors="#c8d8e8")

    # Plot 4: Sentiment
    ax = axes[1, 1]
    sent_df = sentiment["dataframe"]
    colors  = ["#00e5a0" if v > 0.05 else ("#ff4d6d" if v < -0.05 else "#f0c040")
               for v in sent_df["score"]]
    labels  = [h[:28] + "..." for h in sent_df["headline"]]
    ax.barh(labels[::-1], sent_df["score"].values[::-1], color=colors[::-1], edgecolor="#1e3a5f")
    ax.axvline(0, color="#4a7090", lw=1)
    ax.set_title("News Sentiment", color="#e0f0ff", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7, colors="#c8d8e8")

    plt.suptitle(f"Stock ML Analyzer — {ticker}", color="#e0f0ff", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{ticker}_analysis.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#050b14")
    plt.close()
    print(f"   ✅ Chart saved → {out}")


# ── 9. MAIN ──────────────────────────────────
def run_analysis(ticker: str):
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        return

    df        = add_features(df)
    sentiment = analyze_sentiment(ticker)
    tech      = get_tech_reasons(df)
    ml        = train_ml(df, sentiment["avg_score"])
    forecast  = forecast_price(df)

    print_report(ticker, sentiment, ml, forecast, tech)
    plot_results(df, forecast, ml, sentiment, ticker)


def main():
    print("\n" + "★"*55)
    print("  📈  STOCK ML ANALYZER — Lightweight Terminal")
    print("  Enter ticker to analyze. Type 'exit' to quit.")
    print("★"*55)

    while True:
        print()
        ticker = input("  Ticker (e.g. TSLA, AAPL, INFY.NS): ").strip().upper()
        if ticker in ("EXIT", "QUIT", "Q"):
            print("  👋 Goodbye!\n")
            break
        if not ticker:
            print("  ⚠️  Please enter a ticker.")
            continue
        run_analysis(ticker)
        print("-"*55)
        if input("  Analyze another? (y/n): ").strip().lower() != "y":
            print("  👋 Goodbye!\n")
            break

if __name__ == "__main__":
    main()



