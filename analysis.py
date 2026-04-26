






import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

import yfinance as yf
import ta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

import streamlit as st

# આ file ની config import કરો
from config import (
    PERIOD, FORECAST_DAYS,
    GLOBAL_TICKERS, SECTOR_ETFS,
    ML_FEATURES
)


# ─────────────────────────────────────────────
#  1. STOCK DATA FETCH
#  Yahoo Finance પરથી historical price data લાવે
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)   # 5 મિનિટ cache — વારંવાર API call ન થાય
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    ticker: "TSLA", "RELIANCE.NS", "INFY.NS" etc.
    Returns: DataFrame with Open, High, Low, Close, Volume
    3 વખત retry કરે છે
    """
    for attempt in range(1, 4):
        try:
            df = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)
        except Exception:
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df.dropna(inplace=True)
            return df

        # Retry logic: 5s, 10s, 15s wait
        time.sleep(attempt * 5)

    return pd.DataFrame()   # Empty = failure


# ─────────────────────────────────────────────
#  2. TECHNICAL INDICATORS
#  Raw price data માં ML features ઉમેરે
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  Raw OHLCV DataFrame
    Output: Same DataFrame + 20+ technical indicator columns
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # ── Moving Averages ──────────────────────
    # SMA = Simple Moving Average (plain average)
    # EMA = Exponential MA (recent prices ને વધુ weight)
    df["SMA_20"]  = ta.trend.sma_indicator(close, 20)   # 20-day average
    df["SMA_50"]  = ta.trend.sma_indicator(close, 50)   # 50-day average
    df["SMA_200"] = ta.trend.sma_indicator(close, 200)  # 200-day (long-term trend)
    df["EMA_12"]  = ta.trend.ema_indicator(close, 12)   # Short-term EMA
    df["EMA_26"]  = ta.trend.ema_indicator(close, 26)   # Long-term EMA

    # ── Momentum Indicators ──────────────────
    # MACD = EMA(12) - EMA(26), bullish/bearish momentum
    df["MACD"]      = ta.trend.macd(close)
    df["MACD_sig"]  = ta.trend.macd_signal(close)   # 9-day EMA of MACD
    df["MACD_hist"] = ta.trend.macd_diff(close)     # MACD - Signal (histogram)

    # RSI = 0-100 scale, <30 oversold, >70 overbought
    df["RSI"]     = ta.momentum.rsi(close, 14)

    # Stochastic = Price position within High-Low range
    df["Stoch_K"] = ta.momentum.stoch(high, low, close)
    df["Stoch_D"] = ta.momentum.stoch_signal(high, low, close)

    # ── Volatility Indicators ────────────────
    # Bollinger Bands = price range based on standard deviation
    df["BB_upper"] = ta.volatility.bollinger_hband(close)   # Upper band
    df["BB_mid"]   = ta.volatility.bollinger_mavg(close)    # Middle band (SMA20)
    df["BB_lower"] = ta.volatility.bollinger_lband(close)   # Lower band
    df["BB_width"] = ta.volatility.bollinger_wband(close)   # Band width (squeeze indicator)
    df["ATR"]      = ta.volatility.average_true_range(high, low, close)  # Average True Range

    # ── Volume Indicators ────────────────────
    df["OBV"]       = ta.volume.on_balance_volume(close, vol)       # Cumulative volume trend
    df["VWAP_proxy"]= (close * vol).cumsum() / vol.cumsum()          # Volume weighted avg price
    df["Vol_MA20"]  = vol.rolling(20).mean()                         # 20-day avg volume
    df["Vol_ratio"] = vol / df["Vol_MA20"]   # >1.5 = high volume breakout

    # ── Price Returns ────────────────────────
    df["Return_1d"]  = close.pct_change(1)   # Yesterday vs today
    df["Return_3d"]  = close.pct_change(3)
    df["Return_5d"]  = close.pct_change(5)
    df["Return_10d"] = close.pct_change(10)
    df["Volatility"] = df["Return_1d"].rolling(20).std() * np.sqrt(252)  # Annualized vol

    # ── Target (ML Label) ────────────────────
    # 1 = આવતીકાલે price ઉપર જશે
    # 0 = આવતીકાલે price નીચે જશે
    df["Target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
#  3. FUNDAMENTAL DATA
#  Company ની financial health
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)  # 1 કલાક cache (fundamentals ઓછા બદલાય)
def get_fundamentals(ticker: str) -> dict:
    """
    P/E ratio, Revenue growth, Analyst rating etc.
    Yahoo Finance info API વાપરે છે
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "PE Ratio":       info.get("trailingPE", "N/A"),      # Price/Earnings
            "Forward PE":     info.get("forwardPE", "N/A"),       # Future P/E estimate
            "PB Ratio":       info.get("priceToBook", "N/A"),     # Price/Book value
            "Market Cap":     info.get("marketCap", "N/A"),       # Total market value
            "Revenue Growth": info.get("revenueGrowth", "N/A"),   # YoY revenue growth
            "Earnings Growth":info.get("earningsGrowth", "N/A"),  # YoY earnings growth
            "Profit Margin":  info.get("profitMargins", "N/A"),   # Net profit %
            "Debt/Equity":    info.get("debtToEquity", "N/A"),    # Debt level
            "ROE":            info.get("returnOnEquity", "N/A"),  # Return on equity
            "52W High":       info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low":        info.get("fiftyTwoWeekLow", "N/A"),
            "Analyst Target": info.get("targetMeanPrice", "N/A"), # Analyst price target
            "Recommendation": info.get("recommendationKey", "N/A"), # buy/sell/hold
            "Next Earnings":  info.get("earningsTimestamp", None),  # Unix timestamp
            "Sector":         info.get("sector", "N/A"),
            "Industry":       info.get("industry", "N/A"),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
#  4. NEWS SENTIMENT ANALYSIS
#  TextBlob + VADER બે methods combine કરે
# ─────────────────────────────────────────────
@st.cache_data(ttl=600)  # 10 મિનિટ cache
def analyze_sentiment(ticker: str) -> dict:
    """
    Yahoo Finance ની news headlines વાંચે
    TextBlob + VADER ની average score કાઢે
    Returns: avg_score, overall (BULLISH/BEARISH/NEUTRAL), dataframe
    """
    stock = yf.Ticker(ticker)
    news_items = stock.news[:15] if stock.news else []

    # Headlines extract કરો
    headlines = [
        item.get("content", {}).get("title", "") or item.get("title", "")
        for item in news_items if item
    ]
    headlines = [h for h in headlines if h]

    # Fallback: news ન મળે તો generic headlines
    if not headlines:
        headlines = [
            f"{ticker} reports quarterly earnings",
            f"Analysts update price target for {ticker}",
            f"{ticker} announces new product strategy",
        ]

    vader = SentimentIntensityAnalyzer()
    results = []

    for h in headlines:
        # TextBlob: simple polarity (-1 to +1)
        tb_score = TextBlob(h).sentiment.polarity

        # VADER: financial/social media text માટે better
        vd_score = vader.polarity_scores(h)["compound"]

        # Average of both
        score = (tb_score + vd_score) / 2

        label = "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
        results.append({
            "headline": h[:80],
            "score": round(score, 3),
            "label": label
        })

    df_s    = pd.DataFrame(results)
    avg     = df_s["score"].mean()
    overall = "BULLISH" if avg > 0.05 else ("BEARISH" if avg < -0.05 else "NEUTRAL")

    return {
        "dataframe":  df_s,
        "avg_score":  avg,
        "overall":    overall
    }


# ─────────────────────────────────────────────
#  5. GLOBAL MARKET SIGNALS
#  S&P500, Oil, Gold, VIX etc. નો latest change
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)  # 5 મિનિટ cache
def get_global_signals() -> dict:
    """
    દરેક global ticker નો latest price અને % change
    Returns: {"S&P 500": {"price": 5200, "change": +0.5}, ...}
    """
    signals = {}
    for name, sym in GLOBAL_TICKERS.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if not hist.empty:
                close_vals = hist["Close"].squeeze()
                latest = float(close_vals.iloc[-1])
                prev   = float(close_vals.iloc[-2]) if len(close_vals) >= 2 else latest
                change = (latest - prev) / prev * 100
                signals[name] = {"price": latest, "change": round(change, 2)}
        except Exception:
            signals[name] = {"price": "N/A", "change": 0}
    return signals


# ─────────────────────────────────────────────
#  6. SECTOR ROTATION
#  ક્યો sector strong/weak છે?
# ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_sector_rotation() -> dict:
    """
    દરેક sector ની 1-week અને 1-month return
    Returns: {"IT (India)": {"1W Return": 2.1, "1M Return": 5.3, "Momentum": "🟢 Strong"}}
    """
    sectors = {}
    for name, sym in SECTOR_ETFS.items():
        try:
            hist = yf.Ticker(sym).history(period="1mo")
            if not hist.empty:
                close_vals = hist["Close"].squeeze()
                # 1-week return (last 5 trading days)
                ret_1w  = (float(close_vals.iloc[-1]) - float(close_vals.iloc[-5])) \
                           / float(close_vals.iloc[-5]) * 100
                # 1-month return
                ret_1mo = (float(close_vals.iloc[-1]) - float(close_vals.iloc[0])) \
                           / float(close_vals.iloc[0]) * 100
                sectors[name] = {
                    "1W Return": round(ret_1w, 2),
                    "1M Return": round(ret_1mo, 2),
                    "Momentum":  "🟢 Strong" if ret_1w > 1 else ("🔴 Weak" if ret_1w < -1 else "🟡 Neutral")
                }
        except Exception:
            pass
    return sectors


# ─────────────────────────────────────────────
#  7. ML MODEL TRAINING
#  3 Models: Random Forest + XGBoost + GBM
# ─────────────────────────────────────────────
def train_ml(df: pd.DataFrame, sentiment_score: float, global_signals: dict) -> dict:
    """
    Input:
        df              → Technical features DataFrame
        sentiment_score → News sentiment avg score
        global_signals  → VIX, S&P500 change etc.

    Output:
        direction       → "📈 UP" or "📉 DOWN"
        next_day_proba  → 0.0 to 1.0 (UP probability)
        accuracies      → RF, XGB, GBM accuracy %
        feature_importance → Which feature matters most
    """
    df = df.copy()

    # Extra features: sentiment + global signals
    df["Sentiment"]   = sentiment_score
    df["VIX_signal"]  = float(global_signals.get("VIX",     {}).get("change", 0) or 0)
    df["SP500_signal"]= float(global_signals.get("S&P 500", {}).get("change", 0) or 0)

    # Final feature list (only columns that exist in df)
    feats = ML_FEATURES + ["Sentiment", "VIX_signal", "SP500_signal"]
    feats = [f for f in feats if f in df.columns]

    X = df[feats].values
    y = df["Target"].values

    # shuffle=False — time-series data order maintain કરો
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ── Model 1: Random Forest ───────────────
    # 200 decision trees, majority vote
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1          # બધા CPU cores વાપરો
    )
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))

    # ── Model 2: XGBoost ─────────────────────
    # Sequential trees, error-correction based
    xgb_m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,           # 80% data per tree
        colsample_bytree=0.8,    # 80% features per tree
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_acc = accuracy_score(y_te, xgb_m.predict(X_te))

    # ── Model 3: Gradient Boosting ───────────
    # sklearn's own boosting implementation
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    gbm.fit(X_tr, y_tr)
    gbm_acc = accuracy_score(y_te, gbm.predict(X_te))

    # ── Weighted Ensemble ────────────────────
    # XGBoost ને highest weight (સૌથી accurate)
    latest = np.array(df[feats].iloc[-1]).reshape(1, -1)
    rf_p   = rf.predict_proba(latest)[0][1]    # RF: UP probability
    xgb_p  = xgb_m.predict_proba(latest)[0][1] # XGB: UP probability
    gbm_p  = gbm.predict_proba(latest)[0][1]   # GBM: UP probability

    # Weighted average: RF=35%, XGB=40%, GBM=25%
    ens_p     = (rf_p * 0.35 + xgb_p * 0.40 + gbm_p * 0.25)
    direction = "📈 UP" if ens_p >= 0.5 else "📉 DOWN"

    # Feature importance (from Random Forest)
    feat_imp = pd.Series(
        rf.feature_importances_, index=feats
    ).sort_values(ascending=False)

    return {
        "next_day_proba":  ens_p,
        "direction":       direction,
        "rf_accuracy":     rf_acc,
        "xgb_accuracy":    xgb_acc,
        "gbm_accuracy":    gbm_acc,
        "ens_accuracy":    (rf_acc + xgb_acc + gbm_acc) / 3,
        "feature_importance": feat_imp,
        "rf_p": rf_p, "xgb_p": xgb_p, "gbm_p": gbm_p
    }


# ─────────────────────────────────────────────
#  8. 10-DAY PRICE FORECAST
#  Ridge Regression with last 60 days window
# ─────────────────────────────────────────────
def forecast_price(df: pd.DataFrame) -> dict:
    """
    Ridge Regression (LSTM ની જગ્યાએ — fast & lightweight)
    Last 60 days data use કરે better short-term fit માટે
    Returns: future_dates, future_prices, daily_table
    """
    close = df["Close"].squeeze().values.reshape(-1, 1)

    # MinMaxScaler: prices ને 0-1 range માં convert
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close).flatten()

    # Last 60 days window use કરો (better short-term fit)
    window = min(60, len(scaled))
    X = np.arange(len(scaled) - window, len(scaled)).reshape(-1, 1)
    y = scaled[-window:]

    # Ridge Regression (Linear regression + regularization)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Future dates predict
    future_X      = np.arange(len(scaled), len(scaled) + FORECAST_DAYS).reshape(-1, 1)
    future_scaled = model.predict(future_X).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled).flatten()
    future_prices = np.clip(future_prices, 0, None)  # Negative prices ન આવે

    # Business days (weekends skip)
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=FORECAST_DAYS
    )

    current      = float(df["Close"].iloc[-1])
    forecast_end = float(future_prices[-1])
    change_pct   = (forecast_end - current) / current * 100

    # Day-by-day table
    daily = []
    for i, (d, p) in enumerate(zip(future_dates, future_prices)):
        daily.append({
            "Day":            i + 1,
            "Date":           d.strftime("%d %b"),
            "Forecast Price": round(float(p), 2),
            "Change %":       round((float(p) - current) / current * 100, 2)
        })

    return {
        "future_dates":   future_dates,
        "future_prices":  future_prices,
        "current_price":  current,
        "forecast_price": forecast_end,
        "change_pct":     change_pct,
        "trend":          "UPTREND" if change_pct > 0 else "DOWNTREND",
        "daily_table":    pd.DataFrame(daily)
    }


# ─────────────────────────────────────────────
#  9. TECHNICAL REASONS
#  Human-readable bullish/bearish signals
# ─────────────────────────────────────────────
def get_tech_reasons(df: pd.DataFrame) -> dict:
    """
    Latest row ના technical values ચેક કરી
    up[] અને dn[] lists return કરે
    """
    def sf(val, default=0.0):
        """Safe float conversion"""
        try:
            return float(val.iloc[0] if isinstance(val, pd.Series) else val)
        except:
            return default

    row    = df.iloc[-1]
    close  = sf(df["Close"].iloc[-1])
    rsi    = sf(row.get("RSI", 50))
    macd   = sf(row.get("MACD", 0))
    msig   = sf(row.get("MACD_sig", 0))
    mhist  = sf(row.get("MACD_hist", 0))
    sma20  = sf(row.get("SMA_20", 0))
    sma50  = sf(row.get("SMA_50", 0))
    sma200 = sf(row.get("SMA_200", 0))
    bbu    = sf(row.get("BB_upper", 9e9))
    bbl    = sf(row.get("BB_lower", 0))
    bbw    = sf(row.get("BB_width", 0))
    vol_r  = sf(row.get("Vol_ratio", 1))
    ret1   = sf(row.get("Return_1d", 0))
    stk_k  = sf(row.get("Stoch_K", 50))

    up, dn = [], []

    # RSI Analysis
    if rsi < 30:       up.append(f"RSI={rsi:.1f} → Strongly Oversold (strong bounce expected)")
    elif rsi < 40:     up.append(f"RSI={rsi:.1f} → Oversold zone (potential reversal)")
    elif rsi > 70:     dn.append(f"RSI={rsi:.1f} → Overbought (pullback likely)")
    elif rsi > 60:     dn.append(f"RSI={rsi:.1f} → Near overbought zone")

    # MACD Analysis
    if macd > msig and mhist > 0:
        up.append("MACD above signal + positive histogram → Strong bullish momentum")
    elif macd < msig and mhist < 0:
        dn.append("MACD below signal + negative histogram → Strong bearish momentum")
    elif macd > msig:
        up.append("MACD crossed above signal → Bullish crossover")
    else:
        dn.append("MACD below signal line → Bearish momentum")

    # Moving Average Analysis
    if close > sma20 > sma50 > sma200:
        up.append("Price > 20DMA > 50DMA > 200DMA → Perfect uptrend alignment")
    elif close > sma20 > sma50:
        up.append("Price > 20DMA > 50DMA → Short & mid-term uptrend")
    elif close < sma20 < sma50 < sma200:
        dn.append("Price < 20DMA < 50DMA < 200DMA → Perfect downtrend alignment")
    elif close < sma20 < sma50:
        dn.append("Price < 20DMA < 50DMA → Downtrend confirmed")

    # 200 DMA (Long-term trend)
    if close > sma200:
        up.append(f"Price above 200DMA (₹{sma200:.1f}) → Long-term bullish structure")
    else:
        dn.append(f"Price below 200DMA (₹{sma200:.1f}) → Long-term bearish structure")

    # Bollinger Bands
    if close >= bbu * 0.98:
        dn.append("Near Bollinger Upper Band → Strong resistance, may reverse")
    elif close <= bbl * 1.02:
        up.append("Near Bollinger Lower Band → Strong support, bounce expected")
    if bbw < 0.05:
        up.append("Bollinger Band squeeze detected → Big breakout imminent!")

    # Volume Analysis
    if vol_r > 2.0:
        up.append(f"Volume {vol_r:.1f}x above average → Strong institutional interest")
    elif vol_r > 1.5:
        up.append(f"Volume {vol_r:.1f}x average → Above-average buying participation")
    elif vol_r < 0.5:
        dn.append("Very low volume → Weak conviction, move may not sustain")

    # Stochastic
    if stk_k < 20:    up.append(f"Stochastic K={stk_k:.0f} → Oversold, reversal likely")
    elif stk_k > 80:  dn.append(f"Stochastic K={stk_k:.0f} → Overbought, caution needed")

    # 1-Day Return
    if ret1 > 0.03:    up.append(f"1-day return +{ret1*100:.1f}% → Strong buying momentum")
    elif ret1 < -0.03: dn.append(f"1-day return {ret1*100:.1f}% → Selling pressure visible")

    return {"up": up, "dn": dn}


# ─────────────────────────────────────────────
#  10. FINAL VERDICT ENGINE
#  6 factors ની score combine કરી verdict આપે
# ─────────────────────────────────────────────
def compute_verdict(ml, forecast, sentiment, global_signals, fundamentals, tech) -> dict:
    """
    દરેક factor ને +1, 0, -1 score આપે
    Total based on STRONG BUY to STRONG SELL verdict
    """
    scores = {}

    # Factor 1: ML Model Signal
    scores["ML Models"] = (
        +1 if ml["next_day_proba"] >= 0.55 else
        -1 if ml["next_day_proba"] <= 0.45 else 0
    )

    # Factor 2: Price Forecast Signal
    scores["10-Day Forecast"] = (
        +1 if forecast["change_pct"] > 2 else
        -1 if forecast["change_pct"] < -2 else 0
    )

    # Factor 3: News Sentiment Signal
    scores["News Sentiment"] = (
        +1 if sentiment["avg_score"] > 0.05 else
        -1 if sentiment["avg_score"] < -0.05 else 0
    )

    # Factor 4: Technical Signal (bullish vs bearish count)
    tech_score = len(tech["up"]) - len(tech["dn"])
    scores["Technical"] = (
        +1 if tech_score >= 2 else
        -1 if tech_score <= -2 else 0
    )

    # Factor 5: Global Market Signal
    try:
        sp_chg  = float(global_signals.get("S&P 500", {}).get("change", 0) or 0)
        vix_chg = float(global_signals.get("VIX",     {}).get("change", 0) or 0)
        g_score = (
            (1 if sp_chg > 0.5 else -1 if sp_chg < -0.5 else 0) +
            (-1 if vix_chg > 5 else 1 if vix_chg < -5 else 0)
        )
        scores["Global Markets"] = +1 if g_score > 0 else (-1 if g_score < 0 else 0)
    except Exception:
        scores["Global Markets"] = 0

    # Factor 6: Fundamental Signal
    try:
        pe  = fundamentals.get("PE Ratio", "N/A")
        rec = str(fundamentals.get("Recommendation", "")).lower()
        f_score = 0
        if pe != "N/A" and isinstance(pe, (int, float)):
            if pe < 15:  f_score += 1   # Undervalued
            elif pe > 40: f_score -= 1  # Overvalued
        if "buy" in rec or "outperform" in rec:
            f_score += 1
        elif "sell" in rec or "underperform" in rec:
            f_score -= 1
        scores["Fundamentals"] = +1 if f_score > 0 else (-1 if f_score < 0 else 0)
    except Exception:
        scores["Fundamentals"] = 0

    # Final verdict
    total = sum(scores.values())

    if total >= 4:         verdict = "🟢 STRONG BUY"
    elif total == 3:       verdict = "🟢 BUY"
    elif total in [1, 2]:  verdict = "🟡 WEAK BUY"
    elif total == 0:       verdict = "⚪ HOLD / NEUTRAL"
    elif total in [-1,-2]: verdict = "🟠 WEAK SELL"
    elif total == -3:      verdict = "🔴 SELL"
    else:                  verdict = "🔴 STRONG SELL"

    confidence = abs(total) / len(scores) * 100

    return {
        "verdict":    verdict,
        "total":      total,
        "scores":     scores,
        "confidence": round(confidence, 1)
    }