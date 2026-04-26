





PERIOD        = "1y"   # Stock data period (1y = 1 વર્ષ)
FORECAST_DAYS = 10     

# ── Global Market Tickers ─────────────────────
# આ stocks/indices world market ના signals માટે વપરાય છે
GLOBAL_TICKERS = {
    "S&P 500":     "^GSPC",      # US market index
    "Nasdaq":      "^IXIC",      # US tech index
    "Nifty 50":    "^NSEI",      # India market index
    "Crude Oil":   "CL=F",       # Crude oil futures
    "Gold":        "GC=F",       # Gold futures
    "VIX":         "^VIX",       # Fear index (volatility)
    "USD/INR":     "USDINR=X",   # Dollar vs Rupee
    "US 10Y Bond": "^TNX",       # US bond yield
}

# ── Sector ETFs (Sector Rotation માટે) ────────
# આ stocks respective sectors ના proxy છે
SECTOR_ETFS = {
    "IT (India)":      "INFY.NS",       # Infosys = IT sector proxy
    "Banking (India)": "HDFCBANK.NS",   # HDFC = Banking proxy
    "Pharma (India)":  "SUNPHARMA.NS",  # Sun Pharma = Pharma proxy
    "Auto (India)":    "MARUTI.NS",     # Maruti = Auto proxy
    "Energy (India)":  "RELIANCE.NS",   # Reliance = Energy proxy
    "US Tech":         "QQQ",           # Nasdaq 100 ETF
    "US Finance":      "XLF",           # US Financial sector ETF
    "US Energy":       "XLE",           # US Energy sector ETF
}

# ── ML Features List ──────────────────────────

ML_FEATURES = [
    # Moving Averages
    "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",

    # Momentum Indicators
    "MACD", "MACD_sig", "MACD_hist",
    "RSI",
    "Stoch_K", "Stoch_D",

    # Volatility
    "BB_upper", "BB_lower", "BB_width",
    "ATR",

    # Volume
    "OBV", "Vol_ratio",

    # Returns
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    "Volatility",
]

# ── Verdict Thresholds ────────────────────────
# Signal score based on total of all factors
VERDICT_MAP = {
    4:  "🟢 STRONG BUY",
    3:  "🟢 BUY",
    2:  "🟡 WEAK BUY",
    1:  "🟡 WEAK BUY",
    0:  "⚪ HOLD / NEUTRAL",
    -1: "🟠 WEAK SELL",
    -2: "🟠 WEAK SELL",
    -3: "🔴 SELL",
    -4: "🔴 STRONG SELL",
}