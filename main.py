!pip install vaderSentiment -q
!pip install TA-Lib -q

"""
Stock comparator with sentiment analysis + technical analysis (yfinance, TA-Lib)

Features:
- Fetch historical price + fundamentals from Yahoo
- Compute financial metrics (return, volatility, Sharpe, momentum, PE, dividend yield)
- Dual sentiment analysis: TextBlob + VADER
- Technical indicators: RSI, MACD, Bollinger Bands, MA Trend, ADX, Stochastic
- Weighted composite scoring → Final recommendation with confidence level
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis Packages
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except:
    TALIB_AVAILABLE = False

# ---------- Input Settings ----------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
HISTORY_YEARS = 3
RISK_FREE = 0.03
WEIGHTS = {
    "annual_return": 0.18,
    "sharpe": 0.15,
    "volatility": 0.10,
    "pe": 0.07,
    "dividend_yield": 0.07,
    "momentum": 0.10,
    "sentiment": 0.18,
    "technical": 0.15
}
# ------------------------------------

def fetch_data(ticker, period_years=HISTORY_YEARS):
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * period_years))
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
    info = tk.info
    return hist, info, tk

def annualized_return_from_series(price_series):
    if price_series.empty: return np.nan
    n_days = (price_series.index[-1] - price_series.index[0]).days
    if n_days <= 0: return np.nan
    total_return = price_series.iloc[-1] / price_series.iloc[0] - 1
    years = n_days / 365.25
    return (1 + total_return)**(1/years) - 1

def annualized_volatility(daily_returns):
    if daily_returns.dropna().empty: return np.nan
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio(annual_return, ann_vol, risk_free=RISK_FREE):
    if pd.isna(annual_return) or pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    return (annual_return - risk_free) / ann_vol

def momentum_12m(price_series):
    if price_series.empty: return np.nan
    end = price_series.index[-1]
    start_date = end - pd.DateOffset(months=12)
    try:
        start_price = price_series.loc[price_series.index >= start_date].iloc[0]
    except:
        start_price = price_series.iloc[0]
    return price_series.iloc[-1] / start_price - 1

def safe_get(info, key, default=np.nan):
    try:
        return info.get(key, default)
    except:
        return default

# ============================================================
#   SENTIMENT ANALYSIS (TextBlob + VADER)
# ============================================================

def get_comprehensive_sentiment(ticker_obj, ticker_symbol):

    sentiment = {
        'textblob_score': 0.0,
        'vader_score': 0.0,
        'combined_score': 0.0,
        'article_count': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0
    }

    if not TEXTBLOB_AVAILABLE and not VADER_AVAILABLE:
        return sentiment

    try:
        vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        news = ticker_obj.news

        if not news:
            return sentiment

        tb_scores, vd_scores = [], []

        for article in news[:15]:
            title = article.get("title", "")
            summary = article.get("summary", "")
            text = f"{title}. {summary}".strip()
            if not text: continue

            # TextBlob
            if TEXTBLOB_AVAILABLE:
                tb = TextBlob(text).sentiment.polarity
                tb_scores.append(tb)

            # VADER
            if VADER_AVAILABLE:
                vd = vader.polarity_scores(text)["compound"]
                vd_scores.append(vd)

                if vd >= 0.05:
                    sentiment["positive_count"] += 1
                elif vd <= -0.05:
                    sentiment["negative_count"] += 1
                else:
                    sentiment["neutral_count"] += 1

        if tb_scores:
            sentiment["textblob_score"] = np.mean(tb_scores)
        if vd_scores:
            sentiment["vader_score"] = np.mean(vd_scores)

        sentiment["article_count"] = len(tb_scores) or len(vd_scores)

        scores = []
        if tb_scores: scores.append(sentiment["textblob_score"])
        if vd_scores: scores.append(sentiment["vader_score"])

        if scores:
            sentiment["combined_score"] = np.mean(scores)

        return sentiment

    except Exception as e:
        print(f"Sentiment error for {ticker_symbol}: {e}")
        return sentiment

# ============================================================
#   TECHNICAL ANALYSIS (TA-Lib)
# ============================================================

def compute_technical_indicators(hist_df):

    if not TALIB_AVAILABLE or hist_df.empty or len(hist_df) < 50:
        return {key: np.nan for key in
                ["rsi","macd_signal","bb_position","ma_trend","adx","stoch","technical_score"]} | {
                "signal_strength": "N/A"}

    try:
        close = hist_df["Close"].values
        high = hist_df["High"].values
        low = hist_df["Low"].values

        rsi = talib.RSI(close, 14)[-1]
        rsi = 50 if np.isnan(rsi) else rsi

        macd, signal, _ = talib.MACD(close)
        macd_signal = 1 if macd[-1] > signal[-1] else -1

        upper, mid, lower = talib.BBANDS(close, 20)
        bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])

        ma50 = talib.SMA(close, 50)[-1]
        ma200 = talib.SMA(close, 200)[-1]
        ma_trend = 1 if ma50 > ma200 else -1

        adx = talib.ADX(high, low, close, 14)[-1]
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        stoch = stoch_k[-1]

        # ----- Build technical score -----
        rsi_score = 0.9 if rsi < 30 else 0.7 if rsi < 40 else 0.5 if rsi < 60 else 0.3
        macd_score = 0.7 if macd_signal == 1 else 0.3
        bb_score = bb_position
        ma_score = 0.7 if ma_trend == 1 else 0.3
        adx_score = min(adx / 50, 1.0) if adx > 25 else 0.3
        stoch_score = 0.8 if stoch < 20 else 0.5 if stoch < 80 else 0.2

        tech_score = (
            rsi_score * 0.25 +
            macd_score * 0.25 +
            bb_score * 0.15 +
            ma_score * 0.20 +
            adx_score * 0.10 +
            stoch_score * 0.05
        )

        strength = (
            "STRONG BUY" if tech_score >= 0.7 else
            "BUY" if tech_score >= 0.6 else
            "HOLD" if tech_score >= 0.4 else
            "WEAK SELL" if tech_score >= 0.3 else
            "SELL"
        )

        return {
            "rsi": rsi,
            "macd_signal": macd_signal,
            "bb_position": bb_position,
            "ma_trend": ma_trend,
            "adx": adx,
            "stoch": stoch,
            "technical_score": tech_score,
            "signal_strength": strength
        }

    except Exception as e:
        print("TA error:", e)
        return {key: np.nan for key in
                ["rsi","macd_signal","bb_position","ma_trend","adx","stoch","technical_score"]} | {
                "signal_strength": "N/A"}

# ============================================================
#   BUILD DATASET FOR ALL TICKERS
# ============================================================

def build_metrics(tickers):
    rows = []

    for t in tickers:
        print("Processing:", t)
        try:
            hist, info, tkobj = fetch_data(t)
        except:
            hist, info, tkobj = pd.DataFrame(), {}, None

        # Financial metrics
        if hist.empty:
            ann, vol, sr, mom = [np.nan]*4
        else:
            close = hist["Close"]
            ann = annualized_return_from_series(close)
            vol = annualized_volatility(close.pct_change())
            sr = sharpe_ratio(ann, vol)
            mom = momentum_12m(close)

        # Fundamentals
        pe = safe_get(info, "trailingPE")
        dy = safe_get(info, "dividendYield")
        name = safe_get(info, "shortName", t)
        sector = safe_get(info, "sector", "Unknown")
        cap = safe_get(info, "marketCap")
        beta = safe_get(info, "beta")

        # Sentiment
        sentiment = get_comprehensive_sentiment(tkobj, t) if tkobj else \
            {"combined_score":0, "textblob_score":0, "vader_score":0,
             "article_count":0, "positive_count":0, "negative_count":0, "neutral_count":0}

        # Technical Indicators
        tech = compute_technical_indicators(hist)

        rows.append({
            "ticker": t,
            "name": name,
            "sector": sector,
            "marketCap": cap,
            "annual_return": ann,
            "volatility": vol,
            "sharpe": sr,
            "momentum": mom,
            "pe": pe,
            "dividend_yield": dy,
            "beta": beta,
            # Sentiment
            "sentiment": sentiment["combined_score"],
            "textblob_sentiment": sentiment["textblob_score"],
            "vader_sentiment": sentiment["vader_score"],
            "news_articles": sentiment["article_count"],
            "positive_news": sentiment["positive_count"],
            "negative_news": sentiment["negative_count"],
            "neutral_news": sentiment["neutral_count"],
            # Technical
            "technical": tech["technical_score"],
            "rsi": tech["rsi"],
            "macd_signal": tech["macd_signal"],
            "bb_position": tech["bb_position"],
            "ma_trend": tech["ma_trend"],
            "adx": tech["adx"],
            "stoch": tech["stoch"],
            "tech_signal": tech["signal_strength"]
        })

    return pd.DataFrame(rows).set_index("ticker")

# ============================================================
#   NORMALIZATION & SCORING
# ============================================================

def normalize_series_for_score(s, higher_is_better=True):
    s = s.astype(float)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(0.0, index=s.index)

    lo, hi = valid.min(), valid.max()
    norm = (s - lo) / (hi - lo) if hi != lo else pd.Series(0.5, index=s.index)
    norm = norm.fillna(0.0)

    return norm if higher_is_better else 1 - norm

def score_universe(df):
    norm = pd.DataFrame(index=df.index)
    norm["annual_return"] = normalize_series_for_score(df["annual_return"])
    norm["sharpe"] = normalize_series_for_score(df["sharpe"])
    norm["volatility"] = normalize_series_for_score(df["volatility"], higher_is_better=False)
    norm["pe"] = normalize_series_for_score(df["pe"], higher_is_better=False)
    norm["dividend_yield"] = normalize_series_for_score(df["dividend_yield"])
    norm["momentum"] = normalize_series_for_score(df["momentum"])
    norm["sentiment"] = normalize_series_for_score(df["sentiment"])
    norm["technical"] = normalize_series_for_score(df["technical"])

    comp = pd.Series(0.0, index=df.index)
    for metric, w in WEIGHTS.items():
        comp += norm[metric] * w

    df2 = df.copy()
    for col in norm.columns:
        df2[f"norm_{col}"] = norm[col]

    df2["composite_score"] = comp
    return df2.sort_values("composite_score", ascending=False)

# ============================================================
#   FINAL RECOMMENDATION
# ============================================================

def generate_final_recommendation(result_df):

    if result_df.empty:
        print("No data to recommend.")
        return

    top = result_df.iloc[0]
    ticker = top.name
    score = top["composite_score"]

    # SIMPLE PRINT OUTPUT
    print("\n\n========== FINAL RECOMMENDATION ==========")
    print(f"🏆 Best Stock: {ticker} ({top['name']})")
    print(f"🎯 Composite Score: {score:.3f}")
    print(f"📈 Sector: {top['sector']}")
    print(f"👍 Technical Signal: {top['tech_signal']}")
    print(f"📰 Sentiment Score: {top['sentiment']:.3f}")
    print(f"⚡ Sharpe Ratio: {top['sharpe']:.2f}")
    print(f"📊 Annual Return: {top['annual_return']:.2%}")
    print(f"🚀 Momentum (12m): {top['momentum']:.2%}")
    print("==========================================\n")


# ============================================================
#               EXECUTION
# ============================================================

df = build_metrics(TICKERS)
result = score_universe(df)
generate_final_recommendation(result)
