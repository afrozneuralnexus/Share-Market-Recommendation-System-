import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Try TA-Lib
try:
    import talib
    TA_LIB_AVAILABLE = True
except:
    TA_LIB_AVAILABLE = False

# Fallback "ta" library
try:
    import ta
    TA_FALLBACK_AVAILABLE = True
except:
    TA_FALLBACK_AVAILABLE = False

# Sentiment Analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False


# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------
def get_sentiment(text):
    results = {}
    if TEXTBLOB_AVAILABLE:
        results["textblob"] = TextBlob(text).sentiment.polarity
    else:
        results["textblob"] = None

    if VADER_AVAILABLE:
        results["vader"] = vader.polarity_scores(text)["compound"]
    else:
        results["vader"] = None

    return results


# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
def add_indicators(df):

    if TA_LIB_AVAILABLE:
        df["RSI"] = talib.RSI(df["Close"])
        df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
        df["EMA50"] = talib.EMA(df["Close"], timeperiod=50)
        df["MACD"], df["MACD_signal"], _ = talib.MACD(df["Close"])
    elif TA_FALLBACK_AVAILABLE:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
    else:
        df["RSI"] = np.nan
        df["EMA20"] = np.nan
        df["EMA50"] = np.nan
        df["MACD"] = np.nan
        df["MACD_signal"] = np.nan

    return df


# -----------------------------
# SCORING MODEL
# -----------------------------
def score_stock(df, sentiment):
    score = 0

    try:
        rsi = df["RSI"].iloc[-1]
        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_sig = df["MACD_signal"].iloc[-1]
    except:
        return 0

    # RSI
    if rsi < 30:
        score += 2
    elif rsi < 45:
        score += 1

    # EMA crossover
    if ema20 > ema50:
        score += 2

    # MACD momentum
    if macd > macd_sig:
        score += 2

    # Sentiment (VADER)
    if sentiment["vader"] is not None:
        if sentiment["vader"] > 0.3:
            score += 2
        elif sentiment["vader"] > 0:
            score += 1

    return score


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("📈 Stock Comparison & Recommendation System")
st.write("Technical + Sentiment + Scoring")

tickers = st.text_input("Enter 5 stock symbols (comma-separated):", "AAPL, MSFT, AMZN, GOOG, META")
news_text = st.text_area("Paste news or analysis text for sentiment:", "")

if st.button("Compare Stocks"):
    tickers = [t.strip().upper() for t in tickers.split(",")][:5]

    results = []

    for t in tickers:
        try:
            df = yf.download(t, period="6mo", interval="1d", progress=False)

            if df.empty:
                st.warning(f"⚠ No data for {t}. Skipping (possibly rate limit).")
                continue

            df = add_indicators(df)
            sentiment = get_sentiment(news_text if news_text else t)
            score = score_stock(df, sentiment)

            results.append({
                "Ticker": t,
                "RSI": round(df["RSI"].iloc[-1], 2),
                "EMA20": round(df["EMA20"].iloc[-1], 2),
                "EMA50": round(df["EMA50"].iloc[-1], 2),
                "MACD": round(df["MACD"].iloc[-1], 2),
                "Sentiment (VADER)": sentiment["vader"],
                "Final Score": score
            })

        except Exception as e:
            st.error(f"❌ Error fetching {t}: {e}")

    # Prevent crash if all tickers fail
    if not results:
        st.error("❌ No stock data could be loaded (Yahoo rate limit). Try again in 30–60 seconds.")
        st.stop()

    df_results = pd.DataFrame(results)
    st.subheader("📊 Comparison Table")
    st.dataframe(df_results)

    best = df_results.sort_values("Final Score", ascending=False).iloc[0]

    st.success(f"🏆 **Best Stock to Buy: {best['Ticker']}**")
    st.write("### Why?")
    st.write(f"- RSI → `{best['RSI']}`")
    st.write(f"- EMA(20) `{best['EMA20']}` > EMA(50) `{best['EMA50']}`")
    st.write(f"- MACD → `{best['MACD']}`")
    st.write(f"- Sentiment → `{best['Sentiment (VADER)']}`")

