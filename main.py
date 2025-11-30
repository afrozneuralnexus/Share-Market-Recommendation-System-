import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    vader = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False


# -----------------------------
# SENTIMENT ANALYSIS FUNCTION
# -----------------------------
def get_sentiment(text):
    results = {}

    # TextBlob
    if TEXTBLOB_AVAILABLE:
        blob = TextBlob(text)
        results['textblob_sentiment'] = blob.sentiment.polarity
    else:
        results['textblob_sentiment'] = None

    # VADER
    if VADER_AVAILABLE:
        score = vader.polarity_scores(text)
        results['vader_sentiment'] = score['compound']
    else:
        results['vader_sentiment'] = None

    return results


# -----------------------------
# TECHNICAL ANALYSIS FEATURES
# -----------------------------
def add_technical_indicators(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
    return df


# -----------------------------
# STOCK BUY SCORE FUNCTION
# -----------------------------
def stock_score(df, sentiment_scores):
    score = 0

    # RSI rule
    if df['RSI'].iloc[-1] < 30:
        score += 2
    elif df['RSI'].iloc[-1] < 45:
        score += 1

    # EMA crossover rule
    if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
        score += 2

    # MACD rule
    if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        score += 2

    # Sentiment rule
    if sentiment_scores['vader_sentiment'] is not None:
        if sentiment_scores['vader_sentiment'] > 0.3:
            score += 2
        elif sentiment_scores['vader_sentiment'] > 0:
            score += 1

    return score


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📊 Stock Comparison & Recommendation System")
st.write("Compare 5 stocks with Technical + Sentiment Analysis")

tickers = st.text_input("Enter 5 Stock Symbols (comma separated)", "AAPL, MSFT, GOOG, AMZN, META")
news_text = st.text_area("Paste latest news or analysis text for sentiment scoring")

if st.button("Compare Stocks"):
    ticker_list = [t.strip().upper() for t in tickers.split(",")][:5]

    results = []

    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="6mo", interval="1d")
            df = add_technical_indicators(df)

            sentiment_scores = get_sentiment(news_text if news_text else ticker)

            score = stock_score(df, sentiment_scores)

            results.append({
                "Ticker": ticker,
                "RSI": round(df['RSI'].iloc[-1], 2),
                "EMA_20": round(df['EMA_20'].iloc[-1], 2),
                "EMA_50": round(df['EMA_50'].iloc[-1], 2),
                "MACD": round(df['MACD'].iloc[-1], 2),
                "Sentiment (VADER)": sentiment_scores["vader_sentiment"],
                "Final Score": score
            })
        except:
            st.error(f"Error fetching {ticker}")

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    best_stock = results_df.sort_values("Final Score", ascending=False).iloc[0]
    st.success(f"🏆 **Best Stock to Buy: {best_stock['Ticker']}**")
    st.write(f"### Why?")
    st.write(f"- RSI: {best_stock['RSI']}")
    st.write(f"- EMA Trend: {best_stock['EMA_20']} > {best_stock['EMA_50']}")
    st.write(f"- MACD: {best_stock['MACD']}")
    st.write(f"- Sentiment Score: {best_stock['Sentiment (VADER)']}")
