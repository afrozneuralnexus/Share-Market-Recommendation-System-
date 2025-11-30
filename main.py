import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------
# RAPIDAPI FALLBACK
# -----------------------
RAPIDAPI_KEY = st.secrets["rapidapi_key"]  # add key in Streamlit secrets!

def rapidapi_price(symbol):
    try:
        url = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/" + symbol
        headers = {"X-RapidAPI-Key": RAPIDAPI_KEY}
        r = requests.get(url, headers=headers, timeout=5).json()
        price = r["body"][0]["regularMarketPrice"]
        return price
    except:
        return None

# -----------------------
# GET HISTORICAL DATA
# -----------------------
def get_data(symbol):
    try:
        df = yf.download(symbol, period="1y")
        if df.empty:
            raise Exception("Yahoo empty")
        return df
    except:
        # Fallback to RapidAPI for last close only
        price = rapidapi_price(symbol)
        if price:
            df = pd.DataFrame({"Close": [price]})
            return df
        return None

# -----------------------
# SENTIMENT ANALYSIS
# -----------------------
def get_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    polarity = TextBlob(news).sentiment.polarity
    vader_score = analyzer.polarity_scores(news)["compound"]
    return round((polarity + vader_score) / 2, 3)

# -----------------------
# TECHNICAL INDICATORS
# -----------------------
def get_technical_score(df):
    if len(df) < 50:
        return 0.5  # neutral
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    score = 0
    if df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]:
        score += 0.5
    if df["Close"].iloc[-1] > df["SMA20"].iloc[-1]:
        score += 0.5
    return score

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("📈 Stock Comparison + Sentiment + Technicals")
stocks = st.text_input("Enter 5 stock tickers (comma-separated)", "AAPL,MSFT,AMZN,GOOG,META")

if st.button("Compare"):
    symbols = [s.strip().upper() for s in stocks.split(",")]

    results = []
    for symbol in symbols:
        st.write(f"Fetching: **{symbol}** ...")
        df = get_data(symbol)
        if df is None:
            st.error(f"❌ Failed to load data for {symbol}")
            continue

        # Fake news text (replace with actual news API later)
        news_text = f"{symbol} stock is showing investor interest due to market momentum."
        sentiment = get_sentiment(news_text)
        tech_score = get_technical_score(df)

        final_score = round((sentiment + tech_score) / 2, 3)

        results.append({
            "Symbol": symbol,
            "Last Close": df["Close"].iloc[-1],
            "Sentiment": sentiment,
            "Technical Score": tech_score,
            "Final Score": final_score
        })

    if len(results) == 0:
        st.error("❌ No stock data loaded (Yahoo rate limit). Try again later.")
    else:
        df = pd.DataFrame(results)
        st.write(df)

        best = df.loc[df["Final Score"].idxmax()]
        st.success(
            f"### ✅ Recommended Stock: **{best['Symbol']}**\n"
            f"Score: {best['Final Score']}"
        )
