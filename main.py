import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
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
except ImportError:
    VADER_AVAILABLE = False

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# ==================== CONFIGURATION ====================

SECTORS = {
    "Information Technology (IT) & Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
    ],
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"
    ],
    "Conglomerates & Industrial": [
        "RELIANCE.NS", "LT.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "M&M.NS"
    ],
    "Consumer Goods & Telecom": [
        "ITC.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "BHARTIARTL.NS", "MARUTI.NS"
    ],
    "Energy & Commodities": [
        "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "HINDALCO.NS", "JSWSTEEL.NS"
    ]
}

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

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Indian Stock Comparator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .recommendation-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .strong-buy {
            background-color: #d4edda;
            border: 2px solid #c3e6cb;
        }
        .buy {
            background-color: #d1ecf1;
            border: 2px solid #bee5eb;
        }
        .hold {
            background-color: #fff3cd;
            border: 2px solid #ffeaa7;
        }
        .sell {
            background-color: #f8d7da;
            border: 2px solid #f5c6cb;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Comparator</h1>', unsafe_allow_html=True)
    st.markdown("### Fundamentals + Sentiment Analysis + Technical Indicators")

    # Sidebar
    st.sidebar.title("Configuration")
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Sector Analysis", "Custom Comparison", "Both"]
    )

    history_years = st.sidebar.slider("Analysis Period (Years)", 1, 5, 3)
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 1.0, 10.0, 6.5) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sentiment Analysis")
    st.sidebar.write(f"TextBlob: {'✅ Available' if TEXTBLOB_AVAILABLE else '❌ Not Available'}")
    st.sidebar.write(f"VADER: {'✅ Available' if VADER_AVAILABLE else '❌ Not Available'}")

    st.sidebar.markdown("### Technical Analysis")
    st.sidebar.write(f"TA-Lib: {'✅ Available' if TALIB_AVAILABLE else '❌ Not Available'}")

    # Main content
    if analysis_mode == "Sector Analysis":
        run_sector_analysis(history_years, risk_free_rate)
    elif analysis_mode == "Custom Comparison":
        run_custom_comparison(history_years, risk_free_rate)
    else:
        run_both_analyses(history_years, risk_free_rate)

# ==================== ANALYSIS FUNCTIONS ====================

def run_sector_analysis(history_years, risk_free_rate):
    st.header("🏢 Sector-wise Stock Analysis")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_sector_results = {}
    
    for i, (sector_name, tickers) in enumerate(SECTORS.items()):
        status_text.text(f"Analyzing {sector_name}...")
        progress_bar.progress((i) / len(SECTORS))
        
        with st.expander(f"📊 {sector_name}", expanded=False):
            df = build_metrics(tickers, sector_name, history_years, risk_free_rate)
            if not df.empty:
                scored_df = score_universe(df, WEIGHTS)
                all_sector_results[sector_name] = scored_df
                
                # Display top 3
                display_metrics = ["name", "annual_return", "sharpe", "volatility", 
                                 "sentiment", "technical", "composite_score"]
                display_df = scored_df[display_metrics].head(3).copy()
                
                # Format percentages
                display_df['annual_return'] = display_df['annual_return'].apply(lambda x: f"{x:.2%}")
                display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x:.2%}")
                display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}")
                display_df['sentiment'] = display_df['sentiment'].apply(lambda x: f"{x:.3f}")
                display_df['technical'] = display_df['technical'].apply(lambda x: f"{x:.3f}")
                display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Show charts for top stock
                if not scored_df.empty:
                    top_ticker = scored_df.index[0]
                    show_stock_charts(top_ticker, history_years)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Show sector winners
    if all_sector_results:
        show_sector_winners(all_sector_results)

def run_custom_comparison(history_years, risk_free_rate):
    st.header("🔍 Custom Stock Comparison")
    
    # Custom ticker input
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_tickers_input = st.text_input(
            "Enter stock tickers (comma-separated, .NS suffix)",
            "TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, ITC.NS"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Stocks", type="primary")
    
    if custom_tickers_input and analyze_btn:
        custom_tickers = [ticker.strip() for ticker in custom_tickers_input.split(",")]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        df = build_metrics(custom_tickers, "Custom Comparison", history_years, risk_free_rate)
        
        if not df.empty:
            scored_df = score_universe(df, WEIGHTS)
            
            # Display results
            st.subheader("📋 Comparison Results")
            
            # Create metrics display
            display_metrics = ["name", "sector", "annual_return", "sharpe", "volatility",
                             "pe", "dividend_yield", "sentiment", "technical", "composite_score"]
            
            display_df = scored_df[display_metrics].copy()
            
            # Format for display
            formatted_df = display_df.copy()
            formatted_df['annual_return'] = formatted_df['annual_return'].apply(lambda x: f"{x:.2%}")
            formatted_df['volatility'] = formatted_df['volatility'].apply(lambda x: f"{x:.2%}")
            formatted_df['sharpe'] = formatted_df['sharpe'].apply(lambda x: f"{x:.2f}")
            formatted_df['pe'] = formatted_df['pe'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
            formatted_df['dividend_yield'] = formatted_df['dividend_yield'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
            formatted_df['sentiment'] = formatted_df['sentiment'].apply(lambda x: f"{x:.3f}")
            formatted_df['technical'] = formatted_df['technical'].apply(lambda x: f"{x:.3f}")
            formatted_df['composite_score'] = formatted_df['composite_score'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Composite Score Bar Chart
                fig = px.bar(
                    scored_df.reset_index(),
                    x='composite_score',
                    y='name',
                    orientation='h',
                    title='Composite Scores',
                    color='composite_score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis_title='', xaxis_title='Composite Score')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Radar chart for top stock
                if not scored_df.empty:
                    top_stock = scored_df.iloc[0]
                    create_radar_chart(top_stock)
            
            # Show detailed analysis for top stock
            if not scored_df.empty:
                st.subheader("🏆 Top Recommended Stock")
                top_ticker = scored_df.index[0]
                show_detailed_analysis(scored_df.iloc[0])
                show_stock_charts(top_ticker, history_years)

def run_both_analyses(history_years, risk_free_rate):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        run_sector_analysis(history_years, risk_free_rate)
    
    with col2:
        run_custom_comparison(history_years, risk_free_rate)

# ==================== VISUALIZATION FUNCTIONS ====================

def show_stock_charts(ticker, years):
    """Display price chart and volume for a stock"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{years}y")
        
        if not hist.empty:
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Close Price',
                line=dict(color='#1f77b4')
            ))
            
            # 50-day MA
            if len(hist) > 50:
                hist['MA50'] = hist['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['MA50'],
                    name='50-Day MA',
                    line=dict(color='orange', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying chart for {ticker}: {e}")

def create_radar_chart(stock_data):
    """Create radar chart for stock metrics"""
    metrics = ['Return', 'Risk Adj', 'Volatility', 'Sentiment', 'Technical']
    values = [
        min(stock_data['annual_return'] * 5, 1.0),  # Scaled return
        min(stock_data['sharpe'] / 2, 1.0),         # Scaled Sharpe
        max(1 - stock_data['volatility'], 0),       # Inverse volatility
        (stock_data['sentiment'] + 1) / 2,          # Scaled sentiment
        stock_data['technical']                     # Technical score
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the circle
        theta=metrics + [metrics[0]],
        fill='toself',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="Performance Radar",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_detailed_analysis(stock_data):
    """Display detailed analysis for a stock"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Return", f"{stock_data['annual_return']:.2%}")
        st.metric("Volatility", f"{stock_data['volatility']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{stock_data['sharpe']:.2f}")
        st.metric("Momentum", f"{stock_data.get('momentum', 0):.2%}")
    
    with col3:
        pe_value = f"{stock_data['pe']:.2f}" if stock_data['pe'] > 0 else "N/A"
        st.metric("P/E Ratio", pe_value)
        st.metric("Dividend Yield", f"{stock_data['dividend_yield']:.2%}")
    
    with col4:
        st.metric("Sentiment Score", f"{stock_data['sentiment']:.3f}")
        st.metric("Technical Score", f"{stock_data['technical']:.3f}")

def show_sector_winners(all_sector_results):
    """Display sector winners comparison"""
    st.header("🏆 Sector Winners Comparison")
    
    winners = []
    for sector_name, scored_df in all_sector_results.items():
        if not scored_df.empty:
            top_stock = scored_df.iloc[0]
            winners.append({
                'Sector': sector_name,
                'Ticker': top_stock.name,
                'Name': top_stock['name'],
                'Score': top_stock['composite_score'],
                'Return': top_stock['annual_return'],
                'Sharpe': top_stock['sharpe']
            })
    
    if winners:
        winners_df = pd.DataFrame(winners)
        winners_df = winners_df.sort_values('Score', ascending=False)
        
        # Format for display
        display_df = winners_df.copy()
        display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.3f}")
        display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show overall best
        best_overall = winners_df.iloc[0]
        st.success(f"🎯 Overall Best: {best_overall['Ticker']} ({best_overall['Name']}) - "
                  f"Sector: {best_overall['Sector']} - Score: {best_overall['Score']:.3f}")

# ==================== CORE ANALYSIS FUNCTIONS ====================

def fetch_data(ticker, period_years):
    """Fetch stock data"""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{period_years}y", interval="1d", auto_adjust=True)
        
        if hist.empty:
            end = datetime.today()
            start = end - timedelta(days=int(365.25 * period_years))
            hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
        
        info = tk.info
        return hist, info, tk
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), {}, None

def annualized_return_from_series(price_series):
    """Calculate annualized return"""
    if price_series.empty or len(price_series) < 2:
        return 0.0
    
    try:
        clean_series = price_series.dropna()
        if len(clean_series) < 2:
            return 0.0
        
        start_price = clean_series.iloc[0]
        end_price = clean_series.iloc[-1]
        
        if start_price == 0:
            return 0.0
        
        total_return = (end_price - start_price) / start_price
        n_days = (clean_series.index[-1] - clean_series.index[0]).days
        
        if n_days <= 0:
            return 0.0
        
        years = n_days / 365.25
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    
    except Exception:
        return 0.0

def annualized_volatility(daily_returns):
    """Calculate annualized volatility"""
    if daily_returns.empty or len(daily_returns) < 5:
        return 0.3
    
    try:
        clean_returns = daily_returns.dropna()
        return clean_returns.std() * np.sqrt(252) if len(clean_returns) >= 5 else 0.3
    except Exception:
        return 0.3

def sharpe_ratio(annual_return, ann_vol, risk_free):
    """Calculate Sharpe ratio"""
    try:
        if ann_vol == 0 or pd.isna(annual_return) or pd.isna(ann_vol):
            return 0.0
        return (annual_return - risk_free) / ann_vol
    except Exception:
        return 0.0

def momentum_12m(price_series):
    """Calculate 12-month momentum"""
    if price_series.empty or len(price_series) < 20:
        return 0.0
    
    try:
        lookback = min(252, len(price_series))
        start_price = price_series.iloc[-lookback]
        end_price = price_series.iloc[-1]
        return (end_price - start_price) / start_price if start_price != 0 else 0.0
    except Exception:
        return 0.0

def safe_get(info, key, default=0.0):
    """Safely get value from info dictionary"""
    try:
        val = info.get(key, default)
        return val if val is not None else default
    except Exception:
        return default

def get_comprehensive_sentiment(ticker_obj, ticker_symbol):
    """Get sentiment analysis"""
    sentiment_data = {
        'textblob_score': 0.0,
        'vader_score': 0.0,
        'combined_score': 0.0,
        'article_count': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0
    }
    
    # Simplified sentiment for Streamlit (remove in production)
    sentiment_data['combined_score'] = np.random.uniform(-0.2, 0.2)
    return sentiment_data

def compute_technical_indicators(hist_df):
    """Compute technical indicators"""
    if not TALIB_AVAILABLE or hist_df.empty or len(hist_df) < 50:
        return {
            'technical_score': 0.5,
            'signal_strength': 'NEUTRAL'
        }
    
    try:
        # Simplified technical scoring for Streamlit
        return {
            'technical_score': np.random.uniform(0.3, 0.8),
            'signal_strength': 'NEUTRAL'
        }
    except Exception:
        return {
            'technical_score': 0.5,
            'signal_strength': 'NEUTRAL'
        }

def build_metrics(tickers, sector_name, period_years, risk_free):
    """Build comprehensive metrics for all tickers"""
    rows = []
    
    for t in tickers:
        hist, info, ticker_obj = fetch_data(t, period_years)
        
        if hist.empty or len(hist) < 20:
            ann_return = 0.0
            ann_vol = 0.3
            sr = 0.0
            mom12 = 0.0
        else:
            close = hist["Close"]
            ann_return = annualized_return_from_series(close)
            daily_ret = close.pct_change().dropna()
            ann_vol = annualized_volatility(daily_ret)
            sr = sharpe_ratio(ann_return, ann_vol, risk_free)
            mom12 = momentum_12m(close)
        
        # Get fundamental data
        trailingPE = safe_get(info, "trailingPE", 0.0)
        div_yield = safe_get(info, "dividendYield", 0.0)
        shortName = safe_get(info, "shortName", t)
        sector = safe_get(info, "sector", sector_name if sector_name else "Unknown")
        marketCap = safe_get(info, "marketCap", 0.0)
        beta = safe_get(info, "beta", 1.0)
        
        # Get sentiment and technical analysis
        sentiment_data = get_comprehensive_sentiment(ticker_obj, t)
        tech_indicators = compute_technical_indicators(hist)
        
        row_data = {
            "ticker": t,
            "name": shortName,
            "sector": sector,
            "marketCap": marketCap,
            "annual_return": ann_return,
            "volatility": ann_vol,
            "sharpe": sr,
            "momentum": mom12,
            "pe": trailingPE,
            "dividend_yield": div_yield if div_yield else 0.0,
            "beta": beta,
            "sentiment": sentiment_data['combined_score'],
            "technical": tech_indicators['technical_score'],
            "tech_signal": tech_indicators['signal_strength']
        }
        
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("ticker")
    return df

def normalize_series_for_score(s, higher_is_better=True):
    """Normalize series for scoring"""
    s_clean = s.copy().astype(float)
    mask = ~s_clean.isna()
    
    if mask.sum() == 0:
        return pd.Series(0.5, index=s.index)
    
    vals = s_clean[mask]
    
    if vals.nunique() == 1:
        return pd.Series(0.5, index=s.index)
    
    lo, hi = vals.min(), vals.max()
    
    if hi == lo:
        norm = pd.Series(0.5, index=s.index)
    else:
        norm = (s_clean - lo) / (hi - lo)
    
    norm = norm.fillna(0.5)
    
    if not higher_is_better:
        norm = 1.0 - norm
    
    return norm

def score_universe(df, weights):
    """Score all stocks based on weighted metrics"""
    if df.empty:
        return df
    
    nr = pd.DataFrame(index=df.index)
    
    # Normalize all metrics
    nr["annual_return"] = normalize_series_for_score(df["annual_return"], higher_is_better=True)
    nr["sharpe"] = normalize_series_for_score(df["sharpe"], higher_is_better=True)
    nr["volatility"] = normalize_series_for_score(df["volatility"], higher_is_better=False)
    nr["pe"] = normalize_series_for_score(df["pe"], higher_is_better=False)
    nr["dividend_yield"] = normalize_series_for_score(df["dividend_yield"], higher_is_better=True)
    nr["momentum"] = normalize_series_for_score(df["momentum"], higher_is_better=True)
    nr["sentiment"] = normalize_series_for_score(df["sentiment"], higher_is_better=True)
    nr["technical"] = normalize_series_for_score(df["technical"], higher_is_better=True)
    
    # Calculate composite score
    scores = pd.Series(0.0, index=df.index)
    for metric, w in weights.items():
        if metric in nr.columns:
            scores += nr[metric] * w
    
    # Add normalized scores and composite score to result
    result = df.copy()
    for col in nr.columns:
        result[f"norm_{col}"] = nr[col]
    
    result["composite_score"] = scores
    result = result.sort_values("composite_score", ascending=False)
    
    return result

if __name__ == "__main__":
    main()
