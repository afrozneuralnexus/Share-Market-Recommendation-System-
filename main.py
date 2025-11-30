import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import yfinance as yf
from datetime import datetime
import io

# ------------------------------------------------------------------------------------
# SECTOR STOCK LIST
# ------------------------------------------------------------------------------------
SECTORS = {
    "Information Technology (IT) & Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
    ],
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"
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

# ------------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# MAIN APP CLASS
# ------------------------------------------------------------------------------------
class DataAnalysisApp:
    def __init__(self):
        self.df = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

    # ----------------------------------------------------------------------
    # MAIN LAYOUT
    # ----------------------------------------------------------------------
    def main(self):
        st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)

        self.render_sidebar()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Data Explorer", 
            "📈 Visualization", 
            "🤖 ML Modeling", 
            "📋 Statistics", 
            "⚙️ Settings"
        ])

        with tab1: self.data_explorer_tab()
        with tab2: self.visualization_tab()
        with tab3: self.ml_modeling_tab()
        with tab4: self.statistics_tab()
        with tab5: self.settings_tab()

    # ----------------------------------------------------------------------
    # SIDEBAR WITH NEW SECTOR LOADER
    # ----------------------------------------------------------------------
    def render_sidebar(self):
        st.sidebar.title("Navigation")

        st.sidebar.header("📁 Data Management")

        upload_option = st.sidebar.radio(
            "Choose data source:",
            ["Upload CSV", "Use Sample Data", "Load Sector Stocks"]
        )

        # Upload CSV
        if upload_option == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    self.df = pd.read_csv(uploaded_file)
                    st.session_state.data_loaded = True
                    st.sidebar.success("CSV Loaded Successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {e}")

        # Sample Data
        elif upload_option == "Use Sample Data":
            sample_option = st.sidebar.selectbox(
                "Choose Sample Dataset:",
                ["Iris Classification", "Sales Data", "Random Regression"]
            )

            if st.sidebar.button("Generate Sample Data"):
                self.generate_sample_data(sample_option)

        # NEW: Load Sector Stocks
        elif upload_option == "Load Sector Stocks":
            sector = st.sidebar.selectbox("Select NSE Sector:", list(SECTORS.keys()))

            if st.sidebar.button("Fetch Sector Stock Data"):
                try:
                    tickers = SECTORS[sector]
                    dfs = []

                    with st.spinner("📡 Fetching Stock Data for Sector..."):
                        for t in tickers:
                            data = yf.download(t, period="1y", interval="1d")
                            if not data.empty:
                                data["Ticker"] = t
                                dfs.append(data)

                    if dfs:
                        self.df = pd.concat(dfs).reset_index()
                        st.session_state.data_loaded = True
                        st.sidebar.success(f"Loaded {len(dfs)} stocks from {sector}.")
                    else:
                        st.sidebar.error("No data available.")

                except Exception as e:
                    st.sidebar.error(f"Error fetching sector data: {e}")

        # Data info
        if st.session_state.data_loaded and self.df is not None:
            st.sidebar.header("📊 Data Info")
            st.sidebar.write(f"Shape: {self.df.shape}")
            st.sidebar.write(f"Columns: {len(self.df.columns)}")
            st.sidebar.write(f"Memory: {self.df.memory_usage().sum() / 1024**2:.2f} MB")

    # ----------------------------------------------------------------------
    # SAMPLE DATA GENERATOR
    # ----------------------------------------------------------------------
    def generate_sample_data(self, dataset_type):
        if dataset_type == "Iris Classification":
            from sklearn.datasets import load_iris
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df["target"] = iris.target

        elif dataset_type == "Sales Data":
            np.random.seed(42)
            dates = pd.date_range("2023-01-01", periods=1000)
            self.df = pd.DataFrame({
                "date": dates,
                "product": np.random.choice(["A", "B", "C"], 1000),
                "region": np.random.choice(["North", "South", "East", "West"], 1000),
                "sales": np.random.normal(1000, 200, 1000),
                "quantity": np.random.randint(1, 50, 1000)
            })
            self.df["revenue"] = self.df["sales"] * self.df["quantity"]

        else:
            X, y = make_regression(n_samples=1000, n_features=4, noise=0.2)
            self.df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(4)])
            self.df["target"] = y

        st.session_state.data_loaded = True

    # ----------------------------------------------------------------------
    # TAB 1: DATA EXPLORER
    # ----------------------------------------------------------------------
    def data_explorer_tab(self):
        st.header("🔍 Data Explorer")

        if not st.session_state.data_loaded:
            st.info("Upload or Load Data from Sidebar")
            return

        st.subheader("Data Preview")
        st.dataframe(self.df.head(50), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            st.dataframe(self.df.dtypes)

        with col2:
            st.subheader("Missing Values")
            st.dataframe(self.df.isnull().sum())

    # ----------------------------------------------------------------------
    # TAB 2: VISUALIZATION
    # ----------------------------------------------------------------------
    def visualization_tab(self):
        st.header("📈 Visualization")

        if not st.session_state.data_loaded:
            st.info("Load data first.")
            return

        viz_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Histogram", "Bar"])

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)

        if viz_type == "Scatter":
            fig = px.scatter(self.df, x=x_col, y=y_col)
        elif viz_type == "Line":
            fig = px.line(self.df, x=x_col, y=y_col)
        elif viz_type == "Histogram":
            fig = px.histogram(self.df, x=x_col)
        else:
            fig = px.bar(self.df, x=x_col, y=y_col)

        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 3: MACHINE LEARNING
    # ----------------------------------------------------------------------
    def ml_modeling_tab(self):
        st.header("🤖 Machine Learning Modeling")

        if not st.session_state.data_loaded:
            st.info("Load data first.")
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return

        model_type = st.radio("Model Type", ["Classification", "Regression"])
        target = st.selectbox("Target Variable", numeric_cols)
        features = st.multiselect("Features", [c for c in numeric_cols if c != target])

        if not features:
            st.warning("Select at least one feature.")
            return

        if st.button("Train Model"):
            X = self.df[features]
            y = self.df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            if model_type == "Classification":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                st.metric("Accuracy", f"{acc:.3f}")

            else:
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                mse = mean_squared_error(y_test, model.predict(X_test))
                st.metric("RMSE", f"{np.sqrt(mse):.3f}")

    # ----------------------------------------------------------------------
    # TAB 4: STATISTICS
    # ----------------------------------------------------------------------
    def statistics_tab(self):
        st.header("📋 Statistics")

        if not st.session_state.data_loaded:
            st.info("Load data first.")
            return

        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe())

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        st.subheader("Correlation Heatmap")

        fig = px.imshow(self.df[numeric_cols].corr(), text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 5: SETTINGS
    # ----------------------------------------------------------------------
    def settings_tab(self):
        st.header("⚙️ Settings")

        if st.button("Clear Data"):
            st.session_state.data_loaded = False
            self.df = None
            st.rerun()

        st.write("Streamlit Dashboard with Sector-based stock loading.")


# ------------------------------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.main()
