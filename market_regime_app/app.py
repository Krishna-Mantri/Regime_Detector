# app.py
"""
Streamlit web application for Market Regime Detection Dashboard.
"""

import datetime
import logging
import streamlit as st
import pandas as pd
import numpy as np

from services.data_service import get_stock_data
from services.model_service import run_regime_analysis
from utils.plotting import plot_regimes
from regime_model import regime_labels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(layout="wide", page_title="Market Regime Detector")
st.title("📈 Market Regime Detection Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")

sample_tickers = ["^NSEI", "^BSESN", "^NSEBANK", "^INDIAVIX", "USDINR=X", "SPY", "AAPL", "GOOGL", "AMZN"]
ticker_choice = st.sidebar.selectbox("Sample ticker", sample_tickers + ["Custom"], index=0)

if ticker_choice == "Custom":
    ticker = st.sidebar.text_input("Enter ticker", "SPY").upper().strip()
else:
    ticker = ticker_choice

start_date = st.sidebar.date_input(
    "Start Date", 
    datetime.date(2000, 1, 1),
    min_value=datetime.date(1990, 1, 1)
)

run_button = st.sidebar.button("🔍 Run Analysis", width='stretch')

# Main analysis
if run_button:
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        st.stop()
    
    # Fetch data
    with st.spinner(f"📥 Fetching data for {ticker}..."):
        try:
            df = get_stock_data(ticker, start_date)
            if df is None or df.empty:
                st.warning(f"❌ No data available for {ticker} from {start_date}.")
                st.stop()
        except ValueError as e:
            st.error(f"❌ Invalid ticker: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            logger.exception("Data fetch error")
            st.stop()

    # Run analysis
    with st.spinner(f"🔄 Analyzing {len(df)} data points..."):
        try:
            df, regimes, features, kmeans, scaler = run_regime_analysis(df)
        except Exception as e:
            st.error(f"❌ Error running regime analysis: {e}")
            logger.exception("Analysis error")
            st.stop()

    # Visualization
    try:
        fig = plot_regimes(df, regimes)
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.error(f"❌ Error creating visualization: {e}")
        import traceback
        st.error(traceback.format_exc())
        logger.exception("Plotting error")

    # Current regime badge
    try:
        current_regime_id = int(regimes[-1])
        if current_regime_id not in regime_labels:
            raise ValueError(f"Invalid regime ID: {current_regime_id}")
        
        current_label = regime_labels[current_regime_id]["label"]
        current_color = regime_labels[current_regime_id]["color"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Regime", current_label)
        with col2:
            st.markdown(
                f"<div style='color:{current_color}; font-weight:bold; font-size:24px; text-align:center;'>● {current_label}</div>",
                unsafe_allow_html=True
            )
        with col3:
            # Show current price (convert to float to ensure proper formatting)
            current_price = float(df['Close'].iloc[-1])
            st.metric("Current Price", f"{current_price:.2f}")
    except (IndexError, ValueError, KeyError) as e:
        logger.warning(f"Could not display current regime: {e}")

    # Detailed Regime Statistics
    st.subheader("📊 Detailed Regime Analysis")
    try:
        aligned_df = df.copy()
        aligned_df["regime_id"] = regimes.astype(int)
        
        # Create three columns for each regime type
        col1, col2, col3 = st.columns(3)
        
        for regime_id, col in [(0, col1), (1, col2), (2, col3)]:
            regime_data = aligned_df[aligned_df["regime_id"] == regime_id]
            regime_name = regime_labels[regime_id]["label"]
            regime_color = regime_labels[regime_id]["color"]
            
            if len(regime_data) > 0:
                with col:
                    st.markdown(f"**<span style='color:{regime_color}'>● {regime_name} Regime</span>**", unsafe_allow_html=True)
                    
                    # Statistics for this regime (convert to Python native types safely)
                    price_min = float(regime_data['Low'].min()) if hasattr(regime_data['Low'].min(), 'item') else regime_data['Low'].min()
                    price_max = float(regime_data['High'].max()) if hasattr(regime_data['High'].max(), 'item') else regime_data['High'].max()
                    price_range = f"{price_min:.2f} - {price_max:.2f}"
                    
                    # Safely extract scalar from Series
                    close_mean = regime_data['Close'].mean()
                    if hasattr(close_mean, 'iloc'):
                        close_mean = close_mean.iloc[0]
                    avg_price = f"{float(close_mean):.2f}"
                    
                    days_count = int(len(regime_data))
                    
                    pct_change_std = regime_data['Close'].pct_change().std()
                    if hasattr(pct_change_std, 'iloc'):
                        pct_change_std = pct_change_std.iloc[0]
                    volatility = float(pct_change_std * 100) if pct_change_std is not None else 0.0
                    
                    if len(regime_data) > 1:
                        first_price = regime_data['Close'].iloc[0]
                        last_price = regime_data['Close'].iloc[-1]
                        total_return = float(((last_price / first_price) - 1) * 100)
                    else:
                        total_return = 0.0
                    
                    st.metric("Days in Regime", f"{days_count} days")
                    st.metric("Avg Price", avg_price)
                    st.metric("Price Range", price_range, delta=f"{price_max - price_min:.2f}")
                    st.metric("Volatility", f"{volatility:.2f}%")
                    st.metric("Total Return", f"{total_return:.2f}%")
    except Exception as e:
        st.error(f"Error displaying detailed regime analysis: {e}")
        logger.exception("Detailed analysis error")

    # Regime distribution
    st.subheader("📊 Regime Distribution")
    try:
        regime_labels_list = [regime_labels[int(r)]["label"] for r in regimes]
        regime_series = pd.Series(regime_labels_list)
        regime_counts = regime_series.value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(regime_counts)
        with col2:
            st.dataframe(
                pd.DataFrame({
                    "Regime": regime_counts.index,
                    "Days": regime_counts.values
                }),
                width='stretch'
            )
    except Exception as e:
        st.error(f"Error displaying regime distribution: {e}")
        logger.exception("Distribution error")

    # Performance by regime
    st.subheader("📈 Performance by Regime")
    try:
        aligned_df = df.copy()
        aligned_df["regime_id"] = regimes.astype(int)
        aligned_df["returns"] = aligned_df["Close"].pct_change() * 100  # Convert to percentage

        regime_returns = (
            aligned_df
            .groupby("regime_id")["returns"]
            .agg(['mean', 'std', 'min', 'max', 'count'])
            .round(4)
        )
        
        # Rename for readability
        regime_returns.index = [regime_labels[int(i)]["label"] for i in regime_returns.index]
        regime_returns.columns = ['Avg Daily Return (%)', 'Std Dev', 'Min', 'Max', 'Days']
        
        st.dataframe(regime_returns, width='stretch')
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        logger.exception("Performance error")

    # Feature data preview
    st.subheader("📄 Feature Data Preview")
    try:
        display_features = features[['returns', 'volatility', 'rsi', 'ma50', 'ma200']].tail(10)
        st.dataframe(display_features.round(4), width='stretch')
    except Exception as e:
        logger.warning(f"Could not display feature preview: {e}")