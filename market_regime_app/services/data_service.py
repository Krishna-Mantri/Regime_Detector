# services/data_service.py
"""
Data service module for fetching and processing market data.
"""

from typing import Optional
import logging
import yfinance as yf
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def get_stock_data(
    ticker: str, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance with caching.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY')
        start_date: Start date as string 'YYYY-MM-DD' or datetime object
        end_date: End date as string 'YYYY-MM-DD' or datetime object (default: today)
        
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        
    Raises:
        ValueError: If ticker is invalid or no data is found
        Exception: If data fetching fails due to network issues
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.upper().strip()
    
    try:
        # Suppress yfinance progress output
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date,
            progress=False
        )
        
        if df is None or df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Remove rows with NaN Close prices
        df = df[df['Close'].notna()]
        
        if df.empty:
            raise ValueError(f"No valid price data for ticker '{ticker}'")
        
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise