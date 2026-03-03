# services/model_service.py
"""
Model service module for running regime analysis pipeline.
"""

from typing import Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from regime_model import build_features, classify_regimes

logger = logging.getLogger(__name__)


def run_regime_analysis(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, KMeans, StandardScaler]:
    """
    Run complete regime analysis pipeline on stock data.
    
    Pipeline:
    1. Build technical features
    2. Classify market regimes using KMeans
    3. Align original data with computed features
    
    Args:
        df: DataFrame with OHLCV data (must contain 'Close' column)
        
    Returns:
        Tuple of:
        - aligned_df: Original DataFrame aligned to feature indices
        - regimes: Array of regime labels (0=Bull, 1=Bear, 2=Sideways)
        - features: DataFrame of computed technical features
        - kmeans: Fitted KMeans model
        - scaler: Fitted StandardScaler
        
    Raises:
        KeyError: If 'Close' column not found
        ValueError: If input data is empty
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    
    try:
        # Build features from price data
        features = build_features(df)
        
        if features.empty:
            raise ValueError("Feature engineering produced empty DataFrame")
        
        # Classify regimes
        regimes, kmeans, scaler = classify_regimes(features, save_model=True)
        
        # Align original data with features (drop early rows without sufficient data)
        aligned_df = df.loc[features.index].copy()
        
        logger.info(f"Regime analysis complete: {len(regimes)} regimes classified")
        
        return aligned_df, regimes, features, kmeans, scaler
        
    except Exception as e:
        logger.error(f"Error in regime analysis: {str(e)}")
        raise