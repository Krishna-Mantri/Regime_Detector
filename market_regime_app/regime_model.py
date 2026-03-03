# regime_model.py
"""
Market regime detection model using KMeans clustering.
Identifies bull, bear, and sideways market regimes based on technical features.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

regime_labels: Dict[int, Dict[str, str]] = {
    0: {'label': 'Bull', 'color': 'green'},
    1: {'label': 'Bear', 'color': 'red'},
    2: {'label': 'Sideways', 'color': 'blue'}
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features for regime analysis.
    
    Features computed:
    - returns: Daily percentage change
    - volatility: 21-day rolling standard deviation
    - ma50: 50-day moving average
    - ma200: 200-day moving average
    - rsi: 14-period RSI indicator
    
    Args:
        df: DataFrame with 'Close' column
        
    Returns:
        DataFrame with computed features, NaN values removed
        
    Raises:
        KeyError: If 'Close' column not found in input DataFrame
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    
    features = pd.DataFrame(index=df.index)

    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(21).std()
    features['ma50'] = df['Close'].rolling(50).mean()
    features['ma200'] = df['Close'].rolling(200).mean()

    # RSI calculation with division by zero protection
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()

    # Avoid division by zero
    rs = np.where(
        roll_down != 0,
        roll_up / roll_down,
        0
    )
    
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi'] = features['rsi'].fillna(50)  # Fill NaN RSI with neutral value

    return features.dropna()


def classify_regimes(
    features: pd.DataFrame, 
    save_model: bool = True
) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    """
    Classify market regimes using KMeans clustering.
    
    Clusters are labeled as:
    - 0: Bull (highest returns)
    - 1: Bear (lowest returns)  
    - 2: Sideways (middle returns)
    
    Args:
        features: DataFrame with 'returns', 'volatility', 'rsi' columns
        save_model: Whether to save the trained KMeans model
        
    Returns:
        Tuple of:
        - regimes: Array of regime labels (0, 1, or 2) for each sample
        - kmeans: Trained KMeans model
        - scaler: Fitted StandardScaler for feature normalization
        
    Raises:
        KeyError: If required features are missing
        ValueError: If features DataFrame is empty
    """
    if features is None or features.empty:
        raise ValueError("Features DataFrame is empty or None")
    
    required_cols = ['returns', 'volatility', 'rsi']
    missing_cols = set(required_cols) - set(features.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Extract and normalize features
    X = features[required_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans with consistent random state for reproducibility
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X_scaled)

    # Map clusters to regime labels based on average returns
    cluster_stats = pd.DataFrame({
        'returns': features['returns'].groupby(clusters).mean(),
        'volatility': features['volatility'].groupby(clusters).mean()
    })

    # Identify regimes
    bull = cluster_stats['returns'].idxmax()
    bear = cluster_stats['returns'].idxmin()
    sideways = [i for i in cluster_stats.index if i not in [bull, bear]][0]

    mapping = {bull: 0, bear: 1, sideways: 2}
    regimes = np.array([mapping[c] for c in clusters], dtype=int)

    # Save model if requested
    if save_model:
        try:
            os.makedirs("models", exist_ok=True)
            joblib.dump(kmeans, "models/kmeans_model.pkl")
            joblib.dump(scaler, "models/scaler_model.pkl")
            logger.info("Models saved to models/ directory")
        except Exception as e:
            logger.warning(f"Failed to save models: {e}")

    return regimes, kmeans, scaler