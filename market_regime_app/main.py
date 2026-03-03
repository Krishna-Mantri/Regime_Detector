# main.py
"""
Command-line interface for market regime detection analysis.
Useful for non-interactive batch processing and testing.
"""

import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from services.data_service import get_stock_data
from services.model_service import run_regime_analysis
from utils.plotting import plot_regimes
from regime_model import regime_labels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for command-line regime analysis.
    
    Fetches historical data, runs regime analysis, and displays results.
    """
    try:
        # Configuration
        ticker = "SPY"
        start_date = "2000-01-01"
        
        logger.info(f"Starting regime analysis for {ticker} from {start_date}")
        
        # Fetch data
        print(f"📥 Fetching data for {ticker}...")
        df = get_stock_data(ticker, start_date)
        print(f"✓ Fetched {len(df)} data points")
        
        # Run analysis
        print(f"🔄 Running regime analysis...")
        df, regimes, features, kmeans, scaler = run_regime_analysis(df)
        print(f"✓ Analysis complete: {len(regimes)} regimes classified")
        
        # Display results
        print("\n" + "="*60)
        print("REGIME ANALYSIS RESULTS")
        print("="*60)
        
        # Current regime
        current_regime_id = int(regimes[-1])
        current_label = regime_labels[current_regime_id]["label"]
        current_date = df.index[-1].strftime("%Y-%m-%d")
        print(f"\n📊 Current Regime (as of {current_date}): {current_label}")
        
        # Regime distribution
        print(f"\n📈 Regime Distribution:")
        regime_counts = pd.Series(regimes).value_counts().sort_index()
        for regime_id in sorted(regime_counts.index):
            label = regime_labels[int(regime_id)]["label"]
            count = regime_counts[regime_id]
            pct = (count / len(regimes)) * 100
            print(f"   {label:12} : {count:6} days ({pct:5.1f}%)")
        
        # Performance metrics
        print(f"\n💹 Performance by Regime:")
        df_temp = df.copy()
        df_temp["regime_id"] = regimes.astype(int)
        df_temp["returns"] = df_temp["Close"].pct_change() * 100
        
        regime_returns = (
            df_temp
            .groupby("regime_id")["returns"]
            .agg(['mean', 'std', 'min', 'max'])
            .round(4)
        )
        
        for regime_id in sorted(regime_returns.index):
            label = regime_labels[int(regime_id)]["label"]
            avg_return = regime_returns.loc[regime_id, 'mean']
            volatility = regime_returns.loc[regime_id, 'std']
            print(f"   {label:12} : Avg Return: {avg_return:7.4f}% | Volatility: {volatility:.4f}")
        
        # Latest features
        print(f"\n📊 Latest Technical Features:")
        latest_features = features.iloc[-1]
        print(f"   Returns    : {latest_features['returns']:.4f}")
        print(f"   Volatility : {latest_features['volatility']:.4f}")
        print(f"   RSI        : {latest_features['rsi']:.2f}")
        print(f"   MA50       : ${latest_features['ma50']:.2f}")
        print(f"   MA200      : ${latest_features['ma200']:.2f}")
        
        print("\n" + "="*60)
        
        # Display chart
        print("\n📊 Generating visualization...")
        fig = plot_regimes(df, regimes)
        fig.show()
        
        logger.info("Analysis completed successfully")
        return 0
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        logger.error(f"Missing required data: {e}")
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())