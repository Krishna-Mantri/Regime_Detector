# Market Regime Detector Application

This is the main application package for market regime detection and analysis.

## Package Overview

### Core Modules

- **`regime_model.py`** - Machine learning model and feature engineering
  - `build_features()`: Computes technical indicators (returns, volatility, RSI, MAs)
  - `classify_regimes()`: Applies KMeans clustering to identify regimes

- **`services/data_service.py`** - Data layer with caching
  - `get_stock_data()`: Fetches historical OHLCV data via yfinance

- **`services/model_service.py`** - Service layer
  - `run_regime_analysis()`: Orchestrates complete analysis pipeline

- **`utils/plotting.py`** - Visualization utilities
  - `plot_regimes()`: Creates interactive Plotly charts with regime shading

### Entry Points

- **`app.py`** - Streamlit web dashboard
  - Interactive ticker selection with samples
  - Real-time analysis and visualization
  - Performance analytics by regime
  - Technical feature inspection

- **`main.py`** - Command-line interface
  - Batch processing capabilities
  - Detailed text output of analysis
  - Useful for automation and integration

## Quick Start

### Installation
```bash
cd market_regime_app
pip install -r requirements.txt
```

### Run Web Dashboard
```bash
streamlit run app.py
```

### Run as CLI Tool
```bash
python main.py
```

### Use as Module
```python
from services.data_service import get_stock_data
from services.model_service import run_regime_analysis
from utils.plotting import plot_regimes

df = get_stock_data("SPY", "2020-01-01")
df, regimes, features, kmeans, scaler = run_regime_analysis(df)
fig = plot_regimes(df, regimes)
fig.show()
```

## Key Improvements in This Version

✅ **Type Hints**: Full type annotations throughout  
✅ **Error Handling**: Comprehensive exception handling with logging  
✅ **Input Validation**: Validates all inputs before processing  
✅ **Documentation**: Detailed docstrings for all functions  
✅ **Performance**: Data caching and vectorized numpy operations  
✅ **Robustness**: Division by zero protection in RSI calculation  
✅ **Scalability**: Modular design for easy extension  

## Market Regimes

The KMeans model identifies three regimes:

| Regime | Label | Color | Characteristics |
|--------|-------|-------|-----------------|
| 0 | Bull | Green | High average returns, lower volatility |
| 1 | Bear | Red | Low/negative average returns |
| 2 | Sideways | Blue | Medium returns, moderate volatility |

## Configuration

Adjust these parameters in `regime_model.py`:
- **KMeans clusters**: 3 (Bull, Bear, Sideways)
- **RSI period**: 14 days
- **Volatility window**: 21 days
- **Moving averages**: 50 and 200 days
- **Random state**: 42 (reproducibility)

## Performance Tips

- Data is cached for 1 hour by default
- Uses vectorized numpy operations
- KMeans configured with n_init=10, max_iter=300
- Scaler is now saved for consistency across predictions

## Troubleshooting

**"No data found for ticker"**
- Verify ticker symbol is valid (e.g., SPY, AAPL)
- Check date range is valid

**Import errors**
- Ensure you're in the market_regime_app directory
- Check all __init__.py files are present

**Memory issues**
- Consider using date ranges instead of fetching all historical data
- Implement chunking for multi-year datasets

## Version

**Version**: 1.0.0  
**Last Updated**: March 3, 2026  
**Python**: 3.8+
