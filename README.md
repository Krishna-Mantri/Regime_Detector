# Market Regime Detector

A machine learning application that detects and analyzes market regimes using KMeans clustering on technical indicators. Identify Bull, Bear, and Sideways market conditions with interactive visualizations.

## 📋 Features

- **KMeans-based Regime Detection**: Automatically classifies market conditions into Bull, Bear, and Sideways regimes
- **Technical Indicators**: Computes returns, volatility, RSI, and moving averages (MA50, MA200)
- **Interactive Visualizations**: Plotly-based interactive charts with regime shading and moving averages
- **Streamlit Dashboard**: Web-based UI for easy analysis of any stock ticker
- **Performance Analytics**: Detailed performance metrics by regime
- **Command-line Interface**: Batch processing capabilities for automated analysis

## 📁 Project Structure

```
market_regime_app/
├── app.py                     # Streamlit web dashboard
├── main.py                    # Command-line interface
├── regime_model.py            # ML model & feature engineering
├── __init__.py                # Package initialization
│
├── services/
│   ├── data_service.py        # Data fetching with caching
│   ├── model_service.py       # Model pipeline orchestration
│   └── __init__.py
│
├── utils/
│   ├── plotting.py            # Plotly visualization functions
│   └── __init__.py
│
├── models/                    # Saved model artifacts
│   ├── kmeans_model.pkl       # Trained KMeans model
│   └── scaler_model.pkl       # Feature scaler
│
├── requirements.txt           # Python dependencies
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Krishna-Mantri/Regime_Detector.git
cd Regime_Detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Web Dashboard (Streamlit)

Run the interactive web application:

```bash
streamlit run market_regime_app/app.py
```

The dashboard provides:
- Ticker selection with sample stocks (SPY, QQQ, DIA, IWM, AAPL)
- Custom date range selection
- Real-time regime visualization
- Current regime indicator
- Regime distribution analytics
- Performance metrics by regime
- Technical feature inspection

### Command-Line Interface

For batch processing or automated analysis:

```bash
cd market_regime_app
python main.py
```

This outputs:
- Current market regime
- Historical regime distribution
- Average returns and volatility by regime
- Latest technical indicators

### Programmatic Usage

```python
from market_regime_app.services.data_service import get_stock_data
from market_regime_app.services.model_service import run_regime_analysis
from market_regime_app.utils.plotting import plot_regimes

# Fetch data
df = get_stock_data("SPY", "2020-01-01")

# Run analysis
df, regimes, features, kmeans, scaler = run_regime_analysis(df)

# Visualize
fig = plot_regimes(df, regimes)
fig.show()
```

## 📊 Regime Classification

The model identifies three market regimes:

| Regime | Characteristics | Color |
|--------|-----------------|-------|
| **Bull** | High average returns, lower volatility | Green 📈 |
| **Bear** | Low/negative average returns | Red 📉 |
| **Sideways** | Medium returns, moderate volatility | Blue ↔️ |

## 🎯 Technical Indicators

### Features Used for Clustering:
- **Returns**: Daily percentage price change
- **Volatility**: 21-day rolling standard deviation
- **RSI**: 14-period Relative Strength Index

### Additional Indicators Displayed:
- **MA50**: 50-day moving average
- **MA200**: 200-day moving average

## 🏗️ Architecture

The application follows a modular architecture:

1. **Data Layer** (`data_service.py`): Fetches and caches market data via yfinance
2. **Model Layer** (`regime_model.py`): Feature engineering and KMeans clustering
3. **Service Layer** (`model_service.py`): Orchestrates the analysis pipeline
4. **Presentation Layer**: 
   - `app.py`: Streamlit web UI
   - `utils/plotting.py`: Visualization components
   - `main.py`: CLI interface

## 🔍 Key Improvements

- **Type Hints**: Full type annotations throughout codebase
- **Error Handling**: Comprehensive exception handling with logging
- **Input Validation**: Validates all inputs before processing
- **Documentation**: Detailed docstrings for all functions
- **Performance**: Caching for data fetching and efficient numpy operations
- **Robustness**: Division by zero protection in RSI calculation
- **Scalability**: Modular design for easy extension

## 📦 Dependencies

- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning (KMeans, StandardScaler)
- `yfinance` - Market data fetching
- `plotly` - Interactive visualizations
- `joblib` - Model serialization

See `requirements.txt` for version specifications.

## 📈 Example Output

```
Current Regime (as of 2024-03-03): Bull
Regime Distribution:
   Bull       :    450 days (60.5%)
   Sideway    :    200 days (26.9%)
   Bear       :     94 days (12.6%)

Performance by Regime:
   Bull       : Avg Return:  0.0512% | Volatility: 0.0089
   Bear       : Avg Return: -0.0342% | Volatility: 0.0145
   Sideway    : Avg Return:  0.0025% | Volatility: 0.0102
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙋 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Author**: Krishna Mantri  
**Last Updated**: March 3, 2026  
**Version**: 1.0.0
