# LSTM-GRU Stock Price Prediction System

Deep learning model combining LSTM and GRU architectures for financial time series forecasting.

## Features
- Hybrid LSTM-GRU neural network architecture
- Multi-feature engineering (technical indicators, market correlation, sentiment)
- Real-time prediction with GUI interface
- Backtesting framework with performance metrics

## Tech Stack
- TensorFlow/Keras for deep learning
- yfinance for market data
- scikit-learn for preprocessing
- Tkinter for visualization

## Performance Metrics
- Directional Accuracy: 65%+
- Sharpe Ratio: [Your results]
- Feature Set: 8 engineered features including volatility, momentum, market correlation

## Usage
```python
python LSTM-GRU.py
```

Built for systematic trading research at WaverVanir International LLC.
```

### **2. Better Yet - Create a Full GitHub Repo**
Instead of a Gist, create a proper repository:
```
wavervanir-lstm-predictor/
├── README.md (detailed)
├── requirements.txt
├── lstm_gru_predictor.py (main code)
├── backtesting.py (separate module)
├── examples/
│   ├── SPY_prediction.png
│   └── backtest_results.csv
└── docs/
    └── methodology.md
