# Apple Stock Price Prediction 

A time-series machine learning project that predicts Apple Inc. (AAPL) stock closing prices using classical statistical models, tree-based ensembles, and deep learning (LSTM).

##  Project Overview

This project walks through a complete stock price forecasting pipeline:

1. **Data Cleaning & Anomaly Detection** — identifies and interpolates extreme return-based anomalies in the price series.
2. **Feature Engineering** — builds technical indicators including:
   - Simple & Exponential Moving Averages (SMA/EMA, 10 & 30-day)
   - Volatility (rolling standard deviation of returns)
   - Bollinger Bands (Upper/Middle/Lower)
   - Relative Strength Index (RSI)
   - MACD & Signal Line
   - On-Balance Volume (OBV)
   - Lagged close prices (1, 5, 30 days)
   - Earnings report event flags
   - Calendar/seasonality features (Month, Quarter, Day of Week)
   - S&P 500 index data (via `yfinance`) as an external correlated feature
3. **Modeling** — trains and compares multiple approaches:
   - Linear Regression (baseline)
   - Random Forest Regressor
   - XGBoost Regressor
   - LSTM (deep learning, sequence-based)
   - ARIMA
   - SARIMA
4. **Hyperparameter Tuning** — Grid Search (Random Forest) and manual tuning (LSTM) using time-series cross-validation.
5. **Evaluation & Comparison** — models are scored on MAE, RMSE, MAPE, and R², with visual overlays of actual vs. predicted prices.
