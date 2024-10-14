# Stock Price Prediction App

This is a stock price prediction application built using `Streamlit`, `TensorFlow`, and `yfinance`. It allows users to input stock tickers and date ranges to fetch historical data, train prediction models (CNN, LSTM, or Hybrid CNN-LSTM), and predict future stock prices on a specified date.

## Features

- **Input Stock Ticker**: Users can enter stock tickers (e.g., AAPL, GOOG) to fetch data from Yahoo Finance.
- **Date Range Selection**: Users can specify a start and end date for the stock data range to use in model training.
- **Model Selection**: Users can select between three models for prediction:
  - Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)
  - Hybrid CNN-LSTM Model
- **Prediction Date**: Users can input a specific date within the data range to predict the stock price.
- **Real-Time Visualization**: The application displays the latest stock data, model training progress, and predicted stock price.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock-prediction-app
