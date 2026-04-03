# AMZN Stock Prediction — Model Comparison

## Overview
This repository contains a machine learning project designed to forecast Amazon (AMZN) stock prices. The notebook evaluates and compares the predictive performance of a Deep Learning approach (an Enhanced LSTM) against a Tree-Based ensemble method (XGBoost). 

## Key Features
* **Technical Analysis Feature Engineering**: Extracts momentum, trend, and volatility indicators (including RSI, MACD, Bollinger Bands, ATR, OBV, EMA, and ROC) using the `ta` library to enrich the dataset.
* **PyTorch Data Augmentation**: Implements a custom `AugmentedStockDataset` in PyTorch that randomly applies Gaussian noise and magnitude scaling to temporal sequences during training to improve model generalization.
* **Optuna Hyperparameter Tuning**: Automatically searches for the optimal hyperparameters for both the Enhanced LSTM (tuning hidden size, layers, dropout, and learning rate) and the XGBoost model (tuning estimators, depth, learning rate, subsampling, etc.).
* **Model Comparison Dashboard**: Visualizes predictions against actual closing prices, plots the residuals, and generates a bar chart comparing the Root Mean Squared Error (RMSE) of both models.

## Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/drizzy765/AMZN-Stock-Prediction.git>
   cd <AMZN-stock-prediction>
   pip -r install requirements.txt
   ```
