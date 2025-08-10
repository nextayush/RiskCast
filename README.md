# RiskCast — An intraday volatility forecasting system combining econometric models and machine learning to deliver real-time, actionable market risk predictions.

**RiskCast** is a machine learning pipeline for forecasting short-term volatility and risk in stock prices using multiple statistical and ML models.  
It supports multiple symbols, feature engineering, model training, evaluation, and backtesting — all in one run.

---

## Features

- **Preprocessing:** Cleans raw intraday stock price data and aligns it with trading hours.
- **Feature Engineering:** Generates technical indicators, lag features, and integrates external indices (e.g., NIFTY50, NIFTYBANK).
- **Models Supported:**
  - Baseline
  - GARCH
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - XGBoost
- **Evaluation:** Calculates RMSE and other metrics for each model.
- **Backtesting:** Runs realistic trading simulations to test predictive performance.

---

## Folder Structure
```bash
RiskCast/
│
├── data/
│ ├── raw/ # Raw input CSVs (e.g., ICICIBANK.csv, NIFTY50.csv)
│ ├── interim/ # Preprocessed intermediate data
│ ├── features/ # Feature-engineered datasets
│ └── external/ # External index data
│
├── models/ # Trained model files
├── results/
│ ├── metrics/ # Model evaluation metrics
│ ├── backtest/ # Backtest results
│ └── figures/ # Equity curve plots
│
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── backtest.py
│
├── params.yaml # Pipeline parameters
├── main.py # Main entry point
└── requirements.txt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/RiskCast.git
cd RiskCast
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Prepare data
Place your raw stock data CSVs in data/raw/
Example:
```
data/raw/ICICIBANK.csv
data/raw/NIFTY50.csv
data/raw/NIFTYBANK.csv
```
4. Configure parameters
Edit params.yaml to set:

- symbol
- index_symbols
- freq
- prediction_horizon_bars
- annualization_factor
- Trading hours, etc.

---

## Running the Pipeline

Run the full pipeline (**preprocessing → feature engineering → training → evaluation → backtesting**):

```bash
python main.py
```

## Example Output
```
🔸 Running pipeline for symbol: ICICIBANK
🔹 Preprocessing data...
📥 Interim saved -> data/interim/ICICIBANK_interim.parquet
🔹 Feature engineering...
⚙️ Running feature engineering ...
✅ Feature engineering complete. Shape: (130149, 64)
🔹 Training models...
💾 Model saved to: models/ICICIBANK_xgboost.pkl
🔹 Evaluating models...
📄 Metrics saved to: results/metrics/ICICIBANK_xgboost_metrics.json
🔹 Running backtests...
📄 Equity plot saved to: results/figures/ICICIBANK_xgboost_equity_curve.png
✅ Pipeline finished.
```

## Results
- Metrics: results/metrics/
- Backtest summaries: results/backtest/
- Equity curves: results/figures/

## You can compare model performances based on:

- RMSE
- Sharpe ratio
- Maximum drawdown
- Equity curve shape
