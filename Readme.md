## 🚀 Stock Price Prediction & Forecasting Pipeline

An end-to-end **Tesla stock forecasting** project with rigorous **feature engineering**, **baseline & advanced models**, **hyperparameter tuning**, and **walk-forward evaluation**. Built for accuracy and production readiness.

---

### 🎯 Project Overview

* **Goal**: Predict next-day and multi-day values for **Open**, **High**, **Low**, **Close** using historical market, sentiment, and technical features.
* **Approach**:

  1. **Data Collection**: Yahoo Finance (prices), news API / FinBERT sentiment, calendar holidays.
  2. **Preprocessing**: Merge indices (S\&P500, NASDAQ), adjust weekend/holiday sentiment, fill missing.
  3. **Feature Engineering**: Rolling stats, RSI, MACD, Bollinger, OBV, GARCH volatility, spread vs benchmarks, slopes & interactions.
  4. **Feature Selection**: Correlation filtering, importance via RF & XGBoost.
  5. **Modeling**:

     * **Baselines**: MultiOutput RF & XGBoost.
     * **Sequence**: Bidirectional LSTM + GRU.
  6. **Hyperparameter Tuning**: Optuna-driven for RF/XGB & GRU.
  7. **Walk-Forward Validation**: Retrain on-the-fly if recent MAPE > threshold.
  8. **Evaluation**: MAE, RMSE, MAPE, directional accuracy, plots.

---

## 🏗️ File Structure

```text
project-root/
├── main.py                  # CLI entrypoint (prep, train, forecast)
├── requirements.txt         # Pinned dependencies
├── README.md                # You are reading this
├── .gitignore               # data/, __pycache__/, checkpoints/
├── src/
│   ├── data_collection.py   # fetch & merge price + sentiment + holiday data
│   ├── preprocessing.py     # cleaning, adjusting sentiments, merge
│   ├── features.py          # all feature-engineering functions
│   ├── selection.py         # correlation drop + RF/XGB importance
│   ├── models.py            # baseline & sequence models + Optuna routines
│   ├── evaluation.py        # metrics & walk-forward loops
│   └── utils.py             # common helpers (load, save, plotting)
├── data/                    # raw & interim data (ignored via git)
└── checkpoints/             # saved model weights & indices
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<you>/tesla-stock-forecast.git
cd tesla-stock-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Usage

```bash
# 1. Prepare embeddings & data
python main.py prep \
  --start_date 2017-09-30 \
  --ticker TSLA \
  --news_path data/tsla_news.csv

# 2. Train models:
python main.py train \
  --model rf          # or xgb, gru
  --cv_splits 10 \
  --optuna_trials 100

# 3. Walk-forward forecast
python main.py forecast \
  --horizon 15 \
  --retrain_window 5 \
  --mape_thresh 2.0
```

---

## 🧮 Key Concepts

### Feature Engineering

* **RSI**: Relative Strength Index over 14 days.
* **MACD & Signal**: 12/26 EMA crossover & signal 9-day.
* **Bollinger Bands**: 20-day MA ±2σ.
* **OBV**: On-Balance Volume.
* **GARCH(1,1)**: Conditional volatility.
* **Spreads**: Price differences vs NASDAQ & S\&P500.
* **Slopes**: Polynomial fit of weekly windows.
* **Interactions**: e.g. `Sentiment * Returns`, `Close * benchmark`.

### Modeling

* **RandomForest / XGBoost**: MultiOutput for simultaneous OHLC.
* **LSTM+GRU**: Bidirectional LSTM → GRU for sequence forecasting.

### Hyperparameter Tuning

* **Optuna**: Bayesian search for n\_estimators, depth, dropout, units, learning rate.
* Evaluate via **TimeSeriesSplit** to respect temporal order.

### Walk-Forward Validation

1. **Train** on `t0`→`t_end - horizon`
2. **Predict** next slice
3. **Append actual** to train set
4. **Retrain** if recent MAPE > threshold

---

## 📊 Evaluation

* **MAE**, **RMSE**, **MAPE** (avoid zero-division), **Directional Accuracy**.
* Visualize with Plotly: actual vs predicted time series.

---

## 🔗 Connect

* 💼 LinkedIn: https://www.linkedin.com/in/rahul-sharma--/
* 📧 Email: rahul11s@umd.edu

---