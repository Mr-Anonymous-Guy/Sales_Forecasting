# 📈 Sales Forecasting System 📊

A production-grade time series forecasting and business intelligence system using Python.  
Part of the **05_Python_AI_Projects** repository — the Time Series Forecasting project.

---

## What You'll Learn

| Concept | Implementation |
|---|---|
| Time Series Analysis | Date indexing, stationarity, lag features, seasonal decomposition |
| Forecasting Models | Linear Regression, Random Forest, ARIMA, Prophet |
| Trend Detection | Slope analysis, CAGR, month-over-month growth rates |
| Business Intelligence | Automated insights, seasonality profiling, forecast confidence |
| Model Evaluation | MAE, RMSE, MAPE with qualitative ratings |
| Report Generation | CSV exports, business summary reports, chart exports |

---

## Features

### Dataset Management
- **Load CSV** — pandas-based ingestion with date parsing and validation
- **Time Series Validation** — duplicate detection, chronological checks, frequency regularity
- **Missing Values** — interpolation, forward-fill, or drop strategies
- **Summary Statistics** — count, mean, std, quartiles, total sales, date range

### Forecasting Models

Four models trained and compared automatically:

| Model | Approach | Strengths |
|---|---|---|
| Linear Regression | Time-index regression | Fast baseline, interpretable trend |
| Random Forest | Lag + seasonal features | Captures non-linear patterns |
| ARIMA (1,1,1) | Autoregressive + moving average | Classical time series standard |
| Prophet | Additive decomposition | Handles seasonality and holidays |

The **best model** (lowest MAPE) is automatically selected and saved.

### Forecasting Horizons
- **Next Month** — 1-period ahead forecast
- **Next Quarter** — 3-period ahead forecast
- **Next Year** — 12-period ahead forecast

### Evaluation Metrics
- **MAE** — Mean Absolute Error (average prediction error in $)
- **RMSE** — Root Mean Squared Error (penalizes large errors)
- **MAPE** — Mean Absolute Percentage Error (scale-independent accuracy)

| MAPE | Rating |
|---|---|
| < 10% | Excellent |
| 10–20% | Good |
| 20–50% | Acceptable |
| > 50% | Poor |

### Business Analytics
- **Growth Rate** — Compound Annual Growth Rate (CAGR)
- **Trend Detection** — Upward / Downward / Stable with slope
- **Seasonality Profiling** — Average sales by month with bar visualization
- **Business Insights** — Best/worst months, forecast change percentage

### Visualizations
Publication-quality charts using matplotlib + seaborn:

1. **Historical Sales Chart** — full time series with trend line
2. **Forecast Chart** — historical + future predictions overlay
3. **Model Comparison** — MAPE bar chart across all models

### Reporting
- **Forecast CSV** — raw predictions with timestamps
- **Model Comparison CSV** — side-by-side metrics table
- **Business Report** — comprehensive text summary with key takeaways

---

## Project Structure

```
Sales_Forecasting/
├── data/
│   └── sales_data.csv          # 60-month sample dataset (2019–2023)
│
├── models/
│   ├── best_model.pkl          # Auto-selected best model (generated)
│   └── <model_name>.pkl        # Individual model files (generated)
│
├── reports/
│   ├── forecast_*.csv          # Forecast exports (generated)
│   ├── model_comparison_*.csv  # Evaluation exports (generated)
│   └── business_report_*.txt   # Summary reports (generated)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, parameters, thresholds
│   ├── data_loader.py          # CSV loading, validation, statistics
│   ├── forecasters.py          # 4 forecaster classes (ABC pattern)
│   ├── evaluation.py           # MAE, RMSE, MAPE, model comparison
│   ├── analytics.py            # Growth, trend, seasonality, insights
│   ├── model_manager.py        # Save/load/train/auto-select pipeline
│   ├── reporter.py             # CSV exports and business reports
│   └── visualizations.py       # matplotlib charts
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # 28 unit tests
│
├── main.py                     # Interactive CLI application
├── requirements.txt
├── run.sh                      # Linux / macOS launcher
├── run.bat                     # Windows launcher
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Python **3.8+**
- pip (Python package manager)

### Install dependencies

```bash
pip install -r requirements.txt
```

Core packages:
- `numpy` + `pandas` — data manipulation
- `scikit-learn` — ML algorithms
- `statsmodels` — ARIMA time series
- `prophet` — Facebook Prophet forecasting
- `scipy` — statistical analysis
- `matplotlib` + `seaborn` — visualization

### Run the app

**Linux / macOS**
```bash
chmod +x run.sh
./run.sh
```

**Windows**
```bat
run.bat
```

**Direct**
```bash
python main.py
```

---

## Usage Guide

### 1. Load & Explore Dataset

Displays:
- Time series validation status (duplicates, chronological order, data sufficiency)
- Missing value check with auto-interpolation
- Statistical summary (count, mean, std, quartiles, total, date range)

### 2. Train Forecasting Models

Trains all four models on 80% of data, evaluates on held-out 12-month test set:

```
🚀  Training Forecasting Pipeline
════════════════════════════════════════════════════════

  Training 4 models on historical data...
  (Linear Regression, Random Forest, ARIMA, Prophet)

  ────────────────────────────────────────────────────────
  Linear Regression
    MAE  : $3,456.78
    RMSE : $4,123.45
    MAPE : 7.23%
  ────────────────────────────────────────────────────────
  Random Forest
    MAE  : $2,890.12
    RMSE : $3,567.89
    MAPE : 5.67%
  ────────────────────────────────────────────────────────
  ARIMA
    MAE  : $4,012.34
    RMSE : $4,890.56
    MAPE : 8.45%
  ────────────────────────────────────────────────────────

  🏆  Best Model: Random Forest
      MAPE: 5.67%
      Saved to: models/best_model.pkl
```

### 3. Generate Forecasts

Select a horizon and get predictions:

```
🔮  Generate Forecasts
════════════════════════════════════════════════════════

  Using: Random Forest

  Select forecast horizon:
  [1]  Next Month (1 period)
  [2]  Next Quarter (3 periods)
  [3]  Next Year (12 periods)

  Choose: 2

  ────────────────────────────────────────
  📅  QUARTER FORECAST
  ────────────────────────────────────────
    2024-01  :  $52,340.25
    2024-02  :  $54,890.50
    2024-03  :  $57,120.75
  ────────────────────────────────────────
    Total    :  $164,351.50
  ────────────────────────────────────────
```

### 4. Business Analytics

Growth analysis, trend detection, and seasonal profiling:

```
📊  Business Analytics
════════════════════════════════════════════════════════

  ── Growth Analysis ──
  CAGR           : 12.45%
  Trend          : Upward ($634.50/month)

  ── Seasonal Patterns (Average by Month) ──
    Jan: $ 39,584.38  ████████████████████
    Feb: $ 40,850.31  ████████████████████
    ...
    Nov: $ 61,156.25  ██████████████████████████████
    Dec: $ 68,942.69  ██████████████████████████████████
```

### 5. Visualizations

Interactive menu to generate and save charts:
- Historical Sales Chart
- Forecast Chart (historical + predictions)
- Model Comparison Bar Chart

### 6. Export Reports

Generates three files in `reports/`:
- `forecast_<model>_<timestamp>.csv`
- `model_comparison_<timestamp>.csv`
- `business_report_<timestamp>.txt`

---

## Running Tests

```bash
# pytest (recommended)
pip install pytest
python -m pytest tests/ -v

# Built-in unittest — zero dependencies
python -m unittest discover tests/ -v
```

Expected output:
```
test_load_sales_data_success                    PASSED
test_load_sales_data_file_not_found             PASSED
test_load_sales_data_missing_columns            PASSED
test_validate_time_series_valid                  PASSED
test_validate_time_series_insufficient_data      PASSED
test_compute_statistics                          PASSED
test_display_statistics_returns_string           PASSED
test_check_missing_values                        PASSED
test_handle_missing_values_interpolate           PASSED
test_split_train_test                            PASSED
test_linear_regression_fit_predict               PASSED
test_random_forest_fit_predict                   PASSED
test_arima_fit_predict                           PASSED
test_predict_before_fit_raises                   PASSED
test_get_all_forecasters                         PASSED
test_compute_metrics                             PASSED
test_compute_metrics_perfect                     PASSED
test_mape_rating                                 PASSED
test_compare_models                              PASSED
test_select_best_model                           PASSED
test_select_best_model_empty_raises              PASSED
test_calculate_growth_rate                       PASSED
test_detect_trend                                PASSED
test_identify_seasonality                        PASSED
test_month_over_month_growth                     PASSED
test_save_and_load_model                         PASSED
test_save_unfitted_raises                        PASSED
test_load_nonexistent_raises                     PASSED
test_export_forecast_csv                         PASSED
test_export_evaluation_csv                       PASSED
test_export_forecast_csv_auto_path               PASSED

31 passed
```

---

## Sample Dataset

**`data/sales_data.csv`** contains 60 monthly records (2019–2023):

| Column | Description | Range |
|---|---|---|
| `date` | First day of each month | 2019-01 to 2023-12 |
| **`sales`** | **Monthly sales revenue ($)** | $25,430 – $83,560 |

The dataset includes realistic patterns:
- **Upward trend** — steady growth from ~$32K to ~$83K
- **Seasonality** — Q4 holiday peaks (Nov/Dec), summer dips (Jul/Aug)
- **Anomaly** — COVID-19 dip in early 2020 (Mar/Apr)
- **Noise** — natural variation for realistic modeling

---

## Architecture Notes

### Module responsibilities

| Module | Purpose |
|---|---|
| `config.py` | Central configuration — paths, parameters, thresholds |
| `data_loader.py` | Data I/O, validation, statistics — no ML logic |
| `forecasters.py` | 4 forecaster classes with ABC pattern — fit/predict interface |
| `evaluation.py` | Metrics computation — reusable across train/test |
| `analytics.py` | Business intelligence — growth, trends, insights |
| `model_manager.py` | Training pipeline, persistence, auto-selection |
| `reporter.py` | CSV exports and report generation |
| `visualizations.py` | All matplotlib/seaborn code isolated here |
| `main.py` | Orchestration — menu-driven CLI |

### Why four models?

- **Linear Regression** — baseline; captures linear trend only
- **Random Forest** — engineered features (lags, month, quarter); handles non-linearity
- **ARIMA** — classical statistical model; autoregressive + differencing + moving average
- **Prophet** — Facebook's additive model; excels at seasonality and holidays

The pipeline trains all four and picks the winner based on **MAPE** (Mean Absolute Percentage Error).

### Model persistence

Models are serialized to `models/*.pkl` using Python's `pickle` module. Each save includes:
- The fitted forecaster object
- Evaluation metrics (MAE, RMSE, MAPE)
- Model name for identification

The best model is also saved as `best_model.pkl` for quick loading.

---

## Extending the Project

### Add more features
Modify the dataset to include columns like:
- `marketing_spend`
- `promotions`
- `economic_index`

Update `forecasters.py` to incorporate additional features in Random Forest.

### Try other algorithms
In `forecasters.py`, create a new class extending `BaseForecaster`:
```python
class XGBoostForecaster(BaseForecaster):
    def __init__(self) -> None:
        super().__init__("XGBoost")
        # ...
```

Then add it to `get_all_forecasters()`.

### Hyperparameter tuning
Update `config.py` parameters or add grid search:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

---

## License

MIT — free to use, modify, and learn from.
