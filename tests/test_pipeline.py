"""
test_pipeline.py — Unit tests for Sales Forecasting system.

Tests cover: data loading, validation, forecasting, evaluation,
analytics, model persistence, and report generation.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATE_COLUMN, TARGET_COLUMN
from src.data_loader import (
    check_missing_values,
    compute_statistics,
    display_statistics,
    handle_missing_values,
    load_sales_data,
    split_train_test,
    validate_time_series,
)
from src.evaluation import (
    compare_models,
    compute_metrics,
    get_mape_rating,
    mean_absolute_percentage_error,
    select_best_model,
)
from src.analytics import (
    calculate_growth_rate,
    calculate_month_over_month_growth,
    detect_trend,
    identify_seasonality,
)
from src.forecasters import (
    ARIMAForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster,
    get_all_forecasters,
)
from src.model_manager import load_model, save_model
from src.reporter import export_evaluation_csv, export_forecast_csv


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


def _make_sample_df(n: int = 60) -> pd.DataFrame:
    """Create a synthetic sales DataFrame for testing."""
    dates = pd.date_range(start="2019-01-01", periods=n, freq="MS")
    np.random.seed(42)
    trend = np.linspace(30000, 70000, n)
    seasonal = 5000 * np.sin(np.linspace(0, 10 * np.pi, n))
    noise = np.random.normal(0, 1000, n)
    sales = trend + seasonal + noise
    return pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: sales})


def _make_sample_csv(tmp_dir: str, n: int = 60) -> Path:
    """Write a sample DataFrame to CSV and return the path."""
    df = _make_sample_df(n)
    path = Path(tmp_dir) / "test_sales.csv"
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────
# 1. Data Loading Tests
# ──────────────────────────────────────────────────────────────


class TestDataLoading(unittest.TestCase):
    """Tests for data_loader module."""

    def test_load_sales_data_success(self) -> None:
        """CSV loads with correct shape and dtypes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_sample_csv(tmp)
            df = load_sales_data(path)
            self.assertEqual(len(df), 60)
            self.assertIn(DATE_COLUMN, df.columns)
            self.assertIn(TARGET_COLUMN, df.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]))

    def test_load_sales_data_file_not_found(self) -> None:
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_sales_data("/nonexistent/path/data.csv")

    def test_load_sales_data_missing_columns(self) -> None:
        """CSV missing required columns raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_sales_data(path)

    def test_validate_time_series_valid(self) -> None:
        """Valid time series passes validation."""
        df = _make_sample_df()
        is_valid, issues = validate_time_series(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_validate_time_series_insufficient_data(self) -> None:
        """Short time series reports insufficient data."""
        df = _make_sample_df(n=10)
        is_valid, issues = validate_time_series(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Insufficient" in i for i in issues))

    def test_compute_statistics(self) -> None:
        """Statistics contain expected keys."""
        df = _make_sample_df()
        stats = compute_statistics(df)
        for key in ["count", "mean", "std", "min", "max", "total"]:
            self.assertIn(key, stats)
        self.assertEqual(stats["count"], 60)

    def test_display_statistics_returns_string(self) -> None:
        """Display output is a multi-line string."""
        df = _make_sample_df()
        result = display_statistics(df)
        self.assertIsInstance(result, str)
        self.assertIn("Sales Data Summary", result)

    def test_check_missing_values(self) -> None:
        """Clean data reports zero missing values."""
        df = _make_sample_df()
        missing = check_missing_values(df)
        self.assertEqual(missing[TARGET_COLUMN], 0)

    def test_handle_missing_values_interpolate(self) -> None:
        """Interpolation fills NaN values."""
        df = _make_sample_df()
        df.loc[5, TARGET_COLUMN] = np.nan
        result = handle_missing_values(df, method="interpolate")
        self.assertFalse(result[TARGET_COLUMN].isnull().any())

    def test_split_train_test(self) -> None:
        """Train/test split has correct sizes."""
        df = _make_sample_df()
        train, test = split_train_test(df, test_size=12)
        self.assertEqual(len(train), 48)
        self.assertEqual(len(test), 12)


# ──────────────────────────────────────────────────────────────
# 2. Forecasting Tests
# ──────────────────────────────────────────────────────────────


class TestForecasters(unittest.TestCase):
    """Tests for forecaster models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _make_sample_df()

    def test_linear_regression_fit_predict(self) -> None:
        """Linear regression fits and predicts correct number of periods."""
        model = LinearRegressionForecaster()
        model.fit(self.df)
        self.assertTrue(model.is_fitted)

        forecast = model.predict(periods=3)
        self.assertEqual(len(forecast), 3)
        self.assertIn("forecast", forecast.columns)

    def test_random_forest_fit_predict(self) -> None:
        """Random forest fits and predicts correct number of periods."""
        model = RandomForestForecaster()
        model.fit(self.df)
        self.assertTrue(model.is_fitted)

        forecast = model.predict(periods=3)
        self.assertEqual(len(forecast), 3)
        self.assertTrue(all(forecast["forecast"] > 0))

    def test_arima_fit_predict(self) -> None:
        """ARIMA fits and predicts correct number of periods."""
        model = ARIMAForecaster()
        model.fit(self.df)
        self.assertTrue(model.is_fitted)

        forecast = model.predict(periods=3)
        self.assertEqual(len(forecast), 3)

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fitting raises ValueError."""
        model = LinearRegressionForecaster()
        with self.assertRaises(ValueError):
            model.predict(periods=3)

    def test_get_all_forecasters(self) -> None:
        """Factory returns at least 3 forecasters."""
        forecasters = get_all_forecasters()
        self.assertGreaterEqual(len(forecasters), 3)
        names = [f.name for f in forecasters]
        self.assertIn("Linear Regression", names)
        self.assertIn("Random Forest", names)
        self.assertIn("ARIMA", names)


# ──────────────────────────────────────────────────────────────
# 3. Evaluation Tests
# ──────────────────────────────────────────────────────────────


class TestEvaluation(unittest.TestCase):
    """Tests for evaluation metrics."""

    def test_compute_metrics(self) -> None:
        """Metrics contain MAE, RMSE, MAPE."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = compute_metrics(y_true, y_pred)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertGreater(metrics["MAE"], 0)

    def test_compute_metrics_perfect(self) -> None:
        """Perfect predictions yield zero error."""
        y = np.array([100, 200, 300])
        metrics = compute_metrics(y, y)
        self.assertAlmostEqual(metrics["MAE"], 0.0)
        self.assertAlmostEqual(metrics["RMSE"], 0.0)
        self.assertAlmostEqual(metrics["MAPE"], 0.0)

    def test_mape_rating(self) -> None:
        """MAPE ratings follow threshold rules."""
        self.assertEqual(get_mape_rating(5.0), "Excellent")
        self.assertEqual(get_mape_rating(15.0), "Good")
        self.assertEqual(get_mape_rating(35.0), "Acceptable")
        self.assertEqual(get_mape_rating(60.0), "Poor")

    def test_compare_models(self) -> None:
        """Model comparison returns sorted DataFrame."""
        results = [
            {"model_name": "A", "MAE": 10, "RMSE": 12, "MAPE": 20},
            {"model_name": "B", "MAE": 5, "RMSE": 6, "MAPE": 8},
        ]
        df = compare_models(results)
        self.assertEqual(df.iloc[0]["model_name"], "B")  # Lower MAPE first

    def test_select_best_model(self) -> None:
        """Selects model with lowest MAPE."""
        results = [
            {"model_name": "A", "MAPE": 20},
            {"model_name": "B", "MAPE": 8},
            {"model_name": "C", "MAPE": 15},
        ]
        best = select_best_model(results)
        self.assertEqual(best["model_name"], "B")

    def test_select_best_model_empty_raises(self) -> None:
        """Empty results raises ValueError."""
        with self.assertRaises(ValueError):
            select_best_model([])


# ──────────────────────────────────────────────────────────────
# 4. Analytics Tests
# ──────────────────────────────────────────────────────────────


class TestAnalytics(unittest.TestCase):
    """Tests for business analytics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _make_sample_df()

    def test_calculate_growth_rate(self) -> None:
        """Growth rate is a finite number."""
        rate = calculate_growth_rate(self.df)
        self.assertTrue(np.isfinite(rate))

    def test_detect_trend(self) -> None:
        """Trend detection returns valid direction."""
        direction, slope = detect_trend(self.df)
        self.assertIn(direction, ["Upward", "Downward", "Stable"])
        self.assertTrue(np.isfinite(slope))

    def test_identify_seasonality(self) -> None:
        """Seasonality returns 12 months."""
        monthly = identify_seasonality(self.df)
        self.assertEqual(len(monthly), 12)
        self.assertTrue(all(v > 0 for v in monthly.values()))

    def test_month_over_month_growth(self) -> None:
        """MoM growth has correct length."""
        growth = calculate_month_over_month_growth(self.df)
        self.assertEqual(len(growth), len(self.df))


# ──────────────────────────────────────────────────────────────
# 5. Model Persistence Tests
# ──────────────────────────────────────────────────────────────


class TestModelPersistence(unittest.TestCase):
    """Tests for model save/load."""

    def test_save_and_load_model(self) -> None:
        """Model round-trips through pickle correctly."""
        df = _make_sample_df()
        model = LinearRegressionForecaster()
        model.fit(df)

        metrics = {"MAE": 100, "RMSE": 120, "MAPE": 5.0}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_model.pkl"
            save_model(model, metrics, filepath=path)
            self.assertTrue(path.exists())

            loaded = load_model(path)
            self.assertEqual(loaded["model_name"], "Linear Regression")
            self.assertEqual(loaded["metrics"]["MAPE"], 5.0)
            self.assertTrue(loaded["forecaster"].is_fitted)

    def test_save_unfitted_raises(self) -> None:
        """Saving unfitted model raises ValueError."""
        model = LinearRegressionForecaster()
        with self.assertRaises(ValueError):
            save_model(model, {}, filepath="/tmp/test.pkl")

    def test_load_nonexistent_raises(self) -> None:
        """Loading missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_model("/nonexistent/model.pkl")


# ──────────────────────────────────────────────────────────────
# 6. Report Generation Tests
# ──────────────────────────────────────────────────────────────


class TestReports(unittest.TestCase):
    """Tests for CSV and report export."""

    def test_export_forecast_csv(self) -> None:
        """Forecast CSV is created and non-empty."""
        forecast_df = pd.DataFrame({
            DATE_COLUMN: pd.date_range("2024-01-01", periods=3, freq="MS"),
            "forecast": [50000, 52000, 54000],
        })

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecast.csv"
            result = export_forecast_csv(forecast_df, "test", filepath=path)
            self.assertTrue(result.exists())
            loaded = pd.read_csv(result)
            self.assertEqual(len(loaded), 3)

    def test_export_evaluation_csv(self) -> None:
        """Evaluation CSV is created with sorted results."""
        results = [
            {"model_name": "A", "MAE": 10, "RMSE": 12, "MAPE": 20},
            {"model_name": "B", "MAE": 5, "RMSE": 6, "MAPE": 8},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.csv"
            result = export_evaluation_csv(results, filepath=path)
            self.assertTrue(result.exists())
            loaded = pd.read_csv(result)
            self.assertEqual(len(loaded), 2)
            # Should be sorted by MAPE
            self.assertEqual(loaded.iloc[0]["model_name"], "B")

    def test_export_forecast_csv_auto_path(self) -> None:
        """Auto-generated path creates file in reports directory."""
        forecast_df = pd.DataFrame({
            DATE_COLUMN: pd.date_range("2024-01-01", periods=3, freq="MS"),
            "forecast": [50000, 52000, 54000],
        })
        result = export_forecast_csv(forecast_df, "autotest")
        self.assertTrue(result.exists())
        # Clean up
        result.unlink()


if __name__ == "__main__":
    unittest.main()
