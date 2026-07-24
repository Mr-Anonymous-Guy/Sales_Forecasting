"""
forecasters.py — Time series forecasting models.

Implements Linear Regression, Random Forest, ARIMA, and Prophet
with a unified interface for training and prediction.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from src.config import (
    ARIMA_ORDER,
    DATE_COLUMN,
    PROPHET_SEASONALITY_MODE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Base forecaster
# ──────────────────────────────────────────────────────────────


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model: Any = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on historical data."""
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate forecasts for future periods."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ──────────────────────────────────────────────────────────────
# Linear Regression Forecaster
# ──────────────────────────────────────────────────────────────


class LinearRegressionForecaster(BaseForecaster):
    """
    Linear regression with time-based features.

    Uses numeric time index as the single feature.
    """

    def __init__(self) -> None:
        super().__init__("Linear Regression")
        self.model = LinearRegression()
        self.start_date: pd.Timestamp = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit linear regression on time series."""
        self.start_date = df[DATE_COLUMN].iloc[0]

        # Create numeric time feature
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[TARGET_COLUMN].values

        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.name} fitted on {len(df)} samples")

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict future periods using linear trend."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Generate future time indices
        last_idx = self.model.n_features_in_
        future_idx = np.arange(last_idx, last_idx + periods).reshape(-1, 1)

        predictions = self.model.predict(future_idx)

        # Create forecast DataFrame
        future_dates = pd.date_range(
            start=self.start_date,
            periods=last_idx + periods,
            freq="MS",
        )[-periods:]

        return pd.DataFrame({
            DATE_COLUMN: future_dates,
            "forecast": predictions,
        })


# ──────────────────────────────────────────────────────────────
# Random Forest Forecaster
# ──────────────────────────────────────────────────────────────


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest with time-based and lag features.

    Uses multiple features: time index, month, quarter, and lags.
    """

    def __init__(self) -> None:
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
        )
        self.start_date: pd.Timestamp = None
        self.last_values: list[float] = []

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features = df.copy()
        features["time_idx"] = np.arange(len(features))
        features["month"] = features[DATE_COLUMN].dt.month
        features["quarter"] = features[DATE_COLUMN].dt.quarter

        # Lag features
        for lag in [1, 3, 6, 12]:
            features[f"lag_{lag}"] = features[TARGET_COLUMN].shift(lag)

        return features.dropna()

    def fit(self, df: pd.DataFrame) -> None:
        """Fit Random Forest on feature-engineered data."""
        self.start_date = df[DATE_COLUMN].iloc[0]
        self.last_values = df[TARGET_COLUMN].tail(12).tolist()

        features = self._create_features(df)
        X = features[[c for c in features.columns if c not in [DATE_COLUMN, TARGET_COLUMN]]]
        y = features[TARGET_COLUMN]

        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.name} fitted on {len(X)} samples with {X.shape[1]} features")

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict future periods recursively."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        current_values = self.last_values.copy()

        last_time_idx = self.model.n_features_in_
        future_dates = pd.date_range(
            start=self.start_date,
            periods=last_time_idx + periods + 12,  # Account for lags
            freq="MS",
        )[-periods:]

        for i, date in enumerate(future_dates):
            # Create features for this period
            time_idx = last_time_idx + i
            month = date.month
            quarter = date.quarter

            lags = [
                current_values[-1] if len(current_values) >= 1 else 0,
                current_values[-3] if len(current_values) >= 3 else 0,
                current_values[-6] if len(current_values) >= 6 else 0,
                current_values[-12] if len(current_values) >= 12 else 0,
            ]

            X_future = np.array([[time_idx, month, quarter] + lags])
            pred = self.model.predict(X_future)[0]

            predictions.append(pred)
            current_values.append(pred)

        return pd.DataFrame({
            DATE_COLUMN: future_dates,
            "forecast": predictions,
        })


# ──────────────────────────────────────────────────────────────
# ARIMA Forecaster
# ──────────────────────────────────────────────────────────────


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.

    Classical time series forecasting with trend and seasonality.
    """

    def __init__(self, order: Tuple[int, int, int] = ARIMA_ORDER) -> None:
        super().__init__("ARIMA")
        self.order = order
        self.last_date: pd.Timestamp = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit ARIMA model."""
        self.last_date = df[DATE_COLUMN].iloc[-1]
        y = df[TARGET_COLUMN].values

        self.model = ARIMA(y, order=self.order)
        self.model = self.model.fit()
        self.is_fitted = True
        logger.info(f"{self.name}{self.order} fitted on {len(y)} samples")

    def predict(self, periods: int) -> pd.DataFrame:
        """Forecast future periods."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model.forecast(steps=periods)

        future_dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=periods,
            freq="MS",
        )

        return pd.DataFrame({
            DATE_COLUMN: future_dates,
            "forecast": forecast,
        })


# ──────────────────────────────────────────────────────────────
# Prophet Forecaster
# ──────────────────────────────────────────────────────────────


if HAS_PROPHET:
    class ProphetForecaster(BaseForecaster):
        """
        Facebook Prophet — Additive time series model.

        Handles seasonality, holidays, and trends automatically.
        """

        def __init__(self) -> None:
            super().__init__("Prophet")
            self.model = Prophet(
                seasonality_mode=PROPHET_SEASONALITY_MODE,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            self.last_date: pd.Timestamp = None

        def fit(self, df: pd.DataFrame) -> None:
            """Fit Prophet model."""
            self.last_date = df[DATE_COLUMN].iloc[-1]

            # Prophet requires 'ds' and 'y' columns
            prophet_df = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
            prophet_df.columns = ["ds", "y"]

            self.model.fit(prophet_df)
            self.is_fitted = True
            logger.info(f"{self.name} fitted on {len(prophet_df)} samples")

        def predict(self, periods: int) -> pd.DataFrame:
            """Forecast future periods."""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")

            future = self.model.make_future_dataframe(periods=periods, freq="MS")
            forecast = self.model.predict(future)

            # Extract only future predictions
            future_forecast = forecast.tail(periods)[["ds", "yhat"]].copy()
            future_forecast.columns = [DATE_COLUMN, "forecast"]

            return future_forecast


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────


def get_all_forecasters() -> list[BaseForecaster]:
    """Return all available forecasters."""
    forecasters: list[BaseForecaster] = [
        LinearRegressionForecaster(),
        RandomForestForecaster(),
        ARIMAForecaster(),
    ]
    if HAS_PROPHET:
        forecasters.append(ProphetForecaster())
    return forecasters
