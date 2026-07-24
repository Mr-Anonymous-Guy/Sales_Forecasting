"""
evaluation.py — Model evaluation metrics for forecasting.

Computes MAE, RMSE, MAPE and provides model comparison utilities.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import MAPE_ACCEPTABLE, MAPE_EXCELLENT, MAPE_GOOD, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (0-100).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Model predictions.

    Returns:
        Dictionary with MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def get_mape_rating(mape: float) -> str:
    """
    Convert MAPE to a qualitative rating.

    Args:
        mape: MAPE percentage.

    Returns:
        Rating string.
    """
    if mape < MAPE_EXCELLENT:
        return "Excellent"
    elif mape < MAPE_GOOD:
        return "Good"
    elif mape < MAPE_ACCEPTABLE:
        return "Acceptable"
    else:
        return "Poor"


def display_metrics(metrics: dict[str, float]) -> str:
    """
    Format metrics as a display string.

    Args:
        metrics: Dictionary from compute_metrics().

    Returns:
        Formatted string.
    """
    rating = get_mape_rating(metrics["MAPE"])

    lines = [
        "\n" + "=" * 60,
        "  📈  Evaluation Metrics",
        "=" * 60,
        f"  MAE (Mean Absolute Error)      : ${metrics['MAE']:,.2f}",
        f"  RMSE (Root Mean Squared Error) : ${metrics['RMSE']:,.2f}",
        f"  MAPE (Mean Absolute % Error)   : {metrics['MAPE']:.2f}%",
        f"  Rating                          : {rating}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Model comparison
# ──────────────────────────────────────────────────────────────


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models by their metrics.

    Args:
        results: List of dicts with keys: model_name, MAE, RMSE, MAPE.

    Returns:
        DataFrame sorted by MAPE (best first).
    """
    df = pd.DataFrame(results)
    df["Rating"] = df["MAPE"].apply(get_mape_rating)
    df = df.sort_values("MAPE").reset_index(drop=True)
    return df


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select the model with the lowest MAPE.

    Args:
        results: List of model evaluation results.

    Returns:
        Best model result dictionary.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("No results to select from")

    return min(results, key=lambda r: r["MAPE"])
