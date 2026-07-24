"""
model_manager.py — Model persistence and auto-selection.

Handles saving, loading, training all forecasters, and
automatically selecting the best model by MAPE.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATE_COLUMN, MODELS_DIR, TARGET_COLUMN
from src.data_loader import split_train_test
from src.evaluation import compute_metrics, select_best_model
from src.forecasters import BaseForecaster, get_all_forecasters

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Model persistence
# ──────────────────────────────────────────────────────────────


def save_model(
    forecaster: BaseForecaster,
    metrics: dict[str, float],
    filepath: Path | str | None = None,
) -> Path:
    """
    Save a trained forecaster and its metrics to disk.

    Args:
        forecaster: Fitted forecaster instance.
        metrics: Evaluation metrics (MAE, RMSE, MAPE).
        filepath: Target path. Defaults to models/<name>.pkl.

    Returns:
        Path where the model was saved.

    Raises:
        ValueError: If the forecaster is not fitted.
    """
    if not forecaster.is_fitted:
        raise ValueError(f"Cannot save unfitted model: {forecaster.name}")

    if filepath is None:
        safe_name = forecaster.name.lower().replace(" ", "_")
        filepath = MODELS_DIR / f"{safe_name}.pkl"
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "forecaster": forecaster,
        "metrics": metrics,
        "model_name": forecaster.name,
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Model saved: {forecaster.name} → {filepath}")
    return filepath


def load_model(filepath: Path | str) -> dict[str, Any]:
    """
    Load a saved forecaster from disk.

    Args:
        filepath: Path to the .pkl file.

    Returns:
        Dictionary with keys: forecaster, metrics, model_name.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    logger.info(f"Model loaded: {payload['model_name']} ← {path}")
    return payload


# ──────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────


def train_and_evaluate_all(
    df: pd.DataFrame,
    test_size: int = 12,
) -> list[dict[str, Any]]:
    """
    Train all forecasters and evaluate on a held-out test set.

    Args:
        df: Full sales DataFrame (sorted by date).
        test_size: Number of periods for the test split.

    Returns:
        List of result dicts: model_name, forecaster, MAE, RMSE, MAPE.
    """
    train_df, test_df = split_train_test(df, test_size=test_size)
    results: list[dict[str, Any]] = []

    for forecaster in get_all_forecasters():
        try:
            logger.info(f"Training {forecaster.name}...")

            # Fit on training data
            forecaster.fit(train_df)

            # Predict test period
            forecast_df = forecaster.predict(periods=len(test_df))

            # Compute metrics
            y_true = test_df[TARGET_COLUMN].values
            y_pred = forecast_df["forecast"].values[:len(y_true)]
            metrics = compute_metrics(y_true, y_pred)

            result = {
                "model_name": forecaster.name,
                "forecaster": forecaster,
                **metrics,
            }
            results.append(result)

            logger.info(
                f"  {forecaster.name}: MAE=${metrics['MAE']:,.2f}, "
                f"RMSE=${metrics['RMSE']:,.2f}, MAPE={metrics['MAPE']:.2f}%"
            )

        except Exception as exc:
            logger.warning(f"  {forecaster.name} failed: {exc}")

    return results


def auto_select_best(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Select the best model and save it to disk.

    Args:
        results: Output from train_and_evaluate_all().

    Returns:
        Best model result dict.

    Raises:
        ValueError: If no models trained successfully.
    """
    if not results:
        raise ValueError("No models trained successfully")

    best = select_best_model(results)

    # Save best model
    metrics = {k: best[k] for k in ["MAE", "RMSE", "MAPE"]}
    save_path = save_model(best["forecaster"], metrics)

    # Also save as "best_model.pkl" for easy loading
    best_path = MODELS_DIR / "best_model.pkl"
    save_model(best["forecaster"], metrics, best_path)

    logger.info(f"Best model: {best['model_name']} (MAPE={best['MAPE']:.2f}%)")
    return best
