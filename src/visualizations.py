"""
visualizations.py — Charts and graphs for sales forecasting.

Creates professional matplotlib visualizations for historical data,
forecasts, and model comparisons.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import COLORS, DATE_COLUMN, DPI, FIGURE_SIZE, TARGET_COLUMN

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def plot_historical_sales(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot historical sales data.

    Args:
        df: Sales DataFrame.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    ax.plot(
        df[DATE_COLUMN],
        df[TARGET_COLUMN],
        color=COLORS["actual"],
        linewidth=2,
        label="Historical Sales",
    )

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sales ($)", fontsize=12, fontweight="bold")
    ax.set_title("Historical Sales Data", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Chart saved: {save_path}")

    plt.show()


def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot historical data with forecast.

    Args:
        historical_df: Historical sales.
        forecast_df: Forecasted sales.
        model_name: Name for the title.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    # Historical
    ax.plot(
        historical_df[DATE_COLUMN],
        historical_df[TARGET_COLUMN],
        color=COLORS["actual"],
        linewidth=2,
        label="Historical",
    )

    # Forecast
    ax.plot(
        forecast_df[DATE_COLUMN],
        forecast_df["forecast"],
        color=COLORS["forecast"],
        linewidth=2,
        linestyle="--",
        label="Forecast",
    )

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sales ($)", fontsize=12, fontweight="bold")
    ax.set_title(f"Sales Forecast — {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")

    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing model performance.

    Args:
        comparison_df: DataFrame with model_name and MAPE columns.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    bars = ax.bar(
        comparison_df["model_name"],
        comparison_df["MAPE"],
        color=COLORS["forecast"],
        edgecolor="black",
    )

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("MAPE (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")

    plt.show()
