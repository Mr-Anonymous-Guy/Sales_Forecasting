"""
data_loader.py — Data loading and validation for sales forecasting.

Handles CSV loading, date parsing, missing value detection, and
basic time series validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import DATE_COLUMN, FREQ, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────


def load_sales_data(filepath: Path | str) -> pd.DataFrame:
    """
    Load sales data from CSV file.

    Args:
        filepath: Path to CSV file with date and sales columns.

    Returns:
        DataFrame with parsed dates and sorted by date.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading sales data from {path}")
    df = pd.read_csv(path)

    # Validate columns
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TARGET_COLUMN}")

    # Parse dates
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Sort by date
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} records from {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
    return df


# ──────────────────────────────────────────────────────────────
# Data validation
# ──────────────────────────────────────────────────────────────


def validate_time_series(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Validate time series data for forecasting.

    Checks:
    - No duplicate dates
    - Chronological order
    - Regular frequency
    - Sufficient data points

    Args:
        df: DataFrame with date and sales columns.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check for duplicates
    if df[DATE_COLUMN].duplicated().any():
        issues.append("Duplicate dates found")

    # Check chronological order
    if not df[DATE_COLUMN].is_monotonic_increasing:
        issues.append("Dates are not in chronological order")

    # Check minimum data points
    if len(df) < 24:
        issues.append(f"Insufficient data: {len(df)} records (minimum 24 required)")

    # Check for missing values
    if df[TARGET_COLUMN].isnull().any():
        null_count = df[TARGET_COLUMN].isnull().sum()
        issues.append(f"Missing values in sales: {null_count}")

    # Check for negative sales
    if (df[TARGET_COLUMN] < 0).any():
        issues.append("Negative sales values detected")

    return len(issues) == 0, issues


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """
    Count missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary mapping column names to null counts.
    """
    return df.isnull().sum().to_dict()


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "interpolate",
) -> pd.DataFrame:
    """
    Handle missing values in sales data.

    Args:
        df: Input DataFrame.
        method: One of "interpolate", "forward_fill", or "drop".

    Returns:
        DataFrame with handled missing values.

    Raises:
        ValueError: If method is unknown.
    """
    df = df.copy()

    if method == "interpolate":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].interpolate(method="linear")
    elif method == "forward_fill":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].ffill()
    elif method == "drop":
        df = df.dropna(subset=[TARGET_COLUMN])
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


# ──────────────────────────────────────────────────────────────
# Data statistics
# ──────────────────────────────────────────────────────────────


def compute_statistics(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute summary statistics for sales data.

    Args:
        df: Sales DataFrame.

    Returns:
        Dictionary with statistics.
    """
    sales = df[TARGET_COLUMN]
    return {
        "count": len(sales),
        "mean": sales.mean(),
        "std": sales.std(),
        "min": sales.min(),
        "25%": sales.quantile(0.25),
        "median": sales.median(),
        "75%": sales.quantile(0.75),
        "max": sales.max(),
        "total": sales.sum(),
    }


def display_statistics(df: pd.DataFrame) -> str:
    """
    Return a formatted statistics summary.

    Args:
        df: Sales DataFrame.

    Returns:
        Multi-line summary string.
    """
    stats = compute_statistics(df)
    date_range = f"{df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {df[DATE_COLUMN].max().strftime('%Y-%m-%d')}"

    lines = [
        "\n" + "=" * 60,
        "  📊  Sales Data Summary",
        "=" * 60,
        f"  Date Range    : {date_range}",
        f"  Total Records : {stats['count']:.0f}",
        f"  Total Sales   : ${stats['total']:,.2f}",
        "=" * 60,
        f"  Mean          : ${stats['mean']:,.2f}",
        f"  Std Dev       : ${stats['std']:,.2f}",
        f"  Min           : ${stats['min']:,.2f}",
        f"  25th %ile     : ${stats['25%']:,.2f}",
        f"  Median        : ${stats['median']:,.2f}",
        f"  75th %ile     : ${stats['75%']:,.2f}",
        f"  Max           : ${stats['max']:,.2f}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Train/test split
# ──────────────────────────────────────────────────────────────


def split_train_test(
    df: pd.DataFrame,
    test_size: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    Args:
        df: Full dataset.
        test_size: Number of periods to reserve for testing.

    Returns:
        (train_df, test_df)
    """
    split_idx = len(df) - test_size
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(f"Split: {len(train)} train, {len(test)} test")
    return train, test
