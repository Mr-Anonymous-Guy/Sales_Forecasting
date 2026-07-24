"""
analytics.py — Business intelligence and trend analysis.

Provides growth rate calculation, trend detection, and forecast confidence.
"""

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATE_COLUMN, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Growth analysis
# ──────────────────────────────────────────────────────────────


def calculate_growth_rate(df: pd.DataFrame, periods: int = 12) -> float:
    """
    Calculate compound annual growth rate (CAGR).

    Args:
        df: Sales DataFrame.
        periods: Number of periods for growth calculation.

    Returns:
        CAGR as a percentage.
    """
    if len(df) < periods:
        periods = len(df)

    start_value = df[TARGET_COLUMN].iloc[0]
    end_value = df[TARGET_COLUMN].iloc[-1]

    if start_value <= 0:
        return 0.0

    years = periods / 12  # Convert months to years
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100

    return cagr


def calculate_month_over_month_growth(df: pd.DataFrame) -> pd.Series:
    """
    Calculate month-over-month growth rates.

    Args:
        df: Sales DataFrame.

    Returns:
        Series of growth rates (%).
    """
    return df[TARGET_COLUMN].pct_change() * 100


def calculate_year_over_year_growth(df: pd.DataFrame) -> pd.Series:
    """
    Calculate year-over-year growth rates.

    Args:
        df: Sales DataFrame.

    Returns:
        Series of YoY growth rates (%).
    """
    return df[TARGET_COLUMN].pct_change(periods=12) * 100


# ──────────────────────────────────────────────────────────────
# Trend detection
# ──────────────────────────────────────────────────────────────


def detect_trend(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Detect overall trend using linear regression.

    Args:
        df: Sales DataFrame.

    Returns:
        (trend_direction, slope_per_month)
        Direction is one of: "Upward", "Downward", "Stable"
    """
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[TARGET_COLUMN].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y)

    # Classify trend
    if abs(slope) < 0.01 * df[TARGET_COLUMN].mean():
        direction = "Stable"
    elif slope > 0:
        direction = "Upward"
    else:
        direction = "Downward"

    return direction, slope


def identify_seasonality(df: pd.DataFrame) -> dict[int, float]:
    """
    Identify seasonal patterns by month.

    Args:
        df: Sales DataFrame.

    Returns:
        Dictionary mapping month (1-12) to average sales.
    """
    df_copy = df.copy()
    df_copy["month"] = df_copy[DATE_COLUMN].dt.month

    monthly_avg = df_copy.groupby("month")[TARGET_COLUMN].mean().to_dict()
    return monthly_avg


# ──────────────────────────────────────────────────────────────
# Forecast confidence
# ──────────────────────────────────────────────────────────────


def calculate_forecast_confidence(
    historical_std: float,
    forecast_horizon: int,
) -> Tuple[float, float]:
    """
    Calculate confidence intervals for forecasts.

    Args:
        historical_std: Standard deviation of historical data.
        forecast_horizon: Number of periods ahead.

    Returns:
        (lower_bound_multiplier, upper_bound_multiplier)
        Multiply forecast by these to get 95% confidence intervals.
    """
    # Confidence intervals widen with forecast horizon
    z_score = 1.96  # 95% confidence
    uncertainty = historical_std * np.sqrt(forecast_horizon)

    lower = -z_score * uncertainty
    upper = z_score * uncertainty

    return lower, upper


# ──────────────────────────────────────────────────────────────
# Business insights
# ──────────────────────────────────────────────────────────────


def generate_business_insights(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate comprehensive business insights.

    Args:
        df: Historical sales data.
        forecast_df: Forecasted sales.

    Returns:
        Dictionary with insights.
    """
    # Historical analysis
    total_sales = df[TARGET_COLUMN].sum()
    avg_sales = df[TARGET_COLUMN].mean()
    growth_rate = calculate_growth_rate(df)
    trend, slope = detect_trend(df)

    # Forecast analysis
    forecast_total = forecast_df["forecast"].sum()
    forecast_avg = forecast_df["forecast"].mean()
    forecast_change = ((forecast_avg - avg_sales) / avg_sales) * 100

    # Best and worst months
    best_month = df.loc[df[TARGET_COLUMN].idxmax()]
    worst_month = df.loc[df[TARGET_COLUMN].idxmin()]

    return {
        "historical": {
            "total_sales": total_sales,
            "average_monthly_sales": avg_sales,
            "growth_rate": growth_rate,
            "trend_direction": trend,
            "trend_slope": slope,
            "best_month": {
                "date": best_month[DATE_COLUMN],
                "sales": best_month[TARGET_COLUMN],
            },
            "worst_month": {
                "date": worst_month[DATE_COLUMN],
                "sales": worst_month[TARGET_COLUMN],
            },
        },
        "forecast": {
            "total_forecast": forecast_total,
            "average_forecast": forecast_avg,
            "change_percentage": forecast_change,
        },
    }


def format_business_insights(insights: dict[str, Any]) -> str:
    """
    Format business insights as a readable report.

    Args:
        insights: Dictionary from generate_business_insights().

    Returns:
        Formatted report string.
    """
    hist = insights["historical"]
    fore = insights["forecast"]

    lines = [
        "\n" + "=" * 60,
        "  📊  Business Intelligence Report",
        "=" * 60,
        "",
        "  Historical Performance:",
        f"    Total Sales        : ${hist['total_sales']:,.2f}",
        f"    Avg Monthly Sales  : ${hist['average_monthly_sales']:,.2f}",
        f"    Growth Rate (CAGR): {hist['growth_rate']:.2f}%",
        f"    Trend              : {hist['trend_direction']} (${hist['trend_slope']:,.2f}/month)",
        "",
        f"    Best Month         : {hist['best_month']['date'].strftime('%Y-%m')} (${hist['best_month']['sales']:,.2f})",
        f"    Worst Month        : {hist['worst_month']['date'].strftime('%Y-%m')} (${hist['worst_month']['sales']:,.2f})",
        "",
        "  Forecast Outlook:",
        f"    Total Forecast     : ${fore['total_forecast']:,.2f}",
        f"    Avg Monthly        : ${fore['average_forecast']:,.2f}",
        f"    Expected Change    : {fore['change_percentage']:+.2f}%",
        "=" * 60,
    ]
    return "\n".join(lines)
