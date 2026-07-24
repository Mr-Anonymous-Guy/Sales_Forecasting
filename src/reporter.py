"""
reporter.py — Report generation and CSV exports.

Creates timestamped forecast reports, model evaluation CSVs,
and comprehensive business summary text files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATE_COLUMN, REPORTS_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CSV exports
# ──────────────────────────────────────────────────────────────


def export_forecast_csv(
    forecast_df: pd.DataFrame,
    model_name: str = "model",
    filepath: Path | str | None = None,
) -> Path:
    """
    Export forecast data to CSV.

    Args:
        forecast_df: DataFrame with date and forecast columns.
        model_name: Model name for the filename.
        filepath: Custom save path. Auto-generates if None.

    Returns:
        Path to the saved CSV.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.lower().replace(" ", "_")
        filepath = REPORTS_DIR / f"forecast_{safe_name}_{timestamp}.csv"
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(filepath, index=False)

    logger.info(f"Forecast CSV saved: {filepath}")
    return filepath


def export_evaluation_csv(
    results: list[dict[str, Any]],
    filepath: Path | str | None = None,
) -> Path:
    """
    Export model comparison table to CSV.

    Args:
        results: List of model evaluation results.
        filepath: Custom save path. Auto-generates if None.

    Returns:
        Path to the saved CSV.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = REPORTS_DIR / f"model_comparison_{timestamp}.csv"
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Extract only serializable fields
    rows = []
    for r in results:
        rows.append({
            "model_name": r["model_name"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "MAPE": r["MAPE"],
        })

    df = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
    df.to_csv(filepath, index=False)

    logger.info(f"Evaluation CSV saved: {filepath}")
    return filepath


# ──────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────


def generate_summary_report(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    best_model_name: str,
    metrics: dict[str, float],
    insights: dict[str, Any],
    filepath: Path | str | None = None,
) -> Path:
    """
    Generate a comprehensive business summary report.

    Args:
        df: Historical sales data.
        forecast_df: Forecasted sales.
        best_model_name: Name of the best model.
        metrics: Evaluation metrics for the best model.
        insights: Business insights dictionary.
        filepath: Custom save path. Auto-generates if None.

    Returns:
        Path to the saved report.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = REPORTS_DIR / f"business_report_{timestamp}.txt"
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    hist = insights["historical"]
    fore = insights["forecast"]

    # Build report content
    lines = [
        "=" * 70,
        "  SALES FORECASTING — BUSINESS INTELLIGENCE REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "─" * 70,
        "  1. HISTORICAL PERFORMANCE",
        "─" * 70,
        f"  Date Range           : {df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to {df[DATE_COLUMN].max().strftime('%Y-%m-%d')}",
        f"  Total Records        : {len(df)}",
        f"  Total Sales          : ${hist['total_sales']:,.2f}",
        f"  Average Monthly Sales: ${hist['average_monthly_sales']:,.2f}",
        f"  Growth Rate (CAGR)   : {hist['growth_rate']:.2f}%",
        f"  Trend                : {hist['trend_direction']} (${hist['trend_slope']:,.2f}/month)",
        "",
        f"  Best Month           : {hist['best_month']['date'].strftime('%Y-%m')} (${hist['best_month']['sales']:,.2f})",
        f"  Worst Month          : {hist['worst_month']['date'].strftime('%Y-%m')} (${hist['worst_month']['sales']:,.2f})",
        "",
        "─" * 70,
        "  2. FORECAST MODEL",
        "─" * 70,
        f"  Selected Model       : {best_model_name}",
        f"  MAE                  : ${metrics['MAE']:,.2f}",
        f"  RMSE                 : ${metrics['RMSE']:,.2f}",
        f"  MAPE                 : {metrics['MAPE']:.2f}%",
        "",
        "─" * 70,
        "  3. FORECAST OUTLOOK",
        "─" * 70,
        f"  Forecast Periods     : {len(forecast_df)} months",
        f"  Total Forecast Sales : ${fore['total_forecast']:,.2f}",
        f"  Avg Monthly Forecast : ${fore['average_forecast']:,.2f}",
        f"  Expected Change      : {fore['change_percentage']:+.2f}%",
        "",
        "  Forecast Breakdown:",
    ]

    # Add per-period forecast
    for _, row in forecast_df.iterrows():
        date_str = row[DATE_COLUMN].strftime("%Y-%m") if hasattr(row[DATE_COLUMN], "strftime") else str(row[DATE_COLUMN])
        lines.append(f"    {date_str}  :  ${row['forecast']:,.2f}")

    lines.extend([
        "",
        "─" * 70,
        "  4. KEY TAKEAWAYS",
        "─" * 70,
    ])

    # Dynamic takeaways
    if hist["trend_direction"] == "Upward":
        lines.append("  ✅  Sales show a consistent upward trend.")
    elif hist["trend_direction"] == "Downward":
        lines.append("  ⚠️  Sales show a declining trend — review strategy.")
    else:
        lines.append("  ➖  Sales are relatively stable with no strong trend.")

    if fore["change_percentage"] > 5:
        lines.append(f"  ✅  Forecast indicates {fore['change_percentage']:.1f}% growth ahead.")
    elif fore["change_percentage"] < -5:
        lines.append(f"  ⚠️  Forecast suggests {abs(fore['change_percentage']):.1f}% decline ahead.")

    if metrics["MAPE"] < 10:
        lines.append("  ✅  Model accuracy is excellent (MAPE < 10%).")
    elif metrics["MAPE"] < 20:
        lines.append("  ✅  Model accuracy is good (MAPE < 20%).")
    elif metrics["MAPE"] < 50:
        lines.append("  ⚠️  Model accuracy is acceptable but could be improved.")
    else:
        lines.append("  ❌  Model accuracy is poor — consider more data or different models.")

    lines.extend([
        "",
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ])

    report_text = "\n".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Business report saved: {filepath}")
    return filepath


def display_report_summary(filepath: Path | str) -> str:
    """
    Read and return a saved report as a string.

    Args:
        filepath: Path to the report file.

    Returns:
        Report content string.

    Raises:
        FileNotFoundError: If report doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    return path.read_text(encoding="utf-8")
