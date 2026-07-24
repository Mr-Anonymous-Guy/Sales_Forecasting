"""
main.py — CLI entry point for Sales Forecasting System.

Run with:
    python main.py
"""

import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.analytics import (
    calculate_growth_rate,
    calculate_month_over_month_growth,
    calculate_year_over_year_growth,
    detect_trend,
    format_business_insights,
    generate_business_insights,
    identify_seasonality,
)
from src.config import DATA_DIR, FORECAST_PERIODS, MODELS_DIR, REPORTS_DIR
from src.data_loader import (
    check_missing_values,
    compute_statistics,
    display_statistics,
    handle_missing_values,
    load_sales_data,
    validate_time_series,
)
from src.evaluation import compare_models, display_metrics
from src.model_manager import (
    auto_select_best,
    load_model,
    train_and_evaluate_all,
)
from src.reporter import (
    export_evaluation_csv,
    export_forecast_csv,
    generate_summary_report,
)
from src.visualizations import (
    plot_forecast,
    plot_historical_sales,
    plot_model_comparison,
)


# ──────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          📈  SALES FORECASTING SYSTEM  📊            ║
║          Time Series Analysis & Prediction            ║
╚══════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    """Display the application banner."""
    print(BANNER)


def print_menu() -> None:
    """Display the main menu."""
    print("\n" + "=" * 56)
    print("  MAIN MENU")
    print("=" * 56)
    print("  [1]  Load & Explore Dataset")
    print("  [2]  Train Forecasting Models")
    print("  [3]  Generate Forecasts")
    print("  [4]  Business Analytics")
    print("  [5]  Visualizations")
    print("  [6]  Export Reports")
    print("=" * 56)
    print("  [Q]  Quit")
    print("=" * 56)


def get_input(prompt: str) -> str:
    """Safe input wrapper."""
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  👋  Goodbye!\n")
        sys.exit(0)


# ──────────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────────

# Cache to avoid reloading data across menu actions
_state: dict = {
    "df": None,
    "results": None,
    "best": None,
}


def _load_data() -> None:
    """Load and cache sales data if not already loaded."""
    if _state["df"] is None:
        dataset_path = DATA_DIR / "sales_data.csv"
        df = load_sales_data(dataset_path)
        df = handle_missing_values(df, method="interpolate")
        _state["df"] = df


# ──────────────────────────────────────────────────────────────
# Menu actions
# ──────────────────────────────────────────────────────────────


def action_load_and_explore() -> None:
    """Load the dataset and display summary statistics."""
    print("\n" + "=" * 56)
    print("  📂  Loading Sales Dataset...")
    print("=" * 56)

    try:
        dataset_path = DATA_DIR / "sales_data.csv"
        df = load_sales_data(dataset_path)

        # Validate time series
        is_valid, issues = validate_time_series(df)

        if not is_valid:
            print("\n  ⚠️  Validation issues:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print("\n  ✅  Time series validation passed!")

        # Check missing values
        missing = check_missing_values(df)
        has_missing = any(v > 0 for v in missing.values())
        if has_missing:
            print("\n  ⚠️  Missing values detected:")
            for col, count in missing.items():
                if count > 0:
                    print(f"      {col}: {count}")
            print("  Applying interpolation...")
            df = handle_missing_values(df, method="interpolate")
        else:
            print("  ✅  No missing values!")

        # Cache data
        _state["df"] = df

        # Display statistics
        print(display_statistics(df))

        get_input("\n  Press ENTER to continue...")

    except Exception as exc:
        print(f"\n  ❌  Error loading dataset: {exc}\n")


def action_train_models() -> None:
    """Train all forecasting models and select the best one."""
    print("\n" + "=" * 56)
    print("  🚀  Training Forecasting Pipeline")
    print("=" * 56)

    try:
        _load_data()
        df = _state["df"]

        print("\n  Training 4 models on historical data...")
        print("  (Linear Regression, Random Forest, ARIMA, Prophet)\n")

        results = train_and_evaluate_all(df, test_size=12)

        if not results:
            print("  ❌  No models trained successfully.\n")
            return

        # Display results
        print("  " + "─" * 52)
        for r in results:
            print(f"  {r['model_name']}")
            print(f"    MAE  : ${r['MAE']:,.2f}")
            print(f"    RMSE : ${r['RMSE']:,.2f}")
            print(f"    MAPE : {r['MAPE']:.2f}%")
            print("  " + "─" * 52)

        # Auto-select best
        best = auto_select_best(results)
        _state["results"] = results
        _state["best"] = best

        print(f"\n  🏆  Best Model: {best['model_name']}")
        print(f"      MAPE: {best['MAPE']:.2f}%")
        print(f"      Saved to: {MODELS_DIR / 'best_model.pkl'}\n")

        get_input("  Press ENTER to continue...")

    except Exception as exc:
        print(f"\n  ❌  Training failed: {exc}\n")


def action_generate_forecasts() -> None:
    """Generate forecasts for selected horizon."""
    print("\n" + "=" * 56)
    print("  🔮  Generate Forecasts")
    print("=" * 56)

    try:
        # Load best model
        best_path = MODELS_DIR / "best_model.pkl"
        if not best_path.exists():
            print("\n  ⚠️  No trained model found. Please train models first (Option 2).\n")
            return

        _load_data()
        df = _state["df"]

        payload = load_model(best_path)
        forecaster = payload["forecaster"]
        model_name = payload["model_name"]

        print(f"\n  Using: {model_name}")
        print("\n  Select forecast horizon:")
        print("  [1]  Next Month (1 period)")
        print("  [2]  Next Quarter (3 periods)")
        print("  [3]  Next Year (12 periods)")

        choice = get_input("\n  Choose: ")

        horizon_map = {
            "1": ("month", FORECAST_PERIODS["month"]),
            "2": ("quarter", FORECAST_PERIODS["quarter"]),
            "3": ("year", FORECAST_PERIODS["year"]),
        }

        if choice not in horizon_map:
            print("  ⚠️  Invalid choice.")
            return

        horizon_name, periods = horizon_map[choice]

        print(f"\n  Forecasting {periods} period(s)...\n")

        # Re-fit on full data for production forecast
        forecaster.is_fitted = False
        forecaster.__init__() if hasattr(forecaster, '__init__') else None

        # Re-create forecaster of same type
        from src.forecasters import get_all_forecasters
        for f in get_all_forecasters():
            if f.name == model_name:
                forecaster = f
                break

        forecaster.fit(df)
        forecast_df = forecaster.predict(periods=periods)

        # Display forecast
        print("  " + "─" * 40)
        print(f"  📅  {horizon_name.upper()} FORECAST")
        print("  " + "─" * 40)

        for _, row in forecast_df.iterrows():
            date_str = row["date"].strftime("%Y-%m") if hasattr(row["date"], "strftime") else str(row["date"])
            print(f"    {date_str}  :  ${row['forecast']:,.2f}")

        total = forecast_df["forecast"].sum()
        print("  " + "─" * 40)
        print(f"    Total      :  ${total:,.2f}")
        print("  " + "─" * 40)

        get_input("\n  Press ENTER to continue...")

    except Exception as exc:
        print(f"\n  ❌  Forecast failed: {exc}\n")


def action_business_analytics() -> None:
    """Display business intelligence analytics."""
    print("\n" + "=" * 56)
    print("  📊  Business Analytics")
    print("=" * 56)

    try:
        _load_data()
        df = _state["df"]

        # Trend detection
        direction, slope = detect_trend(df)
        growth = calculate_growth_rate(df)
        seasonality = identify_seasonality(df)

        print("\n  ── Growth Analysis ──")
        print(f"  CAGR           : {growth:.2f}%")
        print(f"  Trend          : {direction} (${slope:,.2f}/month)")

        print("\n  ── Seasonal Patterns (Average by Month) ──")
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        for month_num, avg_sales in sorted(seasonality.items()):
            bar = "█" * int(avg_sales / 2000)
            print(f"    {month_names[month_num - 1]:3s}: ${avg_sales:>10,.2f}  {bar}")

        # If model is trained, show forecast insights
        best_path = MODELS_DIR / "best_model.pkl"
        if best_path.exists():
            payload = load_model(best_path)
            forecaster = payload["forecaster"]

            # Re-fit and forecast
            from src.forecasters import get_all_forecasters
            for f in get_all_forecasters():
                if f.name == payload["model_name"]:
                    forecaster = f
                    break

            forecaster.fit(df)
            forecast_df = forecaster.predict(periods=3)

            insights = generate_business_insights(df, forecast_df)
            print(format_business_insights(insights))
        else:
            print("\n  💡  Train models (Option 2) to see forecast insights.")

        get_input("\n  Press ENTER to continue...")

    except Exception as exc:
        print(f"\n  ❌  Analytics failed: {exc}\n")


def action_visualizations() -> None:
    """Generate and display visualizations."""
    print("\n" + "=" * 56)
    print("  📊  Visualizations")
    print("=" * 56)

    try:
        _load_data()
        df = _state["df"]

        print("\n  Select visualization:")
        print("  [1]  Historical Sales Chart")
        print("  [2]  Forecast Chart (requires trained model)")
        print("  [3]  Model Comparison (requires training)")

        choice = get_input("\n  Choose: ")

        if choice == "1":
            save_path = REPORTS_DIR / "historical_sales.png"
            plot_historical_sales(df, save_path=save_path)
            print(f"\n  💾  Chart saved: {save_path}")

        elif choice == "2":
            best_path = MODELS_DIR / "best_model.pkl"
            if not best_path.exists():
                print("\n  ⚠️  No trained model found. Train first (Option 2).")
                return

            payload = load_model(best_path)

            from src.forecasters import get_all_forecasters
            for f in get_all_forecasters():
                if f.name == payload["model_name"]:
                    forecaster = f
                    break

            forecaster.fit(df)
            forecast_df = forecaster.predict(periods=12)

            save_path = REPORTS_DIR / "forecast_chart.png"
            plot_forecast(df, forecast_df, model_name=payload["model_name"], save_path=save_path)
            print(f"\n  💾  Chart saved: {save_path}")

        elif choice == "3":
            if _state["results"] is None:
                print("\n  ⚠️  Please train models first (Option 2).")
                return

            comparison_df = compare_models(_state["results"])
            save_path = REPORTS_DIR / "model_comparison.png"
            plot_model_comparison(comparison_df, save_path=save_path)
            print(f"\n  💾  Chart saved: {save_path}")

        else:
            print("  ⚠️  Invalid choice.")

    except Exception as exc:
        print(f"\n  ❌  Visualization failed: {exc}\n")


def action_export_reports() -> None:
    """Export reports to CSV and text files."""
    print("\n" + "=" * 56)
    print("  📋  Export Reports")
    print("=" * 56)

    try:
        _load_data()
        df = _state["df"]

        best_path = MODELS_DIR / "best_model.pkl"
        if not best_path.exists():
            print("\n  ⚠️  No trained model found. Please train models first (Option 2).\n")
            return

        payload = load_model(best_path)
        model_name = payload["model_name"]
        metrics = payload["metrics"]

        # Re-fit and forecast
        from src.forecasters import get_all_forecasters
        for f in get_all_forecasters():
            if f.name == model_name:
                forecaster = f
                break

        forecaster.fit(df)
        forecast_df = forecaster.predict(periods=12)

        # 1. Export forecast CSV
        csv_path = export_forecast_csv(forecast_df, model_name)
        print(f"\n  ✅  Forecast CSV  : {csv_path}")

        # 2. Export evaluation CSV (if results available)
        if _state["results"]:
            eval_path = export_evaluation_csv(_state["results"])
            print(f"  ✅  Evaluation CSV: {eval_path}")

        # 3. Generate business report
        insights = generate_business_insights(df, forecast_df)
        report_path = generate_summary_report(
            df=df,
            forecast_df=forecast_df,
            best_model_name=model_name,
            metrics=metrics,
            insights=insights,
        )
        print(f"  ✅  Business Report: {report_path}")

        print(f"\n  📁  All reports saved to: {REPORTS_DIR}\n")

        get_input("  Press ENTER to continue...")

    except Exception as exc:
        print(f"\n  ❌  Report export failed: {exc}\n")


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry point."""
    while True:
        print_banner()
        print_menu()

        choice = get_input("  Choose an option: ")

        if choice == "q":
            print("\n" + "=" * 56)
            print("  👋  Thank you for using Sales Forecasting System!")
            print("=" * 56 + "\n")
            sys.exit(0)

        elif choice == "1":
            action_load_and_explore()
        elif choice == "2":
            action_train_models()
        elif choice == "3":
            action_generate_forecasts()
        elif choice == "4":
            action_business_analytics()
        elif choice == "5":
            action_visualizations()
        elif choice == "6":
            action_export_reports()
        else:
            print("\n  ⚠️  Invalid choice. Please try again.\n")
            get_input("  Press ENTER to continue...")


if __name__ == "__main__":
    main()
