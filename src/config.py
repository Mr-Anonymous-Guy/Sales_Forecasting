"""
config.py — Configuration and constants for Sales Forecasting system.

Centralizes all settings for data paths, model parameters, and visualization.
"""

from pathlib import Path
from typing import Final

# ──────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Data configuration
# ──────────────────────────────────────────────────────────────

DATE_COLUMN: Final[str] = "date"
TARGET_COLUMN: Final[str] = "sales"
FREQ: Final[str] = "MS"  # Month Start frequency

# ──────────────────────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────────────────────

# Random Forest
RF_N_ESTIMATORS: Final[int] = 100
RF_MAX_DEPTH: Final[int] = 10
RF_RANDOM_STATE: Final[int] = 42

# ARIMA
ARIMA_ORDER: Final[tuple[int, int, int]] = (1, 1, 1)  # (p, d, q)

# Prophet
PROPHET_SEASONALITY_MODE: Final[str] = "multiplicative"
PROPHET_YEARLY_SEASONALITY: Final[bool] = True
PROPHET_WEEKLY_SEASONALITY: Final[bool] = False
PROPHET_DAILY_SEASONALITY: Final[bool] = False

# ──────────────────────────────────────────────────────────────
# Forecasting horizons
# ──────────────────────────────────────────────────────────────

FORECAST_PERIODS: Final[dict[str, int]] = {
    "month": 1,
    "quarter": 3,
    "year": 12,
}

# ──────────────────────────────────────────────────────────────
# Evaluation thresholds
# ──────────────────────────────────────────────────────────────

MAPE_EXCELLENT: Final[float] = 10.0  # < 10% is excellent
MAPE_GOOD: Final[float] = 20.0       # < 20% is good
MAPE_ACCEPTABLE: Final[float] = 50.0 # < 50% is acceptable

# ──────────────────────────────────────────────────────────────
# Visualization settings
# ──────────────────────────────────────────────────────────────

FIGURE_SIZE: Final[tuple[int, int]] = (12, 6)
DPI: Final[int] = 150
STYLE: Final[str] = "seaborn-v0_8-darkgrid"

COLORS: Final[dict[str, str]] = {
    "actual": "#2E86AB",
    "forecast": "#A23B72",
    "confidence": "#F18F01",
    "trend": "#C73E1D",
}

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: Final[str] = "INFO"
