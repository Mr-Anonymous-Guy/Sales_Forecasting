@echo off
:: ──────────────────────────────────────────────────────────────
:: run.bat — Launch Sales Forecasting System on Windows
:: ──────────────────────────────────────────────────────────────

cd /d "%~dp0"

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8 or newer.
    echo         Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Using:
python --version

echo [INFO] Checking dependencies...
python -c "import numpy, pandas, sklearn, statsmodels" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARN] Missing dependencies. Installing from requirements.txt...
    python -m pip install -r requirements.txt
)

echo [INFO] Starting Sales Forecasting System...
echo.

python main.py

pause
