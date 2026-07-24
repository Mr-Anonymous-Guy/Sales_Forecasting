#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run.sh — Launch Sales Forecasting System on Linux / macOS
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python interpreter detection ──────────────────────────────
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "❌  Python not found. Please install Python 3.8 or newer."
    exit 1
fi

# ── Version guard ─────────────────────────────────────────────
PYVER=$("$PY" -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)")
if [ "$PYVER" -lt 38 ]; then
    echo "❌  Python 3.8+ required. Found: $($PY --version)"
    exit 1
fi

echo "▶  Using $($PY --version)"

# ── Dependency check ──────────────────────────────────────────
if ! "$PY" -c "import numpy, pandas, sklearn, statsmodels" &>/dev/null; then
    echo ""
    echo "⚠️  Missing dependencies. Installing from requirements.txt..."
    "$PY" -m pip install -r requirements.txt
fi

echo "▶  Starting Sales Forecasting System…"
echo ""

"$PY" main.py
