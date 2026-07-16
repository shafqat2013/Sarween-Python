#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
CASES_PATH="${1:-tests/fixtures/tracking_cases.json}"

PYTHONDONTWRITEBYTECODE=1 SARWEEN_TRACKING_CASES="$CASES_PATH" "$PYTHON_BIN" -m unittest tests.test_tracking_regression
