#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON_BIN="$FALL_PREDICTION_PYTHON"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "$PYTHON_BIN was not found. Create the Python 3.11 environment described in README.md first."
  exit 1
fi

echo "[1/3] Verifying final data, models, and reference report"
shasum -a 256 -c FINAL_ARTIFACTS.sha256

echo "[2/3] Regenerating the final five-fold report"
OUTPUT_PATH="/tmp/fall_prediction_reproduced_final.json"
"$PYTHON_BIN" scripts/evaluate_dual_model_cv.py \
  --source-report reports/fusion_grouped_5fold_cv_full_outer.json \
  --input-dir outputs/features \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --model-dir models/cross_validation_tuned \
  --fusion-model-pattern "fold_{fold}_selected_full_outer.pt" \
  --output "$OUTPUT_PATH"

cmp "$OUTPUT_PATH" reports/dual_model_tuned_static_lying_postprocess_5fold_cv.json
echo "The regenerated five-fold report is byte-for-byte identical."

echo "[3/3] Running integrity checks"
if [ "$1" = "--with-tests" ]; then
  if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "pytest is not installed. Install the development dependencies: $PYTHON_BIN -m pip install -e '.[dev]'"
    exit 1
  fi
  "$PYTHON_BIN" -m pytest -q
else
  echo "Tests skipped. Pass --with-tests to run them."
fi

echo "Final-result reproduction passed."
