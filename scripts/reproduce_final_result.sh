#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON_BIN="$FALL_PREDICTION_PYTHON"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 $PYTHON_BIN。请先按照 README.md 建立 Python 3.11 环境。"
  exit 1
fi

echo "[1/3] 校验最终数据、模型与参考报告"
shasum -a 256 -c FINAL_ARTIFACTS.sha256

echo "[2/3] 重新生成最终五折报告"
OUTPUT_PATH="/tmp/fall_prediction_reproduced_final.json"
"$PYTHON_BIN" scripts/evaluate_dual_model_cv.py \
  --source-report reports/fusion_grouped_5fold_cv_full_outer.json \
  --input-dir outputs/features \
  --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
  --model-dir models/cross_validation_tuned \
  --fusion-model-pattern "fold_{fold}_selected_full_outer.pt" \
  --output "$OUTPUT_PATH"

cmp "$OUTPUT_PATH" reports/dual_model_tuned_static_lying_postprocess_5fold_cv.json
echo "最终五折报告逐字节一致。"

echo "[3/3] 基础完整性检查"
if [ "$1" = "--with-tests" ]; then
  if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "当前环境没有 pytest；请安装开发依赖：$PYTHON_BIN -m pip install -e '.[dev]'"
    exit 1
  fi
  "$PYTHON_BIN" -m pytest -q
else
  echo "已跳过自动测试；传入 --with-tests 可运行。"
fi

echo "最终版本复现验证通过。"
