#!/bin/zsh
set -euo pipefail

APP_DIR="${0:A:h}"

cd "$APP_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

export PYTHONPATH="$APP_DIR/src:${PYTHONPATH:-}"

echo "================================================"
echo "  FallGuard — Smart Safety"
echo "================================================"
echo "  Launching native desktop window..."
echo ""

python -m fall_prediction_desktop "$@"
