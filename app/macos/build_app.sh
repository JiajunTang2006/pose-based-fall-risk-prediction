#!/usr/bin/env zsh
set -euo pipefail

APP_DIR="${0:A:h}"

cd "$APP_DIR"

# Activate a local venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

ICON_ICNS="$APP_DIR/assets/FallGuard.icns"
ENTITLEMENTS="$APP_DIR/entitlements.plist"
APP_PLIST="$APP_DIR/dist/FallGuard.app/Contents/Info.plist"
ICON_ARG=()
VERSION="$(python - <<'PY'
from pathlib import Path
import tomllib

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
print(data["project"].get("version", "0.0.0"))
PY
)"
BUILD_NUMBER="$(date +%Y.%m.%d.%H%M)"
SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null | awk -F'\"' '/Developer ID Application|Apple Development|Mac Developer/ {print $2; exit}')"
if [[ -z "$SIGN_IDENTITY" ]]; then
  SIGN_IDENTITY="-"
fi

echo "==> Installing app dependencies..."
if python -c "import wheel" >/dev/null 2>&1; then
  python -m pip install --no-build-isolation -e "$APP_DIR"
else
  echo "Warning: wheel is not installed; using local source path for this build."
fi
export PYTHONPATH="$APP_DIR/src:${PYTHONPATH:-}"
export PYINSTALLER_CONFIG_DIR="$APP_DIR/.pyinstaller"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

echo "==> Installing build dependencies..."
if ! python -c "import PyInstaller" >/dev/null 2>&1; then
  python -m pip install "pyinstaller>=6.0"
fi

# ---- Build the app icon ----
if [[ -f "$APP_DIR/assets/FallGuard.png" ]]; then
  python "$APP_DIR/scripts/create_iconset.py"
  if python "$APP_DIR/scripts/create_icns.py"; then
    ICON_ARG=(--icon "$ICON_ICNS")
  elif iconutil -c icns "$APP_DIR/assets/FallGuard.iconset" -o "$ICON_ICNS"; then
    ICON_ARG=(--icon "$ICON_ICNS")
  else
    sips -s format tiff "$APP_DIR/assets/FallGuard.png" --out "$APP_DIR/assets/FallGuard.tiff" >/dev/null
    if tiff2icns "$APP_DIR/assets/FallGuard.tiff" "$ICON_ICNS"; then
      ICON_ARG=(--icon "$ICON_ICNS")
    else
      echo "Warning: could not generate FallGuard.icns; building without a packaged app icon."
    fi
  fi
fi

if [[ -f "$ICON_ICNS" ]]; then
  ICON_ARG=(--icon "$ICON_ICNS")
fi

echo "==> Building FallGuard.app with PyInstaller (optimized)..."

python -m PyInstaller \
  --noconfirm \
  --windowed \
  --name "FallGuard" \
  "${ICON_ARG[@]}" \
  --paths "$APP_DIR/src" \
  --add-data "$APP_DIR/models:models" \
  --add-data "$APP_DIR/assets:assets" \
  --hidden-import matplotlib \
  --hidden-import objc \
  --hidden-import rumps \
  --hidden-import joblib \
  --hidden-import sklearn \
  --hidden-import sklearn.ensemble \
  --hidden-import sklearn.ensemble._hist_gradient_boosting \
  --hidden-import sklearn.preprocessing \
  --hidden-import sklearn.tree \
  --hidden-import sklearn.utils \
  --hidden-import sklearn.base \
  --hidden-import sklearn.metrics \
  --hidden-import sklearn.model_selection \
  --hidden-import PySide6 \
  --hidden-import PySide6.QtWidgets \
  --hidden-import PySide6.QtGui \
  --hidden-import PySide6.QtCore \
  --hidden-import shiboken6 \
  --hidden-import fall_prediction \
  --hidden-import fall_prediction.camera \
  --hidden-import fall_prediction.runtime \
  --hidden-import fall_prediction.landmarks \
  --hidden-import fall_prediction.features \
  --hidden-import fall_prediction.risk \
  --hidden-import fall_prediction.predictor \
  --hidden-import fall_prediction.pose \
  --hidden-import fall_prediction.ml_features \
  --hidden-import fall_prediction.ml_predictor \
  --hidden-import fall_prediction.window_dataset \
  --hidden-import fall_prediction.config \
  --hidden-import fall_prediction.video_app \
  --hidden-import fall_prediction_desktop \
  --hidden-import fall_prediction_desktop.web_app \
  --hidden-import fall_prediction_desktop.menubar \
  --hidden-import fall_prediction_desktop.runner \
  --hidden-import fall_prediction_desktop.ui \
  --hidden-import fall_prediction_desktop.ui.main_window \
  --hidden-import fall_prediction_desktop.ui.settings_dialog \
  --hidden-import fall_prediction_desktop.ui.widgets \
  --hidden-import fall_prediction_desktop.ui.theme \
  --hidden-import fall_prediction_desktop.ui.i18n \
  --exclude-module IPython \
  --exclude-module jupyter \
  --exclude-module notebook \
  --exclude-module tensorboard \
  --exclude-module pytest \
  --exclude-module mediapipe \
  --exclude-module tkinter \
  --exclude-module _tkinter \
  --exclude-module PyQt5 \
  --exclude-module PyQt6 \
  --exclude-module PySide2 \
  --exclude-module wx \
  --exclude-module sphinx \
  --exclude-module docutils \
  --exclude-module pywebview \
  --exclude-module polars \
  --exclude-module _polars_runtime_32 \
  --exclude-module pyarrow \
  --exclude-module pandas \
  --exclude-module torch._inductor \
  --exclude-module torch._dynamo \
  --osx-bundle-identifier com.fallguard.desktop \
  --osx-entitlements-file "$ENTITLEMENTS" \
  "$APP_DIR/src/fall_prediction_desktop/__main__.py"

echo "==> Pruning packaged build intermediates..."
for RESOURCE_ASSETS in \
  "$APP_DIR/dist/FallGuard.app/Contents/Resources/assets" \
  "$APP_DIR/dist/FallGuard/_internal/assets"
do
  rm -rf "$RESOURCE_ASSETS/FallGuard.iconset"
  rm -f "$RESOURCE_ASSETS/FallGuard.tiff" "$RESOURCE_ASSETS/.DS_Store"
done

set_plist_value() {
  local key="$1"
  local type="$2"
  local value="$3"
  /usr/libexec/PlistBuddy -c "Add :$key $type $value" "$APP_PLIST" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Set :$key $value" "$APP_PLIST"
}

set_plist_bool() {
  local key="$1"
  local value="$2"
  /usr/libexec/PlistBuddy -c "Add :$key bool $value" "$APP_PLIST" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Set :$key $value" "$APP_PLIST"
}

echo "==> Writing professional app metadata..."
set_plist_value CFBundleDisplayName string "FallGuard"
set_plist_value CFBundleName string "FallGuard"
set_plist_value CFBundleIdentifier string "com.fallguard.desktop"
set_plist_value CFBundleShortVersionString string "$VERSION"
set_plist_value CFBundleVersion string "$BUILD_NUMBER"
set_plist_value CFBundleDevelopmentRegion string "zh_CN"
set_plist_value LSMinimumSystemVersion string "11.0"
set_plist_value LSApplicationCategoryType string "public.app-category.healthcare-fitness"
set_plist_value NSCameraUsageDescription string "FallGuard uses the camera to analyze posture locally for real-time fall detection. Video stays on this Mac."
set_plist_value NSHumanReadableCopyright string "Copyright © 2026 FallGuard. All rights reserved."
set_plist_bool NSHighResolutionCapable true
set_plist_bool NSSupportsAutomaticGraphicsSwitching true

echo "==> Signing FallGuard.app with identity: $SIGN_IDENTITY"
codesign --force --deep --options runtime --entitlements "$ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$APP_DIR/dist/FallGuard.app"

echo "==> Verifying app signature..."
codesign --verify --deep --strict --verbose=2 "$APP_DIR/dist/FallGuard.app"

echo "==> Done!  FallGuard.app is in dist/"
echo "    Size: $(du -sh dist/FallGuard.app | cut -f1)"

# Auto-deploy to Desktop so the user always has the latest version.
DESKTOP_APP="$HOME/Desktop/FallGuard.app"
if [ -d "$DESKTOP_APP" ]; then
  rm -rf "$DESKTOP_APP"
fi
cp -R "$APP_DIR/dist/FallGuard.app" "$DESKTOP_APP"
echo "==> Deployed to Desktop: $DESKTOP_APP"
