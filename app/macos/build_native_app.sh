#!/usr/bin/env bash
# Build the SwiftUI shell and bundled Python AI service into FallGuard.app.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/release/FallGuard.app"
CONTENTS="$APP/Contents"
AI_DIST="$ROOT/build/ai-dist/fallguard-ai"
IDENTITY="${CODE_SIGN_IDENTITY:--}"

log() { printf '[FallGuard] %s\n' "$*"; }
fail() { printf '[FallGuard] ERROR: %s\n' "$*" >&2; exit 1; }

command -v swiftc >/dev/null || fail "swiftc not found"
command -v xcrun >/dev/null || fail "xcrun not found"
command -v codesign >/dev/null || fail "codesign not found"
[[ -x "$ROOT/.venv/bin/pyinstaller" ]] || fail "Run: .venv/bin/pip install -e '.[build]'"

mkdir -p "$ROOT/build/ai-dist" "$ROOT/build/pyinstaller" "$ROOT/release"

if [[ "${SKIP_AI_BUILD:-0}" != "1" ]]; then
    log "Building headless Python service"
    PYINSTALLER_CONFIG_DIR="$ROOT/build/pyinstaller-config" \
    "$ROOT/.venv/bin/pyinstaller" \
        --noconfirm \
        --distpath "$ROOT/build/ai-dist" \
        --workpath "$ROOT/build/pyinstaller" \
        "$ROOT/fallguard_ai.spec"
fi

[[ -x "$AI_DIST/fallguard-ai" ]] || fail "AI service build missing: $AI_DIST/fallguard-ai"

log "Compiling SwiftUI application"
CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-/tmp/fallguard-clang-cache}" \
SWIFT_MODULECACHE_PATH="${SWIFT_MODULECACHE_PATH:-/tmp/fallguard-swift-cache}" \
    "$ROOT/native/generate_xcode_project.sh" cli

if [[ -e "$APP" ]]; then
    BACKUP="$ROOT/release/FallGuard.$(date +%Y%m%d-%H%M%S).app"
    mv "$APP" "$BACKUP"
    log "Moved previous build to $BACKUP"
fi
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources/AIService"

ditto "$ROOT/native/FallGuard/.build/FallGuard" "$CONTENTS/MacOS/FallGuard"
ditto "$AI_DIST" "$CONTENTS/Resources/AIService"
ditto "$ROOT/native/FallGuard/Resources/Info.plist" "$CONTENTS/Info.plist"
ditto "$ROOT/assets/FallGuard.icns" "$CONTENTS/Resources/AppIcon.icns"
ditto "$ROOT/native/FallGuard/Resources/en.lproj" "$CONTENTS/Resources/en.lproj"
ditto "$ROOT/native/FallGuard/Resources/zh.lproj" "$CONTENTS/Resources/zh.lproj"
chmod 755 "$CONTENTS/MacOS/FallGuard" "$CONTENTS/Resources/AIService/fallguard-ai"

plutil -lint "$CONTENTS/Info.plist"

if [[ "$IDENTITY" == "-" ]]; then
    log "Applying ad-hoc signature (no Apple signing identity was supplied)"
    codesign --force --deep --sign - "$APP"
else
    log "Signing with identity: $IDENTITY"
    codesign --force --deep --options runtime --timestamp --sign "$IDENTITY" "$APP"
fi

codesign --verify --deep --strict --verbose=2 "$APP"
log "Built $APP"
du -sh "$APP"
