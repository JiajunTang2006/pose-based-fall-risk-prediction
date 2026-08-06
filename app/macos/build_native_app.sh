#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/release/FallGuard.app"
CONTENTS="$APP/Contents"
AI_DIST="$ROOT/build/ai-dist/fallguard-ai"
IDENTITY="${CODE_SIGN_IDENTITY:--}"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"
CREATE_DMG="${CREATE_DMG:-0}"

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
    codesign --force --deep --entitlements "$ROOT/entitlements.plist" --sign - "$APP"
else
    log "Signing with identity: $IDENTITY"
    codesign --force --deep --options runtime --timestamp \
        --entitlements "$ROOT/entitlements.plist" \
        --sign "$IDENTITY" "$APP"
fi

codesign --verify --deep --strict --verbose=2 "$APP"

ARCHS="$(lipo -archs "$CONTENTS/MacOS/FallGuard")"
if [[ "$ARCHS" != *"arm64"* || "$ARCHS" != *"x86_64"* ]]; then
    log "Warning: this build is not Universal 2 (architectures: $ARCHS)"
fi

if [[ -n "$NOTARY_PROFILE" ]]; then
    [[ "$IDENTITY" != "-" ]] || fail "Notarization requires CODE_SIGN_IDENTITY"
    NOTARY_ARCHIVE="$ROOT/release/FallGuard-notarization.zip"
    log "Submitting app for notarization"
    ditto -c -k --keepParent "$APP" "$NOTARY_ARCHIVE"
    xcrun notarytool submit "$NOTARY_ARCHIVE" \
        --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP"
    xcrun stapler validate "$APP"
fi

if [[ "$CREATE_DMG" == "1" ]]; then
    DMG="$ROOT/release/FallGuard-${ARCHS// /-}.dmg"
    if [[ -e "$DMG" ]]; then
        BACKUP_DMG="$ROOT/release/FallGuard-${ARCHS// /-}.$(date +%Y%m%d-%H%M%S).dmg"
        mv "$DMG" "$BACKUP_DMG"
        log "Moved previous disk image to $BACKUP_DMG"
    fi
    log "Creating disk image"
    hdiutil create -volname FallGuard -srcfolder "$APP" -format UDZO "$DMG"
    if [[ -n "$NOTARY_PROFILE" ]]; then
        xcrun notarytool submit "$DMG" \
            --keychain-profile "$NOTARY_PROFILE" --wait
        xcrun stapler staple "$DMG"
    fi
fi

log "Built $APP"
du -sh "$APP"
