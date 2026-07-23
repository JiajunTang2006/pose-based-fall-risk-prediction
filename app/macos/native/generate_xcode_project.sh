#!/usr/bin/env bash
# Generate Xcode project for FallGuard native macOS app.
#
# Usage:
#   ./generate_xcode_project.sh          # Create & open the Xcode project
#   ./generate_xcode_project.sh build    # Build the project
#   ./generate_xcode_project.sh run      # Build & run
#
# Prerequisites:
#   - Xcode 14+ (or Command Line Tools)
#   - macOS 11+
#
# This script uses `xcodebuild` and `swift package` to create a
# working Xcode project.  The generated project lives at
# native/FallGuard/FallGuard.xcodeproj.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/FallGuard"
SCHEME="FallGuard"

# ── colour helpers ──────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── verify tools ────────────────────────────────────────────────────

if ! command -v xcodebuild &>/dev/null; then
    error "xcodebuild not found. Install Xcode or Xcode Command Line Tools."
    exit 1
fi

info "Xcode version: $(xcodebuild -version | head -1)"

# ── locate source files ─────────────────────────────────────────────

SOURCES=(
    "App/FallGuardApp.swift"
    "App/AppDelegate.swift"
    "App/AppStore.swift"
    "App/DesignSystem.swift"
    "App/GlassEffect.swift"
    "App/MenuBarController.swift"
    "App/ThemeManager.swift"
    "Models/ServiceModels.swift"
    "Services/PythonServiceManager.swift"
    "Services/FallGuardAPIClient.swift"
    "Services/StatusPoller.swift"
    "Services/PreviewClient.swift"
    "Services/NotificationService.swift"
    "Services/PermissionService.swift"
    "Features/ContentView.swift"
    "Features/Dashboard/DashboardView.swift"
    "Features/Events/EventsView.swift"
    "Features/ImportMedia/ImportMediaView.swift"
    "Features/Profiles/ProfilesView.swift"
    "Features/Settings/SettingsView.swift"
)

# Verify all sources exist
info "Verifying source files…"
for src in "${SOURCES[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$src" ]]; then
        error "Missing: $src"
        exit 1
    fi
done
info "All ${#SOURCES[@]} source files present."

# ── build using swiftc (command-line) ───────────────────────────────

build_cli() {
    info "Building FallGuard (command-line mode)…"

    local SDK_PATH
    SDK_PATH=$(xcrun --show-sdk-path --sdk macosx)

    # Collect all source files
    local SRC_FILES=()
    for src in "${SOURCES[@]}"; do
        SRC_FILES+=("$PROJECT_DIR/$src")
    done

    # Compile
    swiftc \
        -sdk "$SDK_PATH" \
        -target "$(uname -m)-apple-macos12.0" \
        -F "$SDK_PATH/System/Library/Frameworks" \
        -framework SwiftUI \
        -framework AppKit \
        -framework AVFoundation \
        -framework UserNotifications \
        -framework UniformTypeIdentifiers \
        -framework ServiceManagement \
        -parse-as-library \
        -o "$PROJECT_DIR/.build/FallGuard" \
        "${SRC_FILES[@]}" \
        -Xlinker -rpath -Xlinker @executable_path/../Frameworks

    info "Build successful → native/FallGuard/.build/FallGuard"
}

# ── build using xcodebuild (if .xcodeproj exists) ───────────────────

build_xcode() {
    local XCODEPROJ="$PROJECT_DIR/FallGuard.xcodeproj"

    if [[ ! -d "$XCODEPROJ" ]]; then
        warn "No .xcodeproj found.  Open the sources in Xcode and create one,"
        warn "or use:  ./generate_xcode_project.sh cli"
        warn ""
        warn "To create an Xcode project:"
        warn "  1. Open Xcode"
        warn "  2. File → New → Project → macOS → App"
        warn "  3. Product Name: FallGuard"
        warn "  4. Interface: SwiftUI, Language: Swift"
        warn "  5. Save at: native/FallGuard/"
        warn "  6. Drag all .swift files from Finder into the project navigator"
        return 1
    fi

    info "Building with xcodebuild…"
    xcodebuild build \
        -project "$XCODEPROJ" \
        -scheme "$SCHEME" \
        -destination 'platform=macOS' \
        -configuration Release \
        CODE_SIGN_IDENTITY="-" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        2>&1 | tail -20

    info "Build complete."
}

# ── run the app ─────────────────────────────────────────────────────

run_app() {
    local APP="$PROJECT_DIR/.build/FallGuard"
    if [[ -x "$APP" ]]; then
        info "Launching FallGuard…"
        FALLGUARD_AI_EXECUTABLE="${FALLGUARD_AI_EXECUTABLE:-$SCRIPT_DIR/../.venv/bin/fallguard-ai}" "$APP" &
        info "PID: $!"
    else
        warn "Build first: ./generate_xcode_project.sh cli"
    fi
}

# ── main ────────────────────────────────────────────────────────────

case "${1:-}" in
    cli|build)
        mkdir -p "$PROJECT_DIR/.build"
        build_cli
        ;;
    run)
        mkdir -p "$PROJECT_DIR/.build"
        build_cli && run_app
        ;;
    xcode)
        build_xcode
        ;;
    *)
        echo "Usage: $0 {cli|run|xcode}"
        echo ""
        echo "  cli     Build using swiftc (no Xcode project needed)"
        echo "  run     Build & run using swiftc"
        echo "  xcode   Build using Xcode project (requires .xcodeproj)"
        ;;
esac
