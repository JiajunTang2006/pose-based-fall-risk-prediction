# FallGuard — Known Issues & Feature Status

> Baseline: `baseline-ui-complete` (2026-07-10)
> Based on: FallGuard Development Workflow V1.0

## Current Environment

| Item | Value |
|------|-------|
| Python | 3.11.9 |
| venv | `.venv/` |
| Entry point | `python -m fall_prediction_desktop` |
| Launch script | `./launch.command` |
| Build script | `./build_app.sh` |
| Models | `models/yolo26n-pose.pt`, `models/yolo_tail60_prefall_accel_upperbody_classifier.joblib` |

## Button / Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| Dashboard layout | ✅ Complete | M3 Design, i18n (en/zh), light/dark theme |
| Start/Stop Monitoring | ✅ Working | Camera → YOLO Pose → ML Predictor → Risk |
| Settings UI | ✅ Complete | Sensitivity, Theme, Language, Profiles UI |
| Settings persistence | ✅ SQLite | JSON retained only as legacy migration cache |
| Profiles | ✅ SQLite synced | UI and Session use the same Active Profile |
| Recent Events | ✅ SQLite | Dashboard reads persisted business events |
| Risk Trend chart | ✅ SQLite-backed | Dashboard reads the current/recent Session's latest 60 seconds; historical browsing filters remain future work |
| Risk State Machine | ✅ Working | EMA, confirmation, hysteresis, cooldown, recovery, pose-loss tolerance |
| Import Media | ✅ Working | Video/image → annotated MP4 output |
| Export Logs | ✅ Working | Exports SQLite profiles, Sessions, Events, Samples and Media as JSON or CSV/ZIP |
| Dataset Management | ✅ Working | Lists tracked media, reveals files in Finder and safely deletes managed files |
| Clear History | ✅ Working | Clears SQLite history, legacy profile events and managed media files |
| Sound Alert | ✅ Working | Local macOS sound on confirmed warning/fall escalation with cooldown |
| Popup Alert | ⏸️ Deferred | Extension point retained; macOS notifications intentionally not implemented yet |
| Email Notification | ⏸️ Deferred | Disabled toggle, marked "reserved" |
| Start on Boot | ⏸️ Deferred | Extension point retained and UI disabled; intentionally not implemented yet |
| Minimize to Tray | ✅ Working | Qt menu bar icon provides Show, Start, Stop and Quit actions |
| SQLite Database | ✅ Working | Schema bundled and initialized under Application Support |
| Monitoring Sessions | ✅ Working | Start/Stop/Error lifecycle and interrupted-session recovery |
| Risk Samples | ✅ Working | Periodic writes with final flush on Stop |
| Event video buffer | ✅ Working | Bounded JPEG pre-roll, event thumbnail and post-roll MP4 evidence |

## Known Issues

1. Risk trend uses persisted SQLite samples for the latest 60 seconds, but profile/session/time-range browsing controls are not implemented
2. Event clip encoding is currently synchronous; a bounded background writer would reduce frame-loop latency on slower Macs
3. Export covers complete SQLite table data, but filter controls and optional media-file bundling are not implemented
4. macOS desktop notifications and start-on-boot are intentionally deferred; disabled extension points remain in Settings
5. Dataset ownership is inferred from the managed media root rather than an explicit schema ownership field

## Architecture Gaps (from Workflow Doc)

- [x] Repository layer for Settings, Profiles, Sessions, Samples, Events and Media
- [x] Formal RiskStateMachine and EventService
- [x] Periodic risk sampling instead of per-frame database commits
- [x] Camera and Import Media share `FrameBusinessProcessor` for FSM, Events and Risk Samples
- [x] SoundAlertService, ExportService and event media buffer
- [ ] macOS notification and login-item adapters (intentionally deferred)

## App Size & Optimization

The existing `dist/FallGuard.app` artifact is 693 MiB and predates the latest source changes. Its largest packaged components are approximately: `torch` 274 MiB, `cv2` 118 MiB, PySide6 71 MiB, scikit-learn 14 MiB, and model files 19 MiB.

Recommended cleanup order:

1. **Replace the PyTorch/Ultralytics runtime with Core ML or ONNX inference** — highest potential saving (roughly 200–280 MiB), but requires validating that the exported pose model preserves the current keypoint/output contract.
2. **Reduce the OpenCV package** — evaluate a headless or custom macOS build containing only camera capture, image codecs, drawing and video-writing modules; do not exclude `cv2` blindly because these paths are used at runtime.
3. **Remove the legacy UI stack if PySide6 remains the only production UI** — stop packaging `pywebview`, `rumps`, Cocoa webview imports and legacy web assets after confirming no supported launch mode depends on them.
4. **Prune unused Qt modules and plugins** — retain QtCore, QtGui and QtWidgets, then use the PyInstaller analysis/TOC to identify unused QML, WebEngine, translations and platform plugins before adding exclusions.
5. **Consider lighter chart and classifier runtimes** — replacing Matplotlib with Qt painting and exporting the scikit-learn classifier to a lightweight runtime offers smaller secondary savings.
6. **Do not prioritize model compression yet** — model files are only about 19 MiB, so reducing them has much less impact than removing large runtimes and may unnecessarily reduce detection quality.

Do not apply package exclusions without rebuilding and smoke-testing Camera, Import Media, event clip encoding, sound alerts and the menu-bar lifecycle. A new PyInstaller build and measurement are still required before recording the final application size.

## Engineering review fixes (2026-07-10)

- Fixed startup deadlock in `DatabaseManager.initialize()` that left the app running without a window
- Bundled `schema.sql` and added a visible startup error dialog for database failures
- Prevented worker thread self-join so monitoring Sessions always close cleanly
- Moved all mutable data outside `.app` to preserve the code signature
- Added stale Session recovery, Profile threshold validation, accurate Session aggregates and database-backed Recent Events
- Added database/profile and desktop regression tests; current suite: 35 passing
- Added shared Camera/Import frame orchestration, bound-session EventService and periodic SQLite risk history
- Added event thumbnails and bounded pre/post-roll clips, complete data export, clear-history and Dataset Management
- Added local sound alerts and a Qt menu bar icon; macOS notifications and login items remain intentionally deferred
- Python source/test compilation passes; final PyInstaller rebuild is pending local command permission. The existing `dist/FallGuard.app` artifact is 693 MiB and predates the latest changes.
