# FallGuard — Native macOS Desktop App

FallGuard is a standalone macOS desktop application for AI-powered fall and pre-fall monitoring.

It uses a local camera to analyze human pose in real time and provides warnings when a potential fall-related state is detected.

This app is part of the larger **Pose-Based Fall and Pre-Fall Prediction System** project.

---

## Overview

FallGuard is designed as a local macOS prototype for testing and demonstrating a pose-based fall prediction system.

The application combines:

- A product-style frontend interface
- A Python-based local backend
- YOLO-based human pose estimation
- Machine-learning-based Normal / Pre-fall / Fall prediction
- Real-time camera monitoring
- Local-only video processing

The goal of this app is not only to detect falls after they happen, but also to support early warning through pre-fall prediction.

---

## Design Overview

- The frontend UI is built with **HTML, CSS, and JavaScript**.
- The backend is built with **Python**.
- The backend handles camera input, pose estimation, feature extraction, and fall prediction.
- Video data is processed locally and is not uploaded to external servers.
- The local interface runs through `127.0.0.1`.
- The app uses `pywebview` by default to open a native macOS desktop window.
- The app can be launched using `launch.command`.
- The app can later be packaged into a `.app` bundle using `build_app.sh`.

---

## Quick Start

### 1. Enter the application directory

```bash
cd app/macos
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -e "."
```

### 5. Launch the desktop app

```bash
./launch.command
```

You can also double-click `launch.command` in Finder to open the native macOS desktop window.

---

## Launch Options

| Command | Description |
|---|---|
| `python -m fall_prediction_desktop` | Default mode: native desktop window |
| `python -m fall_prediction_desktop --menubar` | Run as a macOS menu bar app |
| `python -m fall_prediction_desktop --connect http://127.0.0.1:8765/` | Connect to an existing local monitoring service |
| `python -m fall_prediction --source video.mp4 --pose-backend yolo --predictor ml --output-video annotated.mp4` | Process a video or image sequence from the command line |

---

## Directory Structure

```text
macos/
├── src/
│   ├── fall_prediction_desktop/   # Desktop app: window and local server
│   └── fall_prediction/           # AI core: pose estimation and fall prediction
├── web/                           # Frontend UI: HTML/CSS/JS
├── models/                        # Model files used by the app
├── assets/                        # Icons and app resources
├── configs/                       # Configuration files
├── scripts/                       # Utility scripts
├── tests/                         # Test files
├── launch.command                 # Double-click launcher
├── build_app.sh                   # Build script for .app packaging
├── FallGuard.spec                 # PyInstaller specification file
├── Info.plist                     # macOS app metadata
├── entitlements.plist             # macOS app permissions
├── profiles.json                  # App profile configuration
├── pyproject.toml                 # Python dependencies and project configuration
└── README.md                      # This file
```

---

## Current Features

- Product-style dashboard interface
- Native macOS desktop window through `pywebview`
- Real-time camera preview
- YOLO-based human pose estimation
- Machine-learning-based state prediction
- Three-state prediction:
  - `Normal`
  - `Pre-fall`
  - `Fall`
- Skeleton visualization
- Current state display
- Risk score display
- Event log
- FPS display
- Local-only video processing

---

## Model Usage

The app uses pose-based machine learning models for fall and pre-fall prediction.

Model files are stored in:

```text
models/
```

The current prototype is designed around a pose-based prediction pipeline:

```text
Camera / Video Input
        ↓
Pose Estimation
        ↓
Feature Extraction
        ↓
Sliding-Window Prediction
        ↓
Normal / Pre-fall / Fall State Output
        ↓
Warning and Visualization
```

---

## App Icon

Save the designed app icon as:

```text
assets/FallGuard.png
```

A `1024x1024` PNG file is recommended.

In `pywebview` mode, the app can use this PNG file as the window icon.

When packaging the app, `build_app.sh` can generate and use:

```text
FallGuard.icns
```

---

## Packaging as a `.app`

To package the app:

```bash
./build_app.sh
```

The packaged app will be generated at:

```text
dist/FallGuard.app
```

It can be opened like a normal macOS application or moved into the Applications folder.

At this stage, the packaged app is an unsigned local prototype for testing and demonstration. It is not intended for App Store distribution.

---

## Camera Permission

When the app is launched for the first time, macOS may ask for camera permission.

Please allow camera access.

If the camera cannot be opened, check the following:

- Go to **System Settings > Privacy & Security > Camera** and allow access for FallGuard.
- If you launch the app with `launch.command`, also allow access for Terminal or Python.
- Close other apps that may be using the camera, such as FaceTime, Zoom, or browser-based meeting tools.
- When using the packaged version, open `dist/FallGuard.app` directly instead of running the internal executable inside the app bundle.

---

## Notes

This application is currently a local research prototype. And the application is still developing.

It is designed for:

- Testing
- Demonstration
- Real-time interaction
- Further development of the pose-based fall and pre-fall prediction system

The app is not currently intended for clinical use or App Store distribution.
