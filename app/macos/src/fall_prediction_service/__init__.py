"""
FallGuard AI Service — headless Python service for the native Swift macOS app.

This package contains the headless HTTP API that replaces the PySide6/pywebview
desktop UI.  It reuses the existing CameraMonitor, MediaImportProcessor,
FrameBusinessProcessor, and database layer from ``fall_prediction_desktop``.

The service listens on ``127.0.0.1`` with a random port and Bearer-token
authentication.  A structured ``ready`` JSON message is printed to stdout so the
Swift host process can discover the port and token.
"""

__version__ = "0.3.3"
API_VERSION = "v1"
