"""
macOS menu bar app — FallGuard lives in the status bar.

The app starts as a lightweight menu bar icon.  Monitoring runs in the
background and status is reflected in the menu bar title.  The user can
open a live monitor window to see the camera feed.
"""

from __future__ import annotations

import threading

import rumps

from .runner import ensure_repo_on_path, find_app_root
from .web_app import (
    CameraMonitor,
    FallGuardServer,
    MediaImportProcessor,
    ProfileManager,
    find_free_port,
    load_settings,
)


# ── status emoji ──────────────────────────────────────────────────────────
STATUS_ICONS = {
    "Idle":       "⚪",
    "Starting":   "🔵",
    "Normal":     "🟢",
    "Pre-fall":   "🟡",
    "Fall":       "🔴",
    "Error":      "⛔",
    "Unknown":    "⚪",
}


class FallGuardMenuBar(rumps.App):
    def __init__(self) -> None:
        super().__init__(
            name="FallGuard",
            title="⚪ FG",
            quit_button=None,  # We add our own Quit menu item.
        )

        self.app_root = find_app_root()
        ensure_repo_on_path(self.app_root)

        # Shared server state (lazily created).
        self._monitor: CameraMonitor | None = None
        self._server: FallGuardServer | None = None
        self._server_thread: threading.Thread | None = None
        self._url: str = ""
        self._port: int = 0

        # Build the menu.
        self._build_menu()

        # Status-update timer (fires every 1 s while monitoring).
        self._timer: rumps.Timer | None = None

    # ── menu construction ──────────────────────────────────────────────

    def _build_menu(self) -> None:
        self.menu.clear()

        # Status display (non-interactive).
        self.menu.add(rumps.MenuItem("FallGuard — Smart Safety", callback=None))
        self.menu.add(rumps.separator)

        # Controls.
        self._start_btn = rumps.MenuItem("Start Monitoring", callback=self._on_start)
        self._stop_btn = rumps.MenuItem("Stop Monitoring", callback=self._on_stop)
        self._monitor_btn = rumps.MenuItem("Show Monitor", callback=self._on_show_monitor)
        self.menu.add(self._start_btn)
        self.menu.add(self._stop_btn)
        self.menu.add(rumps.separator)
        self.menu.add(self._monitor_btn)
        self.menu.add(rumps.separator)

        # Status info.
        self._status_item = rumps.MenuItem("Status: Idle", callback=None)
        self._risk_item = rumps.MenuItem("Risk: --", callback=None)
        self._fps_item = rumps.MenuItem("FPS: --", callback=None)
        self.menu.add(self._status_item)
        self.menu.add(self._risk_item)
        self.menu.add(self._fps_item)
        self.menu.add(rumps.separator)

        # Quit.
        self.menu.add(rumps.MenuItem("Quit FallGuard", callback=self._on_quit))

        self._update_ui_state(running=False)

    # ── callbacks ──────────────────────────────────────────────────────

    def _on_start(self, sender: rumps.MenuItem) -> None:
        self._ensure_server()
        assert self._monitor is not None
        snap = self._monitor.snapshot()
        if snap.get("running") or snap.get("loading"):
            return

        self._monitor.start()
        self.title = "🔵 FG"
        self._update_ui_state(running=True)

        # Poll status every second.
        if self._timer is None:
            self._timer = rumps.Timer(callback=self._poll_status, interval=1)
            self._timer.start()

    def _on_stop(self, sender: rumps.MenuItem) -> None:
        if self._monitor is None:
            return
        self._monitor.stop()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.title = "⚪ FG"
        self._update_ui_state(running=False)

    def _on_show_monitor(self, sender: rumps.MenuItem) -> None:
        self._ensure_server()
        # Open a native pywebview window in a subprocess, connected to this server.
        import subprocess, sys
        subprocess.Popen(
            [sys.executable, "-m", "fall_prediction_desktop", "--connect", self._url],
            start_new_session=True,
        )

    def _on_quit(self, sender: rumps.MenuItem) -> None:
        if self._monitor is not None:
            self._monitor.stop()
        if self._server is not None:
            self._server.shutdown()
        rumps.quit_application()

    # ── status polling ─────────────────────────────────────────────────

    def _poll_status(self, timer: rumps.Timer) -> None:
        if self._monitor is None:
            return
        snap = self._monitor.snapshot()
        state = str(snap.get("state", "Idle"))
        icon = STATUS_ICONS.get(state, "⚪")
        risk = snap.get("riskPercent", 0)
        fps = snap.get("fps", 0)

        self.title = f"{icon} FG"
        self._status_item.title = f"Status: {snap.get('title', state)}"
        self._risk_item.title = f"Risk: {risk}%"
        self._fps_item.title = f"FPS: {fps:.1f}"

    def _update_ui_state(self, running: bool) -> None:
        # In rumps, setting callback=None dims the menu item.
        self._start_btn.set_callback(None if running else self._on_start)
        self._stop_btn.set_callback(None if not running else self._on_stop)

    # ── server helpers ─────────────────────────────────────────────────

    def _ensure_server(self) -> None:
        """Start the HTTP server + camera monitor (once)."""
        if self._server is not None:
            return

        web_root = self.app_root / "web"
        assets_root = self.app_root / "assets"
        self._port = find_free_port(8765)
        settings = load_settings(self.app_root)
        profile_manager = ProfileManager(self.app_root)
        self._monitor = CameraMonitor(self.app_root, settings)
        self._monitor.profile_manager = profile_manager
        media_processor = MediaImportProcessor(self.app_root, settings)
        self._server = FallGuardServer(
            ("127.0.0.1", self._port),
            web_root, assets_root,
            self._monitor,
            media_processor,
            settings,
            self.app_root,
            profile_manager,
        )
        self._url = f"http://127.0.0.1:{self._port}/"

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._server_thread.start()
        print(f"FallGuard server running at {self._url}")


def main() -> None:
    print("FallGuard is now running in your menu bar (look for ⚪ near the clock).")
    print("Press Ctrl+C in this terminal to quit.")
    FallGuardMenuBar().run()
