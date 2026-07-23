"""
Process lifecycle management for the FallGuard AI Service.

Handles POSIX signals (SIGTERM/SIGINT), graceful shutdown, and optional
parent-PID monitoring so the service exits when the Swift host disappears.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Seconds to wait for in-flight work before forcing exit.
_SHUTDOWN_GRACE_SECONDS = 5.0

# How often to check the parent PID (seconds).
_PARENT_CHECK_INTERVAL = 2.0


class ServiceLifecycle:
    """Encapsulates shutdown signalling and optional parent-PID watchdog."""

    def __init__(self, parent_pid: int | None = None) -> None:
        self._shutdown_requested = threading.Event()
        self._parent_pid = parent_pid
        self._watchdog_thread: threading.Thread | None = None
        self._on_shutdown_requested_callbacks: list[Callable[[], None]] = []
        self._on_shutdown_callbacks: list[Callable[[], None]] = []

    # ── public API ──────────────────────────────────────────────────

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def request_shutdown(self, _signum: int | None = None,
                         _frame: object = None) -> None:
        """Signal handler — may be called from any thread."""
        if self._shutdown_requested.is_set():
            return  # already shutting down
        logger.info("Shutdown requested (signal=%s)", _signum)
        self._shutdown_requested.set()
        for cb in self._on_shutdown_requested_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Shutdown-request callback failed: %s", cb)

    def on_shutdown_requested(self, callback: Callable[[], None]) -> None:
        """Register a non-blocking callback that interrupts the main loop."""
        self._on_shutdown_requested_callbacks.append(callback)

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked during :meth:`shutdown`."""
        self._on_shutdown_callbacks.append(callback)

    def shutdown(self) -> None:
        """Run all registered shutdown callbacks, then stop the watchdog."""
        for cb in self._on_shutdown_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Shutdown callback failed: %s", cb)

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

    def start_watchdog(self) -> None:
        """Begin monitoring the parent PID (if provided)."""
        if self._parent_pid is None:
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True
        )
        self._watchdog_thread.start()

    # ── internal ────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        parent = self._parent_pid
        if parent is None:
            return
        while not self._shutdown_requested.is_set():
            if os.getppid() != parent:
                logger.warning(
                    "Parent relationship changed (expected PID %d, current PPID %d) — shutting down.",
                    parent,
                    os.getppid(),
                )
                self.request_shutdown()
                return
            try:
                os.kill(parent, 0)  # signal 0 = existence check
            except OSError:
                logger.warning(
                    "Parent PID %d no longer exists — shutting down.", parent
                )
                self.request_shutdown()
                return
            time.sleep(_PARENT_CHECK_INTERVAL)


def install_signal_handlers(lifecycle: ServiceLifecycle) -> None:
    """Register SIGTERM and SIGINT handlers that delegate to *lifecycle*."""
    signal.signal(signal.SIGTERM, lifecycle.request_shutdown)
    signal.signal(signal.SIGINT, lifecycle.request_shutdown)
