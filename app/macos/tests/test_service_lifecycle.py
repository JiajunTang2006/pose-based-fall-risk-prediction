"""Tests for the service lifecycle module (signals, watchdog, shutdown)."""

import signal
import threading
import time
import unittest
from unittest.mock import patch

from fall_prediction_service.lifecycle import (
    ServiceLifecycle,
    install_signal_handlers,
)


class TestServiceLifecycle(unittest.TestCase):
    def setUp(self):
        self.lifecycle = ServiceLifecycle()

    def test_initial_state_not_shutdown(self):
        self.assertFalse(self.lifecycle.shutdown_requested)

    def test_request_shutdown_sets_flag(self):
        self.lifecycle.request_shutdown()
        self.assertTrue(self.lifecycle.shutdown_requested)

    def test_double_shutdown_is_idempotent(self):
        self.lifecycle.request_shutdown()
        self.lifecycle.request_shutdown()
        self.assertTrue(self.lifecycle.shutdown_requested)

    def test_shutdown_runs_callbacks(self):
        results = []
        self.lifecycle.on_shutdown(lambda: results.append("a"))
        self.lifecycle.on_shutdown(lambda: results.append("b"))
        self.lifecycle.request_shutdown()
        self.lifecycle.shutdown()
        self.assertEqual(results, ["a", "b"])

    def test_shutdown_callback_exception_does_not_block_others(self):
        results = []
        self.lifecycle.on_shutdown(lambda: 1 / 0)  # raises
        self.lifecycle.on_shutdown(lambda: results.append("ok"))
        self.lifecycle.request_shutdown()
        self.lifecycle.shutdown()
        self.assertIn("ok", results)

    def test_request_shutdown_accepts_signal_args(self):
        self.lifecycle.request_shutdown(signal.SIGTERM, None)
        self.assertTrue(self.lifecycle.shutdown_requested)

    def test_request_shutdown_callback_runs_once(self):
        results = []
        self.lifecycle.on_shutdown_requested(lambda: results.append("stop-server"))
        self.lifecycle.request_shutdown()
        self.lifecycle.request_shutdown()
        self.assertEqual(results, ["stop-server"])


class TestParentWatchdog(unittest.TestCase):
    def test_no_parent_pid_no_watchdog_thread(self):
        lifecycle = ServiceLifecycle(parent_pid=None)
        lifecycle.start_watchdog()
        self.assertIsNone(lifecycle._watchdog_thread)

    def test_parent_pid_starts_watchdog(self):
        stable_parent = 4242
        lifecycle = ServiceLifecycle(parent_pid=stable_parent)
        with (
            patch("fall_prediction_service.lifecycle.os.getppid", return_value=stable_parent),
            patch("fall_prediction_service.lifecycle.os.kill"),
        ):
            lifecycle.start_watchdog()
            self.assertIsNotNone(lifecycle._watchdog_thread)
            self.assertTrue(lifecycle._watchdog_thread.is_alive())
            lifecycle.request_shutdown()
            lifecycle.shutdown()

    def test_watchdog_exits_when_shutdown_requested(self):
        stable_parent = 4242
        lifecycle = ServiceLifecycle(parent_pid=stable_parent)
        with (
            patch("fall_prediction_service.lifecycle.os.getppid", return_value=stable_parent),
            patch("fall_prediction_service.lifecycle.os.kill"),
        ):
            lifecycle.start_watchdog()
            self.assertTrue(lifecycle._watchdog_thread.is_alive())
            lifecycle.request_shutdown()
            lifecycle.shutdown()
            lifecycle._watchdog_thread.join(timeout=1)
            self.assertFalse(lifecycle._watchdog_thread.is_alive())

    def test_watchdog_detects_changed_parent_relationship(self):
        lifecycle = ServiceLifecycle(parent_pid=987654)
        with patch("fall_prediction_service.lifecycle.os.getppid", return_value=1):
            lifecycle.start_watchdog()
            lifecycle._watchdog_thread.join(timeout=1)
        self.assertTrue(lifecycle.shutdown_requested)


class TestInstallSignalHandlers(unittest.TestCase):
    def test_install_does_not_raise(self):
        lifecycle = ServiceLifecycle()
        # Installing signal handlers should succeed
        install_signal_handlers(lifecycle)
        # Verify the handlers were installed (they exist)
        self.assertEqual(
            signal.getsignal(signal.SIGTERM), lifecycle.request_shutdown
        )
        self.assertEqual(
            signal.getsignal(signal.SIGINT), lifecycle.request_shutdown
        )

    def tearDown(self):
        # Restore default handlers
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    unittest.main()
