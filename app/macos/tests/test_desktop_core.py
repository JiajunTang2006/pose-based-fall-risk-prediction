from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fall_prediction_desktop.runner import safe_filename
from fall_prediction_desktop.web_app import (
    CameraMonitor,
    MediaImportProcessor,
    MonitorSnapshot,
    ProfileManager,
)


class ProfileManagerTests(unittest.TestCase):
    def test_loads_legacy_profile_keys_and_saves_canonical_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_id = "abc123"
            (root / "profiles.json").write_text(
                json.dumps({
                    "active_id": profile_id,
                    "profiles": {
                        profile_id: {
                            "id": profile_id,
                            "name": "Alice",
                            "createdAt": "2026-07-05T00:00:00+00:00",
                            "fallEvents": [{
                                "timestamp": "2026-07-05T00:00:00+00:00",
                                "risk_score": 80,
                                "state": "Fall",
                                "detail": "Detected",
                            }],
                        }
                    },
                }),
                encoding="utf-8",
            )

            manager = ProfileManager(root, data_dir=root)
            self.assertEqual(manager.active_id, profile_id)
            self.assertEqual(manager.active.name, "Alice")
            self.assertEqual(len(manager.active.fall_events), 1)

            manager.record_fall("Pre-fall", 42, "Warning")
            saved = json.loads((root / "profiles.json").read_text(encoding="utf-8"))
            profile = saved["profiles"][profile_id]
            self.assertIn("created_at", profile)
            self.assertIn("fall_events", profile)
            self.assertEqual(len(profile["fall_events"]), 2)


class MediaImportProcessorTests(unittest.TestCase):
    def test_detects_video_or_image_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_root = root / "media"
            media_root.mkdir()
            video = media_root / "clip.mp4"
            video.write_bytes(b"placeholder")
            image_dir = media_root / "frames"
            image_dir.mkdir()
            (image_dir / "frame_10.png").write_bytes(b"placeholder")
            (image_dir / "frame_2.png").write_bytes(b"placeholder")

            with patch("pathlib.Path.home", return_value=root):
                processor = MediaImportProcessor(root)

            video_kind, video_paths = processor._detect_path_media_kind([video])
            self.assertEqual(video_kind, "video")
            self.assertEqual(video_paths, [video])

            image_kind, image_paths = processor._detect_path_media_kind([image_dir])
            self.assertEqual(image_kind, "images")
            self.assertEqual([path.name for path in image_paths], ["frame_2.png", "frame_10.png"])

    def test_snapshot_returns_a_copy_of_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("pathlib.Path.home", return_value=root):
                processor = MediaImportProcessor(root)

            snapshot = processor.snapshot()

            self.assertIsInstance(snapshot, dict)
            self.assertFalse(snapshot["running"])
            self.assertEqual(snapshot["state"], "Idle")

    def test_rejects_mixed_video_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            image = root / "frame.png"
            video.write_bytes(b"placeholder")
            image.write_bytes(b"placeholder")

            with patch("pathlib.Path.home", return_value=root):
                processor = MediaImportProcessor(root)

            with self.assertRaises(ValueError):
                processor._detect_path_media_kind([video, image])


class CameraMonitorTests(unittest.TestCase):
    def _monitor(self) -> CameraMonitor:
        monitor = CameraMonitor.__new__(CameraMonitor)
        monitor._lock = threading.Lock()
        monitor._stop_event = threading.Event()
        monitor._worker = None
        monitor._snapshot = MonitorSnapshot()
        monitor._repos = None
        monitor._debug_log = MagicMock()
        return monitor

    def test_snapshot_includes_persisted_recent_events(self) -> None:
        monitor = self._monitor()
        monitor._repos = MagicMock()
        monitor._repos.events.list_recent.return_value = [{
            "id": "event-1",
            "event_type": "fall",
            "started_at": "2026-07-10T09:00:00+00:00",
            "peak_risk": 0.87,
            "status": "closed",
        }]

        snapshot = monitor.snapshot()

        self.assertEqual(snapshot["recentEvents"][0]["id"], "event-1")
        self.assertEqual(snapshot["recentEvents"][0]["risk"], 87)
        monitor._repos.events.list_recent.assert_called_once_with(12)

    def test_stop_does_not_join_current_worker(self) -> None:
        monitor = self._monitor()
        worker = MagicMock()
        worker.is_alive.return_value = True
        monitor._worker = worker

        with patch("threading.current_thread", return_value=worker):
            monitor.stop()

        self.assertTrue(monitor._stop_event.is_set())
        worker.join.assert_not_called()


class RunnerUtilityTests(unittest.TestCase):
    def test_safe_filename_removes_unsafe_characters(self) -> None:
        self.assertEqual(safe_filename(" patient / fall:01.mov "), "patient_fall_01.mov")
        self.assertEqual(safe_filename("..."), "source")


if __name__ == "__main__":
    unittest.main()
