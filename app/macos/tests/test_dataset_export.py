from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fall_prediction_desktop.dataset_export import export_reviewed_dataset


class DatasetExportTests(unittest.TestCase):
    def test_fall_clip_exports_three_training_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.mp4"
            source.write_bytes(b"not-a-real-video")
            destination = root / "destination"
            destination.mkdir()

            result = export_reviewed_dataset(
                [{
                    "id": "event123",
                    "annotation_label": "Fall",
                    "video_clip_path": str(source),
                    "clip_fps": 10.0,
                    "clip_duration_seconds": 20.0,
                    "prefall_start_seconds": 5.0,
                    "fall_start_seconds": 8.0,
                }],
                destination,
                package_name="export",
            )

            with result.annotations_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                rows,
                [
                    {
                        "video": "FallGuard_event123",
                        "start_frame": "20",
                        "end_frame": "49",
                        "label": "Normal",
                    },
                    {
                        "video": "FallGuard_event123",
                        "start_frame": "50",
                        "end_frame": "79",
                        "label": "Pre-fall",
                    },
                    {
                        "video": "FallGuard_event123",
                        "start_frame": "80",
                        "end_frame": "139",
                        "label": "Fall",
                    },
                ],
            )
            self.assertTrue(
                (result.output_directory / "videos" / "FallGuard_event123.mp4").is_file()
            )

    def test_short_context_is_clamped_to_available_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "clip.mp4"
            source.write_bytes(b"not-a-real-video")
            destination = root / "destination"
            destination.mkdir()

            result = export_reviewed_dataset(
                [{
                    "id": "short",
                    "annotation_label": "Fall",
                    "video_clip_path": str(source),
                    "clip_fps": 10.0,
                    "clip_duration_seconds": 10.0,
                    "prefall_start_seconds": 1.0,
                    "fall_start_seconds": 8.0,
                }],
                destination,
                package_name="short",
            )

            with result.annotations_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["start_frame"], "0")
            self.assertEqual(rows[0]["end_frame"], "9")
            self.assertEqual(rows[-1]["start_frame"], "80")
            self.assertEqual(rows[-1]["end_frame"], "99")
            self.assertTrue(
                (result.output_directory / "videos" / "FallGuard_short.mp4").is_file()
            )

    def test_normal_only_events_are_not_exported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            with self.assertRaisesRegex(ValueError, "No reviewed"):
                export_reviewed_dataset(
                    [{
                        "id": "normal",
                        "annotation_label": "Normal",
                        "video_clip_path": str(destination / "missing.mp4"),
                    }],
                    destination,
                    package_name="normal",
                )
            self.assertFalse((destination / "normal").exists())


if __name__ == "__main__":
    unittest.main()
