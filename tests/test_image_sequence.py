import tempfile
import unittest
from pathlib import Path

from fall_prediction.export_dataset_features import iter_dataset_sources
from fall_prediction.video_app import find_image_sequence_files, infer_image_sequence_fps


class ImageSequenceTest(unittest.TestCase):
    def test_find_image_sequence_uses_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame-010.png", "frame-002.png", "frame-001.png"):
                (root / name).write_text("not a real image", encoding="utf-8")

            images = find_image_sequence_files(root)

        self.assertEqual([path.name for path in images], ["frame-001.png", "frame-002.png", "frame-010.png"])

    def test_iter_dataset_sources_finds_image_sequence_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence_dir = root / "fall-01-cam0-rgb"
            sequence_dir.mkdir()
            (sequence_dir / "fall-01-cam0-rgb-001.png").write_text("not a real image", encoding="utf-8")
            (sequence_dir / "fall-01-cam0-rgb-002.png").write_text("not a real image", encoding="utf-8")

            sources = iter_dataset_sources(root)

        self.assertEqual([path.name for path in sources], ["fall-01-cam0-rgb"])

    def test_infer_image_sequence_fps_from_timestamp_names(self):
        paths = [
            Path("2018-07-04T12_04_20.000000.png"),
            Path("2018-07-04T12_04_20.050000.png"),
            Path("2018-07-04T12_04_20.100000.png"),
        ]

        self.assertAlmostEqual(infer_image_sequence_fps(paths), 20.0)

    def test_infer_image_sequence_fps_falls_back_for_frame_numbers(self):
        paths = [
            Path("fall-01-cam0-rgb-001.png"),
            Path("fall-01-cam0-rgb-002.png"),
        ]

        self.assertEqual(infer_image_sequence_fps(paths, default_fps=30.0), 30.0)


if __name__ == "__main__":
    unittest.main()
