import csv
import tempfile
import unittest
from pathlib import Path

from fall_prediction.window_dataset import (
    LabelInterval,
    boundary_distance_for_frame,
    build_window_dataset,
    infer_label_from_filename,
)


class WindowDatasetTest(unittest.TestCase):
    def test_boundary_distance_uses_start_of_changed_interval(self):
        intervals = {
            "fall-01-cam0": [
                LabelInterval("fall-01-cam0", 0, 9, "Normal"),
                LabelInterval("fall-01-cam0", 10, 19, "Pre-fall"),
                LabelInterval("fall-01-cam0", 20, 29, "Fall"),
            ]
        }

        self.assertEqual(
            boundary_distance_for_frame(
                Path("fall-01-cam0.csv"), "fall-01-cam0", 13, intervals
            ),
            3,
        )
        self.assertEqual(
            boundary_distance_for_frame(
                Path("fall-01-cam0.csv"), "fall-01-cam0", 18, intervals
            ),
            2,
        )

    def test_infer_label_from_urfall_filename(self):
        self.assertEqual(infer_label_from_filename("fall-01-cam0.csv"), "Fall")
        self.assertEqual(infer_label_from_filename("adl-01-cam0.csv"), "Normal")
        self.assertIsNone(infer_label_from_filename("subject-01.csv"))

    def test_build_windows_from_feature_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "fall-01-cam0.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "frame",
                        "has_pose",
                        "torso_angle",
                        "torso_angular_velocity",
                        "body_center_y",
                        "body_center_delta",
                        "vertical_velocity",
                        "aspect_ratio",
                        "body_width",
                        "body_height",
                        "visibility_mean",
                        "center_drop",
                    ],
                )
                writer.writeheader()
                for frame in range(4):
                    writer.writerow(
                        {
                            "frame": frame,
                            "has_pose": 1,
                            "torso_angle": frame,
                            "torso_angular_velocity": 0.0,
                            "body_center_y": 0.5,
                            "body_center_delta": 0.0,
                            "vertical_velocity": 0.1,
                            "aspect_ratio": 0.4,
                            "body_width": 0.2,
                            "body_height": 0.5,
                            "visibility_mean": 0.9,
                            "center_drop": 0.0,
                        }
                    )

            dataset = build_window_dataset([csv_path], window_size=2, stride=1)

        self.assertEqual(len(dataset.X), 3)
        self.assertEqual(dataset.y, ["Fall", "Fall", "Fall"])
        self.assertEqual(dataset.groups, ["fall-01-cam0", "fall-01-cam0", "fall-01-cam0"])
        self.assertEqual(len(dataset.X[0]), 22)
        self.assertEqual(dataset.feature_names[0], "t-1_has_pose")

    def test_annotations_mode_requires_annotations_path(self):
        with self.assertRaisesRegex(ValueError, "annotations_path is required"):
            build_window_dataset(["fall-01-cam0.csv"], label_mode="annotations")

    def test_robust_dataset_adds_calibration_masks_and_occlusion_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "fall-01-cam0.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                fieldnames = [
                    "frame", "time", "has_pose", "torso_angle", "torso_angular_velocity",
                    "body_center_y", "body_center_delta", "vertical_velocity", "aspect_ratio",
                    "body_width", "body_height", "visibility_mean", "center_drop",
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for frame in range(6):
                    writer.writerow(
                        {
                            "frame": frame,
                            "time": frame / 30,
                            "has_pose": 1,
                            "torso_angle": 5.0,
                            "torso_angular_velocity": 0.0,
                            "body_center_y": 0.45,
                            "body_center_delta": 0.0,
                            "vertical_velocity": 0.0,
                            "aspect_ratio": 0.4,
                            "body_width": 0.2,
                            "body_height": 0.5,
                            "visibility_mean": 0.9,
                            "center_drop": 0.0,
                        }
                    )

            dataset = build_window_dataset(
                [csv_path],
                window_size=2,
                stride=1,
                use_standing_calibration=True,
                partial_pose_augmentation=True,
                baseline_frames=3,
            )

        self.assertEqual(len(dataset.X), 25)  # 5 windows × original + 4 dropout variants
        self.assertEqual(len(dataset.X[0]), 2 * 15)
        self.assertIn("t-1_torso_valid", dataset.feature_names)
        self.assertIn("t_feature_coverage", dataset.feature_names)

    def test_upfall_camera_views_share_trial_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_paths = []
            for camera in (1, 2):
                csv_path = root / f"Subject1Activity1Trial1Camera{camera}.csv"
                csv_paths.append(csv_path)
                with csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "frame",
                            "has_pose",
                            "torso_angle",
                            "torso_angular_velocity",
                            "body_center_y",
                            "body_center_delta",
                            "vertical_velocity",
                            "aspect_ratio",
                            "body_width",
                            "body_height",
                            "visibility_mean",
                            "center_drop",
                        ],
                    )
                    writer.writeheader()
                    for frame in range(3):
                        writer.writerow(
                            {
                                "frame": frame,
                                "has_pose": 1,
                                "torso_angle": 0.0,
                                "torso_angular_velocity": 0.0,
                                "body_center_y": 0.5,
                                "body_center_delta": 0.0,
                                "vertical_velocity": 0.0,
                                "aspect_ratio": 0.4,
                                "body_width": 0.2,
                                "body_height": 0.5,
                                "visibility_mean": 0.9,
                                "center_drop": 0.0,
                            }
                        )

            annotations_path = root / "annotations.csv"
            annotations_path.write_text(
                "\n".join(
                    [
                        "video,start_frame,end_frame,label",
                        "Subject1Activity1Trial1Camera1,0,2,Fall",
                        "Subject1Activity1Trial1Camera2,0,2,Fall",
                    ]
                ),
                encoding="utf-8",
            )

            dataset = build_window_dataset(
                csv_paths,
                window_size=2,
                stride=1,
                label_mode="annotations",
                annotations_path=annotations_path,
            )

        self.assertEqual(set(dataset.groups), {"subject1activity1trial1"})
        self.assertEqual(dataset.y, ["Fall", "Fall", "Fall", "Fall"])

    def test_annotations_can_be_loaded_from_multiple_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_paths = []
            for name in ("fall-01-cam0.csv", "Subject1Activity1Trial1Camera1.csv"):
                csv_path = root / name
                csv_paths.append(csv_path)
                with csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "frame",
                            "has_pose",
                            "torso_angle",
                            "torso_angular_velocity",
                            "body_center_y",
                            "body_center_delta",
                            "vertical_velocity",
                            "aspect_ratio",
                            "body_width",
                            "body_height",
                            "visibility_mean",
                            "center_drop",
                        ],
                    )
                    writer.writeheader()
                    for frame in range(3):
                        writer.writerow(
                            {
                                "frame": frame,
                                "has_pose": 1,
                                "torso_angle": 0.0,
                                "torso_angular_velocity": 0.0,
                                "body_center_y": 0.5,
                                "body_center_delta": 0.0,
                                "vertical_velocity": 0.0,
                                "aspect_ratio": 0.4,
                                "body_width": 0.2,
                                "body_height": 0.5,
                                "visibility_mean": 0.9,
                                "center_drop": 0.0,
                            }
                        )

            first_annotations = root / "first.csv"
            first_annotations.write_text(
                "video,start_frame,end_frame,label\nfall-01-cam0,0,2,Fall\n",
                encoding="utf-8",
            )
            second_annotations = root / "second.csv"
            second_annotations.write_text(
                "video,start_frame,end_frame,label\nSubject1Activity1Trial1Camera1,0,2,Pre-fall\n",
                encoding="utf-8",
            )

            dataset = build_window_dataset(
                csv_paths,
                window_size=2,
                stride=1,
                label_mode="annotations",
                annotations_path=[first_annotations, second_annotations],
            )

        self.assertEqual(dataset.y, ["Pre-fall", "Pre-fall", "Fall", "Fall"])


if __name__ == "__main__":
    unittest.main()
