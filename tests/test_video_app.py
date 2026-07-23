import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fall_prediction.features import PoseFeatures
from fall_prediction.predictor import Prediction
from fall_prediction.risk import RiskBreakdown
from fall_prediction.video_app import CSV_COLUMNS, prediction_to_row, process_video


class FakeCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 30.0

    def read(self):
        return False, None

    def release(self) -> None:
        self.released = True


class VideoAppTest(unittest.TestCase):
    def test_prediction_to_row_matches_declared_csv_columns(self):
        features = PoseFeatures(
            frame_index=3,
            timestamp=0.1,
            has_pose=True,
            torso_angle_deg=12.0,
            torso_angular_velocity=4.0,
            body_center_y=0.45,
            body_center_delta=0.02,
            vertical_velocity=0.2,
            aspect_ratio=0.6,
            body_width=0.3,
            body_height=0.5,
            visibility_mean=0.9,
        )
        prediction = Prediction(
            frame_index=3,
            timestamp=0.1,
            state="Normal",
            alert_state="Pre-fall",
            instant_state="Normal",
            risk_score=0.2,
            smoothed_risk_score=0.15,
            features=features,
            breakdown=RiskBreakdown(
                risk_score=0.2,
                torso_score=0.1,
                angular_velocity_score=0.0,
                vertical_velocity_score=0.3,
                center_drop_score=0.0,
                aspect_ratio_score=0.0,
                visibility_factor=1.0,
                center_drop=0.05,
            ),
            baseline_center_y=0.4,
        )

        row = prediction_to_row(prediction)

        self.assertEqual(tuple(row.keys()), CSV_COLUMNS)
        self.assertEqual(row["alert_state"], "Pre-fall")
        self.assertEqual(row["advisory_state"], "")

    def test_process_video_releases_capture_when_initialization_fails(self):
        capture = FakeCapture()

        with tempfile.TemporaryDirectory() as directory:
            output_csv = Path(directory) / "predictions.csv"
            with patch("fall_prediction.video_app.open_frame_source", return_value=capture):
                with patch(
                    "fall_prediction.video_app.create_pose_estimator",
                    side_effect=RuntimeError("pose init failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "pose init failed"):
                        process_video(source=0, output_csv=output_csv)

        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
