import unittest

from fall_prediction.robustness import (
    ROBUST_ML_FEATURE_COLUMNS,
    StandingFeatureCalibrator,
    apply_partial_pose_dropout,
    calibrate_feature_rows,
)


def pose_row(
    frame: int,
    *,
    angle: float = 18.0,
    center_y: float = 0.45,
    width: float = 0.20,
    height: float = 0.50,
) -> dict[str, float]:
    return {
        "frame": float(frame),
        "time": frame / 30.0,
        "has_pose": 1.0,
        "torso_angle": abs(angle),
        "torso_signed_angle": angle,
        "torso_angular_velocity": 0.0,
        "body_center_y": center_y,
        "body_center_delta": 0.0,
        "vertical_velocity": 0.0,
        "aspect_ratio": width / height,
        "body_width": width,
        "body_height": height,
        "visibility_mean": 0.9,
        "center_drop": 0.0,
        "torso_valid": 1.0,
        "center_valid": 1.0,
        "bbox_valid": 1.0,
        "shoulder_center_y": center_y - height * 0.35,
        "shoulder_center_delta": 0.0,
        "shoulder_vertical_velocity": 0.0,
        "shoulder_line_angle": 4.0,
        "shoulder_line_angular_velocity": 0.0,
        "upper_body_width": width,
        "upper_body_height": height * 0.45,
        "upper_body_aspect_ratio": width / (height * 0.45),
        "upper_body_valid": 1.0,
    }


class StandingFeatureCalibratorTest(unittest.TestCase):
    def test_standing_pose_maps_to_canonical_relative_values(self):
        rows = [pose_row(frame) for frame in range(5)]
        calibrator = StandingFeatureCalibrator(baseline_frames=3)

        self.assertTrue(calibrator.fit(rows))
        transformed = calibrator.transform(rows[-1])

        self.assertAlmostEqual(transformed["torso_angle"], 0.0)
        self.assertAlmostEqual(transformed["body_center_y"], 0.0)
        self.assertAlmostEqual(transformed["body_width"], 1.0)
        self.assertAlmostEqual(transformed["body_height"], 1.0)
        self.assertAlmostEqual(transformed["aspect_ratio"], 1.0)
        self.assertAlmostEqual(transformed["feature_coverage"], 1.0)

    def test_fixed_roll_baseline_preserves_relative_fall_tilt(self):
        rows = [pose_row(frame, angle=20.0) for frame in range(3)]
        calibrator = StandingFeatureCalibrator(baseline_frames=3)
        self.assertTrue(calibrator.fit(rows))

        tilted = calibrator.transform(pose_row(4, angle=65.0, center_y=0.60, height=0.30))

        self.assertAlmostEqual(tilted["torso_angle"], 45.0)
        self.assertAlmostEqual(tilted["center_drop"], 0.30)
        self.assertAlmostEqual(tilted["body_height"], 0.60)

    def test_different_camera_scale_maps_to_same_relative_pose(self):
        small_rows = [pose_row(frame, width=0.10, height=0.25) for frame in range(3)]
        large_rows = [pose_row(frame, width=0.30, height=0.75) for frame in range(3)]
        small, _ = calibrate_feature_rows(small_rows, baseline_frames=3)
        large, _ = calibrate_feature_rows(large_rows, baseline_frames=3)

        for key in ("body_width", "body_height", "aspect_ratio", "body_center_y"):
            self.assertAlmostEqual(small[-1][key], large[-1][key])

    def test_partial_dropout_uses_explicit_masks(self):
        rows, _ = calibrate_feature_rows([pose_row(frame) for frame in range(4)], baseline_frames=3)
        dropped = apply_partial_pose_dropout(rows, "torso")

        self.assertEqual(dropped[0]["torso_valid"], 0.0)
        self.assertEqual(dropped[0]["torso_angle"], 0.0)
        self.assertEqual(dropped[0]["center_valid"], 1.0)
        self.assertAlmostEqual(dropped[0]["feature_coverage"], 2.0 / 3.0)
        self.assertTrue(set(ROBUST_ML_FEATURE_COLUMNS).issubset(dropped[0]))

    def test_lower_body_dropout_preserves_upper_body_features(self):
        rows, _ = calibrate_feature_rows([pose_row(frame) for frame in range(4)], baseline_frames=3)
        dropped = apply_partial_pose_dropout(rows, "lower_body")

        self.assertEqual(dropped[0]["torso_valid"], 0.0)
        self.assertEqual(dropped[0]["center_valid"], 0.0)
        self.assertEqual(dropped[0]["bbox_valid"], 0.0)
        self.assertEqual(dropped[0]["upper_body_valid"], 1.0)
        self.assertEqual(dropped[0]["has_pose"], 1.0)
        self.assertAlmostEqual(dropped[0]["upper_body_width"], 1.0)

    def test_upper_body_only_rows_can_establish_their_own_calibration(self):
        rows = [pose_row(frame) for frame in range(4)]
        for row in rows:
            row["torso_valid"] = 0.0
            row["center_valid"] = 0.0
            row["bbox_valid"] = 0.0
            row["body_height"] = 0.0
            row["body_width"] = 0.0
            row["aspect_ratio"] = 0.0
        calibrator = StandingFeatureCalibrator(
            baseline_frames=3,
            allow_upper_body_only_calibration=True,
        )

        self.assertTrue(calibrator.fit(rows))
        transformed = calibrator.transform(rows[-1])

        self.assertEqual(transformed["has_pose"], 1.0)
        self.assertEqual(transformed["upper_body_valid"], 1.0)
        self.assertAlmostEqual(transformed["upper_body_width"], 1.0)
        self.assertAlmostEqual(transformed["shoulder_center_y"], 0.0)


if __name__ == "__main__":
    unittest.main()
