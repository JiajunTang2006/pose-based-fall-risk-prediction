

import unittest

from fall_prediction.features import FeatureExtractor
from fall_prediction.landmarks import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    Landmark,
)


def make_landmarks(
    left_shoulder=(0.45, 0.25),
    right_shoulder=(0.55, 0.25),
    left_hip=(0.46, 0.55),
    right_hip=(0.54, 0.55),
    left_knee=(0.46, 0.75),
    right_knee=(0.54, 0.75),
    left_ankle=(0.46, 0.95),
    right_ankle=(0.54, 0.95),
):


    landmarks = [Landmark(0.5, 0.5, visibility=0.9) for _ in range(LANDMARK_COUNT)]


    points = {
        LEFT_SHOULDER: left_shoulder,
        RIGHT_SHOULDER: right_shoulder,
        LEFT_HIP: left_hip,
        RIGHT_HIP: right_hip,
        LEFT_KNEE: left_knee,
        RIGHT_KNEE: right_knee,
        LEFT_ANKLE: left_ankle,
        RIGHT_ANKLE: right_ankle,
    }
    for index, (x, y) in points.items():
        landmarks[index] = Landmark(x, y, visibility=0.95)
    return landmarks


class FeatureExtractorTest(unittest.TestCase):


    def test_standing_torso_angle_is_close_to_vertical(self):

        extractor = FeatureExtractor()
        features = extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)

        self.assertTrue(features.has_pose)
        self.assertLess(features.torso_angle_deg, 5.0)
        self.assertGreater(features.visibility_mean, 0.9)

    def test_downward_motion_has_positive_vertical_velocity(self):

        extractor = FeatureExtractor()

        extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)

        lower_pose = make_landmarks(
            left_shoulder=(0.45, 0.35),
            right_shoulder=(0.55, 0.35),
            left_hip=(0.46, 0.65),
            right_hip=(0.54, 0.65),
            left_knee=(0.46, 0.82),
            right_knee=(0.54, 0.82),
            left_ankle=(0.46, 0.98),
            right_ankle=(0.54, 0.98),
        )
        features = extractor.extract(lower_pose, frame_index=1, timestamp=0.1)

        self.assertGreater(features.vertical_velocity, 0.0)

    def test_tilted_pose_has_large_torso_angle(self):

        extractor = FeatureExtractor()

        tilted = make_landmarks(
            left_shoulder=(0.24, 0.36),
            right_shoulder=(0.34, 0.36),
            left_hip=(0.50, 0.56),
            right_hip=(0.60, 0.56),
            left_knee=(0.62, 0.70),
            right_knee=(0.72, 0.70),
            left_ankle=(0.78, 0.84),
            right_ankle=(0.88, 0.84),
        )
        features = extractor.extract(tilted, frame_index=0, timestamp=0.0)

        self.assertGreater(features.torso_angle_deg, 45.0)
        self.assertGreater(features.aspect_ratio, 0.5)

    def test_missing_hips_keeps_bbox_but_marks_torso_and_center_invalid(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        landmarks = make_landmarks()
        landmarks[LEFT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        landmarks[RIGHT_HIP] = Landmark(0.0, 0.0, visibility=0.0)

        features = extractor.extract(landmarks, frame_index=0, timestamp=0.0)

        self.assertTrue(features.has_pose)
        self.assertFalse(features.torso_valid)
        self.assertFalse(features.center_valid)
        self.assertTrue(features.bbox_valid)
        self.assertTrue(features.upper_body_valid)
        self.assertEqual(features.torso_angle_deg, 0.0)

    def test_motion_after_partial_gap_uses_full_elapsed_time(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)
        partial = make_landmarks()
        partial[LEFT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        partial[RIGHT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        extractor.extract(partial, frame_index=1, timestamp=0.1)
        lower = make_landmarks(
            left_shoulder=(0.45, 0.35),
            right_shoulder=(0.55, 0.35),
            left_hip=(0.46, 0.65),
            right_hip=(0.54, 0.65),
        )

        features = extractor.extract(lower, frame_index=2, timestamp=0.2)

        self.assertAlmostEqual(features.vertical_velocity, 0.5, places=5)

    def test_upper_body_only_pose_does_not_claim_a_full_body_bbox(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        landmarks = make_landmarks()
        for index in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
            landmarks[index] = Landmark(0.0, 0.0, visibility=0.0)

        features = extractor.extract(landmarks, frame_index=0, timestamp=0.0)

        self.assertTrue(features.upper_body_valid)
        self.assertFalse(features.torso_valid)
        self.assertFalse(features.center_valid)
        self.assertFalse(features.bbox_valid)


if __name__ == "__main__":
    unittest.main()
