

import unittest

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
from fall_prediction.predictor import FallPredictor, PredictorConfig


def pose_from_points(shoulder_mid, hip_mid, knee_y, ankle_y, spread=0.05):

    sx, sy = shoulder_mid
    hx, hy = hip_mid


    landmarks = [Landmark(0.5, 0.5, visibility=0.9) for _ in range(LANDMARK_COUNT)]


    landmarks[LEFT_SHOULDER] = Landmark(sx - spread, sy, visibility=0.95)
    landmarks[RIGHT_SHOULDER] = Landmark(sx + spread, sy, visibility=0.95)

    landmarks[LEFT_HIP] = Landmark(hx - spread, hy, visibility=0.95)
    landmarks[RIGHT_HIP] = Landmark(hx + spread, hy, visibility=0.95)

    landmarks[LEFT_KNEE] = Landmark(hx - spread, knee_y, visibility=0.95)
    landmarks[RIGHT_KNEE] = Landmark(hx + spread, knee_y, visibility=0.95)
    landmarks[LEFT_ANKLE] = Landmark(hx - spread, ankle_y, visibility=0.95)
    landmarks[RIGHT_ANKLE] = Landmark(hx + spread, ankle_y, visibility=0.95)
    return landmarks


class FallPredictorTest(unittest.TestCase):


    def test_synthetic_fall_sequence_reaches_prefall_or_fall(self):


        predictor = FallPredictor(
            PredictorConfig(
                baseline_frames=3,
                smoothing_window=3,
                prefall_consecutive_frames=2,
                fall_consecutive_frames=2,
            )
        )

        predictions = []


        for frame in range(4):
            landmarks = pose_from_points((0.50, 0.25), (0.50, 0.55), 0.75, 0.95)
            predictions.append(predictor.predict(landmarks, frame, frame / 10.0))


        falling_poses = [
            pose_from_points((0.42, 0.38), (0.56, 0.58), 0.72, 0.86, spread=0.08),
            pose_from_points((0.34, 0.50), (0.62, 0.66), 0.72, 0.82, spread=0.10),
            pose_from_points((0.28, 0.64), (0.70, 0.74), 0.78, 0.84, spread=0.12),
            pose_from_points((0.22, 0.76), (0.76, 0.80), 0.82, 0.86, spread=0.14),
        ]
        for offset, landmarks in enumerate(falling_poses, start=4):
            predictions.append(predictor.predict(landmarks, offset, offset / 10.0))


        states = [prediction.state for prediction in predictions]

        self.assertTrue(any(state in {"Pre-fall", "Fall"} for state in states))

        self.assertGreater(predictions[-1].smoothed_risk_score, 0.45)


if __name__ == "__main__":
    unittest.main()
