from __future__ import annotations

import unittest
from collections import deque

from fall_prediction.features import PoseFeatures
from fall_prediction.ml_predictor import MachineLearningFallPredictor
from fall_prediction.predictor import Prediction
from fall_prediction.risk import RiskBreakdown


class PredictionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = PoseFeatures(frame_index=7, timestamp=1.5, has_pose=True)
        self.breakdown = RiskBreakdown(
            risk_score=0.25,
            torso_score=0.0,
            angular_velocity_score=0.0,
            vertical_velocity_score=0.0,
            center_drop_score=0.0,
            aspect_ratio_score=0.0,
            visibility_factor=1.0,
            center_drop=0.0,
        )

    def _prediction(self, **overrides) -> Prediction:
        values = {
            "frame_index": 7,
            "timestamp": 1.5,
            "state": "Normal",
            "instant_state": "Normal",
            "risk_score": 0.25,
            "smoothed_risk_score": 0.25,
            "features": self.features,
            "breakdown": self.breakdown,
            "baseline_center_y": None,
        }
        values.update(overrides)
        return Prediction(**values)

    def test_system_status_defaults_to_none(self) -> None:
        prediction = self._prediction()

        self.assertIsNone(prediction.system_status)

    def test_system_status_can_be_stored(self) -> None:
        status = "Calibrating: stand still (1/15)"
        prediction = self._prediction(system_status=status)

        self.assertEqual(prediction.system_status, status)

    def test_ml_prediction_propagates_system_status(self) -> None:
        predictor = MachineLearningFallPredictor.__new__(MachineLearningFallPredictor)
        predictor._risk_history = deque(maxlen=5)
        predictor._baseline_center_y = None
        status = "Calibrating: stand still (1/15)"

        prediction = predictor._prediction(
            frame_index=7,
            timestamp=1.5,
            state="Unknown",
            instant_state="Unknown",
            risk_score=0.25,
            features=self.features,
            center_drop=0.0,
            alert_state="Normal",
            system_status=status,
        )

        self.assertEqual(prediction.system_status, status)
        self.assertEqual(prediction.state, "Unknown")
        self.assertEqual(prediction.alert_state, "Normal")
        self.assertEqual(prediction.smoothed_risk_score, 0.25)


if __name__ == "__main__":
    unittest.main()
