import unittest

from fall_prediction.features import PoseFeatures
from fall_prediction.risk import RiskConfig, RiskScorer, ramp


def make_features(**overrides):
    values = {
        "frame_index": 0,
        "timestamp": 0.0,
        "has_pose": True,
        "torso_angle_deg": 0.0,
        "torso_angular_velocity": 0.0,
        "body_center_y": 0.5,
        "body_center_delta": 0.0,
        "vertical_velocity": 0.0,
        "aspect_ratio": 0.4,
        "body_width": 0.2,
        "body_height": 0.5,
        "visibility_mean": 0.95,
    }
    values.update(overrides)
    return PoseFeatures(**values)


class RiskScorerTest(unittest.TestCase):
    def test_ramp_clamps_values(self):
        self.assertEqual(ramp(0.0, 1.0, 3.0), 0.0)
        self.assertEqual(ramp(4.0, 1.0, 3.0), 1.0)
        self.assertEqual(ramp(2.0, 1.0, 3.0), 0.5)

    def test_score_uses_feature_thresholds_and_center_drop(self):
        scorer = RiskScorer(RiskConfig())
        features = make_features(
            torso_angle_deg=75.0,
            torso_angular_velocity=120.0,
            body_center_y=0.72,
            vertical_velocity=0.85,
            aspect_ratio=1.15,
            visibility_mean=0.95,
        )

        breakdown = scorer.score(features, baseline_center_y=0.5)

        self.assertAlmostEqual(breakdown.torso_score, 1.0)
        self.assertAlmostEqual(breakdown.angular_velocity_score, 1.0)
        self.assertAlmostEqual(breakdown.vertical_velocity_score, 1.0)
        self.assertAlmostEqual(breakdown.center_drop_score, 1.0)
        self.assertAlmostEqual(breakdown.aspect_ratio_score, 1.0)
        self.assertAlmostEqual(breakdown.risk_score, 1.0)

    def test_low_visibility_reduces_but_does_not_zero_risk(self):
        scorer = RiskScorer(RiskConfig(min_visibility=0.35))
        high_visibility = make_features(vertical_velocity=0.85, visibility_mean=0.95)
        low_visibility = make_features(vertical_velocity=0.85, visibility_mean=0.2)

        high = scorer.score(high_visibility, baseline_center_y=None)
        low = scorer.score(low_visibility, baseline_center_y=None)

        self.assertGreater(high.risk_score, low.risk_score)
        self.assertGreater(low.risk_score, 0.0)


if __name__ == "__main__":
    unittest.main()
