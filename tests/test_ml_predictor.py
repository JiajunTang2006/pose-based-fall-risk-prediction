import unittest
from unittest.mock import patch

from fall_prediction.ml_predictor import (
    DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
    DEFAULT_PREFALL_ALERT_THRESHOLD,
    MachineLearningFallPredictor,
    TemporalSequenceGate,
    TemporalFallValidator,
    normalize_state,
)
from fall_prediction.window_dataset import DEFAULT_WINDOW_SIZE


class DummyModel:
    pass


class ProbabilityModel:
    classes_ = ["Fall", "Normal", "Pre-fall"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, sample):
        return self.probabilities


class DetailedNormalProbabilityModel:
    classes_ = ["Bending", "Fall", "Lying", "Pre-fall", "Sitting", "Squatting", "Standing", "Walking"]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, sample):
        return self.probabilities


def standing_rows(length=15):
    return [
        {
            "has_pose": 1.0,
            "torso_angle": 5.0,
            "torso_angular_velocity": 0.0,
            "body_height": 0.55,
            "aspect_ratio": 0.36,
            "center_drop": 0.0,
            "vertical_velocity": 0.0,
        }
        for _ in range(length)
    ]


def lying_rows(length=15):
    return [
        {
            "has_pose": 1.0,
            "torso_angle": 80.0,
            "torso_angular_velocity": 0.0,
            "body_height": 0.22,
            "aspect_ratio": 0.90,
            "center_drop": 0.26,
            "vertical_velocity": 0.0,
        }
        for _ in range(length)
    ]


def fall_motion_rows(length=15):
    rows = standing_rows(length)
    rows[-3]["center_drop"] = 0.02
    rows[-2]["center_drop"] = 0.08
    rows[-1]["center_drop"] = 0.16
    rows[-1]["vertical_velocity"] = 0.82
    rows[-1]["torso_angular_velocity"] = 240.0
    rows[-1]["torso_angle"] = 70.0
    rows[-1]["body_height"] = 0.30
    rows[-1]["aspect_ratio"] = 0.82
    return rows


class MachineLearningFallPredictorTest(unittest.TestCase):
    def test_constructor_arguments_override_artifact_metadata(self):
        artifact = {
            "model": DummyModel(),
            "window_size": 6,
            "baseline_frames": 9,
            "smoothing_window": 4,
            "prefall_alert_threshold": 0.7,
            "prefall_alert_consecutive_frames": 5,
        }

        with patch("fall_prediction.ml_predictor.load_model_artifact", return_value=artifact):
            predictor = MachineLearningFallPredictor(
                "dummy.joblib",
                baseline_frames=3,
                smoothing_window=2,
                prefall_alert_threshold=0.2,
                prefall_alert_consecutive_frames=1,
            )

        self.assertEqual(predictor.window_size, 6)
        self.assertEqual(predictor.baseline_frames, 3)
        self.assertEqual(predictor.smoothing_window, 2)
        self.assertEqual(predictor._risk_history.maxlen, 2)
        self.assertEqual(predictor.prefall_alert_threshold, 0.2)
        self.assertEqual(predictor.prefall_alert_consecutive_frames, 1)

    def test_artifact_metadata_is_used_when_no_override_is_given(self):
        artifact = {
            "model": DummyModel(),
            "baseline_frames": 8,
            "smoothing_window": 6,
        }

        with patch("fall_prediction.ml_predictor.load_model_artifact", return_value=artifact):
            predictor = MachineLearningFallPredictor("dummy.joblib")

        self.assertEqual(predictor.baseline_frames, 8)
        self.assertEqual(predictor.smoothing_window, 6)
        self.assertEqual(predictor.window_size, DEFAULT_WINDOW_SIZE)
        self.assertEqual(predictor.prefall_alert_threshold, DEFAULT_PREFALL_ALERT_THRESHOLD)
        self.assertEqual(
            predictor.prefall_alert_consecutive_frames,
            DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
        )

    def test_prefall_alert_can_be_more_sensitive_than_model_state(self):
        artifact = {
            "model": ProbabilityModel([[0.05, 0.70, 0.25]]),
            "window_size": 2,
        }

        with patch("fall_prediction.ml_predictor.load_model_artifact", return_value=artifact):
            predictor = MachineLearningFallPredictor(
                "dummy.joblib",
                prefall_alert_threshold=0.2,
                prefall_alert_consecutive_frames=1,
            )

        state, risk_score, alert_state = predictor._predict_sample([[0.0, 0.0]])

        self.assertEqual(state, "Normal")
        self.assertEqual(alert_state, "Pre-fall")
        self.assertGreater(risk_score, 0.0)

    def test_prefall_alert_requires_configured_consecutive_frames(self):
        artifact = {
            "model": ProbabilityModel([[0.05, 0.70, 0.25]]),
            "window_size": 2,
        }

        with patch("fall_prediction.ml_predictor.load_model_artifact", return_value=artifact):
            predictor = MachineLearningFallPredictor(
                "dummy.joblib",
                prefall_alert_threshold=0.2,
                prefall_alert_consecutive_frames=2,
            )

        first_state, _first_risk, first_alert = predictor._predict_sample([[0.0, 0.0]])
        second_state, _second_risk, second_alert = predictor._predict_sample([[0.0, 0.0]])

        self.assertEqual(first_state, "Normal")
        self.assertEqual(first_alert, "Normal")
        self.assertEqual(second_state, "Normal")
        self.assertEqual(second_alert, "Pre-fall")

    def test_detailed_normal_labels_are_collapsed_to_normal(self):
        artifact = {
            "model": DetailedNormalProbabilityModel([[0.05, 0.04, 0.07, 0.22, 0.10, 0.41, 0.06, 0.05]]),
            "window_size": 2,
        }

        with patch("fall_prediction.ml_predictor.load_model_artifact", return_value=artifact):
            predictor = MachineLearningFallPredictor(
                "dummy.joblib",
                prefall_alert_threshold=0.2,
                prefall_alert_consecutive_frames=1,
            )

        state, risk_score, alert_state = predictor._predict_sample([[0.0, 0.0]])

        self.assertEqual(state, "Normal")
        self.assertLess(risk_score, 0.2)
        self.assertEqual(alert_state, "Pre-fall")

    def test_normalize_state(self):
        self.assertEqual(normalize_state("fall"), "Fall")
        self.assertEqual(normalize_state("pre_fall"), "Pre-fall")
        self.assertEqual(normalize_state("adl"), "Normal")
        self.assertEqual(normalize_state("standing"), "Normal")
        self.assertEqual(normalize_state("walk"), "Normal")
        self.assertEqual(normalize_state("seated"), "Normal")
        self.assertEqual(normalize_state("crouching"), "Normal")
        self.assertEqual(normalize_state("stooping"), "Normal")
        self.assertEqual(normalize_state("spotting"), "Normal")
        self.assertEqual(normalize_state("laying"), "Normal")


class TemporalSequenceGateTest(unittest.TestCase):
    def test_high_sensitivity_prefall_requires_short_persistence(self):
        gate = TemporalSequenceGate("high")

        for _ in range(2):
            self.assertEqual(
                gate.validate("Normal", "Normal", {"Normal": 0.9, "Pre-fall": 0.05, "Fall": 0.05}, standing_rows()),
                ("Normal", "Normal"),
            )

        prefall_probs = {"Normal": 0.7, "Pre-fall": 0.20, "Fall": 0.10}
        self.assertEqual(gate.validate("Pre-fall", "Pre-fall", prefall_probs, standing_rows()), ("Normal", "Normal"))
        self.assertEqual(
            gate.validate("Pre-fall", "Pre-fall", prefall_probs, standing_rows()),
            ("Pre-fall", "Pre-fall"),
        )

    def test_static_lying_never_bootstraps_prefall_or_fall(self):
        gate = TemporalSequenceGate("high")
        lying = lying_rows()

        for _ in range(4):
            self.assertEqual(
                gate.validate("Fall", "Fall", {"Normal": 0.05, "Pre-fall": 0.10, "Fall": 0.85}, lying),
                ("Normal", "Normal"),
            )

        for _ in range(4):
            self.assertEqual(
                gate.validate("Pre-fall", "Pre-fall", {"Normal": 0.10, "Pre-fall": 0.80, "Fall": 0.10}, lying),
                ("Normal", "Normal"),
            )

    def test_medium_sensitivity_requires_more_prefall_evidence(self):
        gate = TemporalSequenceGate("medium")

        for _ in range(4):
            gate.validate("Normal", "Normal", {"Normal": 0.9, "Pre-fall": 0.05, "Fall": 0.05}, standing_rows())

        prefall_probs = {"Normal": 0.55, "Pre-fall": 0.32, "Fall": 0.13}
        self.assertEqual(gate.validate("Pre-fall", "Pre-fall", prefall_probs, standing_rows()), ("Normal", "Normal"))
        self.assertEqual(gate.validate("Pre-fall", "Pre-fall", prefall_probs, standing_rows()), ("Normal", "Normal"))
        self.assertEqual(
            gate.validate("Pre-fall", "Pre-fall", prefall_probs, standing_rows()),
            ("Pre-fall", "Pre-fall"),
        )

    def test_fall_confirms_after_legal_sequence_from_normal_prefall(self):
        gate = TemporalSequenceGate("medium")

        for _ in range(4):
            gate.validate("Normal", "Normal", {"Normal": 0.9, "Pre-fall": 0.05, "Fall": 0.05}, standing_rows())
        for _ in range(3):
            gate.validate("Pre-fall", "Pre-fall", {"Normal": 0.5, "Pre-fall": 0.35, "Fall": 0.15}, standing_rows())

        self.assertEqual(
            gate.validate("Fall", "Fall", {"Normal": 0.05, "Pre-fall": 0.15, "Fall": 0.80}, fall_motion_rows()),
            ("Fall", "Fall"),
        )

    def test_temporal_fall_validator_filters_stable_fall_without_motion(self):
        validator = TemporalFallValidator()
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.20,
            }
            for _ in range(15)
        ]

        state, alert_state = validator.validate("Fall", "Fall", rows)

        self.assertEqual(state, "Normal")
        self.assertEqual(alert_state, "Normal")

    def test_temporal_fall_validator_downgrades_fast_unconfirmed_fall_to_prefall(self):
        validator = TemporalFallValidator()
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.0,
            }
            for _ in range(15)
        ]
        rows[-2]["vertical_velocity"] = 0.10
        rows[-1]["vertical_velocity"] = 0.75
        rows[-1]["center_drop"] = 0.14

        state, alert_state = validator.validate("Fall", "Fall", rows)

        self.assertEqual(state, "Pre-fall")
        self.assertEqual(alert_state, "Pre-fall")

    def test_temporal_fall_validator_requires_model_fall_state(self):
        validator = TemporalFallValidator()
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.0,
            }
            for _ in range(15)
        ]
        rows[-1]["vertical_velocity"] = 0.80
        rows[-1]["center_drop"] = 0.15

        validator.validate("Normal", "Normal", rows)
        validator.validate("Normal", "Pre-fall", rows)
        state, alert_state = validator.validate("Normal", "Fall", rows)

        self.assertEqual(state, "Normal")
        self.assertEqual(alert_state, "Pre-fall")

    def test_temporal_fall_validator_downgrades_fall_after_prefall_without_motion(self):
        validator = TemporalFallValidator()
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.20,
            }
            for _ in range(15)
        ]

        validator.validate("Normal", "Normal", rows)
        validator.validate("Pre-fall", "Pre-fall", rows)
        state, alert_state = validator.validate("Fall", "Fall", rows)

        self.assertEqual(state, "Pre-fall")
        self.assertEqual(alert_state, "Pre-fall")

    def test_temporal_fall_validator_confirms_repeated_fall_after_prefall(self):
        validator = TemporalFallValidator(fall_after_prefall_confirm_frames=2)
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.20,
            }
            for _ in range(15)
        ]

        validator.validate("Normal", "Normal", rows)
        validator.validate("Pre-fall", "Pre-fall", rows)
        self.assertEqual(validator.validate("Fall", "Fall", rows), ("Pre-fall", "Pre-fall"))
        self.assertEqual(validator.validate("Fall", "Fall", rows), ("Fall", "Fall"))

    def test_temporal_fall_validator_confirms_repeated_fall_when_clip_starts_prefall(self):
        validator = TemporalFallValidator(fall_after_prefall_confirm_frames=2)
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.20,
            }
            for _ in range(15)
        ]

        validator.validate("Pre-fall", "Pre-fall", rows)
        self.assertEqual(validator.validate("Fall", "Fall", rows), ("Pre-fall", "Pre-fall"))
        self.assertEqual(validator.validate("Fall", "Fall", rows), ("Fall", "Fall"))

    def test_temporal_fall_validator_keeps_fall_after_prefall_with_motion(self):
        validator = TemporalFallValidator()
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.0,
            }
            for _ in range(15)
        ]
        rows[-1]["vertical_velocity"] = 0.75
        rows[-1]["center_drop"] = 0.12

        validator.validate("Normal", "Normal", rows)
        validator.validate("Normal", "Pre-fall", rows)
        state, alert_state = validator.validate("Fall", "Fall", rows)

        self.assertEqual(state, "Fall")
        self.assertEqual(alert_state, "Fall")

    def test_temporal_fall_validator_filters_slow_sitting_like_descent(self):
        validator = TemporalFallValidator()
        rows = []
        for index in range(15):
            rows.append(
                {
                    "vertical_velocity": 0.18,
                    "torso_angular_velocity": 18.0,
                    "center_drop": index * 0.01,
                }
            )

        state, alert_state = validator.validate("Fall", "Fall", rows)

        self.assertEqual(state, "Normal")
        self.assertEqual(alert_state, "Normal")

    def test_temporal_fall_validator_holds_confirmed_fall_briefly(self):
        validator = TemporalFallValidator(fall_hold_frames=2)
        rows = [
            {
                "vertical_velocity": 0.0,
                "torso_angular_velocity": 0.0,
                "center_drop": 0.0,
            }
            for _ in range(15)
        ]
        rows[-1]["vertical_velocity"] = 0.80
        rows[-1]["center_drop"] = 0.15

        validator.validate("Normal", "Normal", rows)
        validator.validate("Pre-fall", "Pre-fall", rows)
        self.assertEqual(validator.validate("Fall", "Fall", rows), ("Fall", "Fall"))
        self.assertEqual(validator.validate("Normal", "Normal", rows), ("Fall", "Fall"))
        self.assertEqual(validator.validate("Normal", "Normal", rows), ("Fall", "Fall"))
        self.assertEqual(validator.validate("Normal", "Normal", rows), ("Normal", "Normal"))


if __name__ == "__main__":
    unittest.main()
