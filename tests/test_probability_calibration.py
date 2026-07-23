import unittest

import numpy as np

from fall_prediction.probability_calibration import (
    apply_temperature_scaling,
    fit_temperature_scaling,
    tune_prefall_alert_threshold_with_recall_floor,
)


class ProbabilityCalibrationTest(unittest.TestCase):
    def test_temperature_scaling_preserves_rows_and_argmax(self):
        probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])

        calibrated = apply_temperature_scaling(probabilities, 2.0)

        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(calibrated.argmax(axis=1), probabilities.argmax(axis=1))
        self.assertLess(float(calibrated[0, 0]), 0.8)

    def test_temperature_fit_reduces_overconfident_validation_nll(self):
        probabilities = np.asarray(
            [[0.99, 0.005, 0.005], [0.99, 0.005, 0.005], [0.01, 0.98, 0.01]]
        )
        labels = ["Normal", "Pre-fall", "Pre-fall"]

        result = fit_temperature_scaling(
            probabilities, labels, ["Normal", "Pre-fall", "Fall"]
        )

        self.assertGreater(result["temperature"], 1.0)
        self.assertLess(
            result["after_negative_log_likelihood"],
            result["before_negative_log_likelihood"],
        )

    def test_threshold_tuning_respects_prefall_recall_floor(self):
        labels = ["Normal", "Normal", "Pre-fall", "Pre-fall", "Fall"]
        predictions = ["Normal", "Normal", "Normal", "Pre-fall", "Fall"]
        probabilities = np.asarray(
            [
                [0.80, 0.15, 0.05],
                [0.70, 0.25, 0.05],
                [0.55, 0.40, 0.05],
                [0.20, 0.75, 0.05],
                [0.05, 0.10, 0.85],
            ]
        )

        result = tune_prefall_alert_threshold_with_recall_floor(
            labels,
            predictions,
            ["Normal", "Pre-fall", "Fall"],
            probabilities,
            recall_floor=1.0,
        )

        self.assertTrue(result["recall_floor_satisfied"])
        self.assertEqual(result["best"]["recall"], 1.0)
        self.assertEqual(result["best"]["threshold"], 0.40)


if __name__ == "__main__":
    unittest.main()
