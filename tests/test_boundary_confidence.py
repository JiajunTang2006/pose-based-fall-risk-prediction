import unittest

import numpy as np

from fall_prediction.boundary_confidence import (
    build_boundary_confidence_weights,
    build_boundary_tolerant_metrics,
    build_transition_latency_report,
    combine_label_and_confidence_weights,
)


class BoundaryConfidenceTest(unittest.TestCase):
    def test_boundary_weights_keep_labels_but_lower_boundary_confidence(self):
        weights = build_boundary_confidence_weights([-1, 0, 2, 3, 5, 6])

        np.testing.assert_allclose(weights, [1.0, 0.35, 0.35, 0.65, 0.65, 1.0])

    def test_confidence_multiplies_existing_class_weights(self):
        weights = combine_label_and_confidence_weights(
            ["Normal", "Pre-fall", "Fall"],
            {"Normal": 1.0, "Pre-fall": 4.0, "Fall": 2.0},
            [1.0, 0.35, 0.65],
        )

        np.testing.assert_allclose(weights, [1.0, 1.4, 1.3])

    def test_tolerant_metrics_accept_only_nearby_adjacent_label(self):
        result = build_boundary_tolerant_metrics(
            y_true=["Normal", "Normal", "Pre-fall", "Pre-fall", "Fall"],
            y_pred=["Fall", "Pre-fall", "Normal", "Pre-fall", "Pre-fall"],
            sequences=["s"] * 5,
            end_frames=[4, 7, 10, 13, 20],
            labels=["Normal", "Pre-fall", "Fall"],
            tolerance_frames=5,
        )

        self.assertEqual(result["accepted_boundary_mismatches"], 2)
        self.assertLess(result["metrics"]["accuracy"], 1.0)

    def test_latency_uses_annotation_distance_to_recover_exact_boundary(self):
        report = build_transition_latency_report(
            y_true=["Normal", "Normal", "Pre-fall", "Pre-fall", "Fall", "Fall"],
            y_pred=["Normal", "Pre-fall", "Pre-fall", "Fall", "Fall", "Fall"],
            sequences=["s"] * 6,
            end_frames=[4, 7, 10, 13, 20, 23],
            boundary_distances=[6, 3, 0, 3, 0, 3],
            tolerance_frames=5,
        )

        records = report["records"]
        self.assertEqual(records[0]["boundary_frame"], 10)
        self.assertEqual(records[0]["latency_frames"], -3)
        self.assertEqual(records[1]["boundary_frame"], 20)
        self.assertEqual(records[1]["latency_frames"], -7)


if __name__ == "__main__":
    unittest.main()
