import json
import tempfile
import unittest
from pathlib import Path

from fall_prediction.ml_features import ML_FEATURE_COLUMNS
from fall_prediction.train_model import (
    build_sample_weights,
    default_metrics_output_path,
    train_and_save,
    tune_prefall_alert_threshold_on_validation,
)


class TrainModelTest(unittest.TestCase):
    def test_train_and_save_persists_validation_metrics(self):
        X = []
        y = []
        groups = []
        for group_index, label in enumerate(["Normal", "Normal", "Normal", "Fall", "Fall", "Fall"]):
            base = 0.0 if label == "Normal" else 1.0
            group = f"{label.lower()}-{group_index}"
            for sample_offset in range(2):
                X.append([base + sample_offset * 0.01] * len(ML_FEATURE_COLUMNS))
                y.append(label)
                groups.append(group)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "toy_model.joblib"
            metrics_path = Path(directory) / "toy_metrics.json"
            artifact = train_and_save(
                X=X,
                y=y,
                groups=groups,
                feature_names=list(ML_FEATURE_COLUMNS),
                csv_paths=[Path("normal-01.csv"), Path("fall-01.csv")],
                output_path=output_path,
                window_size=1,
                stride=1,
                baseline_frames=3,
                smoothing_window=2,
                classifier_name="gradient_boosting",
                label_mode="filename",
                test_size=0.5,
                random_state=2,
                metrics_output_path=metrics_path,
            )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.assertTrue(output_path.exists())
            self.assertIsNotNone(artifact["validation_metrics"])
            self.assertIn("macro_f1", artifact["validation_metrics"])
            self.assertEqual(metrics["validation_metrics"]["labels"], ["Fall", "Normal"])
            self.assertGreater(metrics["validation_split"]["validation_samples"], 0)

    def test_train_and_save_can_use_all_samples_without_validation(self):
        X = []
        y = []
        groups = []
        for group_index, label in enumerate(["Normal", "Normal", "Fall", "Fall", "Pre-fall", "Pre-fall"]):
            base = {"Normal": 0.0, "Pre-fall": 0.5, "Fall": 1.0}[label]
            group = f"{label.lower()}-{group_index}"
            for sample_offset in range(2):
                X.append([base + sample_offset * 0.01] * len(ML_FEATURE_COLUMNS))
                y.append(label)
                groups.append(group)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "toy_model.joblib"
            metrics_path = Path(directory) / "toy_metrics.json"
            artifact = train_and_save(
                X=X,
                y=y,
                groups=groups,
                feature_names=list(ML_FEATURE_COLUMNS),
                csv_paths=[Path("normal-01.csv"), Path("fall-01.csv"), Path("prefall-01.csv")],
                output_path=output_path,
                window_size=1,
                stride=1,
                baseline_frames=3,
                smoothing_window=2,
                classifier_name="hist_gradient_boosting",
                label_mode="annotations",
                test_size=0,
                random_state=2,
                metrics_output_path=metrics_path,
                prefall_alert_threshold=0.41,
            )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.assertTrue(output_path.exists())
            self.assertIsNone(artifact["validation_metrics"])
            self.assertEqual(artifact["validation_split"]["train_samples"], len(X))
            self.assertEqual(artifact["validation_split"]["validation_samples"], 0)
            self.assertEqual(artifact["prefall_alert_threshold"], 0.41)
            self.assertEqual(metrics["prefall_alert_threshold"], 0.41)

    def test_default_metrics_path_is_next_to_model(self):
        self.assertEqual(
            default_metrics_output_path(Path("models/model.joblib")),
            Path("models/model.metrics.json"),
        )

    def test_build_sample_weights_uses_class_weights(self):
        weights = build_sample_weights(
            ["Normal", "Pre-fall", "Fall", "Normal"],
            {"Normal": 1.0, "Pre-fall": 4.0, "Fall": 2.0},
        )

        self.assertEqual(weights, [1.0, 4.0, 2.0, 1.0])

    def test_prefall_threshold_search_can_improve_alert_recall(self):
        search = tune_prefall_alert_threshold_on_validation(
            y_true=["Normal", "Pre-fall", "Pre-fall", "Fall"],
            y_pred=["Normal", "Normal", "Pre-fall", "Fall"],
            classes=["Fall", "Normal", "Pre-fall"],
            probabilities=[
                [0.1, 0.7, 0.20],
                [0.1, 0.6, 0.30],
                [0.1, 0.2, 0.70],
                [0.8, 0.1, 0.10],
            ],
            beta=1.5,
        )

        best = search["best"]
        self.assertGreater(best["threshold"], 0.20)
        self.assertLessEqual(best["threshold"], 0.30)
        self.assertEqual(best["recall"], 1.0)
        self.assertEqual(best["precision"], 1.0)
        self.assertEqual(
            search["alert_validation_metrics"]["classification_report"]["Pre-fall"]["recall"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
