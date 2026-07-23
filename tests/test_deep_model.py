import tempfile
import unittest
from pathlib import Path

import numpy as np

from fall_prediction.deep_dataset import (
    fit_feature_normalizer,
    normalize_temporal_features,
    preserve_temporal_shape,
)
from fall_prediction.deep_model import build_temporal_conv_net
from fall_prediction.ml_predictor import load_model_artifact
from fall_prediction.window_dataset import WindowDataset


class DeepDatasetTest(unittest.TestCase):
    def test_flat_windows_preserve_time_axis(self):
        dataset = WindowDataset(
            X=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            y=["Normal"],
            groups=["video-1"],
            feature_names=["a", "b"] * 3,
        )

        temporal = preserve_temporal_shape(dataset, window_size=3, feature_columns=("a", "b"))

        self.assertEqual(temporal.X.shape, (1, 3, 2))
        np.testing.assert_array_equal(temporal.X[0, 1], [3.0, 4.0])

    def test_normalizer_is_fitted_per_feature(self):
        values = np.asarray([[[1.0, 10.0], [3.0, 30.0]]], dtype=np.float32)
        mean, std = fit_feature_normalizer(values)
        normalized = normalize_temporal_features(values, mean, std)

        np.testing.assert_allclose(mean, [2.0, 20.0])
        np.testing.assert_allclose(normalized.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)


class DeepModelTest(unittest.TestCase):
    def test_tcn_returns_one_logit_vector_per_window(self):
        import torch

        model = build_temporal_conv_net(4, 3, channels=(8, 8), dropout=0.0)
        output = model(torch.randn(5, 15, 4))

        self.assertEqual(tuple(output.shape), (5, 3))

    def test_saved_tcn_loads_through_runtime_artifact_loader(self):
        import torch

        model = build_temporal_conv_net(2, 3, channels=(4,), dropout=0.0)
        artifact = {
            "artifact_type": "pytorch_tcn",
            "format_version": 1,
            "state_dict": model.state_dict(),
            "model_config": {
                "num_features": 2,
                "num_classes": 3,
                "channels": [4],
                "kernel_size": 3,
                "dropout": 0.0,
            },
            "classes": ["Normal", "Pre-fall", "Fall"],
            "window_size": 3,
            "stride": 1,
            "feature_columns": ["a", "b"],
            "normalizer_mean": [0.0, 0.0],
            "normalizer_std": [1.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(artifact, path)

            loaded = load_model_artifact(path)
            probabilities = loaded["model"].predict_proba([[0.0] * 6])

        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
