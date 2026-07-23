import tempfile
import unittest
from pathlib import Path

import numpy as np

from fall_prediction.fusion_model import (
    build_skeleton_feature_fusion_net,
    normalized_skeleton_adjacency,
)
from fall_prediction.ml_predictor import load_model_artifact
from fall_prediction.skeleton_dataset import (
    SKELETON_CHANNELS,
    normalize_skeleton_frame,
)
from fall_prediction.train_fusion_model import _normalize_sample_weights


class SkeletonDatasetTest(unittest.TestCase):
    def test_frame_normalization_centers_visible_hips(self):
        coordinates = np.zeros((17, 2), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.float32)
        coordinates[11] = [0.4, 0.7]
        coordinates[12] = [0.6, 0.7]
        coordinates[5] = [0.4, 0.3]
        coordinates[6] = [0.6, 0.3]
        visibility[[5, 6, 11, 12]] = 1.0

        normalized = normalize_skeleton_frame(coordinates, visibility)

        np.testing.assert_allclose(
            normalized[[11, 12]].mean(axis=0), [0.0, 0.0], atol=1e-6
        )
        self.assertLess(float(normalized[[5, 6], 1].mean()), 0.0)
        np.testing.assert_array_equal(normalized[0], [0.0, 0.0])

    def test_adjacency_is_symmetric_and_contains_self_connections(self):
        adjacency = normalized_skeleton_adjacency()

        np.testing.assert_allclose(adjacency, adjacency.T)
        self.assertTrue(np.all(np.diag(adjacency) > 0.0))


class SkeletonFusionModelTest(unittest.TestCase):
    def test_training_sample_weights_must_align_and_be_positive(self):
        np.testing.assert_allclose(_normalize_sample_weights(None, 3), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(_normalize_sample_weights([1.0, 0.5], 2), [1.0, 0.5])
        with self.assertRaisesRegex(ValueError, "one value per training sample"):
            _normalize_sample_weights([1.0], 2)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            _normalize_sample_weights([1.0, 0.0], 2)

    def test_fusion_model_returns_three_class_logits(self):
        import torch

        model = build_skeleton_feature_fusion_net(
            feature_count=13,
            graph_channels=(8, 8),
            temporal_channels=(8,),
            dropout=0.0,
            mode="fusion",
        )
        logits = model(torch.randn(3, 5, 15, 17), torch.randn(3, 15, 13))

        self.assertEqual(tuple(logits.shape), (3, 3))

    def test_skeleton_only_model_does_not_require_features(self):
        import torch

        model = build_skeleton_feature_fusion_net(
            feature_count=13,
            graph_channels=(8,),
            temporal_channels=(8,),
            dropout=0.0,
            mode="skeleton",
        )
        logits = model(torch.randn(2, 5, 15, 17))

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_saved_fusion_model_loads_through_runtime_loader(self):
        import torch

        config = {
            "feature_count": 2,
            "skeleton_channels": 5,
            "num_classes": 3,
            "graph_channels": [4],
            "temporal_channels": [4],
            "dropout": 0.0,
            "mode": "fusion",
        }
        model = build_skeleton_feature_fusion_net(**config)
        artifact = {
            "artifact_type": "pytorch_skeleton_fusion",
            "state_dict": model.state_dict(),
            "model_config": config,
            "classes": ["Normal", "Pre-fall", "Fall"],
            "window_size": 3,
            "feature_columns": ["a", "b"],
            "feature_normalizer_mean": [0.0, 0.0],
            "feature_normalizer_std": [1.0, 1.0],
            "skeleton_normalizer_mean": [0.0] * len(SKELETON_CHANNELS),
            "skeleton_normalizer_std": [1.0] * len(SKELETON_CHANNELS),
            "probability_temperature": 1.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.pt"
            torch.save(artifact, path)
            loaded = load_model_artifact(path)
            loaded["model"].set_skeleton_window(np.zeros((5, 3, 17), dtype=np.float32))
            probabilities = loaded["model"].predict_proba([[0.0] * 6])

        self.assertTrue(loaded["requires_skeleton"])
        self.assertEqual(loaded["model"].probability_temperature, 1.5)
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
