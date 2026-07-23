"""Small-data skeleton GCN and engineered-feature TCN fusion model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .deep_model import _require_torch
from .skeleton_dataset import COCO_SKELETON_EDGES


def normalized_skeleton_adjacency(joint_count: int = 17) -> np.ndarray:
    """Build a symmetric degree-normalized skeleton adjacency matrix."""
    adjacency = np.eye(joint_count, dtype=np.float32)
    for first, second in COCO_SKELETON_EDGES:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(np.power(np.maximum(degree, 1e-6), -0.5))
    return (inv_sqrt @ adjacency @ inv_sqrt).astype(np.float32)


def build_skeleton_feature_fusion_net(
    feature_count: int,
    skeleton_channels: int = 5,
    num_classes: int = 3,
    graph_channels: Sequence[int] = (16, 32, 32),
    temporal_channels: Sequence[int] = (32, 32),
    dropout: float = 0.25,
    mode: str = "fusion",
):
    """Build a compact ST-GCN, optionally fused with a causal feature TCN."""
    torch, nn = _require_torch()
    if mode not in {"skeleton", "fusion"}:
        raise ValueError("mode must be 'skeleton' or 'fusion'")
    base_adjacency = torch.from_numpy(normalized_skeleton_adjacency())

    class SpatialTemporalGraphBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.register_buffer("adjacency", base_adjacency.clone())
            self.edge_importance = nn.Parameter(torch.ones_like(base_adjacency))
            self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.temporal = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 1),
                padding=(dilation, 0),
                dilation=(dilation, 1),
            )
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.activation = nn.ReLU()

        def forward(self, values):
            adjacency = self.adjacency * self.edge_importance
            aggregated = torch.einsum("nctv,vw->nctw", values, adjacency)
            output = self.temporal(self.spatial(aggregated))
            output = self.dropout(self.norm(output))
            return self.activation(output + self.residual(values))

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.padding = 2 * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=self.padding,
                dilation=dilation,
            )

        def forward(self, values):
            output = self.conv(values)
            return output[:, :, : -self.padding]

    class TemporalResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                CausalConv1d(in_channels, out_channels, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConv1d(out_channels, out_channels, dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )

        def forward(self, values):
            return torch.relu(self.network(values) + self.residual(values))

    class SkeletonFeatureFusionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            graph_blocks = []
            in_channels = skeleton_channels
            for level, out_channels in enumerate(graph_channels):
                graph_blocks.append(
                    SpatialTemporalGraphBlock(in_channels, int(out_channels), 2**level)
                )
                in_channels = int(out_channels)
            self.graph_encoder = nn.Sequential(*graph_blocks)
            self.skeleton_projection = nn.Sequential(
                nn.Linear(in_channels * 3, in_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            temporal_blocks = []
            temporal_in = feature_count
            for level, out_channels in enumerate(temporal_channels):
                temporal_blocks.append(
                    TemporalResidualBlock(temporal_in, int(out_channels), 2**level)
                )
                temporal_in = int(out_channels)
            self.feature_encoder = nn.Sequential(*temporal_blocks)
            self.feature_projection = nn.Sequential(
                nn.Linear(temporal_in * 3, temporal_in),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            fusion_width = in_channels + (temporal_in if mode == "fusion" else 0)
            self.classifier = nn.Sequential(
                nn.Linear(fusion_width, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, skeletons, features=None):
            graph_values = self.graph_encoder(skeletons)
            skeleton_summary = torch.cat(
                (
                    graph_values[:, :, -1, :].mean(dim=2),
                    graph_values.mean(dim=(2, 3)),
                    graph_values.amax(dim=(2, 3)),
                ),
                dim=1,
            )
            embeddings = [self.skeleton_projection(skeleton_summary)]
            if mode == "fusion":
                if features is None:
                    raise ValueError("Fusion mode requires engineered feature windows")
                temporal_values = self.feature_encoder(features.transpose(1, 2))
                temporal_summary = torch.cat(
                    (
                        temporal_values[:, :, -1],
                        temporal_values.mean(dim=2),
                        temporal_values.amax(dim=2),
                    ),
                    dim=1,
                )
                embeddings.append(self.feature_projection(temporal_summary))
            return self.classifier(torch.cat(embeddings, dim=1))

    model = SkeletonFeatureFusionNet()
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


class TorchSkeletonFusionClassifier:
    """Sklearn-compatible adapter for a paired skeleton/feature model."""

    def __init__(self, artifact: dict[str, Any], device: str = "cpu") -> None:
        torch, _nn = _require_torch()
        self.window_size = int(artifact["window_size"])
        self.feature_columns = tuple(artifact["feature_columns"])
        self.classes_ = np.asarray(artifact["classes"])
        self.feature_mean = np.asarray(
            artifact["feature_normalizer_mean"], dtype=np.float32
        )
        self.feature_std = np.asarray(
            artifact["feature_normalizer_std"], dtype=np.float32
        )
        self.skeleton_mean = np.asarray(
            artifact["skeleton_normalizer_mean"], dtype=np.float32
        )
        self.skeleton_std = np.asarray(
            artifact["skeleton_normalizer_std"], dtype=np.float32
        )
        self.probability_temperature = float(
            artifact.get("probability_temperature", 1.0)
        )
        if self.probability_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive")
        self.device = torch.device(device)
        self.model = build_skeleton_feature_fusion_net(**artifact["model_config"])
        self.model.load_state_dict(artifact["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._skeleton_window: np.ndarray | None = None

    def set_skeleton_window(self, skeleton_window) -> None:
        values = np.asarray(skeleton_window, dtype=np.float32)
        expected = (len(self.skeleton_mean), self.window_size, 17)
        if values.shape != expected:
            raise ValueError(f"Expected skeleton window {expected}, got {values.shape}")
        self._skeleton_window = values

    def _prepare_features(self, samples):
        torch, _nn = _require_torch()
        values = np.asarray(samples, dtype=np.float32)
        feature_count = len(self.feature_columns)
        if values.ndim == 2:
            expected_width = self.window_size * feature_count
            if values.shape[1] != expected_width:
                raise ValueError(
                    f"Fusion model expected flattened width {expected_width}, got {values.shape[1]}"
                )
            values = values.reshape(-1, self.window_size, feature_count)
        values = (values - self.feature_mean.reshape(1, 1, -1)) / self.feature_std.reshape(
            1, 1, -1
        )
        return torch.from_numpy(values.astype(np.float32)).to(self.device)

    def predict_proba(self, samples) -> np.ndarray:
        torch, _nn = _require_torch()
        if self._skeleton_window is None:
            raise RuntimeError("No skeleton window was supplied for fusion prediction")
        features = self._prepare_features(samples)
        skeleton = self._skeleton_window[np.newaxis, :, :, :]
        if len(features) > 1:
            skeleton = np.repeat(skeleton, len(features), axis=0)
        skeleton = (
            skeleton - self.skeleton_mean.reshape(1, -1, 1, 1)
        ) / self.skeleton_std.reshape(1, -1, 1, 1)
        skeleton_tensor = torch.from_numpy(skeleton.astype(np.float32)).to(self.device)
        with torch.inference_mode():
            logits = self.model(skeleton_tensor, features)
            return torch.softmax(
                logits / self.probability_temperature, dim=1
            ).cpu().numpy()

    def predict(self, samples) -> np.ndarray:
        probabilities = self.predict_proba(samples)
        return self.classes_[probabilities.argmax(axis=1)]


def load_fusion_model_artifact(model_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a weights-only skeleton-fusion artifact."""
    torch, _nn = _require_torch()
    try:
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "pytorch_skeleton_fusion":
        raise RuntimeError(f"Not a fall-prediction skeleton-fusion artifact: {model_path}")
    artifact = dict(artifact)
    artifact["requires_skeleton"] = True
    artifact["model"] = TorchSkeletonFusionClassifier(artifact, device=device)
    return artifact
