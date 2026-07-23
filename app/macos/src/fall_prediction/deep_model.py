"""Lightweight causal TCN for streaming fall-state classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "TCN training and prediction require PyTorch. Install the project deep extra first."
        ) from exc
    return torch, nn


def build_temporal_conv_net(
    num_features: int,
    num_classes: int,
    channels: Sequence[int] = (32, 32, 32),
    kernel_size: int = 3,
    dropout: float = 0.20,
    pooling: str = "last",
):
    """Build a causal residual TCN that consumes ``[batch, time, features]``."""
    torch, nn = _require_torch()

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.padding = padding
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )

        def forward(self, values):
            output = self.conv(values)
            return output[:, :, : -self.padding] if self.padding else output

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
            self.activation = nn.ReLU()

        def forward(self, values):
            return self.activation(self.network(values) + self.residual(values))

    if pooling not in {"last", "last_mean_max"}:
        raise ValueError("pooling must be 'last' or 'last_mean_max'")

    class TemporalConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = num_features
            for level, out_channels in enumerate(channels):
                blocks.append(TemporalResidualBlock(in_channels, int(out_channels), 2**level))
                in_channels = int(out_channels)
            self.encoder = nn.Sequential(*blocks)
            pooled_channels = in_channels * 3 if pooling == "last_mean_max" else in_channels
            self.classifier = nn.Sequential(
                nn.Linear(pooled_channels, in_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, num_classes),
            )

        def forward(self, values):
            if values.ndim != 3:
                raise ValueError("TCN input must have shape [batch, time, features]")
            encoded = self.encoder(values.transpose(1, 2))
            summary = encoded[:, :, -1]
            if pooling == "last_mean_max":
                summary = torch.cat(
                    (summary, encoded.mean(dim=2), encoded.amax(dim=2)), dim=1
                )
            return self.classifier(summary)

    model = TemporalConvNet()
    # Keep construction deterministic across supported PyTorch versions.
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


class TorchTemporalClassifier:
    """Expose a saved TCN through the scikit-learn prediction interface."""

    def __init__(self, artifact: dict[str, Any], device: str = "cpu") -> None:
        torch, _nn = _require_torch()
        config = artifact["model_config"]
        self.window_size = int(artifact["window_size"])
        self.feature_columns = tuple(artifact["feature_columns"])
        self.classes_ = np.asarray(artifact["classes"])
        self.mean = np.asarray(artifact["normalizer_mean"], dtype=np.float32)
        self.std = np.asarray(artifact["normalizer_std"], dtype=np.float32)
        self.device = torch.device(device)
        self.model = build_temporal_conv_net(
            num_features=int(config["num_features"]),
            num_classes=int(config["num_classes"]),
            channels=tuple(config["channels"]),
            kernel_size=int(config["kernel_size"]),
            dropout=float(config["dropout"]),
            pooling=str(config.get("pooling", "last")),
        )
        self.model.load_state_dict(artifact["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _prepare(self, samples):
        torch, _nn = _require_torch()
        values = np.asarray(samples, dtype=np.float32)
        feature_count = len(self.feature_columns)
        if values.ndim == 2:
            expected_width = self.window_size * feature_count
            if values.shape[1] != expected_width:
                raise ValueError(
                    f"TCN expected flattened width {expected_width}, got {values.shape[1]}"
                )
            values = values.reshape(-1, self.window_size, feature_count)
        if values.ndim != 3 or values.shape[1:] != (self.window_size, feature_count):
            raise ValueError(
                "TCN samples must be [batch, time, features] or matching flattened windows"
            )
        values = (values - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)
        return torch.from_numpy(values.astype(np.float32)).to(self.device)

    def predict_proba(self, samples) -> np.ndarray:
        torch, _nn = _require_torch()
        values = self._prepare(samples)
        with torch.inference_mode():
            return torch.softmax(self.model(values), dim=1).cpu().numpy()

    def predict(self, samples) -> np.ndarray:
        probabilities = self.predict_proba(samples)
        return self.classes_[probabilities.argmax(axis=1)]


def load_deep_model_artifact(model_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a safe weights-only TCN artifact and attach its runtime adapter."""
    torch, _nn = _require_torch()
    try:
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "pytorch_tcn":
        raise RuntimeError(f"Not a fall-prediction TCN artifact: {model_path}")
    artifact = dict(artifact)
    artifact["model"] = TorchTemporalClassifier(artifact, device=device)
    return artifact
