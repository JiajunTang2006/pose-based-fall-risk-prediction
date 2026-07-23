"""Utilities that preserve the temporal shape of sliding-window features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .window_dataset import WindowDataset


@dataclass(frozen=True)
class TemporalWindowDataset:
    """A window dataset shaped as ``[samples, time, features]``."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def preserve_temporal_shape(
    dataset: WindowDataset,
    window_size: int,
    feature_columns: Sequence[str],
) -> TemporalWindowDataset:
    """Convert the legacy flattened windows without changing sample order."""
    feature_count = len(feature_columns)
    expected_width = window_size * feature_count
    X = np.asarray(dataset.X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != expected_width:
        actual_width = X.shape[1] if X.ndim == 2 else None
        raise ValueError(
            f"Expected flattened windows with width {expected_width}, got {actual_width}"
        )
    return TemporalWindowDataset(
        X=X.reshape(-1, window_size, feature_count),
        y=np.asarray(dataset.y),
        groups=np.asarray(dataset.groups),
        feature_names=list(dataset.feature_names),
    )


def fit_feature_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature normalization using training windows only."""
    if X.ndim != 3:
        raise ValueError("Expected X shaped as [samples, time, features]")
    mean = X.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = X.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_temporal_features(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply a train-fitted per-feature normalizer."""
    return ((X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
