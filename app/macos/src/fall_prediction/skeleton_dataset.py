"""Paired skeleton/engineered-feature windows for graph-model training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .deep_dataset import TemporalWindowDataset, preserve_temporal_shape
from .ml_features import ACCEL_FEATURE_COLUMNS, ML_FEATURE_COLUMNS
from .window_dataset import (
    _label_for_window,
    _row_frame,
    _video_key,
    boundary_distance_for_frame,
    build_window_dataset,
    infer_label_from_filename,
    load_feature_rows,
    load_label_intervals,
)


# COCO keypoint order expressed as indices in the project's MediaPipe-33 layout.
COCO_MEDIAPIPE_INDICES = (
    0,
    2,
    5,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
)

COCO_SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

COCO_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

SKELETON_CHANNELS = ("x", "y", "visibility", "dx", "dy")


@dataclass(frozen=True)
class PairedTemporalDataset:
    """Aligned feature and skeleton windows."""

    features: np.ndarray
    skeletons: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sequences: np.ndarray
    end_frames: np.ndarray
    boundary_distances: np.ndarray
    feature_columns: tuple[str, ...]


def build_paired_temporal_dataset(
    feature_csv_paths: Sequence[str | Path],
    landmark_dirs: Sequence[str | Path],
    annotations_path: str | Path | Sequence[str | Path],
    window_size: int,
    stride: int,
    use_accel: bool = True,
) -> PairedTemporalDataset:
    """Build exactly aligned legacy-feature and normalized-skeleton windows."""
    feature_paths = sorted(Path(path) for path in feature_csv_paths)
    feature_columns = ACCEL_FEATURE_COLUMNS if use_accel else ML_FEATURE_COLUMNS
    flat_dataset = build_window_dataset(
        csv_paths=feature_paths,
        window_size=window_size,
        stride=stride,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode="annotations",
        annotations_path=annotations_path,
        use_accel=use_accel,
    )
    temporal: TemporalWindowDataset = preserve_temporal_shape(
        flat_dataset,
        window_size=window_size,
        feature_columns=feature_columns,
    )

    landmark_index = index_landmark_csvs(landmark_dirs)
    intervals = load_label_intervals(annotations_path)
    skeleton_windows: list[np.ndarray] = []
    skeleton_labels: list[str] = []
    skeleton_groups: list[str] = []
    skeleton_sequences: list[str] = []
    skeleton_end_frames: list[int] = []
    skeleton_boundary_distances: list[int] = []

    for feature_path in feature_paths:
        landmark_path = landmark_index.get(feature_path.stem.lower())
        if landmark_path is None:
            raise FileNotFoundError(f"No landmark CSV matches feature file: {feature_path}")
        feature_rows = load_feature_rows(feature_path)
        landmark_rows = load_landmark_rows(landmark_path)
        if len(feature_rows) != len(landmark_rows):
            raise ValueError(f"Frame count mismatch: {feature_path} vs {landmark_path}")
        if [row.get("frame") for row in feature_rows] != [row.get("frame") for row in landmark_rows]:
            raise ValueError(f"Frame index mismatch: {feature_path} vs {landmark_path}")

        skeleton_sequence = normalize_skeleton_rows(landmark_rows)
        video_key = _video_key(feature_path)
        file_label = infer_label_from_filename(feature_path)
        for start in range(0, len(feature_rows) - window_size + 1, stride):
            end_row = feature_rows[start + window_size - 1]
            end_frame = _row_frame(end_row, start + window_size - 1)
            label = _label_for_window(
                csv_path=feature_path,
                video_key=video_key,
                end_frame=end_frame,
                file_label=file_label,
                label_mode="annotations",
                intervals=intervals,
            )
            if label is None:
                continue
            skeleton_windows.append(skeleton_sequence[:, start : start + window_size, :])
            skeleton_labels.append(label)
            skeleton_groups.append(video_key)
            skeleton_sequences.append(feature_path.stem.lower())
            skeleton_end_frames.append(end_frame)
            boundary_distance = boundary_distance_for_frame(
                feature_path,
                video_key,
                end_frame,
                intervals,
            )
            skeleton_boundary_distances.append(
                -1 if boundary_distance is None else int(boundary_distance)
            )

    skeletons = np.asarray(skeleton_windows, dtype=np.float32)
    labels = np.asarray(skeleton_labels)
    groups = np.asarray(skeleton_groups)
    sequences = np.asarray(skeleton_sequences)
    end_frames = np.asarray(skeleton_end_frames, dtype=np.int64)
    boundary_distances = np.asarray(skeleton_boundary_distances, dtype=np.int64)
    if not np.array_equal(labels, temporal.y) or not np.array_equal(groups, temporal.groups):
        raise RuntimeError("Skeleton and engineered-feature windows are not aligned")
    return PairedTemporalDataset(
        features=temporal.X,
        skeletons=skeletons,
        y=labels,
        groups=groups,
        sequences=sequences,
        end_frames=end_frames,
        boundary_distances=boundary_distances,
        feature_columns=tuple(feature_columns),
    )


def index_landmark_csvs(landmark_dirs: Sequence[str | Path]) -> dict[str, Path]:
    """Index ``*_landmarks.csv`` files by their matching feature stem."""
    result: dict[str, Path] = {}
    for directory in landmark_dirs:
        for path in sorted(Path(directory).rglob("*_landmarks.csv")):
            stem = path.stem.removesuffix("_landmarks").lower()
            if stem in result:
                raise ValueError(f"Duplicate landmark CSV stem {stem!r}: {result[stem]} and {path}")
            result[stem] = path
    return result


def load_landmark_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize_skeleton_rows(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Return translation/scale-normalized COCO skeletons as ``[5, time, 17]``."""
    frame_count = len(rows)
    joint_count = len(COCO_MEDIAPIPE_INDICES)
    coordinates = np.zeros((frame_count, joint_count, 2), dtype=np.float32)
    visibility = np.zeros((frame_count, joint_count), dtype=np.float32)

    for frame_index, row in enumerate(rows):
        for joint_index, source_index in enumerate(COCO_MEDIAPIPE_INDICES):
            visibility[frame_index, joint_index] = _safe_float(
                row.get(f"kp{source_index:02d}_visibility", 0.0)
            )
            coordinates[frame_index, joint_index, 0] = _safe_float(
                row.get(f"kp{source_index:02d}_x", 0.0)
            )
            coordinates[frame_index, joint_index, 1] = _safe_float(
                row.get(f"kp{source_index:02d}_y", 0.0)
            )

    normalized = np.zeros_like(coordinates)
    for frame_index in range(frame_count):
        normalized[frame_index] = normalize_skeleton_frame(
            coordinates[frame_index], visibility[frame_index]
        )
    velocity = np.zeros_like(normalized)
    if frame_count > 1:
        valid_pair = (
            (visibility[1:] > 0.05) & (visibility[:-1] > 0.05)
        )[..., np.newaxis]
        velocity[1:] = (normalized[1:] - normalized[:-1]) * valid_pair
    return np.concatenate(
        (
            normalized.transpose(2, 0, 1),
            visibility[np.newaxis, :, :],
            velocity.transpose(2, 0, 1),
        ),
        axis=0,
    ).astype(np.float32)


def normalize_skeleton_frame(coordinates: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Center on the hips and scale by torso length, with visibility-aware fallbacks."""
    valid = visibility > 0.05
    if not np.any(valid):
        return np.zeros_like(coordinates)

    hips_valid = valid[11] and valid[12]
    shoulders_valid = valid[5] and valid[6]
    hip_center = coordinates[[11, 12]].mean(axis=0) if hips_valid else None
    shoulder_center = coordinates[[5, 6]].mean(axis=0) if shoulders_valid else None
    if hip_center is not None:
        center = hip_center
    elif shoulder_center is not None:
        center = shoulder_center
    else:
        center = coordinates[valid].mean(axis=0)

    scale = 0.0
    if hip_center is not None and shoulder_center is not None:
        scale = float(np.linalg.norm(shoulder_center - hip_center))
    if scale < 0.03:
        visible_coordinates = coordinates[valid]
        span = visible_coordinates.max(axis=0) - visible_coordinates.min(axis=0)
        scale = float(max(span.max(), 0.05))

    result = (coordinates - center) / scale
    result[~valid] = 0.0
    return np.clip(result, -4.0, 4.0).astype(np.float32)


def landmarks_to_skeleton_frame(
    landmarks,
    previous_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Convert runtime landmarks to one normalized ``[5, 17]`` skeleton frame."""
    joint_count = len(COCO_MEDIAPIPE_INDICES)
    coordinates = np.zeros((joint_count, 2), dtype=np.float32)
    visibility = np.zeros(joint_count, dtype=np.float32)
    if landmarks is not None:
        for joint_index, source_index in enumerate(COCO_MEDIAPIPE_INDICES):
            if source_index >= len(landmarks):
                continue
            point = landmarks[source_index]
            coordinates[joint_index] = (float(point.x), float(point.y))
            visibility[joint_index] = float(point.visibility)
    normalized = normalize_skeleton_frame(coordinates, visibility)
    velocity = np.zeros_like(normalized)
    if previous_frame is not None:
        previous_visibility = previous_frame[2]
        valid = (visibility > 0.05) & (previous_visibility > 0.05)
        velocity[valid] = normalized[valid] - previous_frame[:2, valid].T
    return np.concatenate(
        (normalized.T, visibility[np.newaxis, :], velocity.T), axis=0
    ).astype(np.float32)


def fit_skeleton_normalizer(skeletons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel statistics on training skeleton windows only."""
    mean = skeletons.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = skeletons.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_skeleton_windows(
    skeletons: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (
        (skeletons - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
    ).astype(np.float32)


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0
