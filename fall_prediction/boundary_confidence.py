"""Confidence weighting and tolerant metrics for ambiguous label boundaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np

from .train_model import build_validation_metrics


DEFAULT_BOUNDARY_TOLERANCE_FRAMES = 5
DEFAULT_INNER_BOUNDARY_FRAMES = 2
DEFAULT_INNER_BOUNDARY_WEIGHT = 0.35
DEFAULT_OUTER_BOUNDARY_WEIGHT = 0.65


def build_boundary_confidence_weights(
    boundary_distances: Sequence[int],
    *,
    tolerance_frames: int = DEFAULT_BOUNDARY_TOLERANCE_FRAMES,
    inner_frames: int = DEFAULT_INNER_BOUNDARY_FRAMES,
    inner_weight: float = DEFAULT_INNER_BOUNDARY_WEIGHT,
    outer_weight: float = DEFAULT_OUTER_BOUNDARY_WEIGHT,
) -> np.ndarray:
    """Map distance-to-transition into a non-zero per-window confidence."""
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames cannot be negative")
    if inner_frames < 0 or inner_frames > tolerance_frames:
        raise ValueError("inner_frames must be between 0 and tolerance_frames")
    for name, value in (("inner_weight", inner_weight), ("outer_weight", outer_weight)):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if inner_weight > outer_weight:
        raise ValueError("inner_weight cannot exceed outer_weight")

    distances = np.asarray(boundary_distances, dtype=np.int64)
    weights = np.ones(len(distances), dtype=np.float32)
    annotated = distances >= 0
    weights[annotated & (distances <= tolerance_frames)] = float(outer_weight)
    weights[annotated & (distances <= inner_frames)] = float(inner_weight)
    return weights


def combine_label_and_confidence_weights(
    labels: Sequence[str],
    class_weights: Mapping[str, float] | None,
    confidence_weights: Sequence[float] | None,
) -> np.ndarray:
    """Multiply class-balance and boundary-confidence weights."""
    label_values = [str(value) for value in labels]
    class_weights = class_weights or {}
    result = np.asarray(
        [float(class_weights.get(label, 1.0)) for label in label_values],
        dtype=np.float32,
    )
    if confidence_weights is not None:
        confidence = np.asarray(confidence_weights, dtype=np.float32)
        if confidence.shape != result.shape:
            raise ValueError("confidence_weights must have one value per label")
        if not np.all(np.isfinite(confidence)) or np.any(confidence <= 0.0):
            raise ValueError("confidence_weights must contain finite positive values")
        result *= confidence
    return result


def build_boundary_tolerant_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sequences: Sequence[str],
    end_frames: Sequence[int],
    labels: Sequence[str],
    *,
    tolerance_frames: int = DEFAULT_BOUNDARY_TOLERANCE_FRAMES,
) -> dict[str, Any]:
    """Accept adjacent-state disagreements within the frame tolerance.

    This does not replace strict metrics.  It produces an additional view by
    changing only mismatches where the predicted label is also a true label in
    the same sequence within ``±tolerance_frames``.
    """
    _validate_equal_lengths(y_true, y_pred, sequences, end_frames)
    truths = [str(value) for value in y_true]
    predictions = [str(value) for value in y_pred]
    sequence_names = [str(value) for value in sequences]
    frames = np.asarray(end_frames, dtype=np.int64)
    nearby_true_labels: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for truth, sequence, frame in zip(truths, sequence_names, frames):
        nearby_true_labels[sequence].append((int(frame), truth))

    adjusted = predictions.copy()
    accepted = 0
    accepted_by_transition: Counter[str] = Counter()
    for index, (truth, prediction, sequence, frame) in enumerate(
        zip(truths, predictions, sequence_names, frames)
    ):
        if truth == prediction:
            continue
        nearby = {
            label
            for candidate_frame, label in nearby_true_labels[sequence]
            if abs(candidate_frame - int(frame)) <= tolerance_frames
        }
        if prediction in nearby and truth in nearby and len(nearby) > 1:
            adjusted[index] = truth
            accepted += 1
            accepted_by_transition[f"{truth}<->{prediction}"] += 1

    metrics = build_validation_metrics(truths, adjusted, labels)
    return {
        "tolerance_frames": int(tolerance_frames),
        "accepted_boundary_mismatches": int(accepted),
        "accepted_by_label_pair": dict(sorted(accepted_by_transition.items())),
        "metrics": metrics,
    }


def build_transition_latency_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sequences: Sequence[str],
    end_frames: Sequence[int],
    boundary_distances: Sequence[int],
    *,
    tolerance_frames: int = DEFAULT_BOUNDARY_TOLERANCE_FRAMES,
) -> dict[str, Any]:
    """Report prediction onset latency for every observed true transition."""
    _validate_equal_lengths(
        y_true, y_pred, sequences, end_frames, boundary_distances
    )
    grouped: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    for truth, prediction, sequence, frame, distance in zip(
        y_true, y_pred, sequences, end_frames, boundary_distances
    ):
        grouped[str(sequence)].append(
            (int(frame), str(truth), str(prediction), int(distance))
        )

    records: list[dict[str, Any]] = []
    for sequence, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[0])
        changes: list[tuple[int, str, str, int]] = []
        for index in range(1, len(rows)):
            previous_label = rows[index - 1][1]
            current_frame, current_label, _prediction, distance = rows[index]
            if current_label == previous_label:
                continue
            boundary_frame = current_frame - max(0, distance)
            changes.append((index, previous_label, current_label, boundary_frame))

        previous_boundary = rows[0][0]
        for change_index, (row_index, from_label, to_label, boundary_frame) in enumerate(changes):
            next_boundary = (
                changes[change_index + 1][3]
                if change_index + 1 < len(changes)
                else rows[-1][0] + tolerance_frames
            )
            search_rows = [
                row
                for row in rows
                if previous_boundary <= row[0] <= next_boundary
            ]
            onset_frame = _first_target_onset(search_rows, to_label)
            latency = None if onset_frame is None else int(onset_frame - boundary_frame)
            records.append(
                {
                    "sequence": sequence,
                    "transition": f"{from_label}->{to_label}",
                    "boundary_frame": int(boundary_frame),
                    "predicted_onset_frame": onset_frame,
                    "latency_frames": latency,
                    "within_tolerance": latency is not None and abs(latency) <= tolerance_frames,
                }
            )
            previous_boundary = boundary_frame

    by_transition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_transition[record["transition"]].append(record)
    return {
        "tolerance_frames": int(tolerance_frames),
        "overall": _summarize_latency(records, tolerance_frames),
        "by_transition": {
            name: _summarize_latency(items, tolerance_frames)
            for name, items in sorted(by_transition.items())
        },
        "records": records,
    }


def summarize_confidence_weights(weights: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float32)
    counts = Counter(f"{float(value):.2f}" for value in values)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else 0.0,
        "min": float(values.min()) if len(values) else 0.0,
        "max": float(values.max()) if len(values) else 0.0,
        "counts": dict(sorted(counts.items())),
    }


def _first_target_onset(
    rows: Sequence[tuple[int, str, str, int]],
    target_label: str,
) -> int | None:
    previous_prediction: str | None = None
    for frame, _truth, prediction, _distance in rows:
        if prediction == target_label and previous_prediction != target_label:
            return int(frame)
        previous_prediction = prediction
    return None


def _summarize_latency(
    records: Sequence[Mapping[str, Any]],
    tolerance_frames: int,
) -> dict[str, Any]:
    latencies = [int(item["latency_frames"]) for item in records if item["latency_frames"] is not None]
    return {
        "transitions": int(len(records)),
        "detected": int(len(latencies)),
        "missed": int(len(records) - len(latencies)),
        "within_tolerance": int(sum(abs(value) <= tolerance_frames for value in latencies)),
        "early": int(sum(value < -tolerance_frames for value in latencies)),
        "late": int(sum(value > tolerance_frames for value in latencies)),
        "mean_latency_frames": float(mean(latencies)) if latencies else None,
        "median_latency_frames": float(median(latencies)) if latencies else None,
        "min_latency_frames": min(latencies) if latencies else None,
        "max_latency_frames": max(latencies) if latencies else None,
    }


def _validate_equal_lengths(*values: Sequence[Any]) -> None:
    if len({len(value) for value in values}) != 1:
        raise ValueError("all metric inputs must have equal length")
