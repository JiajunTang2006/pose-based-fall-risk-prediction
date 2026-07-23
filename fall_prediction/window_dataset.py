

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .ml_features import (
    ML_FEATURE_COLUMNS,
    ACCEL_FEATURE_COLUMNS,
    flatten_window,
    make_window_feature_names,
    compute_window_accel_features,
)
from .robustness import (
    ROBUST_ACCEL_FEATURE_COLUMNS,
    ROBUST_ML_FEATURE_COLUMNS,
    UPPER_BODY_ACCEL_FEATURE_COLUMNS,
    UPPER_BODY_ML_FEATURE_COLUMNS,
    apply_partial_pose_dropout,
    calibrate_feature_rows,
)


DEFAULT_WINDOW_SIZE = 15
DEFAULT_STRIDE = 3


@dataclass(frozen=True)
class WindowDataset:


    X: list[list[float]]
    y: list[str]
    groups: list[str]
    feature_names: list[str]


@dataclass(frozen=True)
class LabelInterval:


    video: str
    start_frame: int
    end_frame: int
    label: str


def build_window_dataset(
    csv_paths: Sequence[str | Path],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
    label_mode: str = "filename",
    annotations_path: str | Path | Sequence[str | Path] | None = None,
    use_accel: bool = False,
    use_standing_calibration: bool = False,
    partial_pose_augmentation: bool = False,
    baseline_frames: int = 15,
    use_upper_body_features: bool = False,
) -> WindowDataset:

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if label_mode not in {"filename", "annotations"}:
        raise ValueError("label_mode must be 'filename' or 'annotations'")
    if label_mode == "annotations" and annotations_path is None:
        raise ValueError("annotations_path is required when label_mode='annotations'")

    if use_upper_body_features:
        base_feature_columns = UPPER_BODY_ML_FEATURE_COLUMNS
        feature_columns = (
            UPPER_BODY_ACCEL_FEATURE_COLUMNS if use_accel else UPPER_BODY_ML_FEATURE_COLUMNS
        )
    elif use_standing_calibration:
        base_feature_columns = ROBUST_ML_FEATURE_COLUMNS
        feature_columns = ROBUST_ACCEL_FEATURE_COLUMNS if use_accel else ROBUST_ML_FEATURE_COLUMNS
    else:
        base_feature_columns = ML_FEATURE_COLUMNS
        if use_accel:
            feature_columns = ACCEL_FEATURE_COLUMNS

    intervals = load_label_intervals(annotations_path) if annotations_path else {}
    X: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []

    for csv_path in sorted(Path(path) for path in csv_paths):

        rows = load_feature_rows(csv_path)
        if use_standing_calibration or use_upper_body_features:
            rows, _baseline = calibrate_feature_rows(rows, baseline_frames=baseline_frames)
        if len(rows) < window_size:
            continue

        video_key = _video_key(csv_path)
        file_label = infer_label_from_filename(csv_path)


        for start in range(0, len(rows) - window_size + 1, stride):
            window_rows = rows[start : start + window_size]


            end_frame = _row_frame(window_rows[-1], start + window_size - 1)
            # The final frame is the current state predicted from the preceding window.
            label = _label_for_window(
                csv_path=csv_path,
                video_key=video_key,
                end_frame=end_frame,
                file_label=file_label,
                label_mode=label_mode,
                intervals=intervals,
            )
            if label is None:
                continue

            variants: list[Sequence[Mapping[str, object]]] = [window_rows]
            if (use_standing_calibration or use_upper_body_features) and partial_pose_augmentation:
                patterns = ["torso", "center", "bbox", "temporal"]
                if use_upper_body_features:
                    patterns.extend(("lower_body", "upper_body"))
                variants.extend(
                    apply_partial_pose_dropout(window_rows, pattern)
                    for pattern in patterns
                )

            for variant in variants:
                prepared_rows = list(variant)
                if use_accel:
                    prepared_rows = compute_window_accel_features(
                        prepared_rows,
                        base_feature_columns=base_feature_columns,
                    )
                X.append(flatten_window(prepared_rows, feature_columns))
                y.append(label)
                groups.append(video_key)

    return WindowDataset(
        X=X,
        y=y,
        groups=groups,
        feature_names=make_window_feature_names(window_size, feature_columns),
    )


def load_feature_rows(csv_path: str | Path) -> list[dict[str, str]]:

    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def infer_label_from_filename(path: str | Path) -> str | None:

    stem = Path(path).stem.lower()
    if stem.startswith("fall") or "_fall" in stem or "-fall" in stem:
        return "Fall"
    if stem.startswith("adl") or stem.startswith("normal") or "nonfall" in stem:
        return "Normal"
    return None


def load_label_intervals(
    annotations_path: str | Path | Sequence[str | Path] | None,
) -> dict[str, list[LabelInterval]]:

    if annotations_path is None:
        return {}

    paths: Sequence[str | Path]
    if isinstance(annotations_path, (str, Path)):
        paths = [annotations_path]
    else:
        paths = annotations_path

    intervals: dict[str, list[LabelInterval]] = {}
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"video", "start_frame", "end_frame", "label"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"Annotation CSV is missing columns: {missing_text}")

            for row in reader:
                video = _normalize_video_name(row["video"])
                interval = LabelInterval(
                    video=video,
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    label=row["label"].strip(),
                )
                intervals.setdefault(video, []).append(interval)

    return intervals


def _label_for_window(
    csv_path: Path,
    video_key: str,
    end_frame: int,
    file_label: str | None,
    label_mode: str,
    intervals: Mapping[str, Sequence[LabelInterval]],
) -> str | None:

    if label_mode == "filename":
        return file_label


    for key in _annotation_keys(csv_path, video_key):
        for interval in intervals.get(key, ()):
            if interval.start_frame <= end_frame <= interval.end_frame:
                return interval.label


    return "Normal" if file_label == "Normal" else None


def boundary_distance_for_frame(
    csv_path: str | Path,
    video_key: str,
    frame: int,
    intervals: Mapping[str, Sequence[LabelInterval]],
) -> int | None:
    """Return distance in source frames to the nearest true label transition.

    The first interval is not a transition.  A boundary is the first frame of
    a new interval whose label differs from the preceding interval.  ADL files
    without interval annotations therefore return ``None``.
    """
    path = Path(csv_path)
    matched: Sequence[LabelInterval] = ()
    for key in _annotation_keys(path, video_key):
        if intervals.get(key):
            matched = intervals[key]
            break
    if len(matched) < 2:
        return None
    ordered = sorted(matched, key=lambda item: (item.start_frame, item.end_frame))
    boundaries = [
        current.start_frame
        for previous, current in zip(ordered, ordered[1:])
        if previous.label != current.label
    ]
    if not boundaries:
        return None
    return min(abs(int(frame) - boundary) for boundary in boundaries)


def _annotation_keys(csv_path: Path, video_key: str) -> tuple[str, ...]:

    stem = _normalize_video_name(csv_path.stem)
    name = _normalize_video_name(csv_path.name)
    parent_name = _normalize_video_name(f"{csv_path.parent.name}/{csv_path.stem}")
    return (video_key, stem, name, parent_name)


def _video_key(path: str | Path) -> str:

    stem = _normalize_video_name(Path(path).stem)
    upfall_match = re.match(r"^(subject\d+activity\d+trial\d+)camera\d+$", stem)
    if upfall_match:
        return upfall_match.group(1)
    return stem


def _normalize_video_name(value: str) -> str:

    value = value.replace("\\", "/").strip().lower()
    suffixes = (".csv", ".mp4", ".avi", ".mov", ".mkv")
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _row_frame(row: Mapping[str, str], fallback: int) -> int:

    try:
        return int(float(row.get("frame", fallback)))
    except (TypeError, ValueError):
        return fallback
