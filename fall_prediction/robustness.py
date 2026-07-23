"""Camera/person calibration and partial-pose robustness utilities.

The current classifier consumes image-coordinate features.  This module builds
an alternative, artifact-gated representation whose values are relative to a
short standing calibration period.  It also carries explicit validity masks so
zero never ambiguously means both "missing" and a real measurement.
"""

from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Mapping, Sequence

from .ml_features import ML_FEATURE_COLUMNS, _safe_float


POSE_VALIDITY_COLUMNS = (
    "torso_valid",
    "center_valid",
    "bbox_valid",
    "feature_coverage",
)
ROBUST_ML_FEATURE_COLUMNS = ML_FEATURE_COLUMNS + POSE_VALIDITY_COLUMNS
ROBUST_ACCEL_FEATURE_COLUMNS = ROBUST_ML_FEATURE_COLUMNS + (
    "torso_angular_accel",
    "vertical_accel",
)
UPPER_BODY_FEATURE_COLUMNS = (
    "shoulder_center_y",
    "shoulder_center_delta",
    "shoulder_vertical_velocity",
    "shoulder_line_angle",
    "shoulder_line_angular_velocity",
    "upper_body_width",
    "upper_body_height",
    "upper_body_aspect_ratio",
    "upper_body_valid",
)
UPPER_BODY_ML_FEATURE_COLUMNS = ROBUST_ML_FEATURE_COLUMNS + UPPER_BODY_FEATURE_COLUMNS
UPPER_BODY_ACCEL_FEATURE_COLUMNS = UPPER_BODY_ML_FEATURE_COLUMNS + (
    "torso_angular_accel",
    "vertical_accel",
)


class StandingFeatureCalibrator:
    """Map image-coordinate pose features into a standing-relative space.

    The transform is fitted once and then frozen.  It must never re-orient each
    frame independently, because doing so would erase the tilt produced by a
    real fall.
    """

    def __init__(
        self,
        baseline_frames: int = 15,
        min_visibility: float = 0.35,
        allow_upper_body_only_calibration: bool = False,
    ) -> None:
        self.baseline_frames = max(3, int(baseline_frames))
        self.min_visibility = max(0.0, min(1.0, float(min_visibility)))
        self.allow_upper_body_only_calibration = bool(allow_upper_body_only_calibration)
        self._samples: list[dict[str, float]] = []
        self._baseline: dict[str, float] | None = None
        self._previous_relative_angle: float | None = None
        self._previous_timestamp: float | None = None

    @property
    def ready(self) -> bool:
        return self._baseline is not None

    @property
    def collected_frames(self) -> int:
        return len(self._samples)

    @property
    def baseline(self) -> dict[str, float] | None:
        return dict(self._baseline) if self._baseline is not None else None

    def reset(self) -> None:
        self._samples.clear()
        self._baseline = None
        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        self._previous_relative_angle = None
        self._previous_timestamp = None

    def fit(self, rows: Sequence[Mapping[str, object]]) -> bool:
        """Fit from the first reliable standing rows in a sequence."""
        self.reset()
        for row in rows:
            if self._is_calibration_sample(row):
                self._samples.append(self._sample_values(row))
                if len(self._samples) >= self.baseline_frames:
                    self._finish_fit()
                    return True
        return False

    def update_and_transform(self, row: Mapping[str, object]) -> dict[str, float] | None:
        """Streaming calibration. Return None until enough standing rows exist."""
        if not self.ready:
            if not self._is_calibration_sample(row):
                return None
            self._samples.append(self._sample_values(row))
            if len(self._samples) < self.baseline_frames:
                return None
            self._finish_fit()
        return self.transform(row)

    def transform(self, row: Mapping[str, object]) -> dict[str, float]:
        if self._baseline is None:
            raise RuntimeError("StandingFeatureCalibrator must be fitted before transform().")

        baseline_height = max(self._baseline["body_height"], 1e-6)
        torso_valid, center_valid, bbox_valid = _validity_flags(row)
        has_pose = bool(torso_valid or center_valid or bbox_valid)
        timestamp = _timestamp(row)

        relative_angle = 0.0
        angular_velocity = 0.0
        if torso_valid:
            signed_angle = _safe_float(row.get("torso_signed_angle", row.get("torso_angle", 0.0)))
            relative_angle = abs(_angle_delta(signed_angle, self._baseline["torso_signed_angle"]))
            if self._previous_relative_angle is not None and self._previous_timestamp is not None:
                dt = max(timestamp - self._previous_timestamp, 1e-6)
                angular_velocity = (relative_angle - self._previous_relative_angle) / dt
            self._previous_relative_angle = relative_angle
            self._previous_timestamp = timestamp

        if center_valid:
            body_center_y = (
                _safe_float(row.get("body_center_y", 0.0)) - self._baseline["body_center_y"]
            ) / baseline_height
            body_center_delta = _safe_float(row.get("body_center_delta", 0.0)) / baseline_height
            vertical_velocity = _safe_float(row.get("vertical_velocity", 0.0)) / baseline_height
            center_drop = max(0.0, body_center_y)
        else:
            body_center_y = body_center_delta = vertical_velocity = center_drop = 0.0

        if bbox_valid:
            body_width = _safe_float(row.get("body_width", 0.0)) / max(self._baseline["body_width"], 1e-6)
            body_height = _safe_float(row.get("body_height", 0.0)) / baseline_height
            aspect_ratio = _safe_float(row.get("aspect_ratio", 0.0)) / max(
                self._baseline["aspect_ratio"], 1e-6
            )
        else:
            body_width = body_height = aspect_ratio = 0.0

        coverage = (float(torso_valid) + float(center_valid) + float(bbox_valid)) / 3.0
        upper_body_valid = _upper_body_valid(row) and self._baseline["upper_body_height"] > 1e-6
        upper_body_visibility = _safe_float(
            row.get("upper_body_visibility_mean", row.get("visibility_mean", 0.0))
        )
        if upper_body_valid:
            upper_body_scale = max(self._baseline["upper_body_height"], 1e-6)
            shoulder_center_y = (
                _safe_float(row.get("shoulder_center_y", 0.0))
                - self._baseline["shoulder_center_y"]
            ) / upper_body_scale
            shoulder_center_delta = _safe_float(row.get("shoulder_center_delta", 0.0)) / upper_body_scale
            shoulder_vertical_velocity = _safe_float(
                row.get("shoulder_vertical_velocity", 0.0)
            ) / upper_body_scale
            shoulder_line_angle = _angle_delta(
                _safe_float(row.get("shoulder_line_angle", 0.0)),
                self._baseline["shoulder_line_angle"],
            )
            shoulder_line_angular_velocity = _safe_float(
                row.get("shoulder_line_angular_velocity", 0.0)
            )
            upper_body_width = _safe_float(row.get("upper_body_width", 0.0)) / max(
                self._baseline["upper_body_width"], 1e-6
            )
            upper_body_height = _safe_float(row.get("upper_body_height", 0.0)) / max(
                self._baseline["upper_body_height"], 1e-6
            )
            upper_body_aspect_ratio = _safe_float(
                row.get("upper_body_aspect_ratio", 0.0)
            ) / max(self._baseline["upper_body_aspect_ratio"], 1e-6)
        else:
            shoulder_center_y = shoulder_center_delta = shoulder_vertical_velocity = 0.0
            shoulder_line_angle = shoulder_line_angular_velocity = 0.0
            upper_body_width = upper_body_height = upper_body_aspect_ratio = 0.0

        return {
            "has_pose": 1.0 if has_pose or upper_body_valid else 0.0,
            "torso_angle": relative_angle,
            "torso_angular_velocity": angular_velocity,
            "body_center_y": body_center_y,
            "body_center_delta": body_center_delta,
            "vertical_velocity": vertical_velocity,
            "aspect_ratio": aspect_ratio,
            "body_width": body_width,
            "body_height": body_height,
            "visibility_mean": (
                upper_body_visibility
                if upper_body_valid and not (torso_valid or center_valid or bbox_valid)
                else _safe_float(row.get("visibility_mean", 0.0))
            ),
            "center_drop": center_drop,
            "torso_valid": float(torso_valid),
            "center_valid": float(center_valid),
            "bbox_valid": float(bbox_valid),
            "feature_coverage": coverage,
            "shoulder_center_y": shoulder_center_y,
            "shoulder_center_delta": shoulder_center_delta,
            "shoulder_vertical_velocity": shoulder_vertical_velocity,
            "shoulder_line_angle": shoulder_line_angle,
            "shoulder_line_angular_velocity": shoulder_line_angular_velocity,
            "upper_body_width": upper_body_width,
            "upper_body_height": upper_body_height,
            "upper_body_aspect_ratio": upper_body_aspect_ratio,
            "upper_body_valid": float(upper_body_valid),
            "upper_body_visibility_mean": upper_body_visibility,
        }

    def _finish_fit(self) -> None:
        self._baseline = {
            key: median(sample[key] for sample in self._samples)
            for key in self._samples[0]
        }
        self.reset_temporal_state()

    def _is_calibration_sample(self, row: Mapping[str, object]) -> bool:
        torso_valid, center_valid, bbox_valid = _validity_flags(row)
        full_body_sample = (
            torso_valid
            and center_valid
            and bbox_valid
            and _safe_float(row.get("visibility_mean", 0.0)) >= self.min_visibility
            and _safe_float(row.get("body_height", 0.0)) > 1e-4
            and _safe_float(row.get("body_width", 0.0)) > 1e-4
        )
        upper_body_sample = (
            self.allow_upper_body_only_calibration
            and _upper_body_valid(row)
            and _safe_float(
                row.get("upper_body_visibility_mean", row.get("visibility_mean", 0.0))
            )
            >= self.min_visibility
            and _safe_float(row.get("upper_body_height", 0.0)) > 1e-4
            and _safe_float(row.get("upper_body_width", 0.0)) > 1e-4
        )
        return full_body_sample or upper_body_sample

    @staticmethod
    def _sample_values(row: Mapping[str, object]) -> dict[str, float]:
        return {
            "torso_signed_angle": _safe_float(
                row.get("torso_signed_angle", row.get("torso_angle", 0.0))
            ),
            "body_center_y": _safe_float(row.get("body_center_y", 0.0)),
            "body_width": _safe_float(row.get("body_width", 0.0)),
            "body_height": _safe_float(row.get("body_height", 0.0)),
            "aspect_ratio": _safe_float(row.get("aspect_ratio", 0.0)),
            "shoulder_center_y": _safe_float(row.get("shoulder_center_y", 0.0)),
            "shoulder_line_angle": _safe_float(row.get("shoulder_line_angle", 0.0)),
            "upper_body_width": _safe_float(row.get("upper_body_width", 0.0)),
            "upper_body_height": _safe_float(row.get("upper_body_height", 0.0)),
            "upper_body_aspect_ratio": _safe_float(row.get("upper_body_aspect_ratio", 0.0)),
        }


def calibrate_feature_rows(
    rows: Sequence[Mapping[str, object]],
    baseline_frames: int = 15,
    min_visibility: float = 0.35,
) -> tuple[list[dict[str, float]], dict[str, float] | None]:
    """Fit once, then transform a complete offline video sequence."""
    calibrator = StandingFeatureCalibrator(baseline_frames, min_visibility)
    if not calibrator.fit(rows):
        return [], None
    transformed = [calibrator.transform(row) for row in rows]
    return transformed, calibrator.baseline


def apply_partial_pose_dropout(
    window_rows: Sequence[Mapping[str, object]],
    pattern: str,
) -> list[dict[str, float]]:
    """Create deterministic structured-occlusion training examples."""
    rows = [deepcopy(dict(row)) for row in window_rows]
    if pattern == "torso":
        for row in rows:
            _drop_group(row, "torso")
    elif pattern == "center":
        for row in rows:
            _drop_group(row, "center")
    elif pattern == "bbox":
        for row in rows:
            _drop_group(row, "bbox")
    elif pattern == "temporal":
        # A short internal occlusion, while leaving context on both sides.
        start = max(1, len(rows) // 3)
        end = min(len(rows) - 1, start + max(1, len(rows) // 4))
        for row in rows[start:end]:
            _drop_group(row, "all")
    elif pattern == "lower_body":
        for row in rows:
            _drop_group(row, "torso")
            _drop_group(row, "center")
            _drop_group(row, "bbox")
            row["visibility_mean"] = _safe_float(
                row.get("upper_body_visibility_mean", row.get("visibility_mean", 0.0))
            )
    elif pattern == "upper_body":
        for row in rows:
            _drop_group(row, "upper_body")
    else:
        raise ValueError(f"Unknown partial pose dropout pattern: {pattern}")

    for row in rows:
        masks = [
            _safe_float(row.get("torso_valid", 0.0)),
            _safe_float(row.get("center_valid", 0.0)),
            _safe_float(row.get("bbox_valid", 0.0)),
        ]
        row["feature_coverage"] = sum(masks) / 3.0
        row["has_pose"] = 1.0 if any(mask > 0.0 for mask in masks) or _upper_body_valid(row) else 0.0
    return rows


def _drop_group(row: dict[str, object], group: str) -> None:
    if group in {"torso", "all"}:
        row["torso_angle"] = 0.0
        row["torso_angular_velocity"] = 0.0
        row["torso_valid"] = 0.0
    if group in {"center", "all"}:
        for key in ("body_center_y", "body_center_delta", "vertical_velocity", "center_drop"):
            row[key] = 0.0
        row["center_valid"] = 0.0
    if group in {"bbox", "all"}:
        for key in ("aspect_ratio", "body_width", "body_height"):
            row[key] = 0.0
        row["bbox_valid"] = 0.0
    if group in {"upper_body", "all"}:
        for key in UPPER_BODY_FEATURE_COLUMNS:
            row[key] = 0.0


def _validity_flags(row: Mapping[str, object]) -> tuple[bool, bool, bool]:
    has_pose = _safe_float(row.get("has_pose", 0.0)) > 0.0
    torso = _safe_float(row.get("torso_valid", 1.0 if has_pose else 0.0)) > 0.0
    center = _safe_float(row.get("center_valid", 1.0 if has_pose else 0.0)) > 0.0
    default_bbox = has_pose and _safe_float(row.get("body_height", 0.0)) > 0.0
    bbox = _safe_float(row.get("bbox_valid", 1.0 if default_bbox else 0.0)) > 0.0
    return torso, center, bbox


def _upper_body_valid(row: Mapping[str, object]) -> bool:
    return _safe_float(row.get("upper_body_valid", 0.0)) > 0.0


def _timestamp(row: Mapping[str, object]) -> float:
    return _safe_float(row.get("timestamp", row.get("time", row.get("frame", 0.0))))


def _angle_delta(current: float, baseline: float) -> float:
    return (current - baseline + 180.0) % 360.0 - 180.0
