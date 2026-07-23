

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .features import PoseFeatures


ML_FEATURE_COLUMNS = (

    "has_pose",

    "torso_angle",

    "torso_angular_velocity",

    "body_center_y",

    "body_center_delta",

    "vertical_velocity",

    "aspect_ratio",

    "body_width",

    "body_height",

    "visibility_mean",

    "center_drop",
)


ACCEL_FEATURE_COLUMNS = ML_FEATURE_COLUMNS + (
    "torso_angular_accel",
    "vertical_accel",
)


def pose_features_to_ml_row(features: PoseFeatures, center_drop: float = 0.0) -> dict[str, float]:

    return {
        "has_pose": 1.0 if features.has_pose else 0.0,
        "torso_angle": features.torso_angle_deg,
        "torso_angular_velocity": features.torso_angular_velocity,
        "body_center_y": features.body_center_y,
        "body_center_delta": features.body_center_delta,
        "vertical_velocity": features.vertical_velocity,
        "aspect_ratio": features.aspect_ratio,
        "body_width": features.body_width,
        "body_height": features.body_height,
        "visibility_mean": features.visibility_mean,
        "center_drop": center_drop,
        # Extra metadata used only by robust artifacts; legacy feature columns
        # ignore these keys and therefore remain fully compatible.
        "timestamp": features.timestamp,
        "torso_signed_angle": features.torso_signed_angle_deg,
        "torso_valid": 1.0 if features.torso_valid else 0.0,
        "center_valid": 1.0 if features.center_valid else 0.0,
        "bbox_valid": 1.0 if features.bbox_valid else 0.0,
        "feature_coverage": (
            float(features.torso_valid) + float(features.center_valid) + float(features.bbox_valid)
        )
        / 3.0,
        "shoulder_center_y": features.shoulder_center_y,
        "shoulder_center_delta": features.shoulder_center_delta,
        "shoulder_vertical_velocity": features.shoulder_vertical_velocity,
        "shoulder_line_angle": features.shoulder_line_angle_deg,
        "shoulder_line_angular_velocity": features.shoulder_line_angular_velocity,
        "upper_body_width": features.upper_body_width,
        "upper_body_height": features.upper_body_height,
        "upper_body_aspect_ratio": features.upper_body_aspect_ratio,
        "upper_body_valid": 1.0 if features.upper_body_valid else 0.0,
        "upper_body_visibility_mean": features.upper_body_visibility_mean,
    }


def row_to_feature_values(
    row: Mapping[str, object],
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[float]:

    return [_safe_float(row.get(column, 0.0)) for column in feature_columns]


def flatten_window(
    rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[float]:

    values: list[float] = []
    for row in rows:
        values.extend(row_to_feature_values(row, feature_columns))
    return values


def make_window_feature_names(
    window_size: int,
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[str]:

    names: list[str] = []
    for index in range(window_size):
        relative = index - window_size + 1
        prefix = "t" if relative == 0 else f"t{relative}"
        names.extend(f"{prefix}_{column}" for column in feature_columns)
    return names


def _safe_float(value: object) -> float:

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def compute_window_accel_features(
    window_rows: Sequence[Mapping[str, object]],
    base_feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[dict[str, float]]:

    enhanced: list[dict[str, float]] = []
    for i, row in enumerate(window_rows):
        entry: dict[str, float] = {}

        for col in base_feature_columns:
            if col in {"torso_angular_accel", "vertical_accel"}:
                continue
            entry[col] = _safe_float(row.get(col, 0.0))

        if i == 0:
            entry["torso_angular_accel"] = 0.0
            entry["vertical_accel"] = 0.0
        else:
            prev = window_rows[i - 1]
            entry["torso_angular_accel"] = (
                _safe_float(row.get("torso_angular_velocity", 0.0))
                - _safe_float(prev.get("torso_angular_velocity", 0.0))
            )
            entry["vertical_accel"] = (
                _safe_float(row.get("vertical_velocity", 0.0))
                - _safe_float(prev.get("vertical_velocity", 0.0))
            )
        enhanced.append(entry)
    return enhanced
