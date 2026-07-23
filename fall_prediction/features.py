

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .landmarks import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
    has_landmarks,
    mean_visibility,
    midpoint,
    visible_points,
)


UPPER_BODY_LANDMARKS = (
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)
LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


@dataclass(frozen=True)
class PoseFeatures:

    frame_index: int
    timestamp: float
    has_pose: bool
    torso_angle_deg: float = 0.0
    torso_angular_velocity: float = 0.0
    body_center_y: float = 0.0
    body_center_delta: float = 0.0
    vertical_velocity: float = 0.0
    aspect_ratio: float = 0.0
    body_width: float = 0.0
    body_height: float = 0.0
    visibility_mean: float = 0.0
    torso_signed_angle_deg: float = 0.0
    torso_valid: bool = False
    center_valid: bool = False
    bbox_valid: bool = False
    shoulder_center_y: float = 0.0
    shoulder_center_delta: float = 0.0
    shoulder_vertical_velocity: float = 0.0
    shoulder_line_angle_deg: float = 0.0
    shoulder_line_angular_velocity: float = 0.0
    upper_body_width: float = 0.0
    upper_body_height: float = 0.0
    upper_body_aspect_ratio: float = 0.0
    upper_body_valid: bool = False
    upper_body_visibility_mean: float = 0.0


class FeatureExtractor:


    def __init__(self, min_visibility: float = 0.2) -> None:

        self.min_visibility = min_visibility

        self._previous_center_y: float | None = None
        self._previous_torso_angle: float | None = None
        self._previous_timestamp: float | None = None
        self._previous_shoulder_center_y: float | None = None
        self._previous_shoulder_angle: float | None = None
        self._previous_shoulder_timestamp: float | None = None

    def extract(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> PoseFeatures:


        if not has_landmarks(landmarks):
            return PoseFeatures(frame_index=frame_index, timestamp=timestamp, has_pose=False)

        assert landmarks is not None

        shoulders_valid = self._points_visible(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER))
        hips_valid = self._points_visible(landmarks, (LEFT_HIP, RIGHT_HIP))
        torso_valid = shoulders_valid and hips_valid
        center_valid = torso_valid

        body_center_y = 0.0
        torso_angle = 0.0
        torso_signed_angle = 0.0
        shoulder_center_y = 0.0
        shoulder_center_delta = 0.0
        shoulder_vertical_velocity = 0.0
        shoulder_line_angle = 0.0
        shoulder_line_angular_velocity = 0.0
        if torso_valid:
            shoulder_mid = midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
            hip_mid = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
            body_center_y = (shoulder_mid.y + hip_mid.y) / 2.0
            torso_signed_angle = self._signed_torso_angle_from_vertical(shoulder_mid, hip_mid)
            torso_angle = abs(torso_signed_angle)

        if shoulders_valid:
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            shoulder_mid = midpoint(left_shoulder, right_shoulder)
            shoulder_center_y = shoulder_mid.y
            shoulder_line_angle = math.degrees(
                math.atan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x,
                )
            )
            if self._previous_shoulder_center_y is not None and self._previous_shoulder_timestamp is not None:
                shoulder_dt = max(timestamp - self._previous_shoulder_timestamp, 1e-6)
                shoulder_center_delta = shoulder_center_y - self._previous_shoulder_center_y
                shoulder_vertical_velocity = shoulder_center_delta / shoulder_dt
                if self._previous_shoulder_angle is not None:
                    shoulder_line_angular_velocity = (
                        shoulder_line_angle - self._previous_shoulder_angle
                    ) / shoulder_dt
            self._previous_shoulder_center_y = shoulder_center_y
            self._previous_shoulder_angle = shoulder_line_angle
            self._previous_shoulder_timestamp = timestamp


        body_width, body_height, aspect_ratio = self._body_box(landmarks)
        visible_lower_body_points = sum(
            landmarks[index].visibility >= self.min_visibility
            for index in LOWER_BODY_LANDMARKS
        )
        bbox_valid = (
            body_width > 1e-6
            and body_height > 1e-6
            and visible_lower_body_points >= 2
        )
        upper_body_width, upper_body_height, upper_body_aspect_ratio = self._body_box(
            landmarks,
            indices=UPPER_BODY_LANDMARKS,
        )
        upper_body_valid = (
            shoulders_valid
            and upper_body_width > 1e-6
            and upper_body_height > 1e-6
        )


        visibility = mean_visibility(landmarks)
        upper_body_visibility = mean_visibility(landmarks, indices=UPPER_BODY_LANDMARKS)


        dt = self._delta_time(timestamp)
        center_delta = 0.0
        vertical_velocity = 0.0
        angular_velocity = 0.0

        if center_valid and self._previous_center_y is not None:

            center_delta = body_center_y - self._previous_center_y

            vertical_velocity = center_delta / dt

        if torso_valid and self._previous_torso_angle is not None:

            angular_velocity = (torso_angle - self._previous_torso_angle) / dt


        if center_valid:
            self._previous_center_y = body_center_y
        if torso_valid:
            self._previous_torso_angle = torso_angle
        # Do not advance the motion clock on a bbox-only/fully missing frame.
        # When torso points return, velocity is then divided by the whole gap
        # duration instead of being exaggerated as a one-frame jump.
        if center_valid or torso_valid:
            self._previous_timestamp = timestamp

        return PoseFeatures(
            frame_index=frame_index,
            timestamp=timestamp,
            has_pose=True,
            torso_angle_deg=torso_angle,
            torso_angular_velocity=angular_velocity,
            body_center_y=body_center_y,
            body_center_delta=center_delta,
            vertical_velocity=vertical_velocity,
            aspect_ratio=aspect_ratio,
            body_width=body_width,
            body_height=body_height,
            visibility_mean=visibility,
            torso_signed_angle_deg=torso_signed_angle,
            torso_valid=torso_valid,
            center_valid=center_valid,
            bbox_valid=bbox_valid,
            shoulder_center_y=shoulder_center_y,
            shoulder_center_delta=shoulder_center_delta,
            shoulder_vertical_velocity=shoulder_vertical_velocity,
            shoulder_line_angle_deg=shoulder_line_angle,
            shoulder_line_angular_velocity=shoulder_line_angular_velocity,
            upper_body_width=upper_body_width,
            upper_body_height=upper_body_height,
            upper_body_aspect_ratio=upper_body_aspect_ratio,
            upper_body_valid=upper_body_valid,
            upper_body_visibility_mean=upper_body_visibility,
        )

    def reset(self) -> None:

        self._previous_center_y = None
        self._previous_torso_angle = None
        self._previous_timestamp = None
        self._previous_shoulder_center_y = None
        self._previous_shoulder_angle = None
        self._previous_shoulder_timestamp = None

    def _delta_time(self, timestamp: float) -> float:

        if self._previous_timestamp is None:
            return 1.0 / 30.0
        return max(timestamp - self._previous_timestamp, 1e-6)

    def _body_box(
        self,
        landmarks: Sequence[Landmark],
        indices: Sequence[int] | None = None,
    ) -> tuple[float, float, float]:


        points = (
            visible_points(landmarks, self.min_visibility)
            if indices is None
            else [
                landmarks[index]
                for index in indices
                if landmarks[index].visibility >= self.min_visibility
            ]
        )
        if len(points) < 2:
            return 0.0, 0.0, 0.0


        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        min_y = min(point.y for point in points)
        max_y = max(point.y for point in points)

        width = max_x - min_x
        height = max_y - min_y

        aspect_ratio = width / max(height, 1e-6)
        return width, height, aspect_ratio

    def _points_visible(self, landmarks: Sequence[Landmark], indices: Sequence[int]) -> bool:
        return all(landmarks[index].visibility >= self.min_visibility for index in indices)

    @staticmethod
    def _torso_angle_from_vertical(shoulder_mid: Landmark, hip_mid: Landmark) -> float:

        dx = shoulder_mid.x - hip_mid.x
        dy = shoulder_mid.y - hip_mid.y

        return math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))

    @staticmethod
    def _signed_torso_angle_from_vertical(shoulder_mid: Landmark, hip_mid: Landmark) -> float:
        """Signed tilt around the image vertical; standing is approximately 0°."""
        dx = shoulder_mid.x - hip_mid.x
        upward_dy = hip_mid.y - shoulder_mid.y
        return math.degrees(math.atan2(dx, max(upward_dy, 1e-6)))
