

from __future__ import annotations

from dataclasses import dataclass

from .features import PoseFeatures


@dataclass(frozen=True)
class RiskConfig:


    prefall_threshold: float = 0.45
    fall_threshold: float = 0.72


    min_visibility: float = 0.35


    torso_warn_deg: float = 25.0
    torso_fall_deg: float = 75.0

    angular_velocity_warn: float = 25.0
    angular_velocity_fall: float = 120.0

    vertical_velocity_warn: float = 0.22
    vertical_velocity_fall: float = 0.85

    center_drop_warn: float = 0.06
    center_drop_fall: float = 0.22

    aspect_ratio_warn: float = 0.55
    aspect_ratio_fall: float = 1.15


    torso_weight: float = 0.22
    angular_velocity_weight: float = 0.12
    vertical_velocity_weight: float = 0.34
    center_drop_weight: float = 0.16
    aspect_ratio_weight: float = 0.16


@dataclass(frozen=True)
class RiskBreakdown:

    risk_score: float
    torso_score: float
    angular_velocity_score: float
    vertical_velocity_score: float
    center_drop_score: float
    aspect_ratio_score: float
    visibility_factor: float
    center_drop: float


class RiskScorer:


    def __init__(self, config: RiskConfig | None = None) -> None:

        self.config = config or RiskConfig()

    def score(
        self,
        features: PoseFeatures,
        baseline_center_y: float | None,
    ) -> RiskBreakdown:


        if not features.has_pose:
            return RiskBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        cfg = self.config


        center_drop = 0.0
        if baseline_center_y is not None:
            center_drop = max(0.0, features.body_center_y - baseline_center_y)


        torso_score = ramp(features.torso_angle_deg, cfg.torso_warn_deg, cfg.torso_fall_deg)
        angular_velocity_score = ramp(
            max(0.0, features.torso_angular_velocity),
            cfg.angular_velocity_warn,
            cfg.angular_velocity_fall,
        )
        vertical_velocity_score = ramp(
            max(0.0, features.vertical_velocity),
            cfg.vertical_velocity_warn,
            cfg.vertical_velocity_fall,
        )
        center_drop_score = ramp(center_drop, cfg.center_drop_warn, cfg.center_drop_fall)
        aspect_ratio_score = ramp(
            features.aspect_ratio,
            cfg.aspect_ratio_warn,
            cfg.aspect_ratio_fall,
        )


        raw_score = (
            cfg.torso_weight * torso_score
            + cfg.angular_velocity_weight * angular_velocity_score
            + cfg.vertical_velocity_weight * vertical_velocity_score
            + cfg.center_drop_weight * center_drop_score
            + cfg.aspect_ratio_weight * aspect_ratio_score
        )


        visibility_factor = ramp(features.visibility_mean, cfg.min_visibility, 0.75)
        risk_score = clamp(raw_score * (0.35 + 0.65 * visibility_factor), 0.0, 1.0)

        return RiskBreakdown(
            risk_score=risk_score,
            torso_score=torso_score,
            angular_velocity_score=angular_velocity_score,
            vertical_velocity_score=vertical_velocity_score,
            center_drop_score=center_drop_score,
            aspect_ratio_score=aspect_ratio_score,
            visibility_factor=visibility_factor,
            center_drop=center_drop,
        )

    def state_from_score(self, risk_score: float) -> str:

        if risk_score >= self.config.fall_threshold:
            return "Fall"
        if risk_score >= self.config.prefall_threshold:
            return "Pre-fall"
        return "Normal"


def ramp(value: float, low: float, high: float) -> float:

    if high <= low:
        return 1.0 if value >= high else 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def clamp(value: float, low: float, high: float) -> float:

    return max(low, min(high, value))
