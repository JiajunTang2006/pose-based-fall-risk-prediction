"""Sensitivity profiles shared by the rule and ML predictors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .predictor import PredictorConfig
from .risk import RiskConfig


SensitivityLevel = str

DEFAULT_SENSITIVITY: SensitivityLevel = "medium"
SENSITIVITY_LEVELS: tuple[SensitivityLevel, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class MLSensitivityConfig:
    prefall_alert_threshold: float
    prefall_alert_frames: int
    fall_validator_settings: Mapping[str, float | int]


@dataclass(frozen=True)
class SensitivityProfile:
    level: SensitivityLevel
    predictor_config: PredictorConfig
    ml: MLSensitivityConfig

    def thresholds(self) -> dict[str, float]:
        risk = self.predictor_config.risk
        return {
            "prefall_threshold": risk.prefall_threshold,
            "fall_threshold": risk.fall_threshold,
        }


def normalize_sensitivity(level: str | None) -> SensitivityLevel:
    if isinstance(level, str):
        normalized = level.strip().lower()
        if normalized in SENSITIVITY_LEVELS:
            return normalized
    return DEFAULT_SENSITIVITY


def sensitivity_profile(level: str | None) -> SensitivityProfile:
    return SENSITIVITY_PROFILES[normalize_sensitivity(level)]


def predictor_config_for_sensitivity(level: str | None) -> PredictorConfig:
    return sensitivity_profile(level).predictor_config


def ml_config_for_sensitivity(level: str | None) -> MLSensitivityConfig:
    return sensitivity_profile(level).ml


def sensitivity_thresholds(level: str | None) -> dict[str, float]:
    return sensitivity_profile(level).thresholds()


def _risk_config(**overrides: float) -> RiskConfig:
    return replace(RiskConfig(), **overrides)


SENSITIVITY_PROFILES: dict[SensitivityLevel, SensitivityProfile] = {
    "low": SensitivityProfile(
        level="low",
        predictor_config=PredictorConfig(
            baseline_frames=18,
            smoothing_window=7,
            prefall_consecutive_frames=4,
            fall_consecutive_frames=4,
            risk=_risk_config(
                prefall_threshold=0.55,
                fall_threshold=0.80,
                min_visibility=0.45,
                torso_warn_deg=32.0,
                torso_fall_deg=85.0,
                angular_velocity_warn=40.0,
                angular_velocity_fall=150.0,
                vertical_velocity_warn=0.30,
                vertical_velocity_fall=1.00,
                center_drop_warn=0.09,
                center_drop_fall=0.28,
                aspect_ratio_warn=0.70,
                aspect_ratio_fall=1.30,
            ),
        ),
        ml=MLSensitivityConfig(
            prefall_alert_threshold=0.40,
            prefall_alert_frames=3,
            fall_validator_settings={
                "prefall_memory": 8,
                "fall_hold_frames": 30,
                "fall_after_prefall_confirm_frames": 3,
                "vertical_velocity_threshold": 0.75,
                "vertical_accel_threshold": 0.30,
                "angular_velocity_threshold": 240.0,
                "angular_accel_threshold": 520.0,
                "center_drop_delta_threshold": 0.14,
            },
        ),
    ),
    "medium": SensitivityProfile(
        level="medium",
        predictor_config=PredictorConfig(
            risk=RiskConfig(),
        ),
        ml=MLSensitivityConfig(
            prefall_alert_threshold=0.25,
            prefall_alert_frames=1,
            fall_validator_settings={},
        ),
    ),
    "high": SensitivityProfile(
        level="high",
        predictor_config=PredictorConfig(
            baseline_frames=12,
            smoothing_window=3,
            prefall_consecutive_frames=2,
            fall_consecutive_frames=2,
            risk=_risk_config(
                prefall_threshold=0.35,
                fall_threshold=0.60,
                min_visibility=0.30,
                torso_warn_deg=20.0,
                torso_fall_deg=65.0,
                angular_velocity_warn=18.0,
                angular_velocity_fall=95.0,
                vertical_velocity_warn=0.16,
                vertical_velocity_fall=0.65,
                center_drop_warn=0.04,
                center_drop_fall=0.16,
                aspect_ratio_warn=0.45,
                aspect_ratio_fall=0.95,
            ),
        ),
        ml=MLSensitivityConfig(
            prefall_alert_threshold=0.15,
            prefall_alert_frames=1,
            fall_validator_settings={
                "prefall_memory": 16,
                "fall_hold_frames": 45,
                "fall_after_prefall_confirm_frames": 1,
                "vertical_velocity_threshold": 0.45,
                "vertical_accel_threshold": 0.12,
                "angular_velocity_threshold": 150.0,
                "angular_accel_threshold": 300.0,
                "center_drop_delta_threshold": 0.07,
            },
        ),
    ),
}
