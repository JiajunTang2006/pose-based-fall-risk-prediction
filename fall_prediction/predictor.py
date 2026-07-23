

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .features import FeatureExtractor, PoseFeatures
from .landmarks import Landmark
from .risk import RiskBreakdown, RiskConfig, RiskScorer


@dataclass(frozen=True)
class PredictorConfig:

    baseline_frames: int = 15
    smoothing_window: int = 5
    prefall_consecutive_frames: int = 3
    fall_consecutive_frames: int = 3
    risk: RiskConfig = RiskConfig()


@dataclass(frozen=True)
class Prediction:

    frame_index: int
    timestamp: float
    state: str
    instant_state: str
    risk_score: float
    smoothed_risk_score: float
    features: PoseFeatures
    breakdown: RiskBreakdown
    baseline_center_y: float | None
    alert_state: str | None = None
    system_status: str | None = None
    advisory_state: str | None = None
    decision_tier: str | None = None


class FallPredictor:


    def __init__(self, config: PredictorConfig | None = None) -> None:
        self.config = config or PredictorConfig()

        self.extractor = FeatureExtractor(min_visibility=self.config.risk.min_visibility)

        self.scorer = RiskScorer(self.config.risk)

        self._baseline_samples: list[float] = []

        self._baseline_center_y: float | None = None

        self._risk_history: deque[float] = deque(maxlen=self.config.smoothing_window)

        self._prefall_count = 0
        self._fall_count = 0

    @property
    def baseline_center_y(self) -> float | None:

        return self._baseline_center_y

    def predict(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> Prediction:


        features = self.extractor.extract(landmarks, frame_index, timestamp)


        if features.center_valid and self._baseline_center_y is None:
            self._baseline_samples.append(features.body_center_y)
            if len(self._baseline_samples) >= self.config.baseline_frames:

                self._baseline_center_y = mean(self._baseline_samples)


        fallback_baseline = self._baseline_center_y
        if fallback_baseline is None and self._baseline_samples:
            fallback_baseline = mean(self._baseline_samples)


        breakdown = self.scorer.score(features, fallback_baseline)

        instant_state = self.scorer.state_from_score(breakdown.risk_score)


        self._risk_history.append(breakdown.risk_score)
        smoothed_risk = mean(self._risk_history)


        state = self._temporal_state(smoothed_risk, features)

        return Prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            instant_state=instant_state,
            risk_score=breakdown.risk_score,
            smoothed_risk_score=smoothed_risk,
            features=features,
            breakdown=breakdown,
            baseline_center_y=fallback_baseline,
        )

    def reset(self) -> None:

        self.extractor.reset()
        self._baseline_samples.clear()
        self._baseline_center_y = None
        self._risk_history.clear()
        self._prefall_count = 0
        self._fall_count = 0

    def _temporal_state(self, smoothed_risk: float, features: PoseFeatures) -> str:

        cfg = self.config.risk


        if not features.has_pose or features.visibility_mean < cfg.min_visibility:
            self._prefall_count = 0
            self._fall_count = 0
            return "Unknown"


        if smoothed_risk >= cfg.fall_threshold:
            self._fall_count += 1
        else:
            self._fall_count = 0


        if smoothed_risk >= cfg.prefall_threshold:
            self._prefall_count += 1
        else:
            self._prefall_count = 0


        if self._fall_count >= self.config.fall_consecutive_frames:
            return "Fall"
        if self._prefall_count >= self.config.prefall_consecutive_frames:
            return "Pre-fall"
        return "Normal"
