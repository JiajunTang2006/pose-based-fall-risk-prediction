"""Cooperative tree + skeleton-fusion fall prediction.

The tree model remains the authoritative classifier because grouped cross-
validation showed that it is the more stable model.  The fusion model is used
as an early-warning sensor. A sustained fusion-only Fall may raise the alert
channel, but it does not overwrite the tree model's confirmed state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .landmarks import Landmark
from .lying_adl_filter import StaticLyingADLFilter
from .ml_predictor import MachineLearningFallPredictor, normalize_state
from .predictor import Prediction, PredictorConfig


DEFAULT_TREE_MODEL_PATH = "models/yolo_tail60_prefall_accel_robust_classifier.joblib"
DEFAULT_FUSION_MODEL_PATH = "models/skeleton_feature_fusion_tuned.pt"
DEFAULT_FUSION_FALL_CONFIRMATION_STEPS = 3


@dataclass(frozen=True)
class DualModelDecision:
    """One cooperative decision made from the two model states."""

    state: str
    alert_state: str
    advisory_state: str | None
    tier: str
    status: str


class DualModelDecisionEngine:
    """Apply the product-facing dual-model decision table.

    Rules are intentionally asymmetric:

    * the precise tree model can confirm Pre-fall/Fall directly;
    * a fusion-only Pre-fall becomes an early alert, not a confirmed state;
    * a fusion-only Fall needs a few consecutive outputs before a Fall alert;
    * agreement immediately raises the confidence tier.
    """

    def __init__(self, fusion_fall_confirmation_steps: int = DEFAULT_FUSION_FALL_CONFIRMATION_STEPS) -> None:
        if fusion_fall_confirmation_steps < 1:
            raise ValueError("fusion_fall_confirmation_steps must be at least 1")
        self.fusion_fall_confirmation_steps = int(fusion_fall_confirmation_steps)
        self._fusion_only_fall_count = 0

    def reset(self) -> None:
        self._fusion_only_fall_count = 0

    def decide(
        self,
        tree_state: str,
        fusion_state: str,
        *,
        advance_fusion_counter: bool = True,
    ) -> DualModelDecision:
        tree = normalize_state(tree_state)
        fusion = normalize_state(fusion_state)

        if tree == "Fall":
            self._fusion_only_fall_count = 0
            agreement = fusion == "Fall"
            tier = "critical-agreement" if agreement else "critical-tree-confirmed"
            return self._decision("Fall", "Fall", None, tier, tree, fusion)

        if fusion == "Fall":
            if advance_fusion_counter:
                self._fusion_only_fall_count += 1
            confirmed_state = tree if tree in {"Normal", "Pre-fall", "Unknown"} else "Normal"
            if self._fusion_only_fall_count >= self.fusion_fall_confirmation_steps:
                return self._decision(
                    confirmed_state,
                    "Fall",
                    "Fall",
                    "critical-fusion-alert",
                    tree,
                    fusion,
                )
            tier = (
                f"high-risk-confirming-"
                f"{self._fusion_only_fall_count}/{self.fusion_fall_confirmation_steps}"
            )
            return self._decision(
                confirmed_state,
                confirmed_state,
                "Pre-fall",
                tier,
                tree,
                fusion,
            )

        if advance_fusion_counter:
            self._fusion_only_fall_count = 0

        if tree == "Pre-fall":
            tier = "high-warning-agreement" if fusion == "Pre-fall" else "warning-tree"
            return self._decision("Pre-fall", "Pre-fall", None, tier, tree, fusion)

        if fusion == "Pre-fall":
            # The fusion model is deliberately allowed to warn early without
            # changing the tree model's confirmed state.
            state = "Unknown" if tree == "Unknown" else "Normal"
            return self._decision(state, state, "Pre-fall", "watch-fusion", tree, fusion)

        if tree == "Unknown" and fusion == "Unknown":
            return self._decision("Unknown", "Unknown", None, "pose-unavailable", tree, fusion)
        if tree == "Unknown":
            return self._decision("Unknown", "Unknown", None, "tree-unavailable", tree, fusion)
        if fusion == "Unknown":
            return self._decision(tree, tree, None, "fusion-unavailable", tree, fusion)
        return self._decision("Normal", "Normal", None, "normal", tree, fusion)

    @staticmethod
    def _decision(
        state: str,
        alert_state: str,
        advisory_state: str | None,
        tier: str,
        tree_state: str,
        fusion_state: str,
    ) -> DualModelDecision:
        return DualModelDecision(
            state=state,
            alert_state=alert_state,
            advisory_state=advisory_state,
            tier=tier,
            status=f"Dual {tier}: tree={tree_state}, fusion={fusion_state}",
        )


class DualModelFallPredictor:
    """Run the tree and fusion predictors on each frame and combine them."""

    def __init__(
        self,
        tree_model_path: str | Path = DEFAULT_TREE_MODEL_PATH,
        fusion_model_path: str | Path = DEFAULT_FUSION_MODEL_PATH,
        *,
        predictor_config: PredictorConfig | None = None,
        prefall_alert_threshold: float | None = None,
        prefall_alert_consecutive_frames: int | None = None,
        fusion_use_hmm: bool = True,
        use_accel: bool | None = None,
        fusion_fall_confirmation_steps: int = DEFAULT_FUSION_FALL_CONFIRMATION_STEPS,
        decision_stride: int | None = None,
        use_static_lying_adl_filter: bool = True,
    ) -> None:
        config = predictor_config or PredictorConfig()
        common = {
            "baseline_frames": config.baseline_frames,
            "smoothing_window": config.smoothing_window,
            "min_visibility": config.risk.min_visibility,
            "prefall_alert_threshold": prefall_alert_threshold,
            "prefall_alert_consecutive_frames": prefall_alert_consecutive_frames,
            "use_accel": use_accel,
            # The old strict sequence gate missed Fall events in validation.
            # Cooperative decisions therefore use only the light HMM below.
            "use_temporal_fall_validation": False,
        }
        self.tree_predictor = MachineLearningFallPredictor(
            tree_model_path,
            use_hmm=False,
            **common,
        )
        self.fusion_predictor = MachineLearningFallPredictor(
            fusion_model_path,
            use_hmm=fusion_use_hmm,
            **common,
        )
        if not self.fusion_predictor._requires_skeleton:
            raise ValueError("fusion_model_path must point to a skeleton-fusion model")
        if self.tree_predictor.window_size != self.fusion_predictor.window_size:
            raise ValueError(
                "Tree and fusion models must use the same temporal window size: "
                f"{self.tree_predictor.window_size} != {self.fusion_predictor.window_size}"
            )
        self.decision_engine = DualModelDecisionEngine(
            fusion_fall_confirmation_steps=fusion_fall_confirmation_steps
        )
        self.decision_stride = int(
            decision_stride or self.fusion_predictor.temporal_gate_stride
        )
        if self.decision_stride < 1:
            raise ValueError("decision_stride must be at least 1")
        self.use_static_lying_adl_filter = bool(use_static_lying_adl_filter)
        self.static_lying_adl_filter = StaticLyingADLFilter()
        self._last_fusion_decision_frame: int | None = None

    @property
    def baseline_center_y(self) -> float | None:
        return self.tree_predictor.baseline_center_y

    def predict(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> Prediction:
        tree = self.tree_predictor.predict(landmarks, frame_index, timestamp)
        fusion = self.fusion_predictor.predict(landmarks, frame_index, timestamp)
        advance_fusion_counter = (
            self._last_fusion_decision_frame is None
            or frame_index - self._last_fusion_decision_frame >= self.decision_stride
        )
        decision = self.decision_engine.decide(
            tree.state,
            fusion.state,
            advance_fusion_counter=advance_fusion_counter,
        )
        filtered = None
        if self.use_static_lying_adl_filter:
            filtered = self.static_lying_adl_filter.process(
                decision.state,
                decision.alert_state,
                decision.advisory_state,
                list(self.tree_predictor._raw_window),
                advance=advance_fusion_counter,
            )
        if advance_fusion_counter:
            self._last_fusion_decision_frame = frame_index
        risk_score = max(tree.risk_score, fusion.risk_score)
        smoothed_risk = max(tree.smoothed_risk_score, fusion.smoothed_risk_score)
        instant_state = _more_severe_state(tree.instant_state, fusion.instant_state)
        state = filtered.state if filtered is not None else decision.state
        alert_state = filtered.alert_state if filtered is not None else decision.alert_state
        advisory_state = (
            filtered.advisory_state if filtered is not None else decision.advisory_state
        )
        decision_tier = (
            filtered.tier if filtered is not None and filtered.filtered else decision.tier
        )
        system_status = decision.status
        if filtered is not None:
            system_status = f"{decision.status}; {filtered.status}"
        return replace(
            tree,
            state=state,
            alert_state=alert_state,
            instant_state=instant_state,
            risk_score=risk_score,
            smoothed_risk_score=smoothed_risk,
            breakdown=replace(tree.breakdown, risk_score=risk_score),
            system_status=system_status,
            advisory_state=advisory_state,
            decision_tier=decision_tier,
        )

    def reset(self) -> None:
        self.tree_predictor.reset()
        self.fusion_predictor.reset()
        self.decision_engine.reset()
        self.static_lying_adl_filter.reset()
        self._last_fusion_decision_frame = None

    def acknowledge_fall(self) -> None:
        self.tree_predictor.acknowledge_fall()
        self.fusion_predictor.acknowledge_fall()
        self.decision_engine.reset()
        self.static_lying_adl_filter.acknowledge_fall()
        self._last_fusion_decision_frame = None


def combine_state_sequences(
    tree_states: Sequence[str],
    fusion_states: Sequence[str],
    *,
    fusion_fall_confirmation_steps: int = DEFAULT_FUSION_FALL_CONFIRMATION_STEPS,
) -> tuple[list[str], list[str], list[str]]:
    """Combine one already ordered video sequence for offline evaluation."""
    if len(tree_states) != len(fusion_states):
        raise ValueError("tree_states and fusion_states must have equal length")
    engine = DualModelDecisionEngine(fusion_fall_confirmation_steps)
    states: list[str] = []
    alerts: list[str] = []
    tiers: list[str] = []
    for tree_state, fusion_state in zip(tree_states, fusion_states):
        decision = engine.decide(str(tree_state), str(fusion_state))
        states.append(decision.state)
        alerts.append(decision.advisory_state or decision.alert_state)
        tiers.append(decision.tier)
    return states, alerts, tiers


def _more_severe_state(first: str, second: str) -> str:
    rank = {"Unknown": -1, "Normal": 0, "Pre-fall": 1, "Fall": 2}
    first = normalize_state(first)
    second = normalize_state(second)
    return first if rank.get(first, 0) >= rank.get(second, 0) else second
