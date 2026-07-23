

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .features import FeatureExtractor, PoseFeatures
from .landmarks import Landmark
from .ml_features import (
    ML_FEATURE_COLUMNS,
    flatten_window,
    pose_features_to_ml_row,
    compute_window_accel_features,
)
from .predictor import Prediction, PredictorConfig
from .risk import RiskBreakdown
from .robustness import StandingFeatureCalibrator
from .window_dataset import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()
DEFAULT_PREFALL_ALERT_THRESHOLD = 0.25
DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES = 1
DEFAULT_TEMPORAL_SENSITIVITY = "medium"
DEFAULT_PARTIAL_POSE_GRACE_FRAMES = 5
DEFAULT_TEMPORAL_GATE_STRIDE = DEFAULT_STRIDE
DEFAULT_CONTROLLED_UPRIGHT_VERTICAL_VELOCITY = 0.40
DEFAULT_CONTROLLED_UPRIGHT_ANGULAR_VELOCITY = 100.0
DEFAULT_CONTROLLED_UPRIGHT_CENTER_DROP_DELTA = 0.10
NORMAL_STATES = {"Normal"}


DEFAULT_HMM_BUFFER_SIZE = 25


DEFAULT_FALL_VALIDATION_HISTORY = 30
DEFAULT_FALL_PREFALL_MEMORY = 12
DEFAULT_FALL_HOLD_FRAMES = 45
DEFAULT_FALL_AFTER_PREFALL_CONFIRM_FRAMES = 2
DEFAULT_FALL_VERTICAL_VELOCITY = 0.60
DEFAULT_FALL_VERTICAL_ACCEL = 0.20
DEFAULT_FALL_ANGULAR_VELOCITY = 200.0
DEFAULT_FALL_ANGULAR_ACCEL = 400.0
DEFAULT_FALL_CENTER_DROP_DELTA = 0.10


@dataclass(frozen=True)
class TemporalSensitivityProfile:
    """
    Runtime sensitivity profile for sequence-level gating.

    The classifier still sees every window independently.  This profile controls
    how much temporal evidence is required before a Pre-fall/Fall becomes visible
    to the application.
    """

    name: str
    prefall_probability_threshold: float
    prefall_window: int
    prefall_confirm_count: int
    prefall_consecutive_frames: int
    stable_normal_frames: int
    normal_memory: int
    prefall_memory: int
    fall_probability_threshold: float
    fall_window: int
    fall_confirm_count: int
    fall_hold_frames: int
    fall_recovery_normal_frames: int


TEMPORAL_SENSITIVITY_PROFILES = {
    # High is intentionally still a little calmer than the previous runtime:
    # no single Pre-fall window can alert by itself.
    "high": TemporalSensitivityProfile(
        name="high",
        prefall_probability_threshold=0.06,
        prefall_window=3,
        prefall_confirm_count=2,
        prefall_consecutive_frames=2,
        stable_normal_frames=1,
        normal_memory=30,
        prefall_memory=20,
        # Keep the already well-performing high profile on its original
        # classifier/HMM Fall transition; the probability fallback targets the
        # observed medium-profile argmax tie problem.
        fall_probability_threshold=1.00,
        fall_window=2,
        fall_confirm_count=1,
        fall_hold_frames=15,
        fall_recovery_normal_frames=5,
    ),
    "medium": TemporalSensitivityProfile(
        name="medium",
        prefall_probability_threshold=0.40,
        prefall_window=3,
        prefall_confirm_count=2,
        prefall_consecutive_frames=2,
        stable_normal_frames=1,
        normal_memory=20,
        prefall_memory=15,
        fall_probability_threshold=0.45,
        fall_window=3,
        fall_confirm_count=2,
        fall_hold_frames=20,
        fall_recovery_normal_frames=10,
    ),
    "low": TemporalSensitivityProfile(
        name="low",
        prefall_probability_threshold=0.40,
        prefall_window=5,
        prefall_confirm_count=3,
        prefall_consecutive_frames=2,
        stable_normal_frames=1,
        normal_memory=15,
        prefall_memory=10,
        # Keep low conservative until real-scene low-sensitivity samples are
        # available; a 0.50 fallback reduced offline Fall event detection.
        fall_probability_threshold=1.00,
        fall_window=4,
        fall_confirm_count=3,
        fall_hold_frames=30,
        fall_recovery_normal_frames=15,
    ),
}


def resolve_temporal_sensitivity_profile(
    profile: TemporalSensitivityProfile | str,
) -> TemporalSensitivityProfile:
    """Return a known temporal sensitivity profile by name."""
    if isinstance(profile, TemporalSensitivityProfile):
        return profile
    key = str(profile).strip().lower()
    try:
        return TEMPORAL_SENSITIVITY_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(TEMPORAL_SENSITIVITY_PROFILES))
        raise ValueError(f"Unknown temporal sensitivity {profile!r}; choose one of: {choices}") from exc


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════

def build_hmm_transition_matrix(
    stay_normal: float = 0.92,
    stay_prefall: float = 0.80,
    stay_fall: float = 0.88,
    normal_to_prefall: float = 0.07,
    prefall_to_fall: float = 0.12,
    prefall_to_normal: float = 0.08,
    fall_to_prefall: float = 0.10,
) -> np.ndarray:

    T = np.zeros((3, 3))
    T[0] = [stay_normal, normal_to_prefall, max(0.0, 1.0 - stay_normal - normal_to_prefall)]
    T[1] = [prefall_to_normal, stay_prefall, prefall_to_fall]
    T[2] = [max(0.0, 1.0 - stay_fall - fall_to_prefall), fall_to_prefall, stay_fall]

    T = T / T.sum(axis=1, keepdims=True)
    return T


class HMMStateSmoother:


    def __init__(
        self,
        buffer_size: int = DEFAULT_HMM_BUFFER_SIZE,
        transition_matrix: np.ndarray | None = None,
        initial_probs: list[float] | None = None,
    ) -> None:
        self._buffer_size = max(3, int(buffer_size))
        self._transition = (
            transition_matrix
            if transition_matrix is not None
            else build_hmm_transition_matrix()
        )
        self._initial_probs = (
            initial_probs if initial_probs is not None else [0.98, 0.02, 0.0]
        )

        self._prob_buffer: deque[list[float]] = deque(maxlen=self._buffer_size)
        self._state_idx = {0: "Normal", 1: "Pre-fall", 2: "Fall"}

    def reset(self) -> None:

        self._prob_buffer.clear()

    def smooth(self, probabilities: list[float]) -> str:

        self._prob_buffer.append(list(probabilities))

        if len(self._prob_buffer) < self._buffer_size:

            idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
            return self._state_idx[idx]


        obs = list(self._prob_buffer)
        state_seq = self._viterbi(obs)
        return self._state_idx[state_seq[-1]]

    def _viterbi(self, observations: list[list[float]]) -> list[int]:

        n_states = 3
        T_len = len(observations)
        if T_len == 0:
            return []

        log_T = np.log(np.maximum(self._transition, 1e-12))
        log_init = np.log(np.maximum(self._initial_probs, 1e-12))

        dp = np.full((T_len, n_states), -np.inf)
        bp = np.zeros((T_len, n_states), dtype=int)


        for j in range(n_states):
            obs_prob = max(observations[0][j], 1e-12)
            dp[0][j] = log_init[j] + np.log(obs_prob)


        for t in range(1, T_len):
            for j in range(n_states):
                scores = dp[t - 1] + log_T[:, j]
                best_i = int(np.argmax(scores))
                dp[t][j] = scores[best_i] + np.log(max(observations[t][j], 1e-12))
                bp[t][j] = best_i


        states = [0] * T_len
        states[T_len - 1] = int(np.argmax(dp[T_len - 1]))
        for t in range(T_len - 2, -1, -1):
            states[t] = bp[t + 1][states[t + 1]]
        return states


@dataclass(frozen=True)
class FallMotionEvidence:


    max_vertical_velocity: float
    max_vertical_accel: float
    max_angular_velocity: float
    max_angular_accel: float
    center_drop_delta: float
    has_fast_descent: bool
    has_impact_or_rotation: bool
    has_center_drop_change: bool


@dataclass(frozen=True)
class PostureEvidence:
    """Stable posture cues used to avoid treating lying as a fall event."""

    mean_body_height: float
    mean_aspect_ratio: float
    mean_center_drop: float
    mean_torso_angle: float
    mean_abs_vertical_velocity: float
    is_low_horizontal: bool
    is_low_posture: bool
    is_static_low_posture: bool
    is_upright_normal: bool


class TemporalSequenceGate:
    """
    Product-facing sequence gate for Normal -> Pre-fall -> Fall transitions.

    This layer treats Fall as an event, not just a posture.  A static lying or
    low-posture sequence can look similar to a fallen body in one window, but it
    should not become Fall unless the recent history contains a stable Normal
    start and a short abnormal transition.
    """

    def __init__(
        self,
        profile: TemporalSensitivityProfile | str = DEFAULT_TEMPORAL_SENSITIVITY,
        *,
        allow_warm_start_prefall: bool = True,
        automatic_fall_recovery: bool = False,
    ) -> None:
        self.profile = resolve_temporal_sensitivity_profile(profile)
        self.allow_warm_start_prefall = bool(allow_warm_start_prefall)
        self.automatic_fall_recovery = bool(automatic_fall_recovery)
        self._prefall_candidates: deque[bool] = deque(maxlen=self.profile.prefall_window)
        self._fall_candidates: deque[bool] = deque(maxlen=self.profile.fall_window)
        self._stable_normal_count = 0
        self._normal_age: int | None = None
        self._prefall_consecutive_count = 0
        self._prefall_age: int | None = None
        self._fall_hold_remaining = 0
        self._fall_recovery_normal_count = 0
        self._warm_start_low_posture_count = 0
        self._internal_state = "Normal"

    def reset(self) -> None:
        self._prefall_candidates.clear()
        self._fall_candidates.clear()
        self._stable_normal_count = 0
        self._normal_age = None
        self._prefall_consecutive_count = 0
        self._prefall_age = None
        self._fall_hold_remaining = 0
        self._fall_recovery_normal_count = 0
        self._warm_start_low_posture_count = 0
        self._internal_state = "Normal"

    def acknowledge_fall(self) -> None:
        """Clear a latched Fall after explicit operator acknowledgement."""
        self._prefall_candidates.clear()
        self._fall_candidates.clear()
        self._prefall_consecutive_count = 0
        self._prefall_age = None
        self._fall_hold_remaining = 0
        self._fall_recovery_normal_count = 0
        self._warm_start_low_posture_count = 0
        self._stable_normal_count = 0
        self._normal_age = None
        self._internal_state = "Normal"

    @property
    def fall_latched(self) -> bool:
        """Whether a confirmed Fall is waiting for acknowledgement/recovery."""
        return self._internal_state == "Fall"

    def validate(
        self,
        state: str,
        alert_state: str,
        probabilities: Mapping[str, float],
        window_rows: Sequence[dict[str, float]],
    ) -> tuple[str, str]:


        posture = _posture_evidence(window_rows)
        motion = _motion_evidence(window_rows)

        if self._internal_state == "Fall":
            return self._validate_latched_fall(state, alert_state, posture)

        prefall_probability = max(0.0, min(1.0, float(probabilities.get("Pre-fall", 0.0))))
        fall_probability = max(0.0, min(1.0, float(probabilities.get("Fall", 0.0))))

        # Clear, slowly moving upright posture is trusted as Normal evidence.
        # It bypasses model/HMM false positives while still refreshing temporal
        # memory for a later real transition.
        controlled_upright = (
            posture.is_upright_normal
            and motion.max_vertical_velocity < DEFAULT_CONTROLLED_UPRIGHT_VERTICAL_VELOCITY
            and motion.max_angular_velocity < DEFAULT_CONTROLLED_UPRIGHT_ANGULAR_VELOCITY
            and motion.center_drop_delta < DEFAULT_CONTROLLED_UPRIGHT_CENTER_DROP_DELTA
        )

        # The classifier/HMM can remain on Pre-fall when Fall is a close second
        # (or tied because of smoothing).  Requiring Fall to win the argmax made
        # a sustained ~50% Fall probability invisible to the sequence vote,
        # especially in the medium profile.  Treat a profile-specific Fall
        # probability as a candidate too; the existing Normal -> Pre-fall
        # history and multi-window vote still have to confirm the event.
        classified_fall = state == "Fall" or alert_state == "Fall"
        probability_fall_candidate = (
            not controlled_upright
            and self.profile.fall_probability_threshold < 1.0
            and self._internal_state == "Pre-fall"
            and fall_probability >= self.profile.fall_probability_threshold
        )
        raw_fall = not controlled_upright and (classified_fall or probability_fall_candidate)


        raw_prefall = (
            not controlled_upright
            and not raw_fall
            and (
                state == "Pre-fall"
                or alert_state == "Pre-fall"
                or prefall_probability >= self.profile.prefall_probability_threshold
            )
        )
        normal_signal = controlled_upright or (
            state == "Normal" and not raw_prefall and not raw_fall and not posture.is_static_low_posture
        )

        self._update_normal_memory(normal_signal)

        if posture.is_static_low_posture and not self._has_recent_normal():
            self._warm_start_low_posture_count += 1
        else:
            self._warm_start_low_posture_count = 0

        warm_start_signal = (
            self.allow_warm_start_prefall
            and not self._has_recent_normal()
            and not controlled_upright
            and (raw_prefall or raw_fall)
            and max(prefall_probability, fall_probability) >= self.profile.prefall_probability_threshold
            and (
                not posture.is_static_low_posture
                or self._warm_start_low_posture_count <= self.profile.prefall_memory
            )
        )

        valid_prefall = (
            (raw_prefall or warm_start_signal)
            and (self._has_recent_normal() or warm_start_signal)
            and (not posture.is_static_low_posture or warm_start_signal)
            and prefall_probability >= self.profile.prefall_probability_threshold
        )
        # A raw Fall at warm start has little Pre-fall probability by design.
        # Downgrade it to a temporary Pre-fall warning instead of silently
        # accepting it as Fall or discarding it as Normal.
        if warm_start_signal and raw_fall:
            valid_prefall = True
        self._update_prefall_memory(valid_prefall)

        has_prefall_context = self._has_recent_prefall() or self._prefall_is_confirmed()


        valid_fall_candidate = raw_fall and has_prefall_context and self._has_recent_normal()
        self._fall_candidates.append(valid_fall_candidate)

        if raw_fall:
            if valid_fall_candidate and self._confirms_fall():
                self._fall_hold_remaining = self.profile.fall_hold_frames
                self._fall_recovery_normal_count = 0
                self._prefall_age = 0
                self._internal_state = "Fall"
                return "Fall", "Fall"
            # A probability-only candidate is deliberately weaker than a
            # classifier/HMM Fall.  Keep the already-confirmed warning visible
            # while its multi-window Fall vote accumulates; dropping back to
            # Normal here would prevent medium/low candidates from persisting.
            if probability_fall_candidate and not classified_fall:
                self._internal_state = "Pre-fall"
                return "Pre-fall", "Pre-fall"
            if warm_start_signal or self._prefall_is_confirmed():
                self._internal_state = "Pre-fall"
                return "Pre-fall", "Pre-fall"
            self._internal_state = "Normal"
            return "Normal", "Normal"

        if self._prefall_is_confirmed():
            self._prefall_age = 0
            self._internal_state = "Pre-fall"
            return "Pre-fall", "Pre-fall"

        self._internal_state = "Low-posture" if posture.is_static_low_posture else "Normal"
        return "Normal", "Normal"

    def _update_normal_memory(self, normal_signal: bool) -> None:
        if normal_signal:
            self._stable_normal_count += 1
            if self._stable_normal_count >= self.profile.stable_normal_frames:
                self._normal_age = 0
            return

        self._stable_normal_count = 0
        if self._normal_age is not None:
            self._normal_age += 1
            if self._normal_age > self.profile.normal_memory:
                self._normal_age = None

    def _update_prefall_memory(self, valid_prefall: bool) -> None:
        self._prefall_candidates.append(valid_prefall)
        if valid_prefall:
            self._prefall_consecutive_count += 1
        else:
            self._prefall_consecutive_count = 0
        self._age_event_memory()

    def _age_event_memory(self) -> None:
        if self._prefall_age is not None:
            self._prefall_age += 1
            if self._prefall_age > self.profile.prefall_memory:
                self._prefall_age = None

    def _has_recent_normal(self) -> bool:
        return self._normal_age is not None and self._normal_age <= self.profile.normal_memory

    def _has_recent_prefall(self) -> bool:
        return self._prefall_age is not None and self._prefall_age <= self.profile.prefall_memory

    def _prefall_is_confirmed(self) -> bool:
        return (
            sum(self._prefall_candidates) >= self.profile.prefall_confirm_count
            and self._prefall_consecutive_count >= self.profile.prefall_consecutive_frames
        )

    def _fall_candidate_count_is_enough(self) -> bool:
        return sum(self._fall_candidates) >= self.profile.fall_confirm_count

    def _confirms_fall(self) -> bool:
        return self._has_recent_prefall() and self._fall_candidate_count_is_enough()

    def _validate_latched_fall(
        self,
        state: str,
        alert_state: str,
        posture: PostureEvidence,
    ) -> tuple[str, str]:
        if not self.automatic_fall_recovery:
            return "Fall", "Fall"

        recovery_normal = (
            state == "Normal"
            and alert_state == "Normal"
            and posture.is_upright_normal
        )
        if recovery_normal:
            self._fall_recovery_normal_count += 1
        else:
            self._fall_recovery_normal_count = 0

        if self._fall_hold_remaining > 0:
            self._fall_hold_remaining -= 1
            return "Fall", "Fall"

        if self._fall_recovery_normal_count < self.profile.fall_recovery_normal_frames:
            return "Fall", "Fall"


        self._prefall_candidates.clear()
        self._fall_candidates.clear()
        self._prefall_consecutive_count = 0
        self._prefall_age = None
        self._stable_normal_count = self.profile.stable_normal_frames
        self._normal_age = 0
        self._fall_recovery_normal_count = 0
        self._internal_state = "Normal"
        return "Normal", "Normal"


class TemporalFallValidator:


    def __init__(
        self,
        history_size: int = DEFAULT_FALL_VALIDATION_HISTORY,
        prefall_memory: int = DEFAULT_FALL_PREFALL_MEMORY,
        fall_hold_frames: int = DEFAULT_FALL_HOLD_FRAMES,
        fall_after_prefall_confirm_frames: int = DEFAULT_FALL_AFTER_PREFALL_CONFIRM_FRAMES,
        vertical_velocity_threshold: float = DEFAULT_FALL_VERTICAL_VELOCITY,
        vertical_accel_threshold: float = DEFAULT_FALL_VERTICAL_ACCEL,
        angular_velocity_threshold: float = DEFAULT_FALL_ANGULAR_VELOCITY,
        angular_accel_threshold: float = DEFAULT_FALL_ANGULAR_ACCEL,
        center_drop_delta_threshold: float = DEFAULT_FALL_CENTER_DROP_DELTA,
    ) -> None:
        self._state_history: deque[str] = deque(maxlen=max(1, int(history_size)))
        self._prefall_memory = max(1, int(prefall_memory))
        self._fall_hold_frames = max(0, int(fall_hold_frames))
        self._fall_hold_remaining = 0
        self._fall_after_prefall_confirm_frames = max(1, int(fall_after_prefall_confirm_frames))
        self._fall_after_prefall_count = 0
        self.vertical_velocity_threshold = max(0.0, float(vertical_velocity_threshold))
        self.vertical_accel_threshold = max(0.0, float(vertical_accel_threshold))
        self.angular_velocity_threshold = max(0.0, float(angular_velocity_threshold))
        self.angular_accel_threshold = max(0.0, float(angular_accel_threshold))
        self.center_drop_delta_threshold = max(0.0, float(center_drop_delta_threshold))

    def reset(self) -> None:
        self._state_history.clear()
        self._fall_hold_remaining = 0
        self._fall_after_prefall_count = 0

    def validate(
        self,
        state: str,
        alert_state: str,
        window_rows: Sequence[dict[str, float]],
    ) -> tuple[str, str]:
        if self._fall_hold_remaining > 0:
            self._fall_hold_remaining -= 1
            self._state_history.append("Fall")
            return "Fall", "Fall"

        if state != "Fall":
            self._fall_after_prefall_count = 0
            filtered_alert = "Pre-fall" if alert_state == "Fall" else alert_state
            self._state_history.append(self._history_state(state, filtered_alert))
            return state, filtered_alert

        evidence = self._motion_evidence(window_rows)
        has_prefall_evidence = self._has_recent_prefall_evidence()
        if has_prefall_evidence:
            self._fall_after_prefall_count += 1
        else:
            self._fall_after_prefall_count = 0

        confirmed = self._confirms_fall(evidence, has_prefall_evidence)
        if confirmed:
            self._fall_hold_remaining = self._fall_hold_frames
            self._state_history.append("Fall")
            return "Fall", "Fall"

        filtered_state, filtered_alert = self._unconfirmed_fall_state(alert_state, evidence)
        self._state_history.append(self._history_state(filtered_state, filtered_alert))
        return filtered_state, filtered_alert

    def _has_recent_prefall_evidence(self) -> bool:
        recent_states = list(self._state_history)[-self._prefall_memory :]
        return "Pre-fall" in recent_states

    @staticmethod
    def _history_state(state: str, alert_state: str) -> str:
        return "Pre-fall" if alert_state == "Pre-fall" else state

    def _confirms_fall(self, evidence: FallMotionEvidence, has_prefall_evidence: bool) -> bool:
        if not has_prefall_evidence:
            return False
        if evidence.has_fast_descent and (
            evidence.has_center_drop_change or evidence.has_impact_or_rotation
        ):
            return True
        return self._fall_after_prefall_count >= self._fall_after_prefall_confirm_frames

    def _unconfirmed_fall_state(
        self,
        alert_state: str,
        evidence: FallMotionEvidence,
    ) -> tuple[str, str]:
        suspicious_but_unconfirmed = (
            alert_state == "Pre-fall"
            or self._has_recent_prefall_evidence()
            or evidence.has_fast_descent
            or evidence.has_impact_or_rotation
        )
        if suspicious_but_unconfirmed:
            return "Pre-fall", "Pre-fall"
        return "Normal", "Normal"

    def _motion_evidence(self, window_rows: Sequence[dict[str, float]]) -> FallMotionEvidence:
        return _motion_evidence(
            window_rows,
            vertical_velocity_threshold=self.vertical_velocity_threshold,
            vertical_accel_threshold=self.vertical_accel_threshold,
            angular_velocity_threshold=self.angular_velocity_threshold,
            angular_accel_threshold=self.angular_accel_threshold,
            center_drop_delta_threshold=self.center_drop_delta_threshold,
        )


class MachineLearningFallPredictor:


    def __init__(
        self,
        model_path: str | Path,
        baseline_frames: int | None = None,
        smoothing_window: int | None = None,
        min_visibility: float = 0.35,
        prefall_alert_threshold: float | None = None,
        prefall_alert_consecutive_frames: int | None = None,
        use_hmm: bool = False,
        hmm_buffer_size: int = DEFAULT_HMM_BUFFER_SIZE,
        use_accel: bool | None = None,
        use_temporal_fall_validation: bool = True,
        temporal_sensitivity: str | TemporalSensitivityProfile = DEFAULT_TEMPORAL_SENSITIVITY,
        temporal_gate_stride: int | None = None,
        allow_warm_start_prefall: bool = True,
        automatic_fall_recovery: bool = False,
        fall_validator_settings: Mapping[str, float | int] | None = None,
    ) -> None:


        artifact = load_model_artifact(model_path)


        self.model = artifact["model"]
        self._requires_skeleton = bool(artifact.get("requires_skeleton", False))
        self.window_size = int(artifact.get("window_size", DEFAULT_WINDOW_SIZE))
        self.feature_columns = tuple(artifact.get("feature_columns", ML_FEATURE_COLUMNS))
        self.baseline_frames = _resolve_positive_int_setting(
            artifact=artifact,
            name="baseline_frames",
            explicit_value=baseline_frames,
            default_value=DEFAULT_PREDICTOR_CONFIG.baseline_frames,
        )
        self.smoothing_window = _resolve_positive_int_setting(
            artifact=artifact,
            name="smoothing_window",
            explicit_value=smoothing_window,
            default_value=DEFAULT_PREDICTOR_CONFIG.smoothing_window,
        )
        self.prefall_alert_threshold = _resolve_probability_setting(
            artifact=artifact,
            name="prefall_alert_threshold",
            explicit_value=prefall_alert_threshold,
            default_value=DEFAULT_PREFALL_ALERT_THRESHOLD,
        )
        self.prefall_alert_consecutive_frames = _resolve_positive_int_setting(
            artifact=artifact,
            name="prefall_alert_consecutive_frames",
            explicit_value=prefall_alert_consecutive_frames,
            default_value=DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
        )
        self.min_visibility = min_visibility


        self.extractor = FeatureExtractor(min_visibility=min_visibility)


        self._window: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._raw_window: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self._skeleton_window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._missing_pose_count = 0
        self._uncalibrated_non_upright_count = 0


        self._risk_history: deque[float] = deque(maxlen=self.smoothing_window)


        self._baseline_samples: list[float] = []
        self._baseline_center_y: float | None = None


        self._prefall_alert_count = 0


        self._use_hmm = bool(use_hmm)
        self._hmm: HMMStateSmoother | None = None
        if self._use_hmm:
            self._hmm = HMMStateSmoother(buffer_size=hmm_buffer_size)


        if use_accel is not None:
            self._use_accel = bool(use_accel)
        else:
            self._use_accel = bool(artifact.get("use_accel", False))

        self._use_standing_calibration = bool(artifact.get("use_standing_calibration", False))
        self._use_upper_body_features = bool(artifact.get("use_upper_body_features", False))
        self._standing_calibrator: StandingFeatureCalibrator | None = None
        if self._use_standing_calibration:
            self._standing_calibrator = StandingFeatureCalibrator(
                baseline_frames=self.baseline_frames,
                min_visibility=self.min_visibility,
                allow_upper_body_only_calibration=self._use_upper_body_features,
            )

        self._use_temporal_fall_validation = bool(use_temporal_fall_validation)
        # Kept for compatibility with the macOS app's previous predictor API.
        # The strict TemporalSequenceGate below supersedes the older motion-only
        # Fall validator, so those legacy thresholds are intentionally ignored.
        _ = fall_validator_settings
        self.temporal_sensitivity = resolve_temporal_sensitivity_profile(temporal_sensitivity)
        self.temporal_gate_stride = _resolve_positive_int_setting(
            artifact=artifact,
            name="stride",
            explicit_value=temporal_gate_stride,
            default_value=DEFAULT_TEMPORAL_GATE_STRIDE,
        )
        self._temporal_sequence_gate = TemporalSequenceGate(
            self.temporal_sensitivity,
            allow_warm_start_prefall=allow_warm_start_prefall,
            automatic_fall_recovery=automatic_fall_recovery,
        )
        self._last_temporal_gate_frame: int | None = None
        self._last_temporal_state = "Normal"
        self._last_temporal_alert = "Normal"
        self._last_state_probabilities: dict[str, float] = {
            "Normal": 1.0,
            "Pre-fall": 0.0,
            "Fall": 0.0,
        }

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
        if self._requires_skeleton:
            from .skeleton_dataset import landmarks_to_skeleton_frame

            previous_skeleton = self._skeleton_window[-1] if self._skeleton_window else None
            self._skeleton_window.append(
                landmarks_to_skeleton_frame(landmarks, previous_frame=previous_skeleton)
            )


        center_drop = self._update_baseline_and_center_drop(features)

        raw_row = pose_features_to_ml_row(features, center_drop)


        has_partial_measurement = features.has_pose and (
            features.torso_valid
            or features.center_valid
            or features.bbox_valid
            or (self._use_upper_body_features and features.upper_body_valid)
        )
        pose_is_usable = (
            has_partial_measurement
            if self._use_standing_calibration
            else features.has_pose and features.visibility_mean >= self.min_visibility
        )
        if not pose_is_usable:
            self._prefall_alert_count = 0
            self._missing_pose_count += 1
            if (
                self._standing_calibrator is not None
                and self._standing_calibrator.ready
                and self._missing_pose_count <= DEFAULT_PARTIAL_POSE_GRACE_FRAMES
                and self._window
            ):
                missing_row = self._standing_calibrator.transform(raw_row)
                self._window.append(missing_row)
                self._raw_window.append(raw_row)
                if len(self._window) >= self.window_size:
                    return self._predict_current_window(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        features=features,
                        center_drop=center_drop,
                    )


            latched_fall = self._temporal_sequence_gate.fall_latched
            return self._prediction(
                frame_index=frame_index,
                timestamp=timestamp,
                state="Fall" if latched_fall else "Unknown",
                instant_state="Unknown",
                risk_score=0.0,
                features=features,
                center_drop=center_drop,
                alert_state="Fall" if latched_fall else "Unknown",
                system_status="Pose lost; Fall alarm remains latched" if latched_fall else None,
            )
        self._missing_pose_count = 0

        # Robust artifacts first collect a fixed standing reference.  Partial
        # frames are usable after calibration, but cannot establish the baseline.
        model_row = raw_row
        if self._standing_calibrator is not None:
            raw_posture = _posture_evidence([raw_row])
            if not self._standing_calibrator.ready and not raw_posture.is_upright_normal:
                self._uncalibrated_non_upright_count += 1
                warning_frames = self.temporal_sensitivity.prefall_memory * self.temporal_gate_stride
                warm_warning = self._uncalibrated_non_upright_count <= warning_frames
                return self._prediction(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    state="Pre-fall" if warm_warning else "Normal",
                    instant_state="Unknown",
                    risk_score=0.5 if warm_warning else 0.0,
                    features=features,
                    center_drop=center_drop,
                    alert_state="Pre-fall" if warm_warning else "Normal",
                    system_status=(
                        "Calibration pending: non-upright warm-start warning"
                        if warm_warning
                        else "Calibration pending: non-upright posture settled"
                    ),
                )
            self._uncalibrated_non_upright_count = 0
            calibrated_row = self._standing_calibrator.update_and_transform(raw_row)
            if calibrated_row is None:
                return self._prediction(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    state="Normal",
                    instant_state="Normal",
                    risk_score=0.0,
                    features=features,
                    center_drop=center_drop,
                    alert_state="Normal",
                    system_status=(
                        f"Calibrating: stand still "
                        f"({self._standing_calibrator.collected_frames}/{self._standing_calibrator.baseline_frames})"
                    ),
                )
            model_row = calibrated_row

        # 3. Keep model-space and raw image-space windows separately.  The model
        # sees calibrated ratios; the product state gate still sees physical raw
        # posture values and therefore keeps its existing thresholds meaningful.
        self._window.append(model_row)
        self._raw_window.append(raw_row)


        if len(self._window) < self.window_size:
            self._prefall_alert_count = 0
            return self._prediction(
                frame_index=frame_index,
                timestamp=timestamp,
                state="Normal",
                instant_state="Normal",
                risk_score=0.0,
                features=features,
                center_drop=center_drop,
                alert_state="Normal",
            )

        return self._predict_current_window(
            frame_index=frame_index,
            timestamp=timestamp,
            features=features,
            center_drop=center_drop,
        )

    def _predict_current_window(
        self,
        frame_index: int,
        timestamp: float,
        features: PoseFeatures,
        center_drop: float,
    ) -> Prediction:
        """Run model + temporal gate on the current calibrated/raw windows."""
        window_list = list(self._window)
        if self._use_accel:
            base_feature_cols = tuple(
                column
                for column in self.feature_columns
                if column not in {"torso_angular_accel", "vertical_accel"}
            )
            window_list = compute_window_accel_features(
                window_list,
                base_feature_columns=base_feature_cols,
            )
        feature_cols = self.feature_columns
        sample = [flatten_window(window_list, feature_cols)]
        if self._requires_skeleton:
            if len(self._skeleton_window) < self.window_size:
                raise RuntimeError("Skeleton window is shorter than the model feature window")
            self.model.set_skeleton_window(np.stack(tuple(self._skeleton_window), axis=1))
        model_state, risk_score, model_alert_state = self._predict_sample(sample)
        should_update_gate = (
            self._last_temporal_gate_frame is None
            or frame_index - self._last_temporal_gate_frame >= self.temporal_gate_stride
        )
        if not self._use_temporal_fall_validation:
            state, alert_state = model_state, model_alert_state
        elif should_update_gate:
            state, alert_state = self._apply_temporal_validation(
                model_state,
                model_alert_state,
                window_list,
                list(self._raw_window),
            )
            self._last_temporal_gate_frame = frame_index
            self._last_temporal_state = state
            self._last_temporal_alert = alert_state
        else:
            state, alert_state = self._last_temporal_state, self._last_temporal_alert
        return self._prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            instant_state=model_state,
            risk_score=risk_score,
            features=features,
            center_drop=center_drop,
            alert_state=alert_state,
        )

    def reset(self) -> None:

        self.extractor.reset()
        self._window.clear()
        self._raw_window.clear()
        self._skeleton_window.clear()
        self._risk_history.clear()
        self._baseline_samples.clear()
        self._baseline_center_y = None
        self._prefall_alert_count = 0
        self._missing_pose_count = 0
        self._uncalibrated_non_upright_count = 0
        if self._hmm is not None:
            self._hmm.reset()
        if self._standing_calibrator is not None:
            self._standing_calibrator.reset()
        self._temporal_sequence_gate.reset()
        self._last_temporal_gate_frame = None
        self._last_temporal_state = "Normal"
        self._last_temporal_alert = "Normal"
        self._last_state_probabilities = {
            "Normal": 1.0,
            "Pre-fall": 0.0,
            "Fall": 0.0,
        }

    def acknowledge_fall(self) -> None:
        """Acknowledge a latched Fall without discarding calibration/windows."""
        self._temporal_sequence_gate.acknowledge_fall()
        self._last_temporal_gate_frame = None
        self._last_temporal_state = "Normal"
        self._last_temporal_alert = "Normal"

    def _apply_temporal_validation(
        self,
        model_state: str,
        model_alert_state: str,
        window_list: Sequence[dict[str, float]],
        temporal_window_list: Sequence[dict[str, float]] | None = None,
    ) -> tuple[str, str]:
        if not self._use_temporal_fall_validation:
            return model_state, model_alert_state
        return self._temporal_sequence_gate.validate(
            model_state,
            model_alert_state,
            self._last_state_probabilities,
            temporal_window_list if temporal_window_list is not None else window_list,
        )

    def _update_baseline_and_center_drop(self, features: PoseFeatures) -> float:

        if features.center_valid and self._baseline_center_y is None:
            self._baseline_samples.append(features.body_center_y)
            if len(self._baseline_samples) >= self.baseline_frames:
                self._baseline_center_y = mean(self._baseline_samples)

        baseline = self._baseline_center_y
        if baseline is None and self._baseline_samples:
            baseline = mean(self._baseline_samples)
        if baseline is None:
            return 0.0
        return max(0.0, features.body_center_y - baseline)

    def _predict_sample(self, sample: list[list[float]]) -> tuple[str, float, str]:

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            classes = [str(label) for label in self.model.classes_]
            self._last_state_probabilities = _state_probabilities(classes, probabilities)


            if self._use_hmm and self._hmm is not None:


                hmm_probs = [
                    _normal_probability(classes, probabilities),
                    _state_probability(classes, probabilities, "Pre-fall"),
                    _state_probability(classes, probabilities, "Fall"),
                ]
                state = self._hmm.smooth(hmm_probs)

                risk_score = _fall_probability(classes, probabilities)
                alert_state = self._alert_state_from_probabilities(state, classes, probabilities)
                return state, risk_score, alert_state


            best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
            state = normalize_state(classes[best_index])


            risk_score = _fall_probability(classes, probabilities)
            alert_state = self._alert_state_from_probabilities(state, classes, probabilities)
            return state, risk_score, alert_state

        label = self.model.predict(sample)[0]
        state = normalize_state(str(label))
        risk_score = 1.0 if state in {"Pre-fall", "Fall"} else 0.0
        self._prefall_alert_count = 0
        self._last_state_probabilities = {
            "Normal": 1.0 if state == "Normal" else 0.0,
            "Pre-fall": 1.0 if state == "Pre-fall" else 0.0,
            "Fall": 1.0 if state == "Fall" else 0.0,
        }
        return state, risk_score, state

    def _alert_state_from_probabilities(
        self,
        state: str,
        classes: Sequence[str],
        probabilities: Sequence[float],
    ) -> str:

        if state == "Fall":
            self._prefall_alert_count = 0
            return "Fall"
        if state == "Pre-fall":
            self._prefall_alert_count = self.prefall_alert_consecutive_frames
            return "Pre-fall"

        prefall_probability = _state_probability(classes, probabilities, "Pre-fall")
        if prefall_probability >= self.prefall_alert_threshold:
            self._prefall_alert_count += 1
        else:
            self._prefall_alert_count = 0

        if self._prefall_alert_count >= self.prefall_alert_consecutive_frames:
            return "Pre-fall"
        return state

    def _prediction(
        self,
        frame_index: int,
        timestamp: float,
        state: str,
        instant_state: str,
        risk_score: float,
        features: PoseFeatures,
        center_drop: float,
        alert_state: str | None = None,
        system_status: str | None = None,
    ) -> Prediction:

        self._risk_history.append(risk_score)
        smoothed_risk = mean(self._risk_history) if self._risk_history else 0.0
        return Prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            instant_state=instant_state,
            risk_score=risk_score,
            smoothed_risk_score=smoothed_risk,
            features=features,
            breakdown=RiskBreakdown(
                risk_score=risk_score,
                torso_score=0.0,
                angular_velocity_score=0.0,
                vertical_velocity_score=0.0,
                center_drop_score=0.0,
                aspect_ratio_score=0.0,
                visibility_factor=1.0 if features.has_pose else 0.0,
                center_drop=center_drop,
            ),
            baseline_center_y=self._baseline_center_y,
            alert_state=alert_state or state,
            system_status=system_status,
        )


def load_model_artifact(model_path: str | Path) -> dict:

    model_path = Path(model_path)
    if model_path.suffix.lower() in {".pt", ".pth"}:
        from .deep_model import load_deep_model_artifact
        from .fusion_model import load_fusion_model_artifact

        try:
            return load_deep_model_artifact(model_path)
        except RuntimeError as deep_error:
            try:
                return load_fusion_model_artifact(model_path)
            except RuntimeError:
                raise deep_error

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "ML prediction requires joblib. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        artifact = {
            "model": artifact,
            "window_size": DEFAULT_WINDOW_SIZE,
            "feature_columns": ML_FEATURE_COLUMNS,
            "baseline_frames": DEFAULT_PREDICTOR_CONFIG.baseline_frames,
            "smoothing_window": DEFAULT_PREDICTOR_CONFIG.smoothing_window,
            "prefall_alert_threshold": DEFAULT_PREFALL_ALERT_THRESHOLD,
            "prefall_alert_consecutive_frames": DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
        }
    if "model" not in artifact:
        raise RuntimeError(f"Model artifact is missing the 'model' field: {model_path}")
    return artifact


def normalize_state(label: str) -> str:

    cleaned = label.strip()
    lower = cleaned.lower().replace("_", "-")
    if lower in {"1", "fall", "fallen"}:
        return "Fall"
    if lower in {"pre-fall", "prefall", "pre fall"}:
        return "Pre-fall"
    if lower in {
        "0",
        "normal",
        "adl",
        "nonfall",
        "non-fall",
        "no-fall",
        "standing",
        "stand",
        "normal-standing",
        "normal-stand",
        "walking",
        "walk",
        "normal-walking",
        "normal-walk",
        "sitting",
        "sit",
        "seated",
        "normal-sitting",
        "normal-sit",
        "squatting",
        "squat",
        "crouching",
        "crouch",
        "normal-squatting",
        "normal-squat",
        "normal-crouching",
        "normal-crouch",
        "bending",
        "bend",
        "bent",
        "stooping",
        "stoop",
        "spotting",
        "spot",
        "bowing",
        "bow",
        "normal-bending",
        "normal-bend",
        "normal-stooping",
        "normal-stoop",
        "normal-spotting",
        "normal-spot",
        "lying",
        "lie",
        "laying",
        "laid",
        "reclining",
        "reclined",
        "prone",
        "supine",
        "horizontal",
        "normal-lying",
        "normal-lie",
        "normal-laying",
    }:
        return "Normal"
    if "pre" in lower and "fall" in lower:
        return "Pre-fall"
    if "fall" in lower:
        return "Fall"
    return cleaned


def _motion_evidence(
    window_rows: Sequence[Mapping[str, object]],
    vertical_velocity_threshold: float = DEFAULT_FALL_VERTICAL_VELOCITY,
    vertical_accel_threshold: float = DEFAULT_FALL_VERTICAL_ACCEL,
    angular_velocity_threshold: float = DEFAULT_FALL_ANGULAR_VELOCITY,
    angular_accel_threshold: float = DEFAULT_FALL_ANGULAR_ACCEL,
    center_drop_delta_threshold: float = DEFAULT_FALL_CENTER_DROP_DELTA,
) -> FallMotionEvidence:
    """Build dynamic fall evidence from one runtime window."""
    if not window_rows:
        return FallMotionEvidence(
            max_vertical_velocity=0.0,
            max_vertical_accel=0.0,
            max_angular_velocity=0.0,
            max_angular_accel=0.0,
            center_drop_delta=0.0,
            has_fast_descent=False,
            has_impact_or_rotation=False,
            has_center_drop_change=False,
        )

    max_vertical_velocity = max(max(0.0, _row_float(row, "vertical_velocity")) for row in window_rows)
    max_vertical_accel = max(_positive_delta(window_rows, "vertical_velocity"))

    max_angular_velocity = max(abs(_row_float(row, "torso_angular_velocity")) for row in window_rows)
    max_angular_accel = max(abs(value) for value in _delta(window_rows, "torso_angular_velocity"))

    center_drop_values = [_row_float(row, "center_drop") for row in window_rows]
    center_drop_delta = max(center_drop_values) - min(center_drop_values) if center_drop_values else 0.0

    has_fast_descent = max_vertical_velocity >= vertical_velocity_threshold
    has_impact_or_rotation = (
        max_vertical_accel >= vertical_accel_threshold
        or max_angular_velocity >= angular_velocity_threshold
        or max_angular_accel >= angular_accel_threshold
    )
    has_center_drop_change = center_drop_delta >= center_drop_delta_threshold

    return FallMotionEvidence(
        max_vertical_velocity=max_vertical_velocity,
        max_vertical_accel=max_vertical_accel,
        max_angular_velocity=max_angular_velocity,
        max_angular_accel=max_angular_accel,
        center_drop_delta=center_drop_delta,
        has_fast_descent=has_fast_descent,
        has_impact_or_rotation=has_impact_or_rotation,
        has_center_drop_change=has_center_drop_change,
    )


def _posture_evidence(window_rows: Sequence[Mapping[str, object]]) -> PostureEvidence:
    """Build static posture evidence to separate lying/low posture from fall events."""
    pose_rows = [row for row in window_rows if _row_float(row, "has_pose") > 0.0]
    if not pose_rows:
        return PostureEvidence(
            mean_body_height=0.0,
            mean_aspect_ratio=0.0,
            mean_center_drop=0.0,
            mean_torso_angle=0.0,
            mean_abs_vertical_velocity=0.0,
            is_low_horizontal=False,
            is_low_posture=False,
            is_static_low_posture=False,
            is_upright_normal=False,
        )

    bbox_rows = [row for row in pose_rows if _row_float(row, "bbox_valid", 1.0) > 0.0]
    center_rows = [row for row in pose_rows if _row_float(row, "center_valid", 1.0) > 0.0]
    torso_rows = [row for row in pose_rows if _row_float(row, "torso_valid", 1.0) > 0.0]

    mean_body_height = mean(_row_float(row, "body_height") for row in bbox_rows) if bbox_rows else 0.0
    mean_aspect_ratio = mean(_row_float(row, "aspect_ratio") for row in bbox_rows) if bbox_rows else 0.0
    mean_center_drop = mean(_row_float(row, "center_drop") for row in center_rows) if center_rows else 0.0
    mean_torso_angle = (
        mean(abs(_row_float(row, "torso_angle")) for row in torso_rows) if torso_rows else 0.0
    )
    mean_abs_vertical_velocity = (
        mean(abs(_row_float(row, "vertical_velocity")) for row in center_rows)
        if center_rows
        else 0.0
    )

    is_low_horizontal = bool(bbox_rows) and (
        (mean_body_height <= 0.24 and mean_aspect_ratio >= 0.55)
        or (mean_body_height <= 0.32 and mean_aspect_ratio >= 0.85)
        or (bool(center_rows) and mean_center_drop >= 0.24 and mean_aspect_ratio >= 0.65)
    )
    is_low_posture = (
        is_low_horizontal
        or (
            bool(bbox_rows)
            and bool(center_rows)
            and mean_body_height <= 0.38
            and mean_center_drop >= 0.10
        )
        or (bool(bbox_rows) and mean_body_height <= 0.34)
        or (bool(bbox_rows) and mean_aspect_ratio >= 0.75)
    )
    is_static_low_posture = is_low_posture and (
        not center_rows or mean_abs_vertical_velocity <= 0.25
    )
    is_upright_normal = (
        bool(bbox_rows)
        and bool(center_rows)
        and bool(torso_rows)
        and mean_body_height >= 0.42
        and mean_aspect_ratio <= 0.60
        and mean_center_drop <= 0.12
        and mean_torso_angle <= 25.0
    )

    return PostureEvidence(
        mean_body_height=mean_body_height,
        mean_aspect_ratio=mean_aspect_ratio,
        mean_center_drop=mean_center_drop,
        mean_torso_angle=mean_torso_angle,
        mean_abs_vertical_velocity=mean_abs_vertical_velocity,
        is_low_horizontal=is_low_horizontal,
        is_low_posture=is_low_posture,
        is_static_low_posture=is_static_low_posture,
        is_upright_normal=is_upright_normal,
    )


def _has_fall_like_motion(evidence: FallMotionEvidence) -> bool:
    return evidence.has_fast_descent and (
        evidence.has_center_drop_change or evidence.has_impact_or_rotation
    )


def _has_strong_fall_motion(evidence: FallMotionEvidence) -> bool:
    return (
        evidence.has_fast_descent
        and evidence.has_center_drop_change
        and evidence.has_impact_or_rotation
    )


def _resolve_positive_int_setting(
    artifact: dict,
    name: str,
    explicit_value: int | None,
    default_value: int,
) -> int:
    """
    Resolve integer settings with a clear priority:
    explicit constructor argument > saved model artifact > project default.
    """
    value = explicit_value if explicit_value is not None else artifact.get(name, default_value)
    return max(1, int(value))


def _resolve_probability_setting(
    artifact: dict,
    name: str,
    explicit_value: float | None,
    default_value: float,
) -> float:
    """
    Resolve probability-like settings and keep them inside [0, 1].
    """
    value = explicit_value if explicit_value is not None else artifact.get(name, default_value)
    return max(0.0, min(1.0, float(value)))


def _state_probabilities(classes: Sequence[str], probabilities: Sequence[float]) -> dict[str, float]:
    """Return normalized state probabilities used by temporal gates."""
    return {
        "Normal": _normal_probability(classes, probabilities),
        "Pre-fall": _state_probability(classes, probabilities, "Pre-fall"),
        "Fall": _state_probability(classes, probabilities, "Fall"),
    }


def _fall_probability(classes: Sequence[str], probabilities: Sequence[float]) -> float:

    fall_prob = 0.0
    for label, probability in zip(classes, probabilities):
        state = normalize_state(label)
        if state == "Fall":
            fall_prob += float(probability)
        elif state == "Pre-fall":
            fall_prob += float(probability) * 0.5
    return max(0.0, min(1.0, fall_prob))


def _state_probability(classes: Sequence[str], probabilities: Sequence[float], target_state: str) -> float:
    """Return the summed probability for one normalized state."""
    total = 0.0
    for label, probability in zip(classes, probabilities):
        if normalize_state(label) == target_state:
            total += float(probability)
    return max(0.0, min(1.0, total))


def _normal_probability(classes: Sequence[str], probabilities: Sequence[float]) -> float:
    """Return summed probability for normalized Normal states."""
    total = 0.0
    for label, probability in zip(classes, probabilities):
        if normalize_state(label) in NORMAL_STATES:
            total += float(probability)
    return max(0.0, min(1.0, total))


def _delta(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Return frame-to-frame deltas for one numeric row field."""
    if not rows:
        return [0.0]

    values = [_row_float(row, key) for row in rows]
    deltas = [0.0]
    for index in range(1, len(values)):
        deltas.append(values[index] - values[index - 1])
    return deltas


def _positive_delta(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Return positive frame-to-frame deltas for one numeric row field."""
    return [max(0.0, value) for value in _delta(rows, key)]


def _row_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value
