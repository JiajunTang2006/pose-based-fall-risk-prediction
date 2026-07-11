"""
Risk State Machine — converts per-frame model output into stable business states.

Per the FallGuard Development Workflow (V1.0, Phase 3):
  - EMA (exponential moving average) smoothing
  - Consecutive-frame confirmation (不同阈值进出 = hysteresis)
  - Cooldown period after FALL to prevent duplicate alerts
  - Pose-loss tolerance window
  - Recovery state with event merging

States:  NORMAL → WARNING → FALL → RECOVERY → NORMAL
                NORMAL → FALL  (direct escalation when risk is very high)
                WARNING → NORMAL  (direct de-escalation when risk drops)

Usage::

    fsm = RiskStateMachine(thresholds={"prefall_threshold": 0.45, "fall_threshold": 0.72})
    for frame_index, risk_score, confidence in frames:
        event = fsm.update(risk_score, confidence, person_visible=True)
        if event:
            print(f"State change: {fsm.state} at frame {frame_index}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RiskState(Enum):
    NORMAL = "Normal"
    WARNING = "Pre-fall"
    FALL = "Fall"
    RECOVERY = "Recovery"
    LOST = "Unknown"  # pose tracking lost


@dataclass
class StateChangeEvent:
    """Emitted when the state machine transitions between states."""
    from_state: RiskState
    to_state: RiskState
    frame_index: int
    risk_score: float
    confidence: float
    is_escalation: bool     # True if NORMAL→WARNING or WARNING→FALL
    is_deescalation: bool   # True if FALL→RECOVERY or RECOVERY→NORMAL or WARNING→NORMAL


_DEFAULT_THRESHOLDS = {
    "prefall_threshold": 0.45,
    "fall_threshold": 0.72,
    "min_visibility": 0.35,
    "consecutive_confirm_frames": 3,
    "cooldown_seconds": 30,
    "recovery_frames": 10,
    "lost_tolerance_frames": 15,
    "ema_alpha": 0.5,  # smoothing factor (0=no smoothing, 1=raw)
}


class RiskStateMachine:
    """Converts noisy per-frame risk scores into stable, hysteresis-protected states.

    Implements all mechanisms required by the development workflow document:
      - EMA smoothing to reduce jitter
      - Consecutive-frame confirmation before state transitions
      - Hysteresis (different entry/exit thresholds) via state-dependent checks
      - Cooldown timer to prevent duplicate FALL events
      - Pose-loss tolerance (LOST state after N missing-person frames)
      - RECOVERY state that prevents immediate re-triggering
    """

    def __init__(self, thresholds: dict | None = None,
                 fps: float = 20.0,
                 on_state_change: Callable[[StateChangeEvent], None] | None = None) -> None:
        cfg = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._prefall_threshold: float = cfg["prefall_threshold"]
        self._fall_threshold: float = cfg["fall_threshold"]
        if self._prefall_threshold >= self._fall_threshold:
            raise ValueError(
                f"prefall_threshold ({self._prefall_threshold}) must be less than "
                f"fall_threshold ({self._fall_threshold})"
            )
        self._min_visibility: float = cfg["min_visibility"]
        self._confirm_frames: int = cfg["consecutive_confirm_frames"]
        self._cooldown_frames: int = int(cfg["cooldown_seconds"] * fps)
        self._recovery_frames: int = cfg["recovery_frames"]
        self._lost_tolerance: int = cfg["lost_tolerance_frames"]
        self._ema_alpha: float = cfg["ema_alpha"]
        self._fps: float = fps

        self._state: RiskState = RiskState.NORMAL
        self._smoothed_risk: float = 0.0
        self._consecutive_count: int = 0
        self._deescalation_count: int = 0  # Separate counter for exit-from-elevated
        self._lost_count: int = 0
        self._recovery_count: int = 0
        self._cooldown_remaining: int = 0
        self._frame_index: int = 0
        self._state_before_lost: RiskState | None = None  # saved when entering LOST
        self._on_state_change: Callable[[StateChangeEvent], None] | None = on_state_change

        # Track the last escalation target state (WARNING or FALL) for
        # consecutive-frame counting while already in a higher state.
        self._escalation_target: RiskState | None = None
        self._last_confidence: float = 1.0
        self._in_update: bool = False

    # ── public API ──────────────────────────────────────────────────

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def smoothed_risk(self) -> float:
        return self._smoothed_risk

    @property
    def is_elevated(self) -> bool:
        return self._state in {RiskState.WARNING, RiskState.FALL, RiskState.RECOVERY}

    def update(self, raw_risk: float, confidence: float = 1.0,
               person_visible: bool = True) -> StateChangeEvent | None:
        """Feed one frame's risk score into the state machine.

        Returns a ``StateChangeEvent`` when a state transition occurs, or ``None``.
        """
        if getattr(self, '_in_update', False):
            raise RuntimeError("RiskStateMachine.update() is not re-entrant")
        self._in_update = True
        try:
            self._frame_index += 1
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1

            # ── Pose loss handling ──────────────────────────────────
            if not person_visible or confidence < self._min_visibility:
                self._lost_count += 1
                if self._lost_count >= self._lost_tolerance and self._state != RiskState.LOST:
                    self._state_before_lost = self._state  # remember for recovery
                    return self._transition_to(RiskState.LOST)
                return None
            self._lost_count = 0

            # Recover from LOST — if we were in an elevated state before losing
            # the person, go through RECOVERY to prevent immediate re-trigger.
            if self._state == RiskState.LOST:
                if self._state_before_lost in {RiskState.FALL, RiskState.WARNING}:
                    return self._transition_to(RiskState.RECOVERY)
                return self._transition_to(RiskState.NORMAL)

            # ── EMA smoothing ───────────────────────────────────────
            if self._frame_index == 1:
                self._smoothed_risk = raw_risk
            else:
                self._smoothed_risk = (
                    self._ema_alpha * raw_risk + (1 - self._ema_alpha) * self._smoothed_risk
                )

            risk = self._smoothed_risk
            self._last_confidence = confidence

            # ── State-dependent logic ───────────────────────────────
            if self._state == RiskState.NORMAL:
                return self._handle_normal(risk)
            elif self._state == RiskState.WARNING:
                return self._handle_warning(risk)
            elif self._state == RiskState.FALL:
                return self._handle_fall(risk)
            elif self._state == RiskState.RECOVERY:
                return self._handle_recovery(risk)
            return None
        finally:
            self._in_update = False

    def reset(self) -> None:
        """Reset state machine to initial conditions."""
        self._state = RiskState.NORMAL
        self._smoothed_risk = 0.0
        self._consecutive_count = 0
        self._deescalation_count = 0
        self._lost_count = 0
        self._recovery_count = 0
        self._cooldown_remaining = 0
        self._frame_index = 0
        self._escalation_target = None

    def set_thresholds(self, prefall: float | None = None, fall: float | None = None,
                       cooldown_seconds: float | None = None) -> None:
        """Update thresholds at runtime (e.g. when user changes sensitivity)."""
        if prefall is not None:
            self._prefall_threshold = prefall
        if fall is not None:
            self._fall_threshold = fall
        if cooldown_seconds is not None:
            self._cooldown_frames = int(cooldown_seconds * self._fps)
        if self._prefall_threshold >= self._fall_threshold:
            raise ValueError(
                f"prefall_threshold ({self._prefall_threshold}) must be less than "
                f"fall_threshold ({self._fall_threshold})"
            )

    # ── per-state handlers ──────────────────────────────────────────

    def _handle_normal(self, risk: float) -> StateChangeEvent | None:
        if risk >= self._fall_threshold:
            return self._count_toward(RiskState.FALL, risk)
        elif risk >= self._prefall_threshold:
            return self._count_toward(RiskState.WARNING, risk)
        else:
            self._consecutive_count = 0
            self._escalation_target = None
            return None

    def _handle_warning(self, risk: float) -> StateChangeEvent | None:
        if risk >= self._fall_threshold:
            self._deescalation_count = 0  # reset de-escalation counter
            return self._count_toward(RiskState.FALL, risk)
        elif risk >= self._prefall_threshold:
            # Stay in WARNING
            self._deescalation_count = 0
            self._consecutive_count = 0  # reset escalation counter
            self._escalation_target = None
            return None
        else:
            # Risk dropped below prefall — count toward NORMAL (hysteresis exit)
            # Use separate de-escalation counter to avoid interference with
            # escalation counting that may have been in progress.
            self._deescalation_count += 1
            self._consecutive_count = 0
            self._escalation_target = None
            if self._deescalation_count >= self._confirm_frames:
                return self._transition_to(RiskState.NORMAL)
            return None

    def _handle_fall(self, risk: float) -> StateChangeEvent | None:
        if risk >= self._fall_threshold:
            self._deescalation_count = 0  # stay in FALL
            return None
        # Risk dropped below fall threshold → count toward RECOVERY
        self._deescalation_count += 1
        if self._deescalation_count >= self._confirm_frames:
            return self._transition_to(RiskState.RECOVERY)
        return None

    def _handle_recovery(self, risk: float) -> StateChangeEvent | None:
        # During RECOVERY, ignore risk spikes (prevent immediate re-trigger)
        self._recovery_count += 1
        if self._recovery_count >= self._recovery_frames:
            return self._transition_to(RiskState.NORMAL)
        return None

    # ── internal helpers ────────────────────────────────────────────

    def _count_toward(self, target: RiskState, risk: float) -> StateChangeEvent | None:
        """Count consecutive frames where risk supports escalation to `target`."""
        if self._cooldown_remaining > 0 and target in {RiskState.WARNING, RiskState.FALL}:
            return None  # cooldown active, suppress escalation

        if self._escalation_target == target:
            self._consecutive_count += 1
        else:
            self._escalation_target = target
            self._consecutive_count = 1

        if self._consecutive_count >= self._confirm_frames:
            self._consecutive_count = 0
            self._escalation_target = None
            return self._transition_to(target)
        return None

    def _transition_to(self, new_state: RiskState) -> StateChangeEvent:
        old_state = self._state
        self._state = new_state
        self._consecutive_count = 0
        self._deescalation_count = 0
        self._escalation_target = None

        # Set cooldown only in _transition_to (single source of truth)
        if new_state == RiskState.FALL:
            self._cooldown_remaining = self._cooldown_frames
        if new_state == RiskState.RECOVERY:
            self._recovery_count = 0

        event = StateChangeEvent(
            from_state=old_state,
            to_state=new_state,
            frame_index=self._frame_index,
            risk_score=self._smoothed_risk,
            confidence=self._last_confidence,
            is_escalation=new_state in {RiskState.WARNING, RiskState.FALL},
            is_deescalation=new_state in {RiskState.NORMAL, RiskState.RECOVERY},
        )

        if self._on_state_change:
            self._on_state_change(event)

        return event
