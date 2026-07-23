"""Lightweight product rule for static lying activities of daily living.

The classifiers answer what the current window looks like.  This filter adds
the narrower event-level distinction needed by the product: a person who is
already lying still is not automatically a Fall, while a confirmed dynamic
Fall remains latched after the person becomes still.

Unlike :class:`TemporalSequenceGate`, this rule does not require a complete
Normal -> Pre-fall -> Fall chain.  It intervenes only when posture is both low
and static and the recent windows contain no fall-like motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .ml_predictor import (
    _has_fall_like_motion,
    _motion_evidence,
    _posture_evidence,
    normalize_state,
)


DEFAULT_LYING_SETTLE_STEPS = 3
DEFAULT_LYING_WARM_WARNING_STEPS = 2
DEFAULT_FALL_MOTION_MEMORY_STEPS = 6


@dataclass(frozen=True)
class StaticLyingADLDecision:
    """Postprocessed product states for one sampled window."""

    state: str
    alert_state: str
    advisory_state: str | None
    tier: str
    status: str
    filtered: bool
    fall_latched: bool
    is_static_low_posture: bool
    is_static_lying_posture: bool
    has_recent_fall_motion: bool


class StaticLyingADLFilter:
    """Suppress static-lying false alarms without replacing model training.

    Rules:

    * static low posture + no recent fall motion: confirmed/alert Fall is
      changed to Normal, with only a short Pre-fall advisory at warm start;
    * a confirmed Fall with current/recent motion is latched;
    * a confirmed non-static Fall still passes through and is latched, so this
      narrow ADL rule cannot recreate the misses caused by the old strict gate;
    * once latched, later static lying remains Fall until acknowledged.

    Counters advance at the model artifact stride rather than on every highly
    overlapping video frame.
    """

    def __init__(
        self,
        *,
        lying_settle_steps: int = DEFAULT_LYING_SETTLE_STEPS,
        warm_warning_steps: int = DEFAULT_LYING_WARM_WARNING_STEPS,
        motion_memory_steps: int = DEFAULT_FALL_MOTION_MEMORY_STEPS,
    ) -> None:
        if lying_settle_steps < 1:
            raise ValueError("lying_settle_steps must be at least 1")
        if warm_warning_steps < 0:
            raise ValueError("warm_warning_steps cannot be negative")
        if motion_memory_steps < 1:
            raise ValueError("motion_memory_steps must be at least 1")
        self.lying_settle_steps = int(lying_settle_steps)
        self.warm_warning_steps = min(int(warm_warning_steps), self.lying_settle_steps)
        self.motion_memory_steps = int(motion_memory_steps)
        self._static_lying_count = 0
        self._motion_memory_remaining = 0
        self._fall_latched = False

    @property
    def fall_latched(self) -> bool:
        return self._fall_latched

    def reset(self) -> None:
        self._static_lying_count = 0
        self._motion_memory_remaining = 0
        self._fall_latched = False

    def acknowledge_fall(self) -> None:
        """Clear the event latch without changing model/calibration windows."""
        self._fall_latched = False
        self._static_lying_count = 0
        self._motion_memory_remaining = 0

    def process(
        self,
        state: str,
        alert_state: str,
        advisory_state: str | None,
        window_rows: Sequence[Mapping[str, object]],
        *,
        advance: bool = True,
    ) -> StaticLyingADLDecision:
        state = normalize_state(state)
        alert_state = normalize_state(alert_state)
        advisory = normalize_state(advisory_state) if advisory_state is not None else None
        posture = _posture_evidence(window_rows)
        motion = _motion_evidence(window_rows)
        current_fall_motion = _has_fall_like_motion(motion)

        if advance:
            if current_fall_motion:
                self._motion_memory_remaining = self.motion_memory_steps
            elif self._motion_memory_remaining > 0:
                self._motion_memory_remaining -= 1

        has_recent_fall_motion = current_fall_motion or self._motion_memory_remaining > 0

        if self._fall_latched:
            return self._decision(
                "Fall",
                "Fall",
                None,
                tier="critical-fall-latched",
                status="Postprocess: confirmed Fall remains latched",
                filtered=state != "Fall" or alert_state != "Fall" or advisory is not None,
                posture=posture,
                has_recent_fall_motion=has_recent_fall_motion,
            )

        # `is_low_posture` intentionally also covers a small/crouched person
        # for the legacy sequence gate.  That is too broad for an ADL *lying*
        # override and varies with camera distance.  Requiring its horizontal
        # sub-condition prevents a distant upright person from being treated
        # as already lying merely because the bounding box is short.
        is_static_lying_posture = (
            posture.is_static_low_posture and posture.is_low_horizontal
        )
        static_without_fall_motion = (
            is_static_lying_posture and not has_recent_fall_motion
        )
        has_fall_or_warning_candidate = any(
            value in {"Pre-fall", "Fall"}
            for value in (state, alert_state, advisory)
            if value is not None
        )

        if static_without_fall_motion and has_fall_or_warning_candidate:
            if advance:
                self._static_lying_count += 1
            # The official output is Normal immediately.  A short advisory
            # preserves caution while a warm-start lying posture settles.
            warm_warning = (
                self._static_lying_count <= self.warm_warning_steps
                and self._static_lying_count < self.lying_settle_steps
            )
            return self._decision(
                "Normal",
                "Normal",
                "Pre-fall" if warm_warning else None,
                tier="lying-adl-watch" if warm_warning else "lying-adl-normal",
                status=(
                    "Postprocess: static low posture without fall motion; short advisory"
                    if warm_warning
                    else "Postprocess: settled static lying ADL -> Normal"
                ),
                filtered=True,
                posture=posture,
                has_recent_fall_motion=False,
            )

        if advance:
            self._static_lying_count = 0

        # `state == Fall` is the authoritative/confirmed channel.  A dynamic
        # or non-static confirmed Fall is allowed through and latched.  An
        # alert-only Fall remains an alert and does not create a product latch.
        if state == "Fall":
            self._fall_latched = True
            reason = (
                "recent fall motion"
                if has_recent_fall_motion
                else "authoritative non-static Fall"
            )
            return self._decision(
                "Fall",
                "Fall",
                None,
                tier="critical-fall-latched",
                status=f"Postprocess: {reason}; Fall latched",
                filtered=alert_state != "Fall" or advisory is not None,
                posture=posture,
                has_recent_fall_motion=has_recent_fall_motion,
            )

        return self._decision(
            state,
            alert_state,
            advisory,
            tier="postprocess-pass-through",
            status="Postprocess: no static-lying override",
            filtered=False,
            posture=posture,
            has_recent_fall_motion=has_recent_fall_motion,
        )

    def _decision(
        self,
        state: str,
        alert_state: str,
        advisory_state: str | None,
        *,
        tier: str,
        status: str,
        filtered: bool,
        posture,
        has_recent_fall_motion: bool,
    ) -> StaticLyingADLDecision:
        return StaticLyingADLDecision(
            state=state,
            alert_state=alert_state,
            advisory_state=advisory_state,
            tier=tier,
            status=status,
            filtered=filtered,
            fall_latched=self._fall_latched,
            is_static_low_posture=posture.is_static_low_posture,
            is_static_lying_posture=(
                posture.is_static_low_posture and posture.is_low_horizontal
            ),
            has_recent_fall_motion=has_recent_fall_motion,
        )
