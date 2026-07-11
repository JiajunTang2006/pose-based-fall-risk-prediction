"""Shared business-state processing for camera and imported media frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .event_service import EventService
from .risk_state_machine import RiskStateMachine, StateChangeEvent

if TYPE_CHECKING:
    from fall_prediction.predictor import Prediction

    from .database.init_db import AppRepositories


class StateChangeObserver(Protocol):
    def on_state_change(self, event: StateChangeEvent) -> bool | None: ...


@dataclass(frozen=True)
class ProcessedFrameState:
    state: str
    risk_score: float
    visibility: float
    confidence: float
    state_change: StateChangeEvent | None


class FrameBusinessProcessor:
    """Apply the same FSM, event, and sampling rules to every frame source."""

    def __init__(
        self,
        repos: "AppRepositories | None",
        session_id: str | None,
        profile_id: str | None,
        fps: float,
        state_change_observer: StateChangeObserver | None = None,
    ) -> None:
        self._repos = repos
        self._session_id = session_id
        self._state_change_observer = state_change_observer
        self._fps = max(float(fps), 1.0)
        self._sample_interval = max(1, int(round(self._fps)))
        self._event_service = (
            EventService(repos, session_id=session_id, profile_id=profile_id)
            if repos is not None and session_id is not None
            else None
        )
        self._fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "cooldown_seconds": 30,
                "recovery_frames": 10,
                "lost_tolerance_frames": 15,
                "ema_alpha": 0.5,
            },
            fps=self._fps,
            on_state_change=self._on_state_change,
        )

    def _on_state_change(self, event: StateChangeEvent) -> None:
        if self._event_service is not None:
            self._event_service.on_state_change(event)
        if self._state_change_observer is not None:
            self._state_change_observer.on_state_change(event)

    @property
    def event_service(self) -> EventService | None:
        return self._event_service

    @property
    def state_machine(self) -> RiskStateMachine:
        return self._fsm

    def process(self, prediction: "Prediction", frame_index: int, timestamp: float) -> ProcessedFrameState:
        validated_state = prediction.alert_state or prediction.state
        business_risk = {
            "Normal": 0.0,
            "Pre-fall": 0.55,
            "Fall": 1.0,
        }.get(validated_state, float(prediction.risk_score))
        visibility = max(
            0.0,
            min(1.0, float(getattr(prediction.features, "visibility_mean", 0.8))),
        )
        person_visible = prediction.state != "Unknown"
        state_change = self._fsm.update(business_risk, visibility, person_visible)
        risk_score = max(0.0, min(1.0, float(prediction.risk_score)))

        if self._event_service is not None:
            self._event_service.observe_frame(risk_score)

        state = self._fsm.state.value
        if (
            self._repos is not None
            and self._session_id is not None
            and (frame_index + 1) % self._sample_interval == 0
        ):
            try:
                self._repos.samples.insert(
                    session_id=self._session_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    risk_score=round(risk_score, 4),
                    visibility=round(visibility, 4),
                    state=state,
                    confidence=round(visibility, 4),
                )
                if (frame_index + 1) % (self._sample_interval * 10) == 0:
                    self._repos.samples.commit()
            except Exception:
                # Persistence must not stop live prediction or media processing.
                pass

        return ProcessedFrameState(
            state=state,
            risk_score=risk_score,
            visibility=visibility,
            confidence=visibility,
            state_change=state_change,
        )

    def close(self) -> None:
        if self._event_service is not None:
            self._event_service.close_all()
        if self._repos is not None and self._session_id is not None:
            self._repos.samples.commit()
