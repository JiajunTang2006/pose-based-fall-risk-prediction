from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database.init_db import AppRepositories
    from .risk_state_machine import RiskState, StateChangeEvent


class EventService:

    def __init__(
        self,
        repos: "AppRepositories",
        session_id: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        self._repos = repos
        self._session_id = session_id
        self._profile_id = profile_id
        self._active_event_id: str | None = None
        self._event_start_time: float | None = None
        self._peak_risk: float = 0.0
        self._risk_sum: float = 0.0
        self._risk_count: int = 0


    @property
    def active_event_id(self) -> str | None:
        return self._active_event_id

    def on_state_change(self, event: "StateChangeEvent") -> None:
        from .risk_state_machine import RiskState

        session_id = self._session_id
        if session_id is None:
            session = self._repos.sessions.get_active()
            if session is None:
                return
            session_id = session["id"]

        profile_id = self._profile_id
        if profile_id is None:
            profile = self._repos.profiles.get_active()
            profile_id = profile["id"] if profile else "default"

        if event.is_escalation:
            self._handle_escalation(session_id, profile_id, event)
        elif event.is_deescalation and event.to_state == RiskState.NORMAL:
            self._handle_close(event)

    def observe_frame(self, risk_score: float) -> None:
        if self._active_event_id is None:
            return
        self._peak_risk = max(self._peak_risk, risk_score)
        self._risk_sum += risk_score
        self._risk_count += 1

    def recent_events(self, limit: int = 12) -> list[dict]:
        return self._repos.events.list_recent(limit)

    def close_all(self) -> None:
        if self._active_event_id is not None:
            self._close_current_event()


    def _handle_escalation(self, session_id: str, profile_id: str,
                           event: "StateChangeEvent") -> None:
        from .risk_state_machine import RiskState

        event_type = "fall" if event.to_state == RiskState.FALL else "pre-fall"

        if self._active_event_id is not None:
            db_event = self._repos.events.get(self._active_event_id)
            if db_event is None:
                self._active_event_id = None
            elif db_event.get("event_type") != event_type:
                self._repos.events.update_type(self._active_event_id, event_type)
                self._peak_risk = max(self._peak_risk, event.risk_score)
                return
            else:
                self._peak_risk = max(self._peak_risk, event.risk_score)
                return

        db_event = self._repos.events.create(
            session_id=session_id,
            profile_id=profile_id,
            event_type=event_type,
            risk_score=event.risk_score,
        )
        self._active_event_id = db_event["id"]
        self._event_start_time = time.monotonic()
        self._peak_risk = event.risk_score
        self._risk_sum = 0.0
        self._risk_count = 0

    def _handle_close(self, event: "StateChangeEvent") -> None:
        if self._active_event_id is None:
            return
        self._close_current_event()

    def _close_current_event(self) -> None:
        if self._active_event_id is None:
            return
        duration = 0.0
        if self._event_start_time is not None:
            duration = time.monotonic() - self._event_start_time
        avg_risk = self._risk_sum / max(self._risk_count, 1)

        self._repos.events.update_peak(self._active_event_id, self._peak_risk)
        self._repos.events.close(
            self._active_event_id,
            duration_seconds=round(duration, 1),
            avg_risk=round(avg_risk, 4),
        )
        self._active_event_id = None
        self._event_start_time = None
        self._peak_risk = 0.0
        self._risk_sum = 0.0
        self._risk_count = 0
