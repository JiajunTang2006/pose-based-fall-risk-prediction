"""
Serialization helpers — convert internal Python objects to stable API contracts.

IMPORTANT: Never serialise internal objects (``DashboardSnapshot``, ``Prediction``,
``UserProfile``, etc.) directly.  Always go through the functions in this module
so that Swift's DTO contracts remain stable even when internal field names change.
"""

from __future__ import annotations

import time
from typing import Any

from .contracts import (
    EventDTO,
    HealthResponse,
    ImportJobResponse,
    MonitorCommandResponse,
    PaginatedResponse,
    PerformanceDTO,
    PredictionDTO,
    ProfileDTO,
    ServiceErrorDTO,
    SettingsDTO,
    StatusResponse,
)


# ── sequence counter ────────────────────────────────────────────────

_seq: int = 0


def next_sequence() -> int:
    global _seq
    _seq += 1
    return _seq


def reset_sequence(value: int = 0) -> None:
    global _seq
    _seq = value


# ── value guards ────────────────────────────────────────────────────


def clamp01(value: float) -> float:
    """Clamp *value* to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


# ── state helpers ───────────────────────────────────────────────────


def _model_state_from_internal(raw_state: str | None) -> str:
    """Map an internal state label to the contract model state enum."""
    if raw_state is None:
        return "Unknown"
    state = str(raw_state).strip()
    # Already a valid contract state?
    if state in {"Normal", "Pre-fall", "Fall", "Unknown"}:
        return state
    # Map from various internal labels
    mapping: dict[str, str] = {
        "Idle": "Normal",
        "Starting": "Normal",
        "Ready": "Normal",
        "Monitoring Active": "Normal",
        "Medium Risk Detected": "Pre-fall",
        "High Risk Detected": "Fall",
        "Error": "Unknown",
        "Setup Needed": "Unknown",
        "Person Not Visible": "Unknown",
        # The business FSM uses Recovery briefly after a confirmed Fall.
        # The public contract has no Recovery enum, so expose it as Normal
        # instead of incorrectly falling through to Unknown.
        "Recovery": "Normal",
    }
    return mapping.get(state, "Unknown")


def _business_state_from_model(model_state: str) -> str:
    return {
        "Normal": "safe",
        "Pre-fall": "warning",
        "Fall": "danger",
        "Unknown": "unknown",
    }.get(model_state, "unknown")


# ── main serialisation functions ────────────────────────────────────


def serialize_health(
    *,
    status: str = "starting",
    version: str = "0.3.3",
    api_version: str = "v1",
    models_loaded: bool = False,
    database_ok: bool = False,
    camera_available: bool = False,
) -> dict[str, Any]:
    return HealthResponse(
        status=status,
        version=version,
        api_version=api_version,
        models={"yolo": models_loaded, "classifier": models_loaded},
        database=database_ok,
        camera_available=camera_available,
    ).to_dict()


def serialize_status(
    snapshot: dict[str, object],
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Convert a CameraMonitor/MonitorSnapshot dict to a stable StatusResponse.

    *snapshot* is the dict returned by ``CameraMonitor.snapshot()``.
    """
    risk_percent = int(snapshot.get("riskPercent", 0))
    confidence_percent = int(snapshot.get("confidencePercent", 0))
    monitoring = bool(snapshot.get("running", False))
    raw_state = str(snapshot.get("state", "Idle"))
    model_state = _model_state_from_internal(raw_state)

    risk_score = clamp01(risk_percent / 100.0) if monitoring else 0.0
    confidence_score = clamp01(confidence_percent / 100.0) if monitoring else 0.0
    # A temporally confirmed state must not be paired with a contradictory
    # low percentage in the UI. These floors match the business FSM entry
    # thresholds; the raw score is still retained internally for analytics.
    if monitoring:
        if model_state == "Fall":
            risk_score = max(risk_score, 0.72)
        elif model_state == "Pre-fall":
            risk_score = max(risk_score, 0.45)

    prediction = PredictionDTO(
        state=model_state,
        alert_state=model_state,
        business_state=_business_state_from_model(model_state),
        risk_score=risk_score,
        visibility=confidence_score,
        confidence=confidence_score,
        system_status=snapshot.get("systemStatus"),
    )

    performance = PerformanceDTO(
        fps=max(0.0, float(snapshot.get("fps", 0.0))),
    )

    error_dto = _serialize_error(snapshot.get("error"))

    return StatusResponse(
        schema_version=schema_version,
        sequence=next_sequence(),
        timestamp_ms=int(time.time() * 1000),
        monitoring=monitoring,
        loading=bool(snapshot.get("loading", False)),
        prediction=prediction,
        performance=performance,
        error=error_dto,
    ).to_dict()


def serialize_monitor_command(
    ok: bool,
    monitoring: bool = False,
    session_id: str | None = None,
    error: Any = None,
) -> dict[str, Any]:
    err_dto = _serialize_error(error) if error else None
    return MonitorCommandResponse(
        ok=ok,
        monitoring=monitoring,
        session_id=session_id,
        error=err_dto,
    ).to_dict()


def serialize_import_job(snapshot: dict[str, object]) -> dict[str, Any]:
    """Convert a MediaImportProcessor snapshot dict to ImportJobResponse."""
    return ImportJobResponse(
        id=str(snapshot.get("id", "")),
        state=str(snapshot.get("state", "Idle")).lower(),
        progress=float(snapshot.get("progress", 0.0)),
        current_frame=int(snapshot.get("currentFrame", 0)),
        total_frames=int(snapshot.get("totalFrames", 0)),
        output_video=str(snapshot["outputVideo"]) if snapshot.get("outputVideo") else None,
        error=_serialize_error(snapshot.get("error")) if snapshot.get("error") else None,
    ).to_dict()


def serialize_settings(settings_obj: Any) -> dict[str, Any]:
    """Convert AppSettings to SettingsDTO."""
    thresholds = getattr(settings_obj, "thresholds", None)
    if callable(thresholds):
        thresholds = thresholds()

    return SettingsDTO(
        sensitivity=getattr(settings_obj, "sensitivity", "medium"),
        camera_index=int(getattr(settings_obj, "camera_index", 0)),
        theme=getattr(settings_obj, "theme", "system"),
        lang=getattr(settings_obj, "lang", "en"),
        sound_alert=bool(getattr(settings_obj, "sound_alert", True)),
        thresholds=dict(thresholds) if thresholds else {},
    ).to_dict()


def serialize_profile(profile: Any) -> dict[str, Any]:
    """Convert a UserProfile (or dict row) to ProfileDTO."""
    if hasattr(profile, "to_dict"):
        d = profile.to_dict()  # type: ignore[union-attr]
    elif isinstance(profile, dict):
        d = profile
    else:
        return {}

    return ProfileDTO(
        id=str(d.get("id", "")),
        name=str(d.get("name", "")),
        created_at=str(d.get("createdAt") or d.get("created_at", "")),
        fall_count=int(d.get("fallCount", 0)),
    ).to_dict()


def serialize_event(event_row: dict[str, Any]) -> dict[str, Any]:
    """Convert a database event row to EventDTO."""
    return EventDTO(
        id=str(event_row.get("id", "")),
        event_type=str(event_row.get("event_type", "")),
        status=str(event_row.get("status", "")),
        peak_risk=float(event_row.get("peak_risk", 0.0)),
        started_at=str(event_row.get("started_at", "")),
        ended_at=event_row.get("ended_at"),
        session_id=event_row.get("session_id"),
    ).to_dict()


def serialize_paginated(
    items: list[dict[str, Any]],
    next_cursor: str | None = None,
    has_more: bool = False,
) -> dict[str, Any]:
    return PaginatedResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    ).to_dict()


# ── internal helpers ────────────────────────────────────────────────


def _serialize_error(error_raw: Any) -> ServiceErrorDTO | None:
    """Convert an error string or ServiceError to a ServiceErrorDTO."""
    if error_raw is None:
        return None
    if hasattr(error_raw, "to_dict"):
        d = error_raw.to_dict()  # type: ignore[union-attr]
        return ServiceErrorDTO(
            code=str(d.get("code", "INTERNAL_ERROR")),
            message_key=str(d.get("message_key", "error.internal")),
            retryable=bool(d.get("retryable", False)),
            details=d.get("details"),
        )
    msg = str(error_raw)
    if not msg.strip():
        return None
    return ServiceErrorDTO(
        code="INTERNAL_ERROR",
        message_key="error.internal",
        details=msg,
    )
