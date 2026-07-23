"""
Stable contract types for the ``/api/v1`` service layer.

These DTOs define the exact JSON shapes that Swift expects.  Internal Python
objects (``Prediction``, ``DashboardSnapshot``, etc.) are never serialised
directly — they are mapped through the functions in ``serialization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── enum sets ───────────────────────────────────────────────────────

MODEL_STATES = {"Normal", "Pre-fall", "Fall", "Unknown"}
BUSINESS_STATES = {"safe", "warning", "danger", "unknown"}
SERVICE_HEALTH_STATES = {"starting", "ready", "degraded"}


# ── API DTOs ────────────────────────────────────────────────────────

@dataclass
class PredictionDTO:
    state: str           # "Normal" | "Pre-fall" | "Fall" | "Unknown"
    alert_state: str     # same as state for v1
    business_state: str  # "safe" | "warning" | "danger" | "unknown"
    risk_score: float    # 0.0–1.0
    visibility: float    # 0.0–1.0
    confidence: float    # 0.0–1.0
    system_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "alert_state": self.alert_state,
            "business_state": self.business_state,
            "risk_score": self.risk_score,
            "visibility": self.visibility,
            "confidence": self.confidence,
            "system_status": self.system_status,
        }


@dataclass
class PerformanceDTO:
    fps: float
    frame_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"fps": self.fps, "frame_index": self.frame_index}


@dataclass
class ServiceErrorDTO:
    code: str
    message_key: str
    retryable: bool = False
    details: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message_key": self.message_key,
            "retryable": self.retryable,
        }
        if self.details is not None:
            result["details"] = self.details
        return result


@dataclass
class StatusResponse:
    schema_version: int = 1
    sequence: int = 0
    timestamp_ms: int = 0
    monitoring: bool = False
    loading: bool = False
    prediction: PredictionDTO | None = None
    performance: PerformanceDTO | None = None
    error: ServiceErrorDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "monitoring": self.monitoring,
            "loading": self.loading,
        }
        if self.prediction is not None:
            result["prediction"] = self.prediction.to_dict()
        if self.performance is not None:
            result["performance"] = self.performance.to_dict()
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass
class HealthResponse:
    status: str  # "starting" | "ready" | "degraded"
    version: str
    api_version: str
    models: dict[str, bool] = field(default_factory=dict)
    database: bool = False
    camera_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "api_version": self.api_version,
            "models": self.models,
            "database": self.database,
            "camera_available": self.camera_available,
        }


@dataclass
class MonitorCommandResponse:
    ok: bool
    monitoring: bool = False
    session_id: str | None = None
    error: ServiceErrorDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": self.ok, "monitoring": self.monitoring}
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass
class ImportJobResponse:
    id: str
    state: str  # "running" | "complete" | "error"
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    output_video: str | None = None
    error: ServiceErrorDTO | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
        }
        if self.output_video is not None:
            result["output_video"] = self.output_video
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass
class SettingsDTO:
    sensitivity: str = "medium"
    camera_index: int = 0
    theme: str = "system"
    lang: str = "en"
    sound_alert: bool = True
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity": self.sensitivity,
            "camera_index": self.camera_index,
            "theme": self.theme,
            "lang": self.lang,
            "sound_alert": self.sound_alert,
            "thresholds": self.thresholds,
        }


@dataclass
class ProfileDTO:
    id: str
    name: str
    created_at: str
    fall_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "createdAt": self.created_at,
            "fallCount": self.fall_count,
        }


@dataclass
class EventDTO:
    id: str
    event_type: str
    status: str
    peak_risk: float
    started_at: str
    ended_at: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "event_type": self.event_type,
            "status": self.status,
            "peak_risk": self.peak_risk,
            "started_at": self.started_at,
        }
        if self.ended_at is not None:
            result["ended_at"] = self.ended_at
        if self.session_id is not None:
            result["session_id"] = self.session_id
        return result


@dataclass
class PaginatedResponse:
    items: list[dict[str, Any]]
    next_cursor: str | None = None
    has_more: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }
