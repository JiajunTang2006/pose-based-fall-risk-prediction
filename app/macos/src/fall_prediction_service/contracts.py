from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MODEL_STATES = {"Normal", "Pre-fall", "Fall", "Unknown"}
BUSINESS_STATES = {"safe", "warning", "danger", "unknown"}
SERVICE_HEALTH_STATES = {"starting", "ready", "degraded"}


@dataclass
class PredictionDTO:
    state: str
    alert_state: str
    business_state: str
    risk_score: float
    visibility: float
    confidence: float
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
    active_event_id: str | None = None
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
        if self.active_event_id is not None:
            result["active_event_id"] = self.active_event_id
        if self.performance is not None:
            result["performance"] = self.performance.to_dict()
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass
class HealthResponse:
    status: str
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
    state: str
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
    avg_risk: float = 0.0
    duration_seconds: float = 0.0
    thumbnail_path: str | None = None
    video_clip_path: str | None = None
    clip_fps: float | None = None
    clip_duration_seconds: float | None = None
    user_feedback: str | None = None
    annotation_label: str | None = None
    prefall_start_seconds: float | None = None
    fall_start_seconds: float | None = None
    notes: str | None = None

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
        result["avg_risk"] = self.avg_risk
        result["duration_seconds"] = self.duration_seconds
        if self.thumbnail_path is not None:
            result["thumbnail_path"] = self.thumbnail_path
        if self.video_clip_path is not None:
            result["video_clip_path"] = self.video_clip_path
        if self.clip_fps is not None:
            result["clip_fps"] = self.clip_fps
        if self.clip_duration_seconds is not None:
            result["clip_duration_seconds"] = self.clip_duration_seconds
        if self.user_feedback is not None:
            result["user_feedback"] = self.user_feedback
        if self.annotation_label is not None:
            result["annotation_label"] = self.annotation_label
        if self.prefall_start_seconds is not None:
            result["prefall_start_seconds"] = self.prefall_start_seconds
        if self.fall_start_seconds is not None:
            result["fall_start_seconds"] = self.fall_start_seconds
        if self.notes is not None:
            result["notes"] = self.notes
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
