"""
Stable error codes for the FallGuard AI Service API.

Every error response uses the same envelope so Swift can map codes to
localised messages without parsing free-form text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

# ── error code constants ────────────────────────────────────────────

UNAUTHORIZED = "UNAUTHORIZED"
SERVICE_NOT_READY = "SERVICE_NOT_READY"
CAMERA_PERMISSION_DENIED = "CAMERA_PERMISSION_DENIED"
CAMERA_IN_USE = "CAMERA_IN_USE"
MONITOR_ALREADY_RUNNING = "MONITOR_ALREADY_RUNNING"
MONITOR_NOT_RUNNING = "MONITOR_NOT_RUNNING"
IMPORT_CONFLICT = "IMPORT_CONFLICT"
INVALID_ARGUMENT = "INVALID_ARGUMENT"
MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
DATABASE_ERROR = "DATABASE_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"
NOT_FOUND = "NOT_FOUND"

# ── code → HTTP status mapping ──────────────────────────────────────

_CODE_STATUS: dict[str, HTTPStatus] = {
    UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
    SERVICE_NOT_READY: HTTPStatus.SERVICE_UNAVAILABLE,
    CAMERA_PERMISSION_DENIED: HTTPStatus.FORBIDDEN,
    CAMERA_IN_USE: HTTPStatus.CONFLICT,
    MONITOR_ALREADY_RUNNING: HTTPStatus.OK,
    MONITOR_NOT_RUNNING: HTTPStatus.OK,
    IMPORT_CONFLICT: HTTPStatus.CONFLICT,
    INVALID_ARGUMENT: HTTPStatus.BAD_REQUEST,
    MODEL_LOAD_FAILED: HTTPStatus.INTERNAL_SERVER_ERROR,
    DATABASE_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    NOT_FOUND: HTTPStatus.NOT_FOUND,
}


@dataclass
class ServiceError:
    """Stable error representation for API responses."""

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

    def http_status(self) -> HTTPStatus:
        return _CODE_STATUS.get(self.code, HTTPStatus.INTERNAL_SERVER_ERROR)


# ── convenience constructors ────────────────────────────────────────

def unauthorized(details: Any = None) -> ServiceError:
    return ServiceError(UNAUTHORIZED, "error.auth.unauthorized", details=details)


def service_not_ready(details: Any = None) -> ServiceError:
    return ServiceError(
        SERVICE_NOT_READY, "error.service.not_ready", retryable=True, details=details
    )


def camera_permission_denied(details: Any = None) -> ServiceError:
    return ServiceError(
        CAMERA_PERMISSION_DENIED, "error.camera.permission_denied", details=details
    )


def camera_in_use(details: Any = None) -> ServiceError:
    return ServiceError(
        CAMERA_IN_USE, "error.camera.in_use", retryable=True, details=details
    )


def monitor_already_running() -> ServiceError:
    return ServiceError(
        MONITOR_ALREADY_RUNNING, "error.monitor.already_running"
    )


def monitor_not_running() -> ServiceError:
    return ServiceError(MONITOR_NOT_RUNNING, "error.monitor.not_running")


def import_conflict(details: Any = None) -> ServiceError:
    return ServiceError(
        IMPORT_CONFLICT, "error.import.conflict", details=details
    )


def invalid_argument(details: Any = None) -> ServiceError:
    return ServiceError(
        INVALID_ARGUMENT, "error.invalid_argument", details=details
    )


def model_load_failed(details: Any = None) -> ServiceError:
    return ServiceError(
        MODEL_LOAD_FAILED, "error.model.load_failed", details=details
    )


def database_error(details: Any = None) -> ServiceError:
    return ServiceError(
        DATABASE_ERROR, "error.database", details=details
    )


def internal_error(details: Any = None) -> ServiceError:
    return ServiceError(INTERNAL_ERROR, "error.internal", details=details)


def not_found(details: Any = None) -> ServiceError:
    return ServiceError(NOT_FOUND, "error.not_found", details=details)
