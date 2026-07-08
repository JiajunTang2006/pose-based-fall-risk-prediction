from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CameraAttempt:
    index: int
    backend: str
    opened: bool
    frame_read: bool


@dataclass(frozen=True)
class CameraPermission:
    allowed: bool
    status: str


class CameraOpenError(RuntimeError):
    def __init__(self, attempts: list[CameraAttempt], permission: CameraPermission | None = None) -> None:
        self.attempts = attempts
        self.permission = permission
        if permission and not permission.allowed:
            detail = (
                "Camera access is denied by macOS. Open System Settings > Privacy & Security > Camera "
                "and allow FallGuard, then quit and reopen the app."
            )
            if permission.status == "not_determined":
                detail = "Camera permission was not granted. Please allow camera access when macOS asks."
            elif permission.status == "restricted":
                detail = "Camera access is restricted by macOS Screen Time, MDM, or another system policy."
            super().__init__(detail)
            return

        super().__init__(
            "Camera could not be opened. Allow camera access, close other camera apps, "
            f"and try again. Tried: {summarize_camera_attempts(attempts)}."
        )


def open_camera_capture(
    camera_index: int = 0,
    *,
    probe_indices: Iterable[int] = range(4),
    warmup_reads: int = 6,
):
    """Open a webcam with macOS-friendly backend fallback and frame probing."""
    import cv2

    permission = request_camera_permission()
    if not permission.allowed:
        raise CameraOpenError([], permission=permission)

    attempts: list[CameraAttempt] = []
    for index in _candidate_indices(camera_index, probe_indices):
        for backend_id, backend_name in _candidate_backends(cv2):
            capture = cv2.VideoCapture(index, backend_id)
            opened = bool(capture.isOpened())
            frame_read = _can_read_frame(capture, warmup_reads) if opened else False
            attempts.append(CameraAttempt(index=index, backend=backend_name, opened=opened, frame_read=frame_read))
            if opened and frame_read:
                return capture
            capture.release()

    raise CameraOpenError(attempts)


def request_camera_permission(timeout_seconds: float = 30.0) -> CameraPermission:
    if sys.platform != "darwin":
        return CameraPermission(allowed=True, status="unsupported")

    try:
        import objc

        objc.loadBundle(
            "AVFoundation",
            globals(),
            bundle_path="/System/Library/Frameworks/AVFoundation.framework",
        )
        _register_avfoundation_camera_metadata(objc)
        capture_device = objc.lookUpClass("AVCaptureDevice")
    except Exception:
        return CameraPermission(allowed=True, status="unknown")

    status_code = int(capture_device.authorizationStatusForMediaType_("vide"))
    status = _mac_camera_status_name(status_code)
    if status == "authorized":
        return CameraPermission(allowed=True, status=status)
    if status != "not_determined":
        return CameraPermission(allowed=False, status=status)

    decision: dict[str, bool] = {"allowed": False}
    event = threading.Event()

    def _handler(granted: bool) -> None:
        decision["allowed"] = bool(granted)
        event.set()

    # AVCaptureDevice.requestAccessForMediaType:completionHandler: must be called
    # from the main thread on macOS, otherwise the system permission dialog may
    # never appear and the completion handler may never fire.  When we are already
    # on the main thread we call it directly; otherwise we dispatch to the main
    # queue via PyObjC (the main run loop is always running in a pywebview /
    # PySide6 app).
    import threading as _threading

    def _request() -> None:
        try:
            capture_device.requestAccessForMediaType_completionHandler_("vide", _handler)
        except Exception:
            # If the request itself fails, treat it as an unknown state so the
            # caller will still try to open the camera.
            decision["allowed"] = True
            event.set()

    try:
        if _threading.current_thread() is _threading.main_thread():
            _request()
        else:
            from Foundation import dispatch_async, dispatch_get_main_queue

            dispatch_async(dispatch_get_main_queue(), _request)
    except Exception:
        # Fallback: try calling directly (may work in some configurations).
        _request()

    completed = event.wait(timeout=timeout_seconds)
    if not completed:
        return CameraPermission(allowed=False, status=status)
    return CameraPermission(allowed=decision["allowed"], status="authorized" if decision["allowed"] else "denied")


def _register_avfoundation_camera_metadata(objc) -> None:
    try:
        objc.registerMetaDataForSelector(
            b"AVCaptureDevice",
            b"requestAccessForMediaType:completionHandler:",
            {
                "arguments": {
                    3: {
                        "callable": {
                            "retval": {"type": objc._C_VOID},
                            "arguments": {
                                0: {"type": b"^v"},
                                1: {"type": objc._C_NSBOOL},
                            },
                        },
                        "callable_retained": True,
                    }
                }
            },
        )
    except Exception:
        return


def _mac_camera_status_name(status_code: int) -> str:
    return {
        0: "not_determined",
        1: "restricted",
        2: "denied",
        3: "authorized",
    }.get(status_code, f"unknown:{status_code}")


def summarize_camera_attempts(attempts: Iterable[CameraAttempt]) -> str:
    parts = []
    for attempt in attempts:
        status = "ok" if attempt.frame_read else "opened-no-frame" if attempt.opened else "not-opened"
        parts.append(f"camera {attempt.index} via {attempt.backend} ({status})")
    return ", ".join(parts) if parts else "no camera backends"


def _candidate_indices(camera_index: int, probe_indices: Iterable[int]) -> list[int]:
    indices: list[int] = []
    for index in (camera_index, *probe_indices):
        if index >= 0 and index not in indices:
            indices.append(index)
    return indices


def _candidate_backends(cv2) -> list[tuple[int, str]]:
    backends: list[tuple[int, str]] = []
    if sys.platform == "darwin":
        avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avfoundation is not None:
            backends.append((avfoundation, "AVFoundation"))
    backends.append((getattr(cv2, "CAP_ANY", 0), "default"))
    return backends


def _can_read_frame(capture, warmup_reads: int) -> bool:
    for _ in range(max(1, warmup_reads)):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        time.sleep(0.08)
    return False
