"""
HTTP request handler and server for the FallGuard AI Service.

All ``/api/v1/`` routes require Bearer-token authentication.  This module
provides the ``ServiceRequestHandler`` (per-request) and ``AIServiceServer``
(ThreadingHTTPServer subclass) that wires together the CameraMonitor,
MediaImportProcessor, database, profiles, and settings.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from email.message import Message
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlparse

from . import __version__
from .auth import validate_token
from .errors import (
    ServiceError,
    camera_in_use,
    import_conflict,
    internal_error,
    invalid_argument,
    monitor_already_running,
    monitor_not_running,
    not_found,
    unauthorized,
)
from .serialization import (
    serialize_event,
    serialize_health,
    serialize_import_job,
    serialize_monitor_command,
    serialize_paginated,
    serialize_profile,
    serialize_settings,
    serialize_status,
)

logger = logging.getLogger(__name__)

# Reuse the media constants from the desktop layer.
try:
    from fall_prediction_desktop.web_app import (
        IMAGE_EXTENSIONS,
        SUPPORTED_MEDIA_EXTENSIONS,
        VIDEO_EXTENSIONS,
        natural_media_sort_key,
    )
except ImportError:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
    SUPPORTED_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

    def natural_media_sort_key(value):
        import re
        return [
            int(p) if p.isdigit() else p
            for p in re.split(r"(\d+)", str(value).lower())
        ]


class ServiceRequestHandler(BaseHTTPRequestHandler):
    """Handler for one HTTP request — stateless beyond the shared server."""

    server: "AIServiceServer"

    # ── HTTP method dispatch ────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # Public endpoints (no auth required)
        if path == "/api/v1/health":
            self._handle_health()
            return

        # Authenticated endpoints
        if not self._check_auth():
            return

        if path == "/api/v1/status":
            self._handle_status()
        elif path == "/api/v1/settings":
            self._handle_get_settings()
        elif path == "/api/v1/cameras":
            self._handle_get_cameras()
        elif path == "/api/v1/profiles":
            self._handle_get_profiles()
        elif path == "/api/v1/events":
            self._handle_get_events(qs)
        elif path == "/api/v1/sessions":
            self._handle_get_sessions(qs)
        elif path == "/api/v1/preview.mjpg":
            self._serve_mjpeg_stream()
        elif path == "/api/v1/preview.jpg" or path == "/frame.jpg":
            self._serve_latest_frame()
        elif path.startswith("/api/v1/imports/"):
            import_id = path.split("/")[-1]
            self._handle_get_import(import_id)
        elif path.startswith("/api/v1/media/") and "/content" in path:
            media_id = path.split("/")[-2] if path.endswith("/content") else path.split("/")[-1]
            self._handle_media_content(media_id)
        else:
            self._send_error(not_found(f"Unknown endpoint: {path}"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._check_auth():
            return

        if path == "/api/v1/monitor/start":
            self._handle_monitor_start()
        elif path == "/api/v1/monitor/stop":
            self._handle_monitor_stop()
        elif path == "/api/v1/settings":
            self._handle_update_settings()
        elif path == "/api/v1/profiles":
            self._handle_create_profile()
        elif path.startswith("/api/v1/profiles/") and path.endswith("/activate"):
            profile_id = path.rsplit("/", 2)[-2]
            self._handle_activate_profile(profile_id)
        elif path == "/api/v1/imports":
            self._handle_create_import()
        elif path == "/api/v1/shutdown":
            self._handle_shutdown()
        else:
            self._send_error(not_found(f"Unknown endpoint: {path}"))

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._check_auth():
            return

        if path == "/api/v1/settings":
            self._handle_update_settings()
        elif path.startswith("/api/v1/profiles/") and not path.endswith("/activate"):
            profile_id = path.rsplit("/", 1)[-1]
            self._handle_update_profile(profile_id)
        else:
            self._send_error(not_found(f"Unknown endpoint: {path}"))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._check_auth():
            return

        if path.startswith("/api/v1/profiles/"):
            profile_id = path.rsplit("/", 1)[-1]
            self._handle_delete_profile(profile_id)
        else:
            self._send_error(not_found(f"Unknown endpoint: {path}"))

    def do_OPTIONS(self) -> None:
        """CORS preflight for local development."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    # ── auth ────────────────────────────────────────────────────────

    def _check_auth(self) -> bool:
        header = self.headers.get("Authorization")
        if validate_token(header, self.server.token):
            return True
        self._send_error(unauthorized(), status=HTTPStatus.UNAUTHORIZED)
        return False

    # ── health ──────────────────────────────────────────────────────

    def _handle_health(self) -> None:
        monitor = self.server.monitor
        # Models are loaded when the preload worker has finished (or was never started).
        preload_worker = getattr(monitor, "_preload_worker", None)
        worker_finished = preload_worker is None or not preload_worker.is_alive()
        models_loaded = (
            monitor is not None
            and worker_finished
            and not getattr(monitor, "_preload_error", "")
        )
        database_ok = self.server._repos is not None

        if not models_loaded:
            status = "starting"
        elif database_ok:
            status = "ready"
        else:
            status = "degraded"

        self._send_json(serialize_health(
            status=status,
            version=__version__,
            models_loaded=models_loaded,
            database_ok=database_ok,
        ))

    # ── status ──────────────────────────────────────────────────────

    def _handle_status(self) -> None:
        if self.server.monitor is None:
            self._send_error(internal_error("Monitor not initialised"))
            return
        snapshot = self.server.monitor.snapshot()
        # Include media job snapshot
        if self.server.media_processor is not None:
            snapshot["mediaJob"] = self.server.media_processor.snapshot()
        self._send_json(serialize_status(snapshot))

    # ── monitor start / stop ────────────────────────────────────────

    def _handle_monitor_start(self) -> None:
        monitor = self.server.monitor
        if monitor is None:
            self._send_error(internal_error("Monitor not available"))
            return

        snapshot = monitor.snapshot()
        if snapshot.get("running") or snapshot.get("loading"):
            self._send_json(
                serialize_monitor_command(ok=True, monitoring=True),
                status=HTTPStatus.OK,
            )
            return

        # Check for media import conflict
        if self.server.media_processor is not None:
            media_snap = self.server.media_processor.snapshot()
            if media_snap.get("running"):
                self._send_error(import_conflict(
                    "Stop the current media import before starting monitoring."
                ))
                return

        try:
            monitor.start()
            self._send_json(serialize_monitor_command(
                ok=True, monitoring=True,
                session_id=getattr(monitor, "_session_id", None),
            ))
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    def _handle_monitor_stop(self) -> None:
        monitor = self.server.monitor
        if monitor is None:
            self._send_error(internal_error("Monitor not available"))
            return

        snapshot = monitor.snapshot()
        if not snapshot.get("running") and not snapshot.get("loading"):
            self._send_json(
                serialize_monitor_command(ok=True, monitoring=False),
                status=HTTPStatus.OK,
            )
            return

        try:
            monitor.stop()
            self._send_json(serialize_monitor_command(ok=True, monitoring=False))
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    # ── settings ────────────────────────────────────────────────────

    def _handle_get_settings(self) -> None:
        s = self.server.settings
        self._send_json(serialize_settings(s))

    def _handle_update_settings(self) -> None:
        body = self._read_json_body()
        if body is None:
            return

        s = self.server.settings
        changed = False

        if "sensitivity" in body:
            from fall_prediction.sensitivity import normalize_sensitivity, SENSITIVITY_LEVELS
            level = normalize_sensitivity(str(body.get("sensitivity", "")))
            if level not in SENSITIVITY_LEVELS:
                self._send_error(invalid_argument(f"Unknown sensitivity: {level}"))
                return
            s.sensitivity = level
            changed = True

        if "camera_index" in body:
            index = body["camera_index"]
            if not isinstance(index, int) or index < 0:
                self._send_error(invalid_argument("camera_index must be a non-negative integer"))
                return
            s.camera_index = index
            changed = True

        if "theme" in body:
            theme = str(body["theme"])
            if theme not in ("light", "dark", "system"):
                self._send_error(invalid_argument("theme must be light, dark, or system"))
                return
            s.theme = theme
            changed = True

        if "lang" in body:
            lang = str(body["lang"])
            if lang not in ("en", "zh"):
                self._send_error(invalid_argument("lang must be en or zh"))
                return
            s.lang = lang
            changed = True

        if "sound_alert" in body:
            s.sound_alert = bool(body["sound_alert"])
            changed = True

        if changed:
            try:
                from fall_prediction_desktop.web_app import save_settings
            except ImportError:
                from fall_prediction_desktop.web_app import save_settings  # type: ignore[no-redef]
            save_settings(self.server.app_root, s)
            if self.server.monitor is not None:
                self.server.monitor.update_settings(s)
            if self.server.media_processor is not None:
                self.server.media_processor.update_settings(s)

        self._send_json(serialize_settings(s))

    # ── cameras ─────────────────────────────────────────────────────

    def _handle_get_cameras(self) -> None:
        try:
            from fall_prediction_desktop.web_app import scan_camera_indices
        except ImportError:
            from fall_prediction_desktop.web_app import scan_camera_indices  # type: ignore[no-redef]
        try:
            available = scan_camera_indices()
        except Exception:
            available = [0]
        self._send_json({
            "cameras": available,
            "current": self.server.settings.camera_index,
        })

    # ── profiles ────────────────────────────────────────────────────

    def _handle_get_profiles(self) -> None:
        pm = self.server.profile_manager
        if pm is None:
            self._send_error(internal_error("Profile manager not available"))
            return
        self._send_json(pm.snapshot())

    def _handle_create_profile(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        name = str(body.get("name", "")).strip()
        if not name:
            self._send_error(invalid_argument("Profile name is required"))
            return
        pm = self.server.profile_manager
        if pm is None:
            self._send_error(internal_error("Profile manager not available"))
            return
        try:
            profile = pm.create(name)
            self._send_json({"ok": True, "profile": serialize_profile(profile)})
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    def _handle_activate_profile(self, profile_id: str) -> None:
        pm = self.server.profile_manager
        if pm is None:
            self._send_error(internal_error("Profile manager not available"))
            return
        ok = pm.activate(profile_id)
        if not ok:
            self._send_error(not_found("Profile not found"))
            return
        self._send_json({"ok": True, "activeId": profile_id})

    def _handle_update_profile(self, profile_id: str) -> None:
        body = self._read_json_body()
        if body is None:
            return
        pm = self.server.profile_manager
        if pm is None:
            self._send_error(internal_error("Profile manager not available"))
            return
        name = str(body.get("name", "")).strip()
        if not name:
            self._send_error(invalid_argument("Profile name is required"))
            return
        if profile_id not in pm.profiles:
            self._send_error(not_found("Profile not found"))
            return
        pm.profiles[profile_id].name = name
        pm._save()
        self._send_json({"ok": True, "profile": serialize_profile(pm.profiles[profile_id])})

    def _handle_delete_profile(self, profile_id: str) -> None:
        pm = self.server.profile_manager
        if pm is None:
            self._send_error(internal_error("Profile manager not available"))
            return
        ok = pm.delete(profile_id)
        if not ok:
            self._send_error(invalid_argument("Cannot delete the last profile"))
            return
        self._send_json({"ok": True})

    # ── events ──────────────────────────────────────────────────────

    def _handle_get_events(self, qs: dict[str, list[str]]) -> None:
        repos = self.server._repos
        if repos is None:
            self._send_error(internal_error("Database not available"))
            return

        limit = min(int(qs.get("limit", ["50"])[0]), 200)
        cursor = qs.get("cursor", [None])[0]
        profile_id = qs.get("profile_id", [None])[0]

        try:
            # Decode cursor: base64(created_at|id)
            after_timestamp: str | None = None
            after_id: str | None = None
            if cursor:
                import base64
                try:
                    decoded = base64.urlsafe_b64decode(cursor).decode("utf-8")
                    parts = decoded.split("|", 1)
                    after_timestamp = parts[0] if parts[0] else None
                    after_id = parts[1] if len(parts) > 1 else None
                except Exception:
                    pass

            # Use the events repository for pagination
            rows = repos.events.list_recent(limit)  # fallback: recent events
            items = [serialize_event(dict(r) if hasattr(r, "keys") else r) for r in rows]

            next_cursor = None
            has_more = len(items) >= limit
            if has_more and items:
                import base64
                last = items[-1]
                cursor_raw = f"{last.get('started_at', '')}|{last.get('id', '')}"
                next_cursor = base64.urlsafe_b64encode(cursor_raw.encode()).decode()

            self._send_json(serialize_paginated(items, next_cursor, has_more))
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    # ── sessions ────────────────────────────────────────────────────

    def _handle_get_sessions(self, qs: dict[str, list[str]]) -> None:
        repos = self.server._repos
        if repos is None:
            self._send_error(internal_error("Database not available"))
            return

        limit = min(int(qs.get("limit", ["50"])[0]), 200)
        try:
            rows = repos.sessions.list_recent(limit)
            items = []
            for r in rows:
                d = dict(r) if hasattr(r, "keys") else r
                items.append({
                    "id": str(d.get("id", "")),
                    "profile_id": str(d.get("profile_id", "")),
                    "source_type": str(d.get("source_type", "")),
                    "status": str(d.get("status", "")),
                    "total_frames": int(d.get("total_frames", 0)),
                    "total_events": int(d.get("total_events", 0)),
                    "peak_risk": float(d.get("peak_risk", 0.0)),
                    "started_at": str(d.get("started_at", "")),
                    "ended_at": d.get("ended_at"),
                })
            self._send_json(serialize_paginated(items))
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    # ── imports ─────────────────────────────────────────────────────

    def _handle_create_import(self) -> None:
        monitor = self.server.monitor
        if monitor is not None:
            mon_snap = monitor.snapshot()
            if mon_snap.get("running") or mon_snap.get("loading"):
                self._send_error(import_conflict(
                    "Stop live monitoring before importing media."
                ))
                return

        body = self._read_json_body()
        if body is None:
            return

        paths_raw = body.get("paths", [])
        if not paths_raw:
            self._send_error(invalid_argument("paths is required"))
            return

        output_dir_raw = body.get("output_directory")
        sensitivity = str(body.get("sensitivity", self.server.settings.sensitivity))

        paths = [Path(p).expanduser() for p in paths_raw]
        output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else None

        # Validate paths exist
        for p in paths:
            if not p.exists():
                self._send_error(invalid_argument(f"Path does not exist: {p}"))
                return
            suffix = p.suffix.lower()
            if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
                self._send_error(invalid_argument(f"Unsupported format: {suffix}"))

        processor = self.server.media_processor
        if processor is None:
            self._send_error(internal_error("Media processor not available"))
            return

        # Temporarily set sensitivity
        orig_sensitivity = self.server.settings.sensitivity
        self.server.settings.sensitivity = sensitivity
        processor.update_settings(self.server.settings)

        try:
            job = processor.start_from_paths(paths, output_dir=output_dir)
            self._send_json({"ok": True, "import": serialize_import_job(job)},
                            status=HTTPStatus.ACCEPTED)
        except Exception as exc:
            self.server.settings.sensitivity = orig_sensitivity
            processor.update_settings(self.server.settings)
            self._send_error(internal_error(str(exc)))

    def _handle_get_import(self, import_id: str) -> None:
        processor = self.server.media_processor
        if processor is None:
            self._send_error(internal_error("Media processor not available"))
            return
        job = processor.snapshot()
        self._send_json(serialize_import_job(job))

    # ── media content ───────────────────────────────────────────────

    def _handle_media_content(self, media_id: str) -> None:
        repos = self.server._repos
        if repos is None:
            self._send_error(internal_error("Database not available"))
            return

        try:
            media = repos.media.get(media_id)
            if media is None:
                self._send_error(not_found("Media not found"))
                return

            file_path = Path(media.get("file_path", ""))
            # Security: verify the file is within allowed media directories
            allowed_roots = [
                self.server.monitor.output_dir if self.server.monitor else None,
                getattr(self.server.media_processor, "output_dir", None) if self.server.media_processor else None,
            ]
            allowed_roots = [r for r in allowed_roots if r is not None]
            allowed = any(
                root.resolve() in file_path.resolve().parents
                or file_path.resolve().is_relative_to(root.resolve())
                for root in allowed_roots
            )
            if not allowed and not file_path.is_relative_to(Path.home() / "Movies" / "FallGuard"):
                self._send_error(internal_error("Access denied"))
                return

            if not file_path.is_file():
                self._send_error(not_found("Media file not found"))
                return

            content_type = "video/mp4" if file_path.suffix.lower() in VIDEO_EXTENSIONS else "image/jpeg"
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._send_error(internal_error(str(exc)))

    # ── preview / MJPEG ─────────────────────────────────────────────

    def _serve_latest_frame(self) -> None:
        frame = self.server.monitor.jpeg_frame() if self.server.monitor else None
        if frame is None:
            self.send_error(HTTPStatus.NO_CONTENT, "No frame available yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(frame)

    def _serve_mjpeg_stream(self) -> None:
        monitor = self.server.monitor
        if monitor is None:
            self._send_error(internal_error("Monitor not available"))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()

        while True:
            snap = monitor.snapshot()
            if not snap.get("running"):
                break
            frame = monitor.jpeg_frame()
            if frame is None:
                time.sleep(0.12)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                break

    # ── shutdown ────────────────────────────────────────────────────

    def _handle_shutdown(self) -> None:
        self._send_json({"ok": True, "message": "Shutting down"})
        # Shutdown in a separate thread so the response is sent first.
        def _do_shutdown() -> None:
            self.server.lifecycle.request_shutdown()

        threading.Thread(target=_do_shutdown, daemon=True).start()

    # ── helpers ─────────────────────────────────────────────────────

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error(invalid_argument(f"Invalid JSON: {exc}"))
            return None

    def _send_json(
        self,
        value: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(
        self,
        error: ServiceError,
        status: HTTPStatus | None = None,
    ) -> None:
        http_status = status or error.http_status()
        body = json.dumps({"error": error.to_dict()}, ensure_ascii=False).encode("utf-8")
        self.send_response(http_status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self) -> None:
        """Allow only explicit loopback origins for browser-based dev tools."""
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme in {"http", "https"} and parsed.hostname in {
                "127.0.0.1", "localhost", "::1"
            }:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default stderr logging — we use Python logging instead."""
        logger.debug(format, *args)


class AIServiceServer(ThreadingHTTPServer):
    """Threading HTTP server for the FallGuard AI Service.

    Binds to ``127.0.0.1`` with the requested port (0 = OS-assigned).
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        token: str,
        app_root: Path,
        monitor: Any,
        media_processor: Any,
        settings: Any,
        profile_manager: Any,
        repos: Any,
        lifecycle: Any,
    ) -> None:
        super().__init__(address, ServiceRequestHandler)
        self.token = token
        self.app_root = app_root
        self.monitor = monitor
        self.media_processor = media_processor
        self.settings = settings
        self.profile_manager = profile_manager
        self._repos = repos
        self.lifecycle = lifecycle
        self.base_url = f"http://{address[0]}:{address[1]}"
