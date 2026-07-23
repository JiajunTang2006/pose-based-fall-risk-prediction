from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import Message
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.parse import urlparse

from fall_prediction.sensitivity import (
    DEFAULT_SENSITIVITY,
    SENSITIVITY_LEVELS,
    ml_config_for_sensitivity,
    normalize_sensitivity,
    predictor_config_for_sensitivity,
    sensitivity_thresholds,
)

# Relative imports work when running as `python -m fall_prediction_desktop`.
# Absolute imports work inside a PyInstaller bundle where relative imports fail.
try:
    from .runner import PredictionJob, ensure_repo_on_path, find_app_root, run_prediction_job, safe_filename
    from .paths import media_output_dir, user_data_dir
except ImportError:
    from fall_prediction_desktop.runner import PredictionJob, ensure_repo_on_path, find_app_root, run_prediction_job, safe_filename  # type: ignore[no-redef]
    from fall_prediction_desktop.paths import media_output_dir, user_data_dir  # type: ignore[no-redef]


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
SUPPORTED_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

SENSITIVITY_THRESHOLDS = {
    level: sensitivity_thresholds(level)
    for level in SENSITIVITY_LEVELS
}

SETTINGS_FILENAME = "fallguard_settings.json"
logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    sensitivity: str = DEFAULT_SENSITIVITY   # "low" | "medium" | "high"
    camera_index: int = 0
    theme: str = "system"                    # "light" | "dark" | "system"
    lang: str = "en"                         # "en" | "zh"
    sound_alert: bool = True
    popup_alert: bool = False
    start_on_boot: bool = False
    minimize_to_tray: bool = False

    def thresholds(self) -> dict[str, float]:
        return sensitivity_thresholds(self.sensitivity)


def load_settings(app_root: Path) -> AppSettings:
    # Read the canonical user-data file first, then the legacy app-root file
    # solely for one-time migration from older source builds.
    for path in (user_data_dir() / SETTINGS_FILENAME, app_root / SETTINGS_FILENAME):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AppSettings(
                sensitivity=normalize_sensitivity(data.get("sensitivity")),
                camera_index=int(data.get("camera_index", 0)),
                theme=data.get("theme", "system"),
                lang=data.get("lang", "en"),
                sound_alert=bool(data.get("sound_alert", True)),
                popup_alert=bool(data.get("popup_alert", False)),
                start_on_boot=bool(data.get("start_on_boot", False)),
                minimize_to_tray=bool(data.get("minimize_to_tray", False)),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return AppSettings()


def save_settings(app_root: Path, settings: AppSettings) -> None:
    values = {
        "sensitivity": normalize_sensitivity(settings.sensitivity),
        "camera_index": str(settings.camera_index),
        "theme": settings.theme,
        "language": settings.lang,
        "sound_alert": str(settings.sound_alert).lower(),
        "popup_alert": str(settings.popup_alert).lower(),
        "start_on_boot": str(settings.start_on_boot).lower(),
        "minimize_to_tray": str(settings.minimize_to_tray).lower(),
    }

    # SQLite is the source of truth.  The JSON write below is retained only
    # for migration compatibility with older builds and may be unwritable
    # inside a signed .app bundle.
    try:
        from .database.database import get_database
    except ImportError:
        from fall_prediction_desktop.database.database import get_database  # type: ignore[no-redef]
    try:
        db = get_database()
        try:
            from .database.repositories import ProfilesRepository, SettingsRepository
        except ImportError:
            from fall_prediction_desktop.database.repositories import ProfilesRepository, SettingsRepository  # type: ignore[no-redef]
        repo = SettingsRepository(db)
        for key, value in values.items():
            repo.set(key, value)
        profiles = ProfilesRepository(db)
        active = profiles.get_active()
        if active is not None:
            profiles.update(
                active["id"],
                sensitivity=values["sensitivity"],
                camera_index=settings.camera_index,
            )
    except Exception as exc:
        logger.debug("Settings database persistence unavailable: %s", exc)

    path = user_data_dir() / SETTINGS_FILENAME
    try:
        path.write_text(
            json.dumps({
                "sensitivity": values["sensitivity"],
                "camera_index": settings.camera_index,
                "theme": settings.theme,
                "lang": settings.lang,
                "sound_alert": settings.sound_alert,
                "popup_alert": settings.popup_alert,
                "start_on_boot": settings.start_on_boot,
                "minimize_to_tray": settings.minimize_to_tray,
            }, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # Non-critical — settings just won't persist this session


def scan_camera_indices() -> list[int]:
    """Return only the built-in Mac camera index to avoid activating iPhone Continuity Camera.

    NOTE: This function deliberately returns a hard-coded [0] rather than
    enumerating all available camera devices (e.g. via AVFoundation).
    Enumerating cameras on macOS can trigger iPhone Continuity Camera
    activation prompts, which is disruptive and slow.  Users who need a
    different camera index can change it in settings.
    """
    return [0]
SINGLE_IMAGE_REPEAT_FRAMES = 24


# ---- Profile system ----

PROFILES_FILENAME = "profiles.json"


@dataclass
class FallEvent:
    timestamp: str       # ISO 8601 with timezone
    risk_score: int      # 0-100
    state: str           # "Pre-fall" or "Fall"
    detail: str          # Human-readable description


@dataclass
class UserProfile:
    id: str
    name: str
    created_at: str
    fall_events: list = field(default_factory=list)

    def fall_count(self) -> int:
        return len([e for e in self.fall_events if e.get("state") == "Fall"])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "createdAt": self.created_at,
            "fallEvents": self.fall_events,
            "fallCount": self.fall_count(),
        }


class ProfileManager:
    def __init__(self, app_root: Path, data_dir: Path | None = None, repository=None) -> None:
        self._path = (data_dir or user_data_dir()) / PROFILES_FILENAME
        self._legacy_path = app_root / PROFILES_FILENAME
        self._lock = threading.RLock()
        self._repository = repository
        self.profiles: dict[str, UserProfile] = {}
        self.active_id: str | None = None
        self._load()
        if self._repository is not None:
            self._sync_repository()
        elif not self.profiles:
            self.create("Default")

    def _sync_repository(self) -> None:
        """Migrate legacy profiles, then reload SQLite as the source of truth."""
        legacy_profiles = dict(self.profiles)
        legacy_active_id = self.active_id
        for profile in legacy_profiles.values():
            self._repository.upsert(
                profile.id,
                profile.name,
                is_active=profile.id == legacy_active_id,
            )

        rows = self._repository.list_all()
        if not rows:
            row = self._repository.create("Default")
            rows = [row]
        by_name = {profile.name: profile for profile in legacy_profiles.values()}
        self.profiles = {}
        for row in rows:
            cached = legacy_profiles.get(row["id"]) or by_name.get(row["name"])
            self.profiles[row["id"]] = UserProfile(
                id=row["id"],
                name=row["name"],
                created_at=row["created_at"],
                fall_events=list(cached.fall_events) if cached else [],
            )
        active = self._repository.get_active()
        self.active_id = active["id"] if active else rows[0]["id"]
        if active is None:
            self._repository.activate(self.active_id)
        self._save()

    def _load(self) -> None:
        for path in (self._path, self._legacy_path):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.active_id = data.get("active_id")
                for pid, pd in data.get("profiles", {}).items():
                    self.profiles[pid] = UserProfile(
                        id=pd["id"],
                        name=pd["name"],
                        created_at=pd.get("created_at") or pd.get("createdAt", ""),
                        fall_events=pd.get("fall_events") or pd.get("fallEvents", []),
                    )
                if self.profiles:
                    if path == self._legacy_path:
                        self._save()
                    return
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
        self.profiles = {}
        self.active_id = None

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({
                "active_id": self.active_id,
                "profiles": {
                    pid: {
                        "id": p.id,
                        "name": p.name,
                        "created_at": p.created_at,
                        "fall_events": p.fall_events,
                    }
                    for pid, p in self.profiles.items()
                },
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def create(self, name: str) -> UserProfile:
        with self._lock:
            if self._repository is not None:
                row = self._repository.create(name)
                pid = row["id"]
                now = row["created_at"]
            else:
                pid = uuid.uuid4().hex[:12]
                now = datetime.now(timezone.utc).isoformat()
            profile = UserProfile(id=pid, name=name.strip() or "Unnamed", created_at=now)
            self.profiles[pid] = profile
            if self.active_id is None:
                self.active_id = pid
            self._save()
            return profile

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            if profile_id not in self.profiles:
                return False
            if len(self.profiles) <= 1:
                return False  # Keep at least one profile
            if self._repository is not None and not self._repository.delete(profile_id):
                return False
            del self.profiles[profile_id]
            if self.active_id == profile_id:
                if self._repository is not None:
                    active = self._repository.get_active()
                    self.active_id = active["id"] if active else next(iter(self.profiles.keys()))
                else:
                    self.active_id = next(iter(self.profiles.keys()))
            self._save()
            return True

    def activate(self, profile_id: str) -> bool:
        with self._lock:
            if profile_id not in self.profiles:
                return False
            if self._repository is not None and not self._repository.activate(profile_id):
                return False
            self.active_id = profile_id
            self._save()
            return True

    def record_fall(self, state: str, risk: int, detail: str) -> None:
        with self._lock:
            if self.active_id is None or self.active_id not in self.profiles:
                return
            profile = self.profiles[self.active_id]
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_score": risk,
                "state": state,
                "detail": detail,
            }
            profile.fall_events.append(event)
            self._save()

    @property
    def active(self) -> UserProfile | None:
        with self._lock:
            if self.active_id is None:
                return None
            return self.profiles.get(self.active_id)

    def list_all(self) -> list[UserProfile]:
        with self._lock:
            return list(self.profiles.values())

    def snapshot(self) -> dict:
        with self._lock:
            active = self.profiles.get(self.active_id) if self.active_id else None
            return {
                "profiles": [p.to_dict() for p in self.profiles.values()],
                "activeId": self.active_id,
                "activeProfile": active.to_dict() if active else None,
            }


@dataclass
class ActivityEvent:
    level: str
    title: str
    time: str
    risk: int


@dataclass
class MonitorSnapshot:
    running: bool = False
    loading: bool = False
    camera_connected: bool = False
    model_active: bool = False
    state: str = "Idle"
    title: str = "Ready"
    detail: str = "Click Start Monitoring to begin."
    risk_percent: int = 0
    confidence_percent: int = 0
    fps: float = 0.0
    resolution: str = "--"
    environment: str = "Waiting"
    error: str = ""
    started_at: str = ""
    activities: list[ActivityEvent] = field(default_factory=list)


@dataclass
class MediaJobSnapshot:
    running: bool = False
    state: str = "Idle"
    title: str = "Ready"
    detail: str = "Choose a video, photo, or photo folder to analyze."
    input_name: str = ""
    output_video: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""


@dataclass
class UploadedMediaFile:
    filename: str
    stream: BinaryIO


def writable_output_root(app_root: Path) -> Path:
    return media_output_dir()


class MediaImportProcessor:
    def __init__(self, app_root: Path, settings: AppSettings | None = None) -> None:
        self.app_root = app_root
        self.settings = settings or AppSettings()
        self.output_dir = writable_output_root(app_root) / "imported_media"
        self.upload_dir = self.output_dir / "sources"
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._capture = None
        self._snapshot = MediaJobSnapshot()
        self._repos = None  # AppRepositories — attached by the application bootstrap

    def update_settings(self, settings: AppSettings) -> None:
        with self._lock:
            self.settings = settings

    def _current_sensitivity(self) -> str:
        with self._lock:
            return normalize_sensitivity(self.settings.sensitivity)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            snapshot = self._snapshot
            result = {
                "running": snapshot.running,
                "state": snapshot.state,
                "title": snapshot.title,
                "detail": snapshot.detail,
                "inputName": snapshot.input_name,
                "outputVideo": snapshot.output_video,
                "error": snapshot.error,
                "startedAt": snapshot.started_at,
                "finishedAt": snapshot.finished_at,
            }
        return result

    def start_from_upload(self, filename: str, stream: BinaryIO) -> dict[str, object]:
        return self.start_from_uploads([UploadedMediaFile(filename=filename, stream=stream)])

    def start_from_paths(self, paths: list[Path], output_dir: Path | None = None) -> dict[str, object]:
        if not paths:
            raise ValueError("Please choose a video, photo, or photo folder.")

        media_kind, media_paths = self._detect_path_media_kind(paths)
        display_name = self._display_name_for_paths(paths, media_paths, media_kind)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = safe_filename(self._source_stem(display_name, media_kind))
        upload_folder = self.upload_dir / timestamp
        source_path = media_paths[0] if media_kind == "video" else upload_folder / safe_stem

        with self._lock:
            if self._snapshot.running or (self._worker and self._worker.is_alive()):
                raise RuntimeError("Media is already being processed. Please wait for it to finish.")
            self._snapshot = MediaJobSnapshot(
                running=True,
                state="Uploading",
                title="Importing Media",
                detail=f"Preparing {display_name}.",
                input_name=display_name,
                started_at=datetime.now().strftime("%H:%M:%S"),
            )

        try:
            upload_folder.mkdir(parents=True, exist_ok=True)
            if media_kind == "images":
                source_path.mkdir(parents=True, exist_ok=True)
                self._copy_local_image_sequence(media_paths, source_path)
        except Exception as exc:
            self._update(
                running=False,
                state="Error",
                title="Import Failed",
                detail=str(exc),
                error=str(exc),
                finished_at=datetime.now().strftime("%H:%M:%S"),
            )
            raise

        worker = threading.Thread(
            target=self._run_media_job,
            args=(source_path, display_name, media_kind, output_dir),
            daemon=True,
        )
        with self._lock:
            self._worker = worker
            self._snapshot.state = "Processing"
            self._snapshot.title = "Processing Media"
            self._snapshot.detail = "Analyzing frames and preparing the annotated MP4."
        worker.start()
        return self.snapshot()

    def start_from_uploads(self, uploads: list[UploadedMediaFile], output_dir: Path | None = None) -> dict[str, object]:
        if not uploads:
            raise ValueError("Please choose a video, photo, or photo folder.")

        media_kind = self._detect_media_kind(uploads)
        display_name = self._display_name(uploads, media_kind)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = safe_filename(self._source_stem(display_name, media_kind))
        upload_folder = self.upload_dir / timestamp
        source_path = upload_folder / safe_stem

        with self._lock:
            if self._snapshot.running or (self._worker and self._worker.is_alive()):
                raise RuntimeError("Media is already being processed. Please wait for it to finish.")
            self._snapshot = MediaJobSnapshot(
                running=True,
                state="Uploading",
                title="Importing Media",
                detail=f"Preparing {display_name}.",
                input_name=display_name,
                started_at=datetime.now().strftime("%H:%M:%S"),
            )

        try:
            upload_folder.mkdir(parents=True, exist_ok=True)
            if media_kind == "video":
                upload = uploads[0]
                extension = Path(upload.filename).suffix.lower()
                source_path = upload_folder / f"{safe_stem}{extension}"
                self._copy_upload(upload, source_path)
            else:
                source_path.mkdir(parents=True, exist_ok=True)
                self._copy_image_sequence(uploads, source_path)
        except Exception as exc:
            self._update(
                running=False,
                state="Error",
                title="Import Failed",
                detail=str(exc),
                error=str(exc),
                finished_at=datetime.now().strftime("%H:%M:%S"),
            )
            raise

        worker = threading.Thread(
            target=self._run_media_job,
            args=(source_path, display_name, media_kind, output_dir),
            daemon=True,
        )
        with self._lock:
            self._worker = worker
            self._snapshot.state = "Processing"
            self._snapshot.title = "Processing Media"
            self._snapshot.detail = "Analyzing frames and preparing the annotated MP4."
        worker.start()
        return self.snapshot()

    def _run_media_job(self, source_path: Path, display_name: str, media_kind: str, output_dir: Path | None = None) -> None:
        session_id: str | None = None
        processor = None
        event_media = None
        frame_count = 0
        risk_sum = 0.0
        peak_risk = 0.0
        try:
            target_dir = output_dir if output_dir else self.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            if self._repos is not None:
                active = self._repos.profiles.get_active()
                profile_id = active["id"] if active else "default"
                session = self._repos.sessions.create(
                    profile_id=profile_id,
                    source_type=media_kind,
                    source_path=str(source_path),
                    pose_backend="yolo",
                    predictor_type="ensemble",
                )
                session_id = session["id"]
                from .frame_pipeline import FrameBusinessProcessor
                from .alert_service import SoundAlertService

                processor = FrameBusinessProcessor(
                    repos=self._repos,
                    session_id=session_id,
                    profile_id=profile_id,
                    fps=30.0,
                    state_change_observer=SoundAlertService(
                        enabled=self.settings.sound_alert,
                    ),
                )
                from .event_media_buffer import EventMediaBuffer

                event_media = EventMediaBuffer(
                    repos=self._repos,
                    output_root=writable_output_root(self.app_root),
                    session_id=session_id,
                    fps=30.0,
                )

            def on_prediction(prediction, frame_index: int, timestamp: float, frame) -> None:
                nonlocal frame_count, risk_sum, peak_risk
                frame_count = frame_index + 1
                risk = max(0.0, min(1.0, float(prediction.risk_score)))
                risk_sum += risk
                peak_risk = max(peak_risk, risk)
                if processor is not None:
                    processor.process(prediction, frame_index, timestamp)
                    if event_media is not None:
                        event_media.add_frame(
                            frame,
                            timestamp,
                            processor.event_service.active_event_id
                            if processor.event_service else None,
                        )

            result = run_prediction_job(
                PredictionJob(
                    source=str(source_path),
                    output_dir=target_dir,
                    pose_backend="yolo",
                    predictor="ensemble",
                    sensitivity=self._current_sensitivity(),
                    yolo_model_path=Path("models/yolo26n-pose.pt"),
                    classifier_model_path=Path("models/yolo_tail60_prefall_accel_robust_classifier.joblib"),
                    fusion_model_path=Path("models/skeleton_feature_fusion_tuned.pt"),
                    write_csv=False,
                    write_video=True,
                    show_preview=False,
                    use_hmm=True,
                    use_accel=True,
                    use_temporal_fall_validation=False,
                    fusion_fall_confirmation_steps=3,
                    use_static_lying_adl_filter=True,
                    image_fps=30.0,
                ),
                log=self._handle_job_log,
                on_prediction=on_prediction,
            )
            output_video = str(result.output_video) if result.output_video else ""
            output_name = Path(output_video).name if output_video else "output file"
            self._update(
                running=False,
                state="Complete",
                title="MP4 Ready",
                detail=f"{display_name} is complete. Saved as {output_name}.",
                output_video=output_video,
                finished_at=datetime.now().strftime("%H:%M:%S"),
            )
            reveal_path = result.output_video
            if reveal_path:
                subprocess.run(["open", "-R", str(reveal_path)], check=False)
        except Exception as exc:
            self._update(
                running=False,
                state="Error",
                title="Import Failed",
                detail=str(exc),
                error=str(exc),
                finished_at=datetime.now().strftime("%H:%M:%S"),
            )
        finally:
            if event_media is not None:
                try:
                    event_media.close()
                except Exception:
                    pass
            if processor is not None:
                try:
                    processor.close()
                except Exception:
                    pass
            if session_id is not None and self._repos is not None:
                try:
                    snapshot = self.snapshot()
                    self._repos.sessions.stop(
                        session_id=session_id,
                        total_frames=frame_count,
                        total_events=self._repos.events.count_for_session(session_id),
                        peak_risk=round(peak_risk, 4),
                        avg_risk=round(risk_sum / frame_count, 4) if frame_count else 0.0,
                        error_message=str(snapshot["error"]) or None,
                    )
                except Exception:
                    pass

    def _detect_media_kind(self, uploads: list[UploadedMediaFile]) -> str:
        suffixes = [Path(upload.filename).suffix.lower() for upload in uploads]
        videos = [suffix for suffix in suffixes if suffix in VIDEO_EXTENSIONS]
        images = [suffix for suffix in suffixes if suffix in IMAGE_EXTENSIONS]
        unsupported = sorted({suffix or "(no extension)" for suffix in suffixes if suffix not in SUPPORTED_MEDIA_EXTENSIONS})

        if unsupported:
            raise ValueError(f"Unsupported media format: {', '.join(unsupported)}.")
        if len(videos) == 1 and len(uploads) == 1:
            return "video"
        if images and len(images) == len(uploads):
            return "images"
        raise ValueError("Choose one video file, or choose photos only.")

    def _detect_path_media_kind(self, paths: list[Path]) -> tuple[str, list[Path]]:
        videos: list[Path] = []
        images: list[Path] = []
        unsupported: list[str] = []

        for raw_path in paths:
            path = raw_path.expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Selected item does not exist: {path}")
            if path.is_dir():
                folder_images = self._find_images_in_folder(path)
                if not folder_images:
                    raise ValueError(f"No supported photos found in folder: {path.name}")
                images.extend(folder_images)
                continue
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                videos.append(path)
            elif suffix in IMAGE_EXTENSIONS:
                images.append(path)
            else:
                unsupported.append(suffix or "(no extension)")

        if unsupported:
            raise ValueError(f"Unsupported media format: {', '.join(sorted(set(unsupported)))}.")
        if len(videos) == 1 and not images:
            return "video", videos
        if videos:
            raise ValueError("Choose one video file, or choose photos/folders only.")
        if images:
            return "images", sorted(images, key=natural_media_sort_key)
        raise ValueError("Please choose a video, photo, or photo folder.")

    def _find_images_in_folder(self, folder: Path) -> list[Path]:
        images = [
            child
            for child in folder.iterdir()
            if child.is_file()
            and not child.name.startswith(".")
            and child.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(images, key=natural_media_sort_key)

    def _display_name(self, uploads: list[UploadedMediaFile], media_kind: str) -> str:
        if media_kind == "video":
            return Path(uploads[0].filename).name or "video"
        if len(uploads) == 1:
            return Path(uploads[0].filename).name or "photo"

        folders = {self._top_level_folder(upload.filename) for upload in uploads}
        folders.discard("")
        if len(folders) == 1:
            return f"{sorted(folders)[0]} ({len(uploads)} photos)"
        return f"{len(uploads)} photos"

    def _display_name_for_paths(self, selected_paths: list[Path], media_paths: list[Path], media_kind: str) -> str:
        if media_kind == "video":
            return media_paths[0].name
        if len(media_paths) == 1:
            return media_paths[0].name

        selected_dirs = [path for path in selected_paths if path.expanduser().is_dir()]
        if len(selected_dirs) == 1:
            return f"{selected_dirs[0].name} ({len(media_paths)} photos)"
        return f"{len(media_paths)} photos"

    def _source_stem(self, display_name: str, media_kind: str) -> str:
        if media_kind == "video":
            return Path(display_name).stem or "video"
        if display_name.endswith(" photos"):
            return "photo_sequence"
        if " (" in display_name:
            return display_name.split(" (", 1)[0]
        return Path(display_name).stem or "photo"

    def _top_level_folder(self, filename: str) -> str:
        parts = filename.replace("\\", "/").split("/")
        return parts[0] if len(parts) > 1 else ""

    def _copy_upload(self, upload: UploadedMediaFile, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        upload.stream.seek(0)
        with target_path.open("wb") as target:
            shutil.copyfileobj(upload.stream, target, length=1024 * 1024)

    def _copy_image_sequence(self, uploads: list[UploadedMediaFile], sequence_dir: Path) -> None:
        sorted_uploads = sorted(uploads, key=lambda upload: natural_media_sort_key(upload.filename))
        if len(sorted_uploads) == 1:
            upload = sorted_uploads[0]
            suffix = self._image_output_suffix(upload.filename)
            for index in range(SINGLE_IMAGE_REPEAT_FRAMES):
                target = sequence_dir / f"frame_{index + 1:06d}{suffix}"
                self._copy_image_upload(upload, target)
            return

        for index, upload in enumerate(sorted_uploads, start=1):
            suffix = self._image_output_suffix(upload.filename)
            stem = safe_filename(Path(upload.filename).stem or "photo")
            target = sequence_dir / f"{index:06d}_{stem}{suffix}"
            self._copy_image_upload(upload, target)

    def _copy_local_image_sequence(self, image_paths: list[Path], sequence_dir: Path) -> None:
        sorted_paths = sorted(image_paths, key=natural_media_sort_key)
        if len(sorted_paths) == 1:
            source = sorted_paths[0]
            suffix = self._image_output_suffix(source.name)
            for index in range(SINGLE_IMAGE_REPEAT_FRAMES):
                target = sequence_dir / f"frame_{index + 1:06d}{suffix}"
                self._copy_image_path(source, target)
            return

        for index, source in enumerate(sorted_paths, start=1):
            suffix = self._image_output_suffix(source.name)
            stem = safe_filename(source.stem or "photo")
            target = sequence_dir / f"{index:06d}_{stem}{suffix}"
            self._copy_image_path(source, target)

    def _image_output_suffix(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return ".png" if suffix in {".heic", ".heif"} else suffix

    def _copy_image_upload(self, upload: UploadedMediaFile, target_path: Path) -> None:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".heic", ".heif"}:
            self._copy_upload(upload, target_path)
            return

        raw_path = target_path.with_suffix(suffix)
        self._copy_upload(upload, raw_path)
        try:
            subprocess.run(
                ["sips", "-s", "format", "png", str(raw_path), "--out", str(target_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            raw_path.unlink(missing_ok=True)

    def _copy_image_path(self, source_path: Path, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower()
        if suffix not in {".heic", ".heif"}:
            shutil.copy2(source_path, target_path)
            return

        subprocess.run(
            ["sips", "-s", "format", "png", str(source_path), "--out", str(target_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _handle_job_log(self, message: str) -> None:
        if message.startswith("Starting prediction"):
            self._update(detail="Analyzing the media. This can take a few minutes.")
        elif message == "Done.":
            self._update(detail="Finalizing the MP4 output.")

    def _update(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                if hasattr(self._snapshot, key):
                    setattr(self._snapshot, key, value)


VideoImportProcessor = MediaImportProcessor


def natural_media_sort_key(value: str | Path) -> list[int | str]:
    name = str(value).replace("\\", "/").lower()
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def media_picker_applescript() -> list[str]:
    return [
        'use framework "AppKit"',
        "use scripting additions",
        "set panel to current application's NSOpenPanel's openPanel()",
        "panel's setCanChooseFiles:true",
        "panel's setCanChooseDirectories:true",
        "panel's setAllowsMultipleSelection:true",
        "panel's setTreatsFilePackagesAsDirectories:false",
        # Allow common video and image file types so files are selectable
        # rather than greyed out on newer macOS versions.
        "set allowedTypes to current application's NSMutableArray's array()",
        # Video UTIs
        "allowedTypes's addObject:\"public.mpeg-4\"",
        "allowedTypes's addObject:\"com.apple.quicktime-movie\"",
        "allowedTypes's addObject:\"public.avi\"",
        "allowedTypes's addObject:\"org.matroska.mkv\"",
        "allowedTypes's addObject:\"public.mpeg\"",
        # Image UTIs
        "allowedTypes's addObject:\"public.png\"",
        "allowedTypes's addObject:\"public.jpeg\"",
        "allowedTypes's addObject:\"com.microsoft.bmp\"",
        "allowedTypes's addObject:\"public.tiff\"",
        "allowedTypes's addObject:\"public.heic\"",
        "allowedTypes's addObject:\"public.heif\"",
        # Also allow public.data so folders and any file can still be picked.
        "allowedTypes's addObject:\"public.data\"",
        "panel's setAllowedFileTypes:allowedTypes",
        'panel\'s setPrompt:"Import"',
        'panel\'s setMessage:"Choose a video, photos, or a photo folder."',
        "set response to panel's runModal()",
        "if response = (current application's NSModalResponseOK) then",
        "set selectedURLs to panel's URLs()",
        "set selectedPaths to {}",
        "repeat with selectedURL in selectedURLs",
        "set end of selectedPaths to (selectedURL's |path|()) as text",
        "end repeat",
        "set AppleScript's text item delimiters to linefeed",
        "return selectedPaths as text",
        "else",
        'return ""',
        "end if",
    ]


def save_panel_applescript(default_name: str = "annotated_video.mp4") -> str:
    """Open a native NSSavePanel so the user can choose where to save the output.

    Returns the chosen path as a string, or an empty string if the user cancels.
    """
    script = [
        'use framework "AppKit"',
        "use scripting additions",
        "set panel to current application's NSSavePanel's savePanel()",
        f'panel\'s setNameFieldStringValue:"{default_name}"',
        'panel\'s setTitle:"Save Annotated Video"',
        'panel\'s setMessage:"Choose a location to save the annotated MP4 output."',
        'panel\'s setPrompt:"Save"',
        'panel\'s setCanCreateDirectories:true',
        "set response to panel's runModal()",
        "if response = (current application's NSModalResponseOK) then",
        "return (panel's URL()'s |path|()) as text",
        "else",
        'return ""',
        "end if",
    ]
    try:
        result = subprocess.run(
            ["osascript", *sum([["-e", line] for line in script], [])],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


class CameraMonitor:
    def __init__(self, app_root: Path, settings: AppSettings | None = None) -> None:
        self.app_root = app_root
        self.settings = settings or AppSettings()
        self.profile_manager: ProfileManager | None = None
        self._repos = None  # AppRepositories — set by main_native() after DB init
        self._session_id: str | None = None  # current monitoring session ID
        self._profile_id: str | None = None  # profile bound to the current session
        self.output_dir = writable_output_root(app_root) / "camera_sessions"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._preload_worker: threading.Thread | None = None
        self._preload_error = ""
        self._jpeg_frame: bytes | None = None
        self._snapshot = MonitorSnapshot()
        self._last_activity_state = ""
        self._fsm = None       # RiskStateMachine — created in _run_camera_loop
        self._event_svc = None  # EventService — created in _run_camera_loop
        import atexit
        atexit.register(self._cleanup_db)
        self._debug_log("CameraMonitor.__init__", f"app_root={app_root}")
        self.preload_models()

    @staticmethod
    def _debug_log(step: str, detail: str = "") -> None:
        """Write a debug trace to a temp file so we can diagnose the built .app."""
        try:
            from datetime import datetime
            from pathlib import Path
            log_path = Path("/tmp/fallguard_debug.log")
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line = f"[{ts}] {step}"
            if detail:
                line += f"  |  {detail}"
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _cleanup_db(self) -> None:
        try:
            from .database.database import get_database as _get_db
        except ImportError:
            from fall_prediction_desktop.database.database import get_database as _get_db
        try:
            _get_db().close_all()
        except Exception:
            pass

    def update_settings(self, settings: AppSettings) -> None:
        with self._lock:
            self.settings = settings

    def _current_sensitivity(self) -> str:
        with self._lock:
            return normalize_sensitivity(self.settings.sensitivity)

    def _create_predictor_for_sensitivity(self, sensitivity: str):
        from fall_prediction.video_app import create_predictor

        ml_config = ml_config_for_sensitivity(sensitivity)
        return create_predictor(
            predictor_type="ensemble",
            classifier_model_path=self.app_root / "models" / "yolo_tail60_prefall_accel_robust_classifier.joblib",
            fusion_model_path=self.app_root / "models" / "skeleton_feature_fusion_tuned.pt",
            predictor_config=predictor_config_for_sensitivity(sensitivity),
            prefall_alert_threshold=ml_config.prefall_alert_threshold,
            prefall_alert_frames=ml_config.prefall_alert_frames,
            use_hmm=True,
            use_accel=True,
            use_temporal_fall_validation=False,
            fall_validator_settings=ml_config.fall_validator_settings,
            temporal_sensitivity=sensitivity,
            fusion_fall_confirmation_steps=3,
            use_static_lying_adl_filter=True,
        )

    def start(self) -> None:
        self._debug_log("start", "entering")
        with self._lock:
            if self._worker and self._worker.is_alive():
                self._debug_log("start", "worker already alive, returning")
                return
            self._stop_event.clear()
            self._snapshot = MonitorSnapshot(
                running=True,
                loading=True,
                title="Starting",
                detail="Loading camera and AI model.",
                state="Starting",
                started_at=datetime.now().strftime("%H:%M:%S"),
            )
            self._jpeg_frame = None
            self._last_activity_state = ""
            self._fsm = None       # Reset FSM for fresh session
            self._event_svc = None  # Reset EventService for fresh session
            self._debug_log("start", "snapshot set to loading=True")

        # Create a monitoring session in the database (Task 4)
        self._session_id = None
        self._profile_id = None
        try:
            if self._repos is not None:
                active = self._repos.profiles.get_active()
                profile_id = active["id"] if active else "default"
                self._profile_id = profile_id
                sess = self._repos.sessions.create(
                    profile_id=profile_id,
                    source_type="camera",
                    pose_backend="yolo",
                    predictor_type="ensemble",
                )
                self._session_id = sess["id"]
                self._debug_log("start", f"DB session created: {self._session_id[:12]}...")
        except Exception as exc:
            self._debug_log("start", f"DB session creation failed (non-fatal): {exc}")

        self._worker = threading.Thread(target=self._run_camera_loop, daemon=True)
        self._worker.start()
        self._debug_log("start", "worker thread started")

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if (
            worker
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=2.0)
            if worker.is_alive():
                # Unblock a driver-level capture.read() that did not observe
                # the stop event promptly, then wait for the finally block.
                with self._lock:
                    capture = self._capture
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
                worker.join(timeout=3.0)
        with self._lock:
            self._jpeg_frame = None

    def preload_models(self) -> None:
        with self._lock:
            if self._preload_worker and self._preload_worker.is_alive():
                return
            self._preload_error = ""
            self._preload_worker = threading.Thread(target=self._preload_models, daemon=True)
            self._preload_worker.start()

    def _preload_models(self) -> None:
        try:
            from fall_prediction.ml_predictor import load_model_artifact
            from fall_prediction.pose import preload_yolo_model

            preload_yolo_model(self.app_root / "models" / "yolo26n-pose.pt", warmup=True)
            load_model_artifact(self.app_root / "models" / "yolo_tail60_prefall_accel_robust_classifier.joblib")
            load_model_artifact(self.app_root / "models" / "skeleton_feature_fusion_tuned.pt")
        except Exception as exc:
            with self._lock:
                self._preload_error = str(exc)

    def jpeg_frame(self) -> bytes | None:
        with self._lock:
            return self._jpeg_frame

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            snapshot = self._snapshot
            result = {
                "running": snapshot.running,
                "loading": snapshot.loading,
                "cameraConnected": snapshot.camera_connected,
                "modelActive": snapshot.model_active,
                "state": snapshot.state,
                "title": snapshot.title,
                "detail": snapshot.detail,
                "riskPercent": snapshot.risk_percent,
                "confidencePercent": snapshot.confidence_percent,
                "fps": round(snapshot.fps, 1),
                "resolution": snapshot.resolution,
                "environment": snapshot.environment,
                "error": snapshot.error,
                "startedAt": snapshot.started_at,
                "activities": [
                    {
                        "level": activity.level,
                        "title": activity.title,
                        "time": activity.time,
                        "risk": activity.risk,
                    }
                    for activity in snapshot.activities[-6:]
                ],
            }
        if self._repos is not None:
            try:
                result["recentEvents"] = [
                    {
                        "id": event["id"],
                        "level": "danger" if event["event_type"] == "fall" else "warning",
                        "title": "Fall detected" if event["event_type"] == "fall" else "Pre-fall warning",
                        "time": event["started_at"],
                        "risk": int(round(float(event["peak_risk"]) * 100)),
                        "eventType": event["event_type"],
                        "status": event["status"],
                    }
                    for event in self._repos.events.list_recent(12)
                ]
            except Exception as exc:
                self._debug_log("snapshot.recent_events", str(exc))
        return result

    def _run_camera_loop(self) -> None:
        capture = None
        estimator = None
        frame_processor = None
        event_media = None
        frame_index = 0  # Initialized early so finally block always has it
        risk_sum = 0.0
        risk_count = 0
        peak_risk = 0.0
        fps_sum = 0.0
        fps_count = 0
        resolution = ""
        try:
            self._debug_log("_run_camera_loop", "thread started, importing cv2...")
            import cv2
            self._debug_log("_run_camera_loop", "cv2 imported ok")

            from fall_prediction.video_app import create_pose_estimator
            self._debug_log("_run_camera_loop", "create_pose_estimator imported ok")

            self._update(
                loading=True,
                title="Loading Model",
                detail="Preparing YOLO pose detection and fall prediction.",
                environment="Checking",
            )
            self._debug_log("_run_camera_loop", "loading YOLO model...")
            estimator = create_pose_estimator(
                pose_backend="yolo",
                yolo_model_path=self.app_root / "models" / "yolo26n-pose.pt",
            )
            self._debug_log("_run_camera_loop", "YOLO model loaded ok")
            sensitivity = self._current_sensitivity()
            predictor = self._create_predictor_for_sensitivity(sensitivity)
            self._debug_log("_run_camera_loop", "predictor created ok")

            from fall_prediction.camera import CameraOpenError, open_camera_capture, summarize_camera_attempts

            try:
                camera_idx = self.settings.camera_index
                self._debug_log("_run_camera_loop", f"opening camera index={camera_idx}...")
                capture = open_camera_capture(camera_idx)
                with self._lock:
                    self._capture = capture
                self._debug_log("_run_camera_loop", "camera opened successfully")
            except CameraOpenError as exc:
                if exc.permission and not exc.permission.allowed:
                    raise RuntimeError(str(exc)) from exc
                raise RuntimeError(
                    "Camera could not be opened. Allow camera access for FallGuard "
                    "(or Terminal/Python when using launch.command), close other camera apps, "
                    "and start dist/FallGuard.app instead of the inner executable when using a packaged build. "
                    f"Tried: {summarize_camera_attempts(exc.attempts)}."
                ) from exc

            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 1e-6:
                fps = 20.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            resolution = f"{width}x{height}"
            if self._session_id is not None and self._repos is not None:
                self._repos.sessions.update_stats(self._session_id, resolution=resolution)

            self._add_activity("normal", "Monitoring started", 0)
            self._update(
                running=True,
                loading=False,
                camera_connected=True,
                model_active=True,
                state="Normal",
                title="Monitoring Active",
                detail="System is running normally.",
                resolution=resolution,
                environment="Good",
            )

            frame_index = 0
            fps_frames = 0
            last_fps_at = time.monotonic()
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed.")

                timestamp = frame_index / fps
                landmarks = estimator.process_bgr(frame, timestamp_ms=int(timestamp * 1000))
                current_sensitivity = self._current_sensitivity()
                if current_sensitivity != sensitivity:
                    sensitivity = current_sensitivity
                    predictor = self._create_predictor_for_sensitivity(sensitivity)
                prediction = predictor.predict(landmarks, frame_index, timestamp)

                # Lazily initialize the shared business frame processor.
                if frame_processor is None:
                    try:
                        from .alert_service import SoundAlertService
                        from .frame_pipeline import FrameBusinessProcessor
                    except ImportError:
                        from fall_prediction_desktop.alert_service import SoundAlertService  # type: ignore[no-redef]
                        from fall_prediction_desktop.frame_pipeline import FrameBusinessProcessor  # type: ignore[no-redef]
                    frame_processor = FrameBusinessProcessor(
                        repos=self._repos,
                        session_id=self._session_id,
                        profile_id=self._profile_id,
                        fps=fps,
                        state_change_observer=SoundAlertService(
                            enabled=self.settings.sound_alert,
                        ),
                    )
                    self._fsm = frame_processor.state_machine
                    self._event_svc = frame_processor.event_service
                    try:
                        from .event_media_buffer import EventMediaBuffer
                    except ImportError:
                        from fall_prediction_desktop.event_media_buffer import EventMediaBuffer  # type: ignore[no-redef]
                    event_media = EventMediaBuffer(
                        repos=self._repos,
                        output_root=writable_output_root(self.app_root),
                        session_id=self._session_id,
                        fps=fps,
                    )

                processed = frame_processor.process(prediction, frame_index, timestamp)
                fsm_event = processed.state_change
                display_state = processed.state
                if event_media is not None:
                    event_media.add_frame(
                        frame,
                        timestamp,
                        self._event_svc.active_event_id if self._event_svc else None,
                    )

                risk_percent = max(0, min(100, int(round(processed.risk_score * 100))))
                risk_value = processed.risk_score
                risk_sum += risk_value
                risk_count += 1
                peak_risk = max(peak_risk, risk_value)
                confidence = max(0, min(100, int(round(processed.confidence * 100))))
                title, detail, level = state_copy(display_state)
                if display_state != self._last_activity_state and display_state in {"Normal", "Pre-fall", "Fall"}:
                    self._last_activity_state = display_state
                    self._add_activity(level, title, risk_percent)
                    # Record fall/pre-fall events to the active profile (legacy JSON)
                    if display_state in {"Pre-fall", "Fall"} and self.profile_manager:
                        self.profile_manager.record_fall(display_state, risk_percent, detail)
                _, buffer = cv2.imencode(".jpg", frame)
                with self._lock:
                    self._jpeg_frame = buffer.tobytes()

                fps_frames += 1
                now = time.monotonic()
                if now - last_fps_at >= 1.0:
                    live_fps = fps_frames / (now - last_fps_at)
                    fps_sum += live_fps
                    fps_count += 1
                    fps_frames = 0
                    last_fps_at = now
                    self._update(fps=live_fps)

                self._update(
                    state=display_state,
                    title=title,
                    detail=detail,
                    risk_percent=risk_percent,
                    confidence_percent=confidence,
                    environment="Good" if confidence >= 45 else "Needs Better View",
                )
                frame_index += 1

        except Exception as exc:
            self._debug_log("_run_camera_loop", f"EXCEPTION: {type(exc).__name__}: {exc}")
            self._add_activity("danger", "Monitoring error", 0)
            self._update(
                running=False,
                loading=False,
                camera_connected=False,
                model_active=False,
                state="Error",
                title="Setup Needed",
                detail=str(exc),
                error=str(exc),
                environment="Check Setup",
            )
        finally:
            # Each cleanup step is individually protected so one failure
            # does not skip the remaining steps.
            try:
                if estimator:
                    estimator.close()
            except Exception:
                pass
            try:
                if capture:
                    capture.release()
            except Exception:
                pass
            with self._lock:
                self._capture = None
                self._jpeg_frame = None
            with self._lock:
                has_error = bool(self._snapshot.error)
            if not has_error:
                self._add_activity("normal", "Monitoring stopped", self._snapshot.risk_percent)
                self._update(
                    running=False,
                    loading=False,
                    camera_connected=False,
                    model_active=False,
                    state="Idle",
                    title="Ready",
                    detail="Click Start Monitoring to begin.",
                    fps=0.0,
                    risk_percent=0,
                    confidence_percent=0,
                    environment="Waiting",
                )

            if event_media is not None:
                try:
                    event_media.close()
                except Exception:
                    pass
            # Close the shared frame pipeline before finalizing its session.
            if frame_processor is not None:
                try:
                    frame_processor.close()
                except Exception:
                    pass
            # ── Task 4: close the monitoring session in the database ──
            if self._session_id is not None and self._repos is not None:
                try:
                    self._repos.samples.commit()  # flush remaining samples
                    snap = self._snapshot
                    total_frames = frame_index  # always defined (initialized before try block)
                    total_events = self._repos.events.count_for_session(self._session_id)
                    self._repos.sessions.stop(
                        session_id=self._session_id,
                        total_frames=total_frames,
                        total_events=total_events,
                        peak_risk=round(peak_risk, 4),
                        avg_risk=round(risk_sum / risk_count, 4) if risk_count else 0.0,
                        fps_avg=round(fps_sum / fps_count, 1) if fps_count else round(snap.fps, 1),
                        error_message=snap.error if snap.state == "Error" else None,
                    )
                    self._debug_log("_run_camera_loop", f"DB session closed: {self._session_id[:12]}...")
                except Exception as exc:
                    self._debug_log("_run_camera_loop", f"DB session close failed (non-fatal): {exc}")
                finally:
                    self._session_id = None

    def _update(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                if hasattr(self._snapshot, key):
                    setattr(self._snapshot, key, value)

    def _add_activity(self, level: str, title: str, risk: int) -> None:
        with self._lock:
            self._snapshot.activities.append(
                ActivityEvent(level=level, title=title, time=datetime.now().strftime("%H:%M:%S"), risk=risk)
            )


def state_copy(state: str) -> tuple[str, str, str]:
    if state == "Fall":
        return "High Risk Detected", "Possible fall detected. Please check immediately.", "danger"
    if state == "Pre-fall":
        return "Medium Risk Detected", "Instability detected. Stay alert.", "warning"
    if state == "Unknown":
        return "Person Not Visible", "Please keep the full body in view.", "muted"
    return "Monitoring Active", "System is running normally.", "normal"


class FallGuardRequestHandler(BaseHTTPRequestHandler):
    server: "FallGuardServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if path == "/stream.mjpg":
            self._serve_stream()
            return
        if path == "/frame.jpg":
            self._serve_latest_frame()
            return
        if path == "/api/status":
            self._send_json(self.server.snapshot())
            return
        if path == "/api/settings":
            self._handle_get_settings()
            return
        if path == "/api/settings/cameras":
            self._handle_get_cameras()
            return
        if path == "/api/profiles":
            self._send_json(self.server.profile_manager.snapshot())
            return
        if path == "/settings":
            self._serve_static("settings.html", "text/html; charset=utf-8")
            return
        if path == "/api/open-settings":
            self._handle_open_settings()
            return
        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            content_type = content_type_for(relative)
            self._serve_static(relative, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/start":
            if self.server.media_processor.snapshot().get("running"):
                self._send_json(
                    {"ok": False, "error": "Wait for the imported media to finish before starting monitoring."},
                    status=HTTPStatus.CONFLICT,
                )
                return
            self.server.monitor.start()
            self._send_json({"ok": True})
            return
        if path == "/api/stop":
            self.server.monitor.stop()
            self._send_json({"ok": True})
            return
        if path == "/api/camera/repair":
            self._handle_camera_repair()
            return
        if path == "/api/media/pick":
            self._handle_media_pick()
            return
        if path in {"/api/media/import", "/api/video/import"}:
            self._handle_media_import()
            return
        if path == "/api/settings/sensitivity":
            self._handle_set_sensitivity()
            return
        if path == "/api/settings/theme":
            self._handle_set_theme()
            return
        if path == "/api/settings/language":
            self._handle_set_language()
            return
        if path == "/api/settings/camera":
            self._handle_set_camera()
            return
        if path == "/api/profiles":
            self._handle_create_profile()
            return
        if path.startswith("/api/profiles/") and path.endswith("/activate"):
            self._handle_activate_profile(path)
            return
        if path.startswith("/api/profiles/"):
            self._handle_delete_profile(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        """Handle DELETE requests — used for profile deletion."""
        path = urlparse(self.path).path
        if path.startswith("/api/profiles/"):
            self._handle_delete_profile(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    # ---- Settings API ----

    def _handle_get_settings(self) -> None:
        s = self.server.settings
        self._send_json({
            "sensitivity": s.sensitivity,
            "cameraIndex": s.camera_index,
            "thresholds": s.thresholds(),
            "theme": s.theme,
            "lang": s.lang,
            "version": getattr(self.server, "app_version", "0.3.3"),
        })

    def _handle_get_cameras(self) -> None:
        try:
            available = scan_camera_indices()
        except Exception:
            available = [0]
        self._send_json({"cameras": available, "current": self.server.settings.camera_index})

    def _handle_set_sensitivity(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json({"ok": False, "error": "Invalid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return

        level = str(body.get("level", "")).strip().lower()
        if level not in SENSITIVITY_LEVELS:
            self._send_json(
                {"ok": False, "error": f"Unknown sensitivity level: {level}. Use low, medium, or high."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.server.settings.sensitivity = level
        self.server.monitor.update_settings(self.server.settings)
        self.server.media_processor.update_settings(self.server.settings)
        save_settings(self.server.app_root, self.server.settings)
        self._send_json({"ok": True, "sensitivity": level, "thresholds": sensitivity_thresholds(level)})

    def _handle_set_camera(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json({"ok": False, "error": "Invalid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return

        index = body.get("index", -1)
        if not isinstance(index, int) or index < 0:
            self._send_json({"ok": False, "error": "Invalid camera index."}, status=HTTPStatus.BAD_REQUEST)
            return

        self.server.settings.camera_index = index
        self.server.monitor.update_settings(self.server.settings)
        save_settings(self.server.app_root, self.server.settings)
        self._send_json({"ok": True, "cameraIndex": index})

    def _handle_set_theme(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json({"ok": False, "error": "Invalid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return
        mode = body.get("mode", "system")
        if mode not in ("light", "dark", "system"):
            self._send_json({"ok": False, "error": "Invalid theme mode."}, status=HTTPStatus.BAD_REQUEST)
            return
        self.server.settings.theme = mode
        save_settings(self.server.app_root, self.server.settings)
        self._send_json({"ok": True, "theme": mode})

    def _handle_set_language(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json({"ok": False, "error": "Invalid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return
        lang = body.get("lang", "en")
        if lang not in ("en", "zh"):
            self._send_json({"ok": False, "error": "Invalid language."}, status=HTTPStatus.BAD_REQUEST)
            return
        self.server.settings.lang = lang
        save_settings(self.server.app_root, self.server.settings)
        self._send_json({"ok": True, "lang": lang})

    def _handle_open_settings(self) -> None:
        """Open the Settings page in a separate native window."""
        import tempfile
        import traceback

        url = f"{self.server.base_url}/settings"
        log_path = Path(tempfile.gettempdir()) / "fallguard_settings_debug.log"

        lines: list[str] = []
        def _log(msg: str) -> None:
            lines.append(msg)

        _log(f"=== Opening settings window === url={url}")

        # Approach 1: dispatch_async via pyobjc Foundation (standard Cocoa)
        try:
            from Foundation import dispatch_async, dispatch_get_main_queue

            def _create():
                try:
                    import webview

                    webview.create_window(
                        title="FallGuard — Settings",
                        url=url,
                        width=640,
                        height=700,
                        min_size=(500, 500),
                        resizable=True,
                    )
                    # Append success marker after window creation
                    with open(str(log_path), "a") as f:
                        f.write("SUCCESS: window created via dispatch_async\n")
                except Exception as inner_exc:
                    with open(str(log_path), "a") as f:
                        f.write(f"FAIL in _create: {inner_exc}\n{traceback.format_exc()}\n")

            dispatch_async(dispatch_get_main_queue(), _create)
            _log("dispatch_async: block queued to main thread")
            self._send_json({"ok": True})
        except Exception as exc:
            _log(f"dispatch_async setup failed: {exc}\n{traceback.format_exc()}")

            # Approach 2: try direct call (may work in some pywebview versions)
            try:
                import webview

                _log("Attempting direct webview.create_window call")
                webview.create_window(
                    title="FallGuard — Settings",
                    url=url,
                    width=640,
                    height=700,
                    min_size=(500, 500),
                    resizable=True,
                )
                _log("Direct call succeeded")
                self._send_json({"ok": True})
            except Exception as exc2:
                _log(f"Direct call also failed: {exc2}")
                self._send_json(
                    {"ok": False, "error": str(exc2)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        # Write all log lines
        try:
            log_path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass

    # ---- Profile API ----

    def _handle_create_profile(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json({"ok": False, "error": "Invalid JSON."}, status=HTTPStatus.BAD_REQUEST)
            return

        name = body.get("name", "").strip()
        if not name:
            self._send_json({"ok": False, "error": "Profile name is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        profile = self.server.profile_manager.create(name)
        self._send_json({"ok": True, "profile": profile.to_dict()})

    def _handle_activate_profile(self, path: str) -> None:
        # Path format: /api/profiles/{id}/activate
        parts = path.split("/")
        if len(parts) < 4:
            self._send_json({"ok": False, "error": "Missing profile ID."}, status=HTTPStatus.BAD_REQUEST)
            return
        profile_id = parts[3]
        ok = self.server.profile_manager.activate(profile_id)
        if not ok:
            self._send_json({"ok": False, "error": "Profile not found."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"ok": True, "activeId": profile_id})

    def _handle_delete_profile(self, path: str) -> None:
        # Path format: /api/profiles/{id}
        parts = path.split("/")
        if len(parts) < 3:
            self._send_json({"ok": False, "error": "Missing profile ID."}, status=HTTPStatus.BAD_REQUEST)
            return
        profile_id = parts[3]
        ok = self.server.profile_manager.delete(profile_id)
        if not ok:
            self._send_json({"ok": False, "error": "Cannot delete the last profile."}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True})

    # ---- Static file serving ----

    def _serve_static(self, relative: str, content_type: str) -> None:
        path = (self.server.web_root / relative).resolve()
        if not path.is_file() and relative == "FallGuard.png":
            path = (self.server.assets_root / "FallGuard.png").resolve()
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self.server.web_root not in path.parents and self.server.assets_root not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        while True:
            if not self.server.monitor._snapshot.running:
                break
            frame = self.server.monitor.jpeg_frame()
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

    def _handle_camera_repair(self) -> None:
        errors: list[str] = []
        reset_cmd = shutil.which("tccutil")
        if reset_cmd:
            result = subprocess.run(
                [reset_cmd, "reset", "Camera", "com.fallguard.desktop"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append((result.stderr or result.stdout or "Camera permission reset failed.").strip())
        else:
            errors.append("Camera permission reset tool is not available on this Mac.")

        settings_url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
        open_cmd = shutil.which("open")
        if open_cmd:
            subprocess.run([open_cmd, settings_url], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if errors:
            self._send_json({"ok": False, "error": " ".join(errors)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "ok": True,
                "detail": "Camera access was reset. Quit and reopen FallGuard, then allow camera access.",
            }
        )

    def _serve_latest_frame(self) -> None:
        """Serve the most recent camera frame as a single JPEG image.

        WKWebView (Safari's engine) does not render MJPEG streams in <img>
        tags, so the front-end falls back to polling this endpoint."""
        frame = self.server.monitor.jpeg_frame()
        if frame is None:
            self.send_error(HTTPStatus.NO_CONTENT, "No frame available yet")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(frame)

    def _handle_media_import(self) -> None:
        monitor_state = self.server.monitor.snapshot()
        if monitor_state.get("running") or monitor_state.get("loading"):
            self._send_json(
                {"ok": False, "error": "Stop live monitoring before importing media."},
                status=HTTPStatus.CONFLICT,
            )
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = self.headers.get("Content-Length", "0")
        if "multipart/form-data" not in content_type:
            self._send_json(
                {"ok": False, "error": "Please choose a video, photo, or photo folder."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            length = int(content_length)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json({"ok": False, "error": "The selected media is empty."}, status=HTTPStatus.BAD_REQUEST)
            return

        uploads: list[UploadedMediaFile] = []
        try:
            uploads = self._parse_media_upload(content_type, length)
            try:
                # Let the user choose where to save the output.
                output_dir = self._pick_output_dir_for_uploads(uploads)
                if output_dir is None:
                    self._send_json({"ok": True, "canceled": True})
                    return
                job = self.server.media_processor.start_from_uploads(uploads, output_dir=output_dir)
            finally:
                for upload in uploads:
                    upload.stream.close()
            self._send_json({"ok": True, "mediaJob": job})
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            for upload in uploads:
                upload.stream.close()
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_media_pick(self) -> None:
        monitor_state = self.server.monitor.snapshot()
        if monitor_state.get("running") or monitor_state.get("loading"):
            self._send_json(
                {"ok": False, "error": "Stop live monitoring before importing media."},
                status=HTTPStatus.CONFLICT,
            )
            return

        try:
            paths = self._pick_media_paths()
            if not paths:
                self._send_json({"ok": True, "canceled": True})
                return

            # Let the user choose where to save the annotated output video.
            output_dir = self._pick_output_dir(paths)
            if output_dir is None:
                self._send_json({"ok": True, "canceled": True})
                return

            job = self.server.media_processor.start_from_paths(paths, output_dir=output_dir)
            self._send_json({"ok": True, "mediaJob": job})
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _pick_output_dir(self, paths: list[Path]) -> Path | None:
        """Show a save panel so the user can choose where to output the annotated video.

        Returns the parent directory of the chosen path, or None if cancelled.
        """
        # Build a sensible default filename based on the input.
        input_name = paths[0].stem if len(paths) == 1 and paths[0].is_file() else "annotated"
        default_name = safe_filename(f"{input_name}_annotated.mp4")
        chosen = save_panel_applescript(default_name)
        if not chosen:
            return None
        output_path = Path(chosen)
        # Use the parent directory as the output dir; the filename is auto-generated.
        return output_path.parent

    def _pick_output_dir_for_uploads(self, uploads: list[UploadedMediaFile]) -> Path | None:
        """Show a save panel for browser-uploaded media."""
        if not uploads:
            return None
        input_stem = Path(uploads[0].filename).stem or "media"
        default_name = safe_filename(f"{input_stem}_annotated.mp4")
        chosen = save_panel_applescript(default_name)
        if not chosen:
            return None
        return Path(chosen).parent

    def _pick_media_paths(self) -> list[Path]:
        script = media_picker_applescript()
        try:
            result = subprocess.run(
                ["osascript", *sum([["-e", line] for line in script], [])],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Native picker unavailable.") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Native picker unavailable.").strip()
            if "User canceled" in detail or "(-128)" in detail:
                return []
            raise RuntimeError(detail)

        return [Path(line) for line in result.stdout.splitlines() if line.strip()]

    def _parse_media_upload(self, content_type: str, length: int) -> list[UploadedMediaFile]:
        message = Message()
        message["Content-Type"] = content_type
        boundary = message.get_param("boundary", header="content-type")
        if not boundary:
            raise ValueError("Please choose a video, photo, or photo folder.")

        remaining = length
        boundary_line = f"--{boundary}".encode("utf-8")
        closing_boundary_line = boundary_line + b"--"

        def read_line() -> bytes:
            nonlocal remaining
            if remaining <= 0:
                return b""
            line = self.rfile.readline(min(remaining, 1024 * 1024))
            remaining -= len(line)
            return line

        line = read_line()
        while line:
            marker = self._boundary_marker(line, boundary_line, closing_boundary_line)
            if marker == "part":
                break
            if marker == "end":
                raise ValueError("Please choose a video, photo, or photo folder.")
            line = read_line()

        uploads: list[UploadedMediaFile] = []
        while remaining > 0:
            headers = self._read_part_headers(read_line)
            disposition = Message()
            disposition["Content-Disposition"] = headers.get("content-disposition", "")
            name = disposition.get_param("name", header="content-disposition")
            filename = disposition.get_filename()

            if name in {"media", "video"} and filename:
                payload = SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b")
                marker = self._copy_part_payload(read_line, boundary_line, closing_boundary_line, payload)
                payload.seek(0, 2)
                size = payload.tell()
                payload.seek(0)
                if size <= 0:
                    payload.close()
                    raise ValueError("The selected media is empty.")
                uploads.append(UploadedMediaFile(filename=filename, stream=payload))
                if marker == "end":
                    break
                continue

            marker = self._copy_part_payload(read_line, boundary_line, closing_boundary_line, None)
            if marker == "end":
                break

        if not uploads:
            raise ValueError("Please choose a video, photo, or photo folder.")
        return uploads

    def _read_part_headers(self, read_line) -> dict[str, str]:
        headers: dict[str, str] = {}
        while True:
            line = read_line()
            if not line:
                raise ValueError("Upload ended before the video file was received.")
            if line in (b"\r\n", b"\n"):
                return headers
            text = line.decode("utf-8", "replace")
            if ":" in text:
                key, value = text.split(":", 1)
                headers[key.strip().lower()] = value.strip()

    def _copy_part_payload(
        self,
        read_line,
        boundary_line: bytes,
        closing_boundary_line: bytes,
        target: BinaryIO | None,
    ) -> str:
        previous: bytes | None = None
        while True:
            line = read_line()
            if not line:
                raise ValueError("Upload ended before the video file was received.")

            marker = self._boundary_marker(line, boundary_line, closing_boundary_line)
            if marker:
                if previous is not None and target is not None:
                    target.write(self._strip_payload_newline(previous))
                return marker

            if previous is not None and target is not None:
                target.write(previous)
            previous = line

    def _boundary_marker(self, line: bytes, boundary_line: bytes, closing_boundary_line: bytes) -> str:
        stripped = line.rstrip(b"\r\n")
        if stripped == boundary_line:
            return "part"
        if stripped == closing_boundary_line:
            return "end"
        return ""

    def _strip_payload_newline(self, payload: bytes) -> bytes:
        if payload.endswith(b"\r\n"):
            return payload[:-2]
        if payload.endswith(b"\n"):
            return payload[:-1]
        return payload

    def _send_json(self, value: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FallGuardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        web_root: Path,
        assets_root: Path,
        monitor: CameraMonitor,
        media_processor: MediaImportProcessor,
        settings: AppSettings,
        app_root: Path,
        profile_manager: ProfileManager,
    ) -> None:
        super().__init__(address, FallGuardRequestHandler)
        self.web_root = web_root.resolve()
        self.assets_root = assets_root.resolve()
        self.monitor = monitor
        self.media_processor = media_processor
        self.video_processor = media_processor
        self.settings = settings
        self.app_root = app_root
        self.app_version = "0.3.3"
        self.profile_manager = profile_manager
        self.base_url = f"http://{address[0]}:{address[1]}"

    def snapshot(self) -> dict[str, object]:
        snapshot = self.monitor.snapshot()
        snapshot["mediaJob"] = self.media_processor.snapshot()
        active = self.profile_manager.active
        if active:
            snapshot["activeProfile"] = active.to_dict()
        return snapshot


def content_type_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".png":
        return "image/png"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((DEFAULT_HOST, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


def _run_native_window(
    server: FallGuardServer,
    monitor: CameraMonitor,
    url: str,
) -> None:
    """Launch the FallGuard UI inside a native macOS window using pywebview."""
    import webview

    # Start the HTTP server in a daemon thread so it runs alongside the GUI loop.
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webview.create_window(
        title="FallGuard — Smart Safety",
        url=url,
        width=1280,
        height=860,
        min_size=(1024, 680),
        resizable=True,
    )

    # Block until the user closes the window.
    webview.start()

    # Cleanup after the window is closed.
    monitor.stop()
    server.shutdown()
    server_thread.join(timeout=3)


def connect_and_show(url: str) -> None:
    """Open a native pywebview window connected to an already-running server."""
    import webview

    webview.create_window(
        title="FallGuard — Live Monitor",
        url=url,
        width=1280,
        height=860,
        min_size=(1024, 680),
        resizable=True,
    )
    webview.start()


def main_native(argv: list[str] | None = None) -> None:
    """Launch FallGuard with native PySide6 UI (no web/HTTP layer)."""
    from PySide6.QtGui import QAction, QIcon
    from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon
    # Relative imports work when running as `python -m fall_prediction_desktop`.
    # Absolute imports work inside a PyInstaller bundle where relative imports fail.
    try:
        from .ui.main_window import MainWindow
    except ImportError:
        from fall_prediction_desktop.ui.main_window import MainWindow  # type: ignore[no-redef]

    app_root = find_app_root()
    ensure_repo_on_path(app_root)

    # Create QApplication before database initialization so startup failures
    # are visible in a normal macOS dialog even in a windowed bundle.
    qt_app = QApplication(sys.argv[:1] if argv is None else argv)
    qt_app.setApplicationName("FallGuard")
    qt_app.setOrganizationName("FallGuard")

    # Resolve resource directories — inside a PyInstaller bundle all data
    # folders (assets, web, models, configs) are at the _MEIPASS root.
    bundle_root = Path(getattr(sys, "_MEIPASS", app_root))
    assets_root = bundle_root / "assets"
    if not assets_root.is_dir():
        assets_root = app_root / "assets"

    # locales may live under assets/locales/ (build artifact) or web/locales/ (source).
    locales_dir = assets_root / "locales"
    if not locales_dir.is_dir():
        web_root = bundle_root / "web"
        if not web_root.is_dir():
            web_root = app_root / "web"
        locales_dir = web_root / "locales"

    # Init backend components — database layer (Tasks 1-4 from dev workflow)
    try:
        from .database.init_db import init_app_database
    except ImportError:
        from fall_prediction_desktop.database.init_db import init_app_database  # type: ignore[no-redef]
    try:
        repos = init_app_database(app_root)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "FallGuard Startup Error",
            f"FallGuard could not initialize its local database.\n\n{exc}",
        )
        return

    # Load settings from DB (with JSON fallback for migration)
    settings = load_settings(app_root)
    # Migrate: if JSON has settings but DB doesn't, seed DB from JSON
    if repos.settings.get("language", "") == "":
        repos.settings.set("language", settings.lang)
        repos.settings.set("theme", settings.theme)
        repos.settings.set("sensitivity", normalize_sensitivity(settings.sensitivity))
    else:
        # Override JSON settings from DB (single source of truth)
        settings.lang = repos.settings.get("language", settings.lang)
        settings.theme = repos.settings.get("theme", settings.theme)
        settings.sensitivity = normalize_sensitivity(repos.settings.get("sensitivity", settings.sensitivity))
        settings.sound_alert = repos.settings.get_bool("sound_alert", settings.sound_alert)
        settings.popup_alert = repos.settings.get_bool("popup_alert", settings.popup_alert)
        settings.start_on_boot = repos.settings.get_bool("start_on_boot", settings.start_on_boot)
        settings.minimize_to_tray = repos.settings.get_bool("minimize_to_tray", settings.minimize_to_tray)

    profile_manager = ProfileManager(app_root, repository=repos.profiles)
    monitor = CameraMonitor(app_root, settings)
    monitor.profile_manager = profile_manager
    monitor._repos = repos  # Attach DB repositories for session/event tracking
    media_processor = MediaImportProcessor(app_root, settings)
    media_processor._repos = repos

    icon_path = assets_root / "FallGuard.icns"
    if not icon_path.exists():
        icon_path = assets_root / "FallGuard.png"
    app_icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
    if not app_icon.isNull():
        qt_app.setWindowIcon(app_icon)

    # Create main window
    window = MainWindow(
        monitor=monitor,
        media_processor=media_processor,
        profile_manager=profile_manager,
        settings=settings,
        app_root=app_root,
        locales_dir=locales_dir,
        assets_dir=assets_root,
        app_version="0.3.3",
        repos=repos,
    )
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)

    tray_icon = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray_icon = QSystemTrayIcon(app_icon, qt_app)
        tray_icon.setToolTip("FallGuard")
        tray_menu = QMenu()
        show_action = QAction("Show FallGuard", tray_menu)
        start_action = QAction("Start Monitoring", tray_menu)
        stop_action = QAction("Stop Monitoring", tray_menu)
        quit_action = QAction("Quit FallGuard", tray_menu)
        show_action.triggered.connect(lambda: (window.show(), window.raise_(), window.activateWindow()))
        start_action.triggered.connect(window._start_monitoring)
        stop_action.triggered.connect(window._stop_monitoring)
        quit_action.triggered.connect(window.request_quit)
        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(start_action)
        tray_menu.addAction(stop_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        tray_icon.setContextMenu(tray_menu)
        tray_icon.activated.connect(
            lambda reason: show_action.trigger()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        tray_icon.show()
        window._tray_icon = tray_icon

    window.show()

    sys.exit(qt_app.exec())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the FallGuard desktop app.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    app_root = find_app_root()
    ensure_repo_on_path(app_root)
    bundle_root = Path(getattr(sys, "_MEIPASS", app_root))
    web_root = bundle_root / "web"
    assets_root = bundle_root / "assets"
    if not web_root.exists():
        web_root = app_root / "web"
    if not assets_root.exists():
        assets_root = app_root / "assets"
    port = find_free_port(args.port)
    settings = load_settings(app_root)
    profile_manager = ProfileManager(app_root)
    monitor = CameraMonitor(app_root, settings)
    monitor.profile_manager = profile_manager
    media_processor = MediaImportProcessor(app_root, settings)
    server = FallGuardServer((args.host, port), web_root, assets_root, monitor, media_processor, settings, app_root, profile_manager)
    url = f"http://{args.host}:{port}/"

    print(f"FallGuard is running at {url}")
    _run_native_window(server, monitor, url)
