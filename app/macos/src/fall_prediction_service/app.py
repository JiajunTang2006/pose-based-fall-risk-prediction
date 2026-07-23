"""
Service assembly — wires together the database, CameraMonitor, MediaImportProcessor,
and HTTP server into a runnable ``AIServiceServer``.

``create_service()`` is the single entry point called from ``__main__.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import API_VERSION, __version__
from .lifecycle import ServiceLifecycle
from .serialization import serialize_health
from .server import AIServiceServer

logger = logging.getLogger(__name__)


def create_service(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str,
    data_dir: Path,
    resource_root: Path,
    parent_pid: int | None = None,
    debug: bool = False,
) -> AIServiceServer:
    """Create and return a fully-wired :class:`AIServiceServer`.

    The server is ready to call ``serve_forever()`` on.  The caller is
    responsible for printing the ``ready`` line and installing signal
    handlers *after* this function returns (so the port is known).

    Parameters
    ----------
    host:
        Bind address — must be ``127.0.0.1`` in production.
    port:
        ``0`` for OS-assigned, or a specific port for dev/testing.
    token:
        Bearer token that clients must present on every ``/api/v1/`` request.
    data_dir:
        Writable directory for the SQLite database and logs.
    resource_root:
        Read-only directory containing models, configs, etc.
    parent_pid:
        Optional PID of the parent Swift process — the service will exit
        when this PID disappears.
    debug:
        Enable debug-level logging.
    """
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            stream=__import__("sys").stderr,
        )

    logger.info("FallGuard AI Service %s starting (api=%s)", __version__, API_VERSION)

    # ── ensure import paths ─────────────────────────────────────────
    import sys
    src_dir = str(resource_root / "src")
    if src_dir not in sys.path and Path(src_dir).is_dir():
        sys.path.insert(0, src_dir)

    # ── lifecycle ───────────────────────────────────────────────────
    lifecycle = ServiceLifecycle(parent_pid=parent_pid)

    # ── database ────────────────────────────────────────────────────
    repos = _init_database(data_dir, resource_root)

    # ── settings ────────────────────────────────────────────────────
    from fall_prediction_desktop.web_app import AppSettings, load_settings
    settings = _load_settings(repos, resource_root, data_dir)

    # ── profile manager ─────────────────────────────────────────────
    from fall_prediction_desktop.web_app import ProfileManager
    profile_manager = ProfileManager(resource_root, data_dir=data_dir, repository=repos.profiles)

    # ── camera monitor ──────────────────────────────────────────────
    from fall_prediction_desktop.web_app import CameraMonitor
    monitor = CameraMonitor(resource_root, settings)
    monitor.profile_manager = profile_manager
    monitor._repos = repos

    # ── media import processor ──────────────────────────────────────
    from fall_prediction_desktop.web_app import MediaImportProcessor
    media_processor = MediaImportProcessor(resource_root, settings)
    media_processor._repos = repos

    # ── HTTP server ─────────────────────────────────────────────────
    server = AIServiceServer(
        address=(host, port),
        token=token,
        app_root=resource_root,
        monitor=monitor,
        media_processor=media_processor,
        settings=settings,
        profile_manager=profile_manager,
        repos=repos,
        lifecycle=lifecycle,
    )

    # Interrupt serve_forever from a helper thread for API, signal, and
    # parent-watchdog shutdown paths. BaseServer.shutdown must not run on the
    # serving thread and must not be called again after serve_forever returns.
    import threading
    lifecycle.on_shutdown_requested(
        lambda: threading.Thread(target=server.shutdown, daemon=True).start()
    )

    # Register cleanup hooks run once after serve_forever has returned.
    lifecycle.on_shutdown(lambda: monitor.stop())
    lifecycle.on_shutdown(lambda: _safe_close_db(repos))

    logger.info("Service assembly complete on %s:%s", host, server.server_address[1])
    return server


# ── internal helpers ────────────────────────────────────────────────


def _init_database(data_dir: Path, resource_root: Path):
    """Initialise SQLite and return AppRepositories."""
    from fall_prediction_desktop.database.init_db import init_app_database
    try:
        repos = init_app_database(resource_root, data_dir=data_dir)
        logger.info("Database initialised: %s", data_dir / "fallguard.db")
        return repos
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        raise


def _load_settings(repos, resource_root: Path, data_dir: Path):
    """Load settings from DB (with JSON fallback)."""
    from fall_prediction_desktop.web_app import AppSettings, load_settings
    from fall_prediction.sensitivity import normalize_sensitivity

    settings = load_settings(resource_root)

    # Migrate: if JSON has settings but DB doesn't, seed DB from JSON
    if repos.settings.get("language", "") == "":
        repos.settings.set("language", settings.lang)
        repos.settings.set("theme", settings.theme)
        repos.settings.set("sensitivity", normalize_sensitivity(settings.sensitivity))
    else:
        # Override JSON settings from DB (single source of truth)
        settings.lang = repos.settings.get("language", settings.lang)
        settings.theme = repos.settings.get("theme", settings.theme)
        settings.sensitivity = normalize_sensitivity(
            repos.settings.get("sensitivity", settings.sensitivity)
        )
        settings.sound_alert = repos.settings.get_bool("sound_alert", settings.sound_alert)

    return settings


def _safe_close_db(repos) -> None:
    """Close all database connections without raising."""
    if repos is None:
        return
    try:
        repos.db.close_all()
    except Exception:
        pass
