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

    import sys
    src_dir = str(resource_root / "src")
    if src_dir not in sys.path and Path(src_dir).is_dir():
        sys.path.insert(0, src_dir)

    lifecycle = ServiceLifecycle(parent_pid=parent_pid)

    repos = _init_database(data_dir, resource_root)

    from fall_prediction_desktop.web_app import AppSettings, load_settings
    settings = _load_settings(repos, resource_root, data_dir)

    from fall_prediction_desktop.web_app import ProfileManager
    profile_manager = ProfileManager(resource_root, data_dir=data_dir, repository=repos.profiles)

    from fall_prediction_desktop.web_app import CameraMonitor
    monitor = CameraMonitor(resource_root, settings)
    monitor.profile_manager = profile_manager
    monitor._repos = repos

    from fall_prediction_desktop.web_app import MediaImportProcessor
    media_processor = MediaImportProcessor(resource_root, settings)
    media_processor._repos = repos

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

    import threading
    lifecycle.on_shutdown_requested(
        lambda: threading.Thread(target=server.shutdown, daemon=True).start()
    )

    lifecycle.on_shutdown(lambda: monitor.stop())
    lifecycle.on_shutdown(lambda: _safe_close_db(repos))

    logger.info("Service assembly complete on %s:%s", host, server.server_address[1])
    return server


def _init_database(data_dir: Path, resource_root: Path):
    from fall_prediction_desktop.database.init_db import init_app_database
    try:
        repos = init_app_database(resource_root, data_dir=data_dir)
        logger.info("Database initialised: %s", data_dir / "fallguard.db")
        return repos
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        raise


def _load_settings(repos, resource_root: Path, data_dir: Path):
    from fall_prediction_desktop.web_app import AppSettings, load_settings
    from fall_prediction.sensitivity import normalize_sensitivity

    settings = load_settings(resource_root)

    if repos.settings.get("language", "") == "":
        repos.settings.set("language", settings.lang)
        repos.settings.set("theme", settings.theme)
        repos.settings.set("sensitivity", normalize_sensitivity(settings.sensitivity))
    else:
        settings.lang = repos.settings.get("language", settings.lang)
        settings.theme = repos.settings.get("theme", settings.theme)
        settings.sensitivity = normalize_sensitivity(
            repos.settings.get("sensitivity", settings.sensitivity)
        )
        settings.sound_alert = repos.settings.get_bool("sound_alert", settings.sound_alert)

    return settings


def _safe_close_db(repos) -> None:
    if repos is None:
        return
    try:
        repos.db.close_all()
    except Exception:
        pass
