from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_writable(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".fallguard-write-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return directory


def user_data_dir() -> Path:
    override = os.environ.get("FALLGUARD_DATA_DIR")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "FallGuard")
        candidates.append(Path.home() / "AppData" / "Local" / "FallGuard")
    elif sys.platform == "darwin":
        candidates.append(Path.home() / "Library" / "Application Support" / "FallGuard")
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        candidates.append(
            Path(xdg_data_home) / "FallGuard" if xdg_data_home else Path.home() / ".local" / "share" / "FallGuard"
        )
    candidates.append(Path.home() / ".fallguard")
    errors: list[str] = []
    for candidate in candidates:
        try:
            return _ensure_writable(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("No writable FallGuard data directory is available. " + "; ".join(errors))


def media_output_dir() -> Path:
    if sys.platform == "win32":
        videos = os.environ.get("USERPROFILE") or str(Path.home())
        candidates = [Path(videos) / "Videos" / "FallGuard"]
    elif sys.platform == "darwin":
        candidates = [Path.home() / "Movies" / "FallGuard"]
    else:
        candidates = [Path.home() / "Videos" / "FallGuard"]
    candidates.append(user_data_dir() / "media")
    errors: list[str] = []
    for candidate in candidates:
        try:
            return _ensure_writable(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("No writable FallGuard media directory is available. " + "; ".join(errors))
