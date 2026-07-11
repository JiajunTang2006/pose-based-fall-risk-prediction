"""Writable user-data paths for source and packaged FallGuard builds."""

from __future__ import annotations

import os
from pathlib import Path


def _ensure_writable(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".fallguard-write-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return directory


def user_data_dir() -> Path:
    """Return a writable directory that is never inside the .app bundle."""
    override = os.environ.get("FALLGUARD_DATA_DIR")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([
        Path.home() / "Library" / "Application Support" / "FallGuard",
        Path.home() / ".fallguard",
    ])
    errors: list[str] = []
    for candidate in candidates:
        try:
            return _ensure_writable(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("No writable FallGuard data directory is available. " + "; ".join(errors))


def media_output_dir() -> Path:
    """Return a writable media directory, falling back to user app data."""
    candidates = [
        Path.home() / "Movies" / "FallGuard",
        user_data_dir() / "media",
    ]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return _ensure_writable(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("No writable FallGuard media directory is available. " + "; ".join(errors))
