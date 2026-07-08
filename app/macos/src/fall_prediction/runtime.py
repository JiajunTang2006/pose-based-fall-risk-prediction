from __future__ import annotations

import os
import tempfile
from pathlib import Path


def prepare_runtime_cache() -> Path:
    """Point heavy libraries at a stable writable cache directory."""
    cache_root = _cache_root()
    _set_cache_env("MPLCONFIGDIR", cache_root / "matplotlib")
    _set_cache_env("YOLO_CONFIG_DIR", cache_root / "ultralytics")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return cache_root


def _cache_root() -> Path:
    explicit = os.environ.get("FALLGUARD_CACHE_DIR")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    home = Path.home()
    if os.name == "posix":
        candidates.append(home / "Library" / "Caches" / "FallGuard")
        candidates.append(home / ".cache" / "fallguard")
    candidates.append(Path(tempfile.gettempdir()) / "fallguard")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue

    return Path(tempfile.gettempdir())


def _set_cache_env(name: str, path: Path) -> None:
    if os.environ.get(name):
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ[name] = str(path)
