"""
One-stop database initialization for the FallGuard application.

Call ``init_app_database(app_root)`` once at startup.  It returns all six
repository instances so the rest of the application never touches SQL directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..paths import media_output_dir, user_data_dir
from .database import init_database
from .repositories import (
    SettingsRepository,
    ProfilesRepository,
    SessionsRepository,
    RiskSamplesRepository,
    EventsRepository,
    MediaFilesRepository,
)

# Default data directory under the app root.
DEFAULT_DATA_DIR_NAME = "data"
DB_FILENAME = "fallguard.db"


def default_data_dir(app_root: Path) -> Path:
    """Return the database directory outside the source or .app bundle.

    Existing databases from the early Movies-based build remain supported so
    users do not lose history during upgrade.
    """
    legacy = Path.home() / "Movies" / "FallGuard"
    if sys.platform == "darwin" and (legacy / DB_FILENAME).is_file():
        try:
            probe = legacy / ".fallguard-db-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return legacy
        except OSError:
            pass
    return user_data_dir()


def init_app_database(app_root: Path, data_dir: Path | None = None) -> "AppRepositories":
    """Initialize the SQLite database and return all repository instances.

    Called once at application startup.  Creates the database file and
    default profile if this is the first run.
    """
    data_dir = data_dir or default_data_dir(app_root)
    db_path = data_dir / DB_FILENAME
    schema_path = Path(__file__).resolve().parent / "schema.sql"

    db = init_database(db_path, schema_path)

    repos = AppRepositories(
        settings=SettingsRepository(db),
        profiles=ProfilesRepository(db),
        sessions=SessionsRepository(db),
        samples=RiskSamplesRepository(db),
        events=EventsRepository(db),
        media=MediaFilesRepository(db),
    )

    # A force-quit or power loss can leave a previous session marked as
    # running.  Recover it before any new monitoring session is created so
    # EventService can never attach events to stale history.
    repos.sessions.recover_interrupted()

    # Ensure at least one default profile exists
    _ensure_default_profile(repos)

    return repos


def _ensure_default_profile(repos: "AppRepositories") -> None:
    if repos.profiles.count() == 0:
        repos.profiles.create("Default")
        # Seed default settings
        repos.settings.set("language", "en")
        repos.settings.set("theme", "system")
        repos.settings.set("sensitivity", "medium")


class AppRepositories:
    """Container for all repository instances — passed through the app as one object."""

    __slots__ = ("db", "settings", "profiles", "sessions", "samples", "events", "media")

    def __init__(self, settings: SettingsRepository, profiles: ProfilesRepository,
                 sessions: SessionsRepository, samples: RiskSamplesRepository,
                 events: EventsRepository, media: MediaFilesRepository) -> None:
        self.db = settings._db
        self.settings = settings
        self.profiles = profiles
        self.sessions = sessions
        self.samples = samples
        self.events = events
        self.media = media

    def clear_history(self) -> dict[str, int]:
        """Delete monitoring history and its locally generated evidence files.

        Only regular files located under FallGuard's own data/media roots are
        removed.  This keeps a corrupted or manually edited database path from
        deleting an arbitrary user file.
        """
        conn = self.db.get_connection()
        rows = conn.execute(
            "SELECT thumbnail_path, video_clip_path FROM events"
        ).fetchall()
        media_rows = conn.execute(
            "SELECT file_path FROM media_files"
        ).fetchall()

        allowed_roots = {
            user_data_dir().resolve(),
            media_output_dir().resolve(),
        }
        candidates = {
            str(path)
            for row in rows
            for path in (row["thumbnail_path"], row["video_clip_path"])
            if path
        }
        candidates.update(
            str(row["file_path"]) for row in media_rows if row["file_path"]
        )

        removed_files = 0
        for raw_path in candidates:
            candidate = Path(raw_path).expanduser().resolve(strict=False)
            if not any(candidate.is_relative_to(root) for root in allowed_roots):
                continue
            try:
                if candidate.is_file():
                    candidate.unlink()
                    removed_files += 1
            except OSError:
                # Database history should still be removable if an evidence
                # file is locked or was already removed outside the app.
                pass

        event_count = int(
            conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        session_count = int(
            conn.execute("SELECT COUNT(*) FROM monitoring_sessions").fetchone()[0]
        )
        with self.db.transaction() as transaction:
            transaction.execute("DELETE FROM media_files")
            transaction.execute("DELETE FROM monitoring_sessions")

        return {
            "events": event_count,
            "sessions": session_count,
            "files": removed_files,
        }
