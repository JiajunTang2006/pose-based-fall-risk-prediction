"""
Repository for the ``media_files`` table — imported videos/images and event clips.

Media files are stored on disk; this table tracks their metadata and status.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..database import DatabaseManager


class MediaFilesRepository:
    """Track imported media and event media clips."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def create(self, file_path: str, media_type: str, session_id: str | None = None,
               event_id: str | None = None, file_size_bytes: int = 0,
               width: int | None = None, height: int | None = None,
               fps: float | None = None, duration_seconds: float | None = None) -> dict:
        mid = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "INSERT INTO media_files "
            "(id, session_id, event_id, file_path, media_type, file_size_bytes, "
            " width, height, fps, duration_seconds, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (mid, session_id, event_id, file_path, media_type, file_size_bytes,
             width, height, fps, duration_seconds, now),
        )
        self._db.get_connection().commit()
        return self.get(mid)  # type: ignore[return-value]

    def get(self, media_id: str) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM media_files WHERE id = ?", (media_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_status(self, media_id: str, status: str, error_message: str | None = None) -> None:
        self._db.get_connection().execute(
            "UPDATE media_files SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, media_id),
        )
        self._db.get_connection().commit()

    def list_for_session(self, session_id: str) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM media_files WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_for_event(self, event_id: str) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM media_files WHERE event_id = ? ORDER BY created_at",
            (event_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self, limit: int = 500) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM media_files ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, media_id: str) -> bool:
        cur = self._db.get_connection().execute(
            "DELETE FROM media_files WHERE id = ?", (media_id,)
        )
        self._db.get_connection().commit()
        return cur.rowcount > 0

    def delete_by_session(self, session_id: str) -> int:
        """Delete all media file records for a session. Returns count."""
        cur = self._db.get_connection().execute(
            "DELETE FROM media_files WHERE session_id = ?", (session_id,)
        )
        self._db.get_connection().commit()
        return cur.rowcount
