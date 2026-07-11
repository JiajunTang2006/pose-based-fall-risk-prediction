"""
Repository for the ``monitoring_sessions`` table.

A session spans one Start→Stop cycle (camera) or one media import job.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..database import DatabaseManager


class SessionsRepository:
    """Manage monitoring session lifecycle."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def create(self, profile_id: str, source_type: str = "camera",
               source_path: str | None = None, model_version: str | None = None,
               pose_backend: str = "yolo", predictor_type: str = "ml",
               resolution: str | None = None) -> dict:
        sid = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "INSERT INTO monitoring_sessions "
            "(id, profile_id, source_type, source_path, status, model_version, "
            " pose_backend, predictor_type, resolution, started_at) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)",
            (sid, profile_id, source_type, source_path, model_version,
             pose_backend, predictor_type, resolution, now),
        )
        self._db.get_connection().commit()
        return self.get(sid)  # type: ignore[return-value]

    def get(self, session_id: str) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM monitoring_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_active(self) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM monitoring_sessions WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def recover_interrupted(self) -> int:
        """Mark sessions left running by an earlier process as errors."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._db.get_connection().execute(
            "UPDATE monitoring_sessions "
            "SET status = 'error', ended_at = ?, "
            "error_message = COALESCE(error_message, 'Application exited before the session was closed') "
            "WHERE status = 'running'",
            (now,),
        )
        self._db.get_connection().commit()
        return cur.rowcount

    def stop(self, session_id: str, total_frames: int = 0,
             total_events: int = 0, peak_risk: float = 0.0,
             avg_risk: float = 0.0, fps_avg: float = 0.0,
             error_message: str | None = None) -> dict | None:
        status = "error" if error_message else "stopped"
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "UPDATE monitoring_sessions SET status = ?, ended_at = ?, "
            "total_frames = ?, total_events = ?, peak_risk = ?, avg_risk = ?, "
            "fps_avg = ?, error_message = ? WHERE id = ?",
            (status, now, total_frames, total_events, peak_risk, avg_risk,
             fps_avg, error_message, session_id),
        )
        self._db.get_connection().commit()
        return self.get(session_id)

    def cancel(self, session_id: str) -> dict | None:
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "UPDATE monitoring_sessions SET status = 'cancelled', ended_at = ? WHERE id = ?",
            (now, session_id),
        )
        self._db.get_connection().commit()
        return self.get(session_id)

    def update_stats(self, session_id: str, **fields) -> None:
        allowed = {"total_frames", "total_events", "peak_risk", "avg_risk", "fps_avg", "resolution"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        self._db.get_connection().execute(
            f"UPDATE monitoring_sessions SET {set_clause} WHERE id = ?", values
        )
        self._db.get_connection().commit()

    def list_recent(self, limit: int = 20) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM monitoring_sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self._db.get_connection().execute(
            "SELECT COUNT(*) AS cnt FROM monitoring_sessions"
        ).fetchone()
        return row["cnt"]
