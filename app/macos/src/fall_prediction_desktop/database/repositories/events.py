"""
Repository for the ``events`` table — business-level fall/pre-fall detection events.

Each event represents a continuous risk episode (not a single frame).
The state machine is responsible for creating, updating, and closing events.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..database import DatabaseManager


class EventsRepository:
    """Manage the lifecycle of fall/pre-fall detection events."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def create(self, session_id: str, profile_id: str, event_type: str,
               risk_score: float = 0.0, confidence: float = 0.0) -> dict:
        eid = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "INSERT INTO events "
            "(id, session_id, profile_id, event_type, status, peak_risk, avg_risk, "
            " min_confidence, started_at, created_at) "
            "VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)",
            (eid, session_id, profile_id, event_type, risk_score, risk_score,
             confidence, now, now),
        )
        self._db.get_connection().commit()
        return self.get(eid)  # type: ignore[return-value]

    def get(self, event_id: str) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_open_for_session(self, session_id: str) -> dict | None:
        """Return the currently-open event for a session, if any."""
        row = self._db.get_connection().execute(
            "SELECT * FROM events WHERE session_id = ? AND status = 'open' "
            "ORDER BY started_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_peak(self, event_id: str, risk_score: float) -> None:
        self._db.get_connection().execute(
            "UPDATE events SET peak_risk = MAX(peak_risk, ?) WHERE id = ?",
            (risk_score, event_id),
        )
        self._db.get_connection().commit()

    def update_type(self, event_id: str, event_type: str) -> None:
        """Update the event type (e.g. pre-fall → fall upgrade)."""
        self._db.get_connection().execute(
            "UPDATE events SET event_type = ? WHERE id = ?",
            (event_type, event_id),
        )
        self._db.get_connection().commit()

    def close(self, event_id: str, duration_seconds: float = 0.0,
              avg_risk: float | None = None) -> dict | None:
        now = datetime.now(timezone.utc).isoformat()
        updates = {"status": "ended", "ended_at": now, "duration_seconds": duration_seconds}
        if avg_risk is not None:
            updates["avg_risk"] = avg_risk
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [event_id]
        self._db.get_connection().execute(
            f"UPDATE events SET {set_clause} WHERE id = ?", values
        )
        self._db.get_connection().commit()
        return self.get(event_id)

    def update_media_paths(
        self,
        event_id: str,
        thumbnail_path: str | None = None,
        video_clip_path: str | None = None,
    ) -> None:
        updates: dict[str, str] = {}
        if thumbnail_path is not None:
            updates["thumbnail_path"] = thumbnail_path
        if video_clip_path is not None:
            updates["video_clip_path"] = video_clip_path
        if not updates:
            return
        set_clause = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [event_id]
        self._db.get_connection().execute(
            f"UPDATE events SET {set_clause} WHERE id = ?", values
        )
        self._db.get_connection().commit()

    def set_feedback(self, event_id: str, feedback: str, notes: str = "") -> dict | None:
        self._db.get_connection().execute(
            "UPDATE events SET user_feedback = ?, notes = ?, status = 'reviewed' WHERE id = ?",
            (feedback, notes, event_id),
        )
        self._db.get_connection().commit()
        return self.get(event_id)

    def list_for_session(self, session_id: str) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY started_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_recent(self, limit: int = 12) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM events ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def count_for_session(self, session_id: str) -> int:
        row = self._db.get_connection().execute(
            "SELECT COUNT(*) AS cnt FROM events WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["cnt"]

    def delete(self, event_id: str) -> bool:
        cur = self._db.get_connection().execute(
            "DELETE FROM events WHERE id = ?", (event_id,)
        )
        self._db.get_connection().commit()
        return cur.rowcount > 0

    def delete_all(self) -> None:
        """Clear all events (for 'Clear History')."""
        self._db.get_connection().execute("DELETE FROM events")
        self._db.get_connection().commit()
