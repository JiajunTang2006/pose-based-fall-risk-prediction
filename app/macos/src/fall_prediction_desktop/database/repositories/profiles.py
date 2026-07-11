"""
Repository for the ``profiles`` table — per-person detection configs.

Only ONE profile may be ``is_active = 1`` at any time (enforced in Python,
not via a DB trigger, to keep the schema simple).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..database import DatabaseManager


class ProfilesRepository:
    """CRUD for user profiles with detection thresholds."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── create / read / update / delete ────────────────────────────

    def create(self, name: str) -> dict:
        pid = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM profiles").fetchone()
            is_active = 1 if row["cnt"] == 0 else 0
            conn.execute(
                "INSERT INTO profiles (id, name, is_active, sensitivity, created_at, updated_at) "
                "VALUES (?, ?, ?, 'medium', ?, ?)",
                (pid, name.strip() or "Unnamed", is_active, now, now),
            )
        return self.get(pid)  # type: ignore[return-value]

    def upsert(self, profile_id: str, name: str, *, is_active: bool = False) -> dict:
        """Import or update a profile while preserving its legacy ID."""
        if not profile_id:
            raise ValueError("profile_id is required")
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO profiles (id, name, is_active, sensitivity, created_at, updated_at) "
                "VALUES (?, ?, 0, 'medium', ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
                (profile_id, name.strip() or "Unnamed", now, now),
            )
            if is_active:
                conn.execute("UPDATE profiles SET is_active = 0")
                conn.execute("UPDATE profiles SET is_active = 1 WHERE id = ?", (profile_id,))
        return self.get(profile_id)  # type: ignore[return-value]

    def get(self, profile_id: str) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_active(self) -> dict | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM profiles WHERE is_active = 1"
        ).fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM profiles ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        row = self._db.get_connection().execute(
            "SELECT COUNT(*) AS cnt FROM profiles"
        ).fetchone()
        return row["cnt"]

    def update(self, profile_id: str, **fields) -> dict | None:
        allowed = {"name", "sensitivity", "prefall_threshold", "fall_threshold",
                    "consecutive_frames", "cooldown_seconds", "camera_index"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get(profile_id)
        current = self.get(profile_id)
        if current is None:
            return None
        merged = {**current, **updates}
        sensitivity = str(merged["sensitivity"])
        if sensitivity not in {"low", "medium", "high"}:
            raise ValueError("sensitivity must be low, medium, or high")
        prefall = float(merged["prefall_threshold"])
        fall = float(merged["fall_threshold"])
        if not (0.0 <= prefall < fall <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= prefall < fall <= 1")
        consecutive = int(merged["consecutive_frames"])
        cooldown = int(merged["cooldown_seconds"])
        camera_index = int(merged["camera_index"])
        if not (1 <= consecutive <= 300):
            raise ValueError("consecutive_frames must be between 1 and 300")
        if not (0 <= cooldown <= 3600):
            raise ValueError("cooldown_seconds must be between 0 and 3600")
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [profile_id]
        self._db.get_connection().execute(
            f"UPDATE profiles SET {set_clause} WHERE id = ?", values
        )
        self._db.get_connection().commit()
        return self.get(profile_id)

    def delete(self, profile_id: str) -> bool:
        if self.count() <= 1:
            return False
        try:
            with self._db.transaction() as conn:
                conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
                if self.get_active() is None:
                    first = conn.execute(
                        "SELECT id FROM profiles ORDER BY created_at LIMIT 1"
                    ).fetchone()
                    if first:
                        conn.execute(
                            "UPDATE profiles SET is_active = 1 WHERE id = ?",
                            (first["id"],)
                        )
            return True
        except Exception:
            return False

    def activate(self, profile_id: str) -> bool:
        profile = self.get(profile_id)
        if profile is None:
            return False
        with self._db.transaction() as conn:
            conn.execute("UPDATE profiles SET is_active = 0")
            conn.execute("UPDATE profiles SET is_active = 1 WHERE id = ?", (profile_id,))
        return True

    # ── threshold helpers ──────────────────────────────────────────

    def get_thresholds(self, profile_id: str | None = None) -> dict:
        """Return {prefall_threshold, fall_threshold, consecutive_frames, cooldown_seconds}."""
        profile = self.get(profile_id) if profile_id else self.get_active()
        if profile is None:
            return {
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_frames": 3,
                "cooldown_seconds": 30,
            }
        return {
            "prefall_threshold": profile.get("prefall_threshold", 0.45),
            "fall_threshold": profile.get("fall_threshold", 0.72),
            "consecutive_frames": profile.get("consecutive_frames", 3),
            "cooldown_seconds": profile.get("cooldown_seconds", 30),
        }
