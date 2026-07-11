"""
Repository for the ``app_settings`` table — key-value application config.

Usage::

    repo = SettingsRepository(db)
    repo.set("language", "zh")
    lang = repo.get("language", default="en")
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..database import DatabaseManager


class SettingsRepository:
    """Persist and retrieve application-level key-value settings."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def get(self, key: str, default: str = "") -> str:
        row = self._db.get_connection().execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._db.get_connection().execute(
            "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now),
        )
        self._db.get_connection().commit()

    def get_all(self) -> dict[str, str]:
        rows = self._db.get_connection().execute(
            "SELECT key, value FROM app_settings"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete(self, key: str) -> None:
        self._db.get_connection().execute(
            "DELETE FROM app_settings WHERE key = ?", (key,)
        )
        self._db.get_connection().commit()

    def set_bool(self, key: str, value: bool) -> None:
        self.set(key, "1" if value else "0")

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, "1" if default else "0")
        return val == "1"
