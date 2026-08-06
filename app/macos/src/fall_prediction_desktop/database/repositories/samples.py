from __future__ import annotations

from datetime import datetime, timezone

from ..database import DatabaseManager


class RiskSamplesRepository:

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def insert(self, session_id: str, frame_index: int, timestamp: float,
               risk_score: float, visibility: float, state: str,
               confidence: float = 0.0) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._db.get_connection().execute(
            "INSERT INTO risk_samples "
            "(session_id, frame_index, timestamp, risk_score, visibility, state, confidence, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, frame_index, timestamp, risk_score, visibility, state, confidence, now),
        )

    def insert_batch(self, rows: list[tuple]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        values = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], now) for r in rows]
        self._db.get_connection().executemany(
            "INSERT INTO risk_samples "
            "(session_id, frame_index, timestamp, risk_score, visibility, state, confidence, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        self._db.get_connection().commit()

    def get_for_session(self, session_id: str, limit: int = 120) -> list[dict]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM risk_samples WHERE session_id = ? "
            "ORDER BY frame_index DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_latest_for_session(self, session_id: str, seconds: float = 60.0) -> list[dict]:
        latest = self._db.get_connection().execute(
            "SELECT MAX(timestamp) AS max_ts FROM risk_samples WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not latest or latest["max_ts"] is None:
            return []
        min_ts = max(0.0, float(latest["max_ts"]) - seconds)
        rows = self._db.get_connection().execute(
            "SELECT * FROM risk_samples WHERE session_id = ? AND timestamp >= ? "
            "ORDER BY frame_index ASC LIMIT 120",
            (session_id, min_ts),
        ).fetchall()
        return [dict(r) for r in rows]

    def begin_batch(self) -> None:
        self._db.get_connection().execute("BEGIN")

    def commit(self) -> None:
        self._db.get_connection().commit()

    def end_batch(self) -> None:
        self.commit()
