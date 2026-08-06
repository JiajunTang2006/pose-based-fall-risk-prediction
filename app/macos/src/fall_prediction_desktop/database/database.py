from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class DatabaseError(Exception):
    """Raised when a database operation fails in a non-recoverable way."""


class DatabaseManager:

    def __init__(self, db_path: Path, schema_path: Path | None = None) -> None:
        self._db_path = db_path
        self._schema_path = schema_path
        self._local = threading.local()
        self._lock = threading.RLock()
        self._initialized = False
        self._connections: list[sqlite3.Connection] = []


    @property
    def path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                if not self._schema_path or not self._schema_path.is_file():
                    raise DatabaseError(
                        f"FallGuard database schema was not found: {self._schema_path}"
                    )
                conn.executescript(self._schema_path.read_text(encoding="utf-8"))
                self._migrate(conn)
                conn.commit()
                self._initialized = True
                logger.info("Database initialized: %s (v%d)", self._db_path, SCHEMA_VERSION)
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
                if conn in self._connections:
                    self._connections.remove(conn)

    def get_connection(self) -> sqlite3.Connection:
        if not self._initialized:
            self.initialize()
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            conn = self._local.conn
            try:
                conn.close()
            except Exception:
                pass
            finally:
                self._local.conn = None
                with self._lock:
                    if conn in self._connections:
                        self._connections.remove(conn)

    def close_all(self) -> None:
        with self._lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            self.close()


    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        with self._lock:
            self._connections.append(conn)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        event_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        additions = {
            "annotation_label": "TEXT",
            "prefall_start_seconds": "REAL",
            "fall_start_seconds": "REAL",
        }
        for name, column_type in additions.items():
            if name not in event_columns:
                conn.execute(f"ALTER TABLE events ADD COLUMN {name} {column_type}")


_db_manager: DatabaseManager | None = None
_db_lock = threading.Lock()


def get_database() -> DatabaseManager:
    global _db_manager
    with _db_lock:
        if _db_manager is None:
            raise DatabaseError("Database not initialized. Call init_database() first.")
        return _db_manager


def init_database(db_path: Path, schema_path: Path | None = None) -> DatabaseManager:
    global _db_manager
    with _db_lock:
        if _db_manager is not None:
            return _db_manager
        manager = DatabaseManager(db_path, schema_path)
        manager.initialize()
        _db_manager = manager
        return manager
