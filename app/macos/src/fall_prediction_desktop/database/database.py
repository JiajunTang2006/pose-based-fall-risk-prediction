"""
DatabaseManager — singleton SQLite connection with schema init and migration.

All database access goes through this module.  Frontend code MUST NOT
execute SQL directly; use the repository classes in ``repositories/``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class DatabaseError(Exception):
    """Raised when a database operation fails in a non-recoverable way."""


class DatabaseManager:
    """Thread-safe SQLite connection manager.

    Creates the database file and runs the schema on first open.
    Uses WAL mode for better concurrent read performance.
    """

    def __init__(self, db_path: Path, schema_path: Path | None = None) -> None:
        self._db_path = db_path
        self._schema_path = schema_path
        self._local = threading.local()
        # ``initialize()`` holds this lock while opening its first connection.
        # Opening a connection also registers it under the same lock, so the
        # lock must be re-entrant.  A plain Lock deadlocks before the UI is
        # created and leaves macOS showing an app that is "running" forever.
        self._lock = threading.RLock()
        self._initialized = False
        self._connections: list[sqlite3.Connection] = []  # tracked for close_all

    # ── public API ──────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create the database file and run the schema (idempotent)."""
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
        """Return a thread-local connection.  Creates one on first access."""
        if not self._initialized:
            self.initialize()
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager with explicit BEGIN...COMMIT/ROLLBACK.

        Only the work inside this block is atomic; unrelated pending
        statements on the same connection are NOT committed.
        """
        conn = self.get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the thread-local connection if open."""
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
        """Close all tracked connections (call at app shutdown)."""
        with self._lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            # Also clear the thread-local for the calling thread
            self.close()

    # ── internal ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        with self._lock:
            self._connections.append(conn)
        return conn


# ── Singleton access ──────────────────────────────────────────────────

_db_manager: DatabaseManager | None = None
_db_lock = threading.Lock()


def get_database() -> DatabaseManager:
    """Return the global DatabaseManager singleton (must call ``init_database`` first)."""
    global _db_manager
    with _db_lock:
        if _db_manager is None:
            raise DatabaseError("Database not initialized. Call init_database() first.")
        return _db_manager


def init_database(db_path: Path, schema_path: Path | None = None) -> DatabaseManager:
    """Create and return the global DatabaseManager singleton (thread-safe)."""
    global _db_manager
    with _db_lock:
        if _db_manager is not None:
            return _db_manager
        manager = DatabaseManager(db_path, schema_path)
        manager.initialize()
        _db_manager = manager
        return manager
