from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from fall_prediction_desktop.database.database import DatabaseError, DatabaseManager
from fall_prediction_desktop.database.repositories import (
    ProfilesRepository,
    RiskSamplesRepository,
    SessionsRepository,
)
from fall_prediction_desktop.web_app import ProfileManager


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fall_prediction_desktop"
    / "database"
    / "schema.sql"
)


class DatabaseManagerTests(unittest.TestCase):
    def test_initialize_does_not_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "fallguard.db", SCHEMA_PATH)
            errors: list[BaseException] = []

            def initialize() -> None:
                try:
                    db.initialize()
                except BaseException as exc:  # pragma: no cover - assertion captures it
                    errors.append(exc)

            worker = threading.Thread(target=initialize)
            worker.start()
            worker.join(timeout=2.0)

            self.assertFalse(worker.is_alive(), "database initialization deadlocked")
            self.assertEqual(errors, [])
            table = db.get_connection().execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='profiles'"
            ).fetchone()
            self.assertIsNotNone(table)
            db.close_all()

    def test_missing_schema_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "fallguard.db", Path(tmp) / "missing.sql")
            with self.assertRaises(DatabaseError):
                db.initialize()


class RepositoryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = DatabaseManager(Path(self.tempdir.name) / "fallguard.db", SCHEMA_PATH)
        self.db.initialize()
        self.profiles = ProfilesRepository(self.db)
        self.sessions = SessionsRepository(self.db)
        self.samples = RiskSamplesRepository(self.db)
        self.profile = self.profiles.create("Default")

    def tearDown(self) -> None:
        self.db.close_all()
        self.tempdir.cleanup()

    def test_invalid_thresholds_are_not_written(self) -> None:
        with self.assertRaises(ValueError):
            self.profiles.update(
                self.profile["id"],
                prefall_threshold=0.9,
                fall_threshold=0.5,
            )
        stored = self.profiles.get(self.profile["id"])
        self.assertEqual(stored["prefall_threshold"], 0.45)
        self.assertEqual(stored["fall_threshold"], 0.72)

    def test_interrupted_sessions_are_recovered(self) -> None:
        session = self.sessions.create(self.profile["id"])
        self.assertEqual(session["status"], "running")
        self.assertEqual(self.sessions.recover_interrupted(), 1)
        recovered = self.sessions.get(session["id"])
        self.assertEqual(recovered["status"], "error")
        self.assertIsNotNone(recovered["ended_at"])

    def test_risk_samples_commit_pending_inserts(self) -> None:
        session = self.sessions.create(self.profile["id"])
        self.samples.insert(session["id"], 10, 1.0, 0.3, 0.9, "Normal")

        self.assertTrue(self.db.get_connection().in_transaction)
        self.samples.commit()

        self.assertFalse(self.db.get_connection().in_transaction)
        stored = self.samples.get_for_session(session["id"])
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["frame_index"], 10)

    def test_risk_samples_end_batch_uses_commit_contract(self) -> None:
        session = self.sessions.create(self.profile["id"])
        self.samples.insert(session["id"], 11, 2.0, 0.4, 0.8, "Pre-fall")

        self.samples.end_batch()

        self.assertFalse(self.db.get_connection().in_transaction)
        self.assertEqual(len(self.samples.get_for_session(session["id"])), 1)

    def test_profile_manager_and_repository_share_active_profile(self) -> None:
        root = Path(self.tempdir.name)
        manager = ProfileManager(root, data_dir=root, repository=self.profiles)
        alice = manager.create("Alice")
        self.assertTrue(manager.activate(alice.id))
        self.assertEqual(manager.active_id, alice.id)
        self.assertEqual(self.profiles.get_active()["id"], alice.id)


if __name__ == "__main__":
    unittest.main()
