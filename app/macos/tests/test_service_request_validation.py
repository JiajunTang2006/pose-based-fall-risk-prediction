from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fall_prediction_service.server import ServiceRequestHandler


class ImportRequestValidationTests(unittest.TestCase):
    def test_unsupported_existing_file_stops_before_processor(self) -> None:
        handler = object.__new__(ServiceRequestHandler)
        processor = MagicMock()
        handler.server = SimpleNamespace(
            monitor=None,
            settings=SimpleNamespace(sensitivity="medium"),
            media_processor=processor,
        )
        handler._read_json_body = MagicMock()
        handler._send_error = MagicMock()
        handler._send_json = MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            unsupported = Path(directory, "notes.txt")
            unsupported.write_text("not media", encoding="utf-8")
            handler._read_json_body.return_value = {"paths": [str(unsupported)]}

            handler._handle_create_import()

        handler._send_error.assert_called_once()
        processor.start_from_paths.assert_not_called()
        handler._send_json.assert_not_called()


class EventRequestValidationTests(unittest.TestCase):
    def _handler(self) -> tuple[ServiceRequestHandler, MagicMock]:
        handler = object.__new__(ServiceRequestHandler)
        events = MagicMock()
        handler.server = SimpleNamespace(
            _repos=SimpleNamespace(events=events),
        )
        handler._read_json_body = MagicMock()
        handler._send_error = MagicMock()
        handler._send_json = MagicMock()
        return handler, events

    def test_invalid_feedback_is_rejected(self) -> None:
        handler, events = self._handler()
        handler._read_json_body.return_value = {
            "feedback": "definitely_not_valid",
            "notes": "",
        }

        handler._handle_update_event_feedback("event-1")

        handler._send_error.assert_called_once()
        events.set_feedback.assert_not_called()

    def test_valid_feedback_updates_event(self) -> None:
        handler, events = self._handler()
        handler._read_json_body.return_value = {
            "feedback": "false_alarm",
            "notes": "Person sat down quickly.",
        }
        events.set_feedback.return_value = {
            "id": "event-1",
            "event_type": "fall",
            "status": "reviewed",
            "peak_risk": 0.9,
            "started_at": "2026-01-01T00:00:00Z",
            "user_feedback": "false_alarm",
            "notes": "Person sat down quickly.",
        }

        handler._handle_update_event_feedback("event-1")

        events.set_feedback.assert_called_once_with(
            "event-1", "false_alarm", "Person sat down quickly."
        )
        handler._send_json.assert_called_once()

    def test_training_annotation_boundaries_are_saved(self) -> None:
        handler, events = self._handler()
        handler._read_json_body.return_value = {
            "feedback": "confirmed",
            "notes": "",
            "annotation_label": "Fall",
            "prefall_start_seconds": 4.5,
            "fall_start_seconds": 7.2,
            "clip_duration_seconds": 18.0,
        }
        events.set_feedback.return_value = {
            "id": "event-1",
            "event_type": "fall",
            "status": "reviewed",
            "peak_risk": 0.9,
            "started_at": "2026-01-01T00:00:00Z",
        }

        handler._handle_update_event_feedback("event-1")

        events.set_feedback.assert_called_once_with(
            "event-1",
            "confirmed",
            "",
            annotation_label="Fall",
            prefall_start_seconds=4.5,
            fall_start_seconds=7.2,
        )
        handler._send_json.assert_called_once()

    def test_invalid_training_boundaries_are_rejected(self) -> None:
        handler, events = self._handler()
        handler._read_json_body.return_value = {
            "feedback": "confirmed",
            "notes": "",
            "annotation_label": "Fall",
            "prefall_start_seconds": 8.0,
            "fall_start_seconds": 7.0,
            "clip_duration_seconds": 18.0,
        }

        handler._handle_update_event_feedback("event-1")

        handler._send_error.assert_called_once()
        events.set_feedback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
