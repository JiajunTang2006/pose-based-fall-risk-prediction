from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fall_prediction_desktop.event_media_buffer import EventMediaBuffer


class EventMediaBufferTests(unittest.TestCase):
    def test_event_creates_thumbnail_and_clip_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repos = MagicMock()
            repos.media.create.side_effect = [
                {"id": "thumbnail-media"},
                {"id": "clip-media"},
            ]
            buffer = EventMediaBuffer(
                repos=repos,
                output_root=Path(tmp),
                session_id="session-1",
                fps=20.0,
                pre_seconds=1.0,
                post_seconds=0.0,
                clip_fps=10.0,
            )
            frame = object()

            def write_clip(capture) -> None:
                capture.clip_path.write_bytes(b"mp4")

            with patch.object(buffer, "_encode_jpeg", return_value=b"jpeg"), patch.object(
                buffer, "_write_clip", side_effect=write_clip
            ) as write_clip_mock:
                buffer.add_frame(frame, 0.0, None)
                buffer.add_frame(frame, 0.1, "event-1")
                buffer.close()

            thumbnail = Path(tmp) / "events" / "event-1" / "thumbnail.jpg"
            self.assertEqual(thumbnail.read_bytes(), b"jpeg")
            write_clip_mock.assert_called_once()
            self.assertEqual(repos.media.create.call_count, 2)
            repos.events.update_media_paths.assert_any_call(
                "event-1", thumbnail_path=str(thumbnail)
            )

    def test_pre_roll_is_bounded(self) -> None:
        buffer = EventMediaBuffer(
            repos=None,
            output_root=Path("/tmp/fallguard-test"),
            session_id=None,
            fps=30.0,
            pre_seconds=1.0,
            clip_fps=2.0,
        )
        with patch.object(buffer, "_encode_jpeg", return_value=b"jpeg"):
            for index in range(10):
                buffer.add_frame(object(), float(index), None)

        self.assertLessEqual(len(buffer._pre_roll), 4)


if __name__ == "__main__":
    unittest.main()
