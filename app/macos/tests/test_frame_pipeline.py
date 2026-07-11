from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from fall_prediction.features import PoseFeatures
from fall_prediction_desktop.frame_pipeline import FrameBusinessProcessor


class FrameBusinessProcessorTests(unittest.TestCase):
    def _prediction(self, state: str, risk: float) -> SimpleNamespace:
        return SimpleNamespace(
            state=state,
            alert_state=state,
            risk_score=risk,
            features=PoseFeatures(
                frame_index=0,
                timestamp=0.0,
                has_pose=True,
                visibility_mean=0.9,
            ),
        )

    def test_processes_frames_without_repositories(self) -> None:
        processor = FrameBusinessProcessor(None, None, None, fps=20.0)

        result = processor.process(self._prediction("Normal", 0.1), 0, 0.0)

        self.assertEqual(result.state, "Normal")
        self.assertEqual(result.risk_score, 0.1)
        self.assertEqual(result.visibility, 0.9)

    def test_persists_periodic_samples_for_bound_session(self) -> None:
        repos = MagicMock()
        repos.events.list_recent.return_value = []
        processor = FrameBusinessProcessor(
            repos,
            session_id="session-1",
            profile_id="profile-1",
            fps=2.0,
        )

        processor.process(self._prediction("Normal", 0.1), 0, 0.0)
        processor.process(self._prediction("Normal", 0.2), 1, 0.5)

        repos.samples.insert.assert_called_once()
        call = repos.samples.insert.call_args.kwargs
        self.assertEqual(call["session_id"], "session-1")
        self.assertEqual(call["frame_index"], 1)

    def test_event_service_uses_bound_session_and_observes_frames(self) -> None:
        repos = MagicMock()
        repos.events.create.return_value = {"id": "event-1"}
        repos.events.get.return_value = {"id": "event-1", "event_type": "fall"}
        processor = FrameBusinessProcessor(
            repos,
            session_id="session-1",
            profile_id="profile-1",
            fps=20.0,
        )

        for frame_index in range(8):
            processor.process(self._prediction("Fall", 0.9), frame_index, frame_index / 20.0)

        repos.events.create.assert_called_once()
        call = repos.events.create.call_args.kwargs
        self.assertEqual(call["session_id"], "session-1")
        self.assertEqual(call["profile_id"], "profile-1")
        repos.sessions.get_active.assert_not_called()
    def test_notifies_optional_state_change_observer(self) -> None:
        observer = MagicMock()
        processor = FrameBusinessProcessor(
            None,
            session_id=None,
            profile_id=None,
            fps=20.0,
            state_change_observer=observer,
        )

        for frame_index in range(8):
            processor.process(self._prediction("Fall", 0.9), frame_index, frame_index / 20.0)

        observer.on_state_change.assert_called_once()
        self.assertEqual(observer.on_state_change.call_args.args[0].to_state.value, "Fall")


if __name__ == "__main__":
    unittest.main()
