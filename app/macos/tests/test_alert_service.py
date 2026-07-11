from __future__ import annotations

import unittest
from unittest.mock import patch

from fall_prediction_desktop.alert_service import SoundAlertService
from fall_prediction_desktop.risk_state_machine import RiskState, StateChangeEvent


class SoundAlertServiceTests(unittest.TestCase):
    @staticmethod
    def _event(to_state: RiskState, escalation: bool = True) -> StateChangeEvent:
        return StateChangeEvent(
            from_state=RiskState.NORMAL,
            to_state=to_state,
            frame_index=3,
            risk_score=0.8,
            confidence=0.9,
            is_escalation=escalation,
            is_deescalation=not escalation,
        )

    def test_disabled_service_does_not_schedule_sound(self) -> None:
        player = unittest.mock.MagicMock()
        service = SoundAlertService(enabled=False, player=player)

        self.assertFalse(service.on_state_change(self._event(RiskState.FALL)))
        player.assert_not_called()

    def test_warning_alert_respects_cooldown(self) -> None:
        player = unittest.mock.MagicMock()
        service = SoundAlertService(
            cooldown_seconds=5.0,
            clock=unittest.mock.MagicMock(side_effect=[10.0, 12.0]),
            player=player,
        )

        with patch("fall_prediction_desktop.alert_service.threading.Thread") as thread:
            self.assertTrue(service.on_state_change(self._event(RiskState.WARNING)))
            self.assertFalse(service.on_state_change(self._event(RiskState.WARNING)))

        thread.assert_called_once()

    def test_fall_escalation_bypasses_warning_cooldown(self) -> None:
        service = SoundAlertService(
            cooldown_seconds=5.0,
            clock=unittest.mock.MagicMock(side_effect=[10.0, 11.0]),
        )

        with patch("fall_prediction_desktop.alert_service.threading.Thread") as thread:
            self.assertTrue(service.on_state_change(self._event(RiskState.WARNING)))
            self.assertTrue(service.on_state_change(self._event(RiskState.FALL)))

        self.assertEqual(thread.call_count, 2)


if __name__ == "__main__":
    unittest.main()
