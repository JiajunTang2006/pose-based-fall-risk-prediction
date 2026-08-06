from __future__ import annotations

import unittest

from fall_prediction_desktop.risk_state_machine import (
    RiskState,
    RiskStateMachine,
    StateChangeEvent,
)

_BUILDUP = 6


class TestFullLifecycle(unittest.TestCase):

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "cooldown_seconds": 5,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )

    def test_full_lifecycle(self):
        events: list[StateChangeEvent] = []

        for _ in range(10):
            e = self.fsm.update(0.10, confidence=0.9)
            self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        for _ in range(_BUILDUP):
            e = self.fsm.update(0.55, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.WARNING)
        self.assertTrue(events[0].is_escalation)

        for _ in range(_BUILDUP):
            e = self.fsm.update(0.85, confidence=0.85)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.FALL)
        self.assertTrue(events[-1].is_escalation)

        for _ in range(_BUILDUP):
            e = self.fsm.update(0.30, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.RECOVERY)
        self.assertTrue(events[-1].is_deescalation)

        for _ in range(15):
            e = self.fsm.update(0.10, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        self.assertGreaterEqual(len(events), 4)


class TestSingleFrameSuppression(unittest.TestCase):

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )

    def test_single_high_frame_does_not_trigger(self):
        e = self.fsm.update(0.90, confidence=0.9)
        self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        for _ in range(5):
            self.fsm.update(0.10, confidence=0.9)

        self.fsm.update(0.90, confidence=0.9)
        e = self.fsm.update(0.90, confidence=0.9)
        self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

    def test_consecutive_frames_trigger(self):
        for _ in range(_BUILDUP):
            e = self.fsm.update(0.90, confidence=0.9)
            if e and e.to_state == RiskState.FALL:
                break
        self.assertEqual(self.fsm.state, RiskState.FALL)


class TestHysteresis(unittest.TestCase):

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "recovery_frames": 10,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )

    def test_no_flicker_near_threshold(self):
        for _ in range(_BUILDUP):
            self.fsm.update(0.50, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.WARNING)

        transition_count = 0
        for _ in range(2):
            e = self.fsm.update(0.40, confidence=0.9)
            if e:
                transition_count += 1
        self.assertEqual(transition_count, 0, "Should not transition on brief drop")
        self.assertEqual(self.fsm.state, RiskState.WARNING)


class TestCooldown(unittest.TestCase):

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "cooldown_seconds": 2,
                "recovery_frames": 5,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )

    def test_cooldown_after_fall(self):
        for _ in range(_BUILDUP):
            self.fsm.update(0.90, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.FALL)

        for _ in range(_BUILDUP):
            self.fsm.update(0.10, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.RECOVERY)
        for _ in range(10):
            self.fsm.update(0.10, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        transitions = []
        for _ in range(10):
            e = self.fsm.update(0.90, confidence=0.9)
            if e:
                transitions.append(e)
        self.assertEqual(len(transitions), 0,
                         "Cooldown should prevent immediate re-trigger after FALL")


class TestPoseLossTolerance(unittest.TestCase):

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "lost_tolerance_frames": 10,
            },
            fps=20.0,
        )

    def test_brief_loss_tolerated(self):
        for _ in range(5):
            e = self.fsm.update(0.10, confidence=0.9, person_visible=False)
            self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

    def test_sustained_loss_triggers_lost(self):
        for _ in range(12):
            e = self.fsm.update(0.10, confidence=0.3, person_visible=False)
            if e:
                self.assertEqual(e.to_state, RiskState.LOST)
                break
        self.assertEqual(self.fsm.state, RiskState.LOST)


class TestEMASmoothing(unittest.TestCase):

    def test_ema_smooths_spikes(self):
        fsm = RiskStateMachine(
            thresholds={"ema_alpha": 0.5},
            fps=20.0,
        )
        for _ in range(10):
            fsm.update(0.10, confidence=0.9)
        self.assertLess(fsm.smoothed_risk, 0.15)

        fsm.update(0.90, confidence=0.9)
        self.assertLess(fsm.smoothed_risk, 0.60,
                        "EMA should dampen single-frame spikes")


class TestSetThresholds(unittest.TestCase):

    def test_update_thresholds(self):
        fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )
        fsm.set_thresholds(prefall=0.30, fall=0.55)
        for _ in range(_BUILDUP):
            fsm.update(0.40, confidence=0.9)
        self.assertEqual(fsm.state, RiskState.WARNING)


if __name__ == "__main__":
    unittest.main()
