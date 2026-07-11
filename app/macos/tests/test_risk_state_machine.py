"""
Unit tests for RiskStateMachine — per FallGuard Development Workflow V1.0, Phase 3.

Covers:
  - Full NORMAL → WARNING → FALL → RECOVERY → NORMAL lifecycle
  - Single-frame anomaly suppression (consecutive-frame confirmation)
  - Hysteresis (different entry/exit thresholds)
  - Cooldown after FALL
  - Pose-loss tolerance
  - EMA smoothing
  - Recovery prevents immediate re-trigger

Note: EMA alpha=0.5 means about 3 frames to cross a threshold from 0,
plus 3 consecutive confirmations = ~6 frames per transition.
"""

from __future__ import annotations

import unittest

from fall_prediction_desktop.risk_state_machine import (
    RiskState,
    RiskStateMachine,
    StateChangeEvent,
)

# Number of frames to feed so EMA builds past threshold + confirm count
_BUILDUP = 6


class TestFullLifecycle(unittest.TestCase):
    """Test the complete NORMAL → WARNING → FALL → RECOVERY → NORMAL path."""

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

        # Phase 1: NORMAL — stay normal with low risk
        for _ in range(10):
            e = self.fsm.update(0.10, confidence=0.9)
            self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        # Phase 2: NORMAL → WARNING (risk 0.55 > prefall 0.45, need ~6 frames)
        for _ in range(_BUILDUP):
            e = self.fsm.update(0.55, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.WARNING)
        self.assertTrue(events[0].is_escalation)

        # Phase 3: WARNING → FALL (risk 0.85 > fall 0.72)
        for _ in range(_BUILDUP):
            e = self.fsm.update(0.85, confidence=0.85)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.FALL)
        self.assertTrue(events[-1].is_escalation)

        # Phase 4: FALL → RECOVERY (risk drops to 0.30)
        for _ in range(_BUILDUP):
            e = self.fsm.update(0.30, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.RECOVERY)
        self.assertTrue(events[-1].is_deescalation)

        # Phase 5: RECOVERY → NORMAL
        for _ in range(15):
            e = self.fsm.update(0.10, confidence=0.9)
            if e:
                events.append(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        # At least 4 transitions: N→W, W→F, F→R, R→N
        self.assertGreaterEqual(len(events), 4)


class TestSingleFrameSuppression(unittest.TestCase):
    """Single-frame anomaly MUST NOT trigger FALL (per doc requirement)."""

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
        # One frame of high risk → no transition
        e = self.fsm.update(0.90, confidence=0.9)
        self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        # Back to normal
        for _ in range(5):
            self.fsm.update(0.10, confidence=0.9)

        # 2 high frames → still no transition (need 3 consecutive confirms)
        self.fsm.update(0.90, confidence=0.9)
        e = self.fsm.update(0.90, confidence=0.9)
        self.assertIsNone(e)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

    def test_consecutive_frames_trigger(self):
        """Enough consecutive high-risk frames must trigger FALL."""
        # Feed 6 frames of 0.90: ~3 to cross threshold + 3 confirms
        for _ in range(_BUILDUP):
            e = self.fsm.update(0.90, confidence=0.9)
            if e and e.to_state == RiskState.FALL:
                break
        self.assertEqual(self.fsm.state, RiskState.FALL)


class TestHysteresis(unittest.TestCase):
    """State must not flicker when risk hovers near a threshold."""

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
        # Enter WARNING
        for _ in range(_BUILDUP):
            self.fsm.update(0.50, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.WARNING)

        # Brief drop below prefall threshold — should NOT immediately go to NORMAL
        transition_count = 0
        for _ in range(2):
            e = self.fsm.update(0.40, confidence=0.9)
            if e:
                transition_count += 1
        self.assertEqual(transition_count, 0, "Should not transition on brief drop")
        self.assertEqual(self.fsm.state, RiskState.WARNING)


class TestCooldown(unittest.TestCase):
    """After FALL→RECOVERY→NORMAL, cooldown must prevent immediate re-trigger."""

    def setUp(self):
        self.fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "consecutive_confirm_frames": 3,
                "cooldown_seconds": 2,  # 2s = 40 frames at 20 fps
                "recovery_frames": 5,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )

    def test_cooldown_after_fall(self):
        # Trigger FALL
        for _ in range(_BUILDUP):
            self.fsm.update(0.90, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.FALL)

        # Exit to RECOVERY → NORMAL
        for _ in range(_BUILDUP):
            self.fsm.update(0.10, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.RECOVERY)
        for _ in range(10):
            self.fsm.update(0.10, confidence=0.9)
        self.assertEqual(self.fsm.state, RiskState.NORMAL)

        # Immediately try to re-trigger — cooldown should suppress
        transitions = []
        for _ in range(10):
            e = self.fsm.update(0.90, confidence=0.9)
            if e:
                transitions.append(e)
        self.assertEqual(len(transitions), 0,
                         "Cooldown should prevent immediate re-trigger after FALL")


class TestPoseLossTolerance(unittest.TestCase):
    """Short pose loss should not trigger LOST; sustained loss should."""

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
    """EMA should smooth out noise."""

    def test_ema_smooths_spikes(self):
        fsm = RiskStateMachine(
            thresholds={"ema_alpha": 0.5},
            fps=20.0,
        )
        for _ in range(10):
            fsm.update(0.10, confidence=0.9)
        self.assertLess(fsm.smoothed_risk, 0.15)

        # One spike should not jump the smoothed value too high
        fsm.update(0.90, confidence=0.9)
        # With alpha=0.5: smoothed = 0.5*0.9 + 0.5*~0.1 ≈ 0.50
        self.assertLess(fsm.smoothed_risk, 0.60,
                        "EMA should dampen single-frame spikes")


class TestSetThresholds(unittest.TestCase):
    """Runtime threshold updates."""

    def test_update_thresholds(self):
        fsm = RiskStateMachine(
            thresholds={
                "prefall_threshold": 0.45,
                "fall_threshold": 0.72,
                "ema_alpha": 0.5,
            },
            fps=20.0,
        )
        # High sensitivity: lower thresholds
        fsm.set_thresholds(prefall=0.30, fall=0.55)
        for _ in range(_BUILDUP):
            fsm.update(0.40, confidence=0.9)
        self.assertEqual(fsm.state, RiskState.WARNING)


if __name__ == "__main__":
    unittest.main()
