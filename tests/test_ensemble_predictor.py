import unittest

from fall_prediction.ensemble_predictor import (
    DualModelDecisionEngine,
    combine_state_sequences,
)


class DualModelDecisionEngineTest(unittest.TestCase):
    def test_fusion_prefall_is_an_early_alert_not_a_confirmed_state(self):
        decision = DualModelDecisionEngine().decide("Normal", "Pre-fall")

        self.assertEqual(decision.state, "Normal")
        self.assertEqual(decision.alert_state, "Normal")
        self.assertEqual(decision.advisory_state, "Pre-fall")
        self.assertEqual(decision.tier, "watch-fusion")

    def test_tree_prefall_confirms_warning_without_fusion_agreement(self):
        decision = DualModelDecisionEngine().decide("Pre-fall", "Normal")

        self.assertEqual(decision.state, "Pre-fall")
        self.assertEqual(decision.alert_state, "Pre-fall")
        self.assertEqual(decision.tier, "warning-tree")

    def test_tree_fall_is_immediately_authoritative(self):
        decision = DualModelDecisionEngine(3).decide("Fall", "Normal")

        self.assertEqual(decision.state, "Fall")
        self.assertEqual(decision.alert_state, "Fall")
        self.assertEqual(decision.tier, "critical-tree-confirmed")

    def test_fusion_only_fall_requires_consecutive_confirmation(self):
        engine = DualModelDecisionEngine(3)

        first = engine.decide("Normal", "Fall")
        second = engine.decide("Normal", "Fall")
        third = engine.decide("Normal", "Fall")

        self.assertEqual((first.state, second.state, third.state), (
            "Normal", "Normal", "Normal"
        ))
        self.assertEqual((first.alert_state, second.alert_state, third.alert_state), (
            "Normal", "Normal", "Fall"
        ))
        self.assertEqual((first.advisory_state, second.advisory_state, third.advisory_state), (
            "Pre-fall", "Pre-fall", "Fall"
        ))
        self.assertEqual(third.tier, "critical-fusion-alert")

    def test_non_fall_output_breaks_fusion_only_fall_streak(self):
        engine = DualModelDecisionEngine(2)
        engine.decide("Normal", "Fall")
        engine.decide("Normal", "Normal")

        decision = engine.decide("Normal", "Fall")

        self.assertEqual(decision.state, "Normal")
        self.assertEqual(decision.alert_state, "Normal")
        self.assertEqual(decision.advisory_state, "Pre-fall")
        self.assertIn("1/2", decision.tier)

    def test_unsampled_frames_do_not_advance_fusion_fall_confirmation(self):
        engine = DualModelDecisionEngine(2)

        first = engine.decide("Normal", "Fall")
        repeated = engine.decide(
            "Normal", "Fall", advance_fusion_counter=False
        )

        self.assertIn("1/2", first.tier)
        self.assertIn("1/2", repeated.tier)

    def test_offline_sequence_helper_resets_between_calls(self):
        first = combine_state_sequences(
            ["Normal", "Normal"],
            ["Fall", "Fall"],
            fusion_fall_confirmation_steps=2,
        )[0]
        second = combine_state_sequences(
            ["Normal"],
            ["Fall"],
            fusion_fall_confirmation_steps=2,
        )[0]

        self.assertEqual(first, ["Normal", "Normal"])
        self.assertEqual(second, ["Normal"])


if __name__ == "__main__":
    unittest.main()
