import unittest

from fall_prediction.lying_adl_filter import StaticLyingADLFilter


def standing_rows(length=15):
    return [
        {
            "has_pose": 1.0,
            "torso_angle": 5.0,
            "torso_angular_velocity": 0.0,
            "body_height": 0.55,
            "aspect_ratio": 0.36,
            "center_drop": 0.0,
            "vertical_velocity": 0.0,
        }
        for _ in range(length)
    ]


def lying_rows(length=15):
    return [
        {
            "has_pose": 1.0,
            "torso_angle": 80.0,
            "torso_angular_velocity": 0.0,
            "body_height": 0.22,
            "aspect_ratio": 0.90,
            "center_drop": 0.26,
            "vertical_velocity": 0.0,
        }
        for _ in range(length)
    ]


def fall_motion_rows(length=15):
    rows = standing_rows(length)
    rows[-3]["center_drop"] = 0.02
    rows[-2]["center_drop"] = 0.08
    rows[-1].update(
        {
            "center_drop": 0.16,
            "vertical_velocity": 0.82,
            "torso_angular_velocity": 240.0,
            "torso_angle": 70.0,
            "body_height": 0.30,
            "aspect_ratio": 0.82,
        }
    )
    return rows


class StaticLyingADLFilterTest(unittest.TestCase):
    def test_static_lying_fall_is_normal_with_only_short_advisory(self):
        rule = StaticLyingADLFilter(lying_settle_steps=3, warm_warning_steps=2)

        first = rule.process("Fall", "Fall", None, lying_rows())
        second = rule.process("Fall", "Fall", None, lying_rows())
        settled = rule.process("Fall", "Fall", None, lying_rows())

        self.assertEqual((first.state, first.alert_state, first.advisory_state), (
            "Normal", "Normal", "Pre-fall"
        ))
        self.assertEqual(second.advisory_state, "Pre-fall")
        self.assertEqual((settled.state, settled.alert_state, settled.advisory_state), (
            "Normal", "Normal", None
        ))
        self.assertFalse(rule.fall_latched)

    def test_unsampled_frames_do_not_advance_lying_settlement(self):
        rule = StaticLyingADLFilter(lying_settle_steps=2, warm_warning_steps=1)

        first = rule.process("Fall", "Fall", None, lying_rows(), advance=True)
        repeated = rule.process("Fall", "Fall", None, lying_rows(), advance=False)

        self.assertEqual(first.advisory_state, "Pre-fall")
        self.assertEqual(repeated.advisory_state, "Pre-fall")

    def test_dynamic_fall_is_latched_and_later_lying_cannot_clear_it(self):
        rule = StaticLyingADLFilter()

        fall = rule.process("Fall", "Fall", None, fall_motion_rows())
        later = rule.process("Normal", "Normal", None, lying_rows())

        self.assertEqual((fall.state, fall.alert_state), ("Fall", "Fall"))
        self.assertTrue(fall.fall_latched)
        self.assertEqual((later.state, later.alert_state), ("Fall", "Fall"))

    def test_motion_memory_can_confirm_fall_after_motion_window(self):
        rule = StaticLyingADLFilter(motion_memory_steps=2)

        warning = rule.process("Pre-fall", "Pre-fall", None, fall_motion_rows())
        fall = rule.process("Fall", "Fall", None, lying_rows())

        self.assertEqual(warning.state, "Pre-fall")
        self.assertTrue(fall.has_recent_fall_motion)
        self.assertEqual(fall.state, "Fall")
        self.assertTrue(rule.fall_latched)

    def test_fusion_only_static_fall_alert_is_suppressed_without_latching(self):
        rule = StaticLyingADLFilter(lying_settle_steps=1, warm_warning_steps=0)

        result = rule.process("Normal", "Fall", "Fall", lying_rows())

        self.assertEqual((result.state, result.alert_state, result.advisory_state), (
            "Normal", "Normal", None
        ))
        self.assertFalse(rule.fall_latched)

    def test_non_static_authoritative_fall_is_not_blocked_by_narrow_rule(self):
        rule = StaticLyingADLFilter()

        result = rule.process("Fall", "Fall", None, standing_rows())

        self.assertEqual(result.state, "Fall")
        self.assertTrue(result.fall_latched)

    def test_small_upright_person_is_not_mistaken_for_static_lying(self):
        rule = StaticLyingADLFilter()
        rows = standing_rows()
        for row in rows:
            row["body_height"] = 0.31
            row["aspect_ratio"] = 0.40

        result = rule.process("Fall", "Fall", None, rows)

        self.assertTrue(result.is_static_low_posture)
        self.assertFalse(result.is_static_lying_posture)
        self.assertEqual(result.state, "Fall")

    def test_acknowledge_clears_latch(self):
        rule = StaticLyingADLFilter()
        rule.process("Fall", "Fall", None, fall_motion_rows())

        rule.acknowledge_fall()
        result = rule.process("Normal", "Normal", None, standing_rows())

        self.assertFalse(result.fall_latched)
        self.assertEqual(result.state, "Normal")


if __name__ == "__main__":
    unittest.main()
