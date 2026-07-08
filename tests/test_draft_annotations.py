import unittest
from pathlib import Path

from fall_prediction.draft_annotations import draft_intervals_for_csv


class DraftAnnotationsTest(unittest.TestCase):
    def test_adl_sequence_is_all_normal(self):
        rows = [{"frame": "0"}, {"frame": "1"}, {"frame": "2"}]

        intervals = draft_intervals_for_csv(Path("adl-01-cam0-rgb.csv"), rows)

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].label, "Normal")
        self.assertEqual(intervals[0].start_frame, 0)
        self.assertEqual(intervals[0].end_frame, 2)

    def test_fall_sequence_gets_normal_prefall_fall(self):
        rows = []
        for frame in range(10):
            rows.append(
                {
                    "frame": str(frame),
                    "has_pose": "1",
                    "instant_state": "Fall" if frame == 7 else "Normal",
                    "risk_score": "0.0",
                    "smoothed_risk_score": "0.0",
                    "torso_angle": "0.0",
                    "center_drop": "0.0",
                }
            )

        intervals = draft_intervals_for_csv(Path("fall-01-cam0-rgb.csv"), rows, prefall_frames=3)

        self.assertEqual([(item.start_frame, item.end_frame, item.label) for item in intervals], [
            (0, 3, "Normal"),
            (4, 6, "Pre-fall"),
            (7, 9, "Fall"),
        ])


if __name__ == "__main__":
    unittest.main()
