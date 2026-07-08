import json
import tempfile
import unittest
from pathlib import Path

from fall_prediction.config import load_predictor_config


class ConfigTest(unittest.TestCase):
    def test_load_predictor_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "state_thresholds": {
                            "prefall_threshold": 0.4,
                            "fall_threshold": 0.8,
                            "min_visibility": 0.5,
                        },
                        "temporal_smoothing": {
                            "baseline_frames": 7,
                            "smoothing_window": 3,
                            "prefall_consecutive_frames": 2,
                            "fall_consecutive_frames": 4,
                        },
                        "risk_scoring": {
                            "vertical_velocity_warn": 0.2,
                            "vertical_velocity_fall": 0.9,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_predictor_config(config_path)

        self.assertEqual(config.baseline_frames, 7)
        self.assertEqual(config.smoothing_window, 3)
        self.assertEqual(config.prefall_consecutive_frames, 2)
        self.assertEqual(config.fall_consecutive_frames, 4)
        self.assertEqual(config.risk.prefall_threshold, 0.4)
        self.assertEqual(config.risk.fall_threshold, 0.8)
        self.assertEqual(config.risk.min_visibility, 0.5)
        self.assertEqual(config.risk.vertical_velocity_warn, 0.2)
        self.assertEqual(config.risk.vertical_velocity_fall, 0.9)


if __name__ == "__main__":
    unittest.main()
