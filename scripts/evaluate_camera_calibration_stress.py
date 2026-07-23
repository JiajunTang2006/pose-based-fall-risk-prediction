"""Numerically verify standing calibration under synthetic 2-D camera changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.ml_features import _safe_float
from fall_prediction.robustness import ROBUST_ML_FEATURE_COLUMNS, calibrate_feature_rows
from fall_prediction.window_dataset import load_feature_rows


SCENARIOS = {
    "camera_roll_20deg": {"roll": 20.0},
    "farther_uniform_scale": {"scale_x": 0.65, "scale_y": 0.65},
    "vertical_reframe": {"offset_y": 0.18},
    "mild_pitch_proxy": {"roll": -12.0, "scale_x": 0.82, "scale_y": 0.68, "offset_y": 0.10},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=["outputs/features/urfall_yolo", "outputs/features/upfall_yolo"],
    )
    parser.add_argument("--baseline-frames", type=int, default=15)
    parser.add_argument("--output", default="reports/camera_calibration_stress_eval.json")
    args = parser.parse_args()

    csv_paths = sorted(
        path
        for directory in args.feature_dir
        for path in Path(directory).glob("*.csv")
    )
    report = {"baseline_frames": args.baseline_frames, "videos": len(csv_paths), "scenarios": {}}

    for name, params in SCENARIOS.items():
        absolute_errors: list[float] = []
        compared_rows = 0
        skipped_videos = 0
        for path in csv_paths:
            raw_rows = load_feature_rows(path)
            original, _ = calibrate_feature_rows(raw_rows, baseline_frames=args.baseline_frames)
            changed, _ = calibrate_feature_rows(
                [_transform_row(row, **params) for row in raw_rows],
                baseline_frames=args.baseline_frames,
            )
            if not original or len(original) != len(changed):
                skipped_videos += 1
                continue
            for left, right in zip(original, changed):
                for column in ROBUST_ML_FEATURE_COLUMNS:
                    absolute_errors.append(abs(left[column] - right[column]))
                compared_rows += 1

        report["scenarios"][name] = {
            "parameters": params,
            "compared_rows": compared_rows,
            "skipped_videos": skipped_videos,
            "max_absolute_feature_error": max(absolute_errors, default=0.0),
            "mean_absolute_feature_error": (
                sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, item in report["scenarios"].items():
        print(
            f"{name:24s} rows={item['compared_rows']} "
            f"max_error={item['max_absolute_feature_error']:.3e} "
            f"mean_error={item['mean_absolute_feature_error']:.3e}"
        )
    print(f"Wrote {output}")


def _transform_row(
    row,
    roll: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_y: float = 0.0,
):
    changed = dict(row)
    raw_angle = _safe_float(row.get("torso_signed_angle", row.get("torso_angle", 0.0)))
    changed["torso_signed_angle"] = raw_angle + roll
    changed["torso_angle"] = abs(raw_angle + roll)
    changed["body_center_y"] = _safe_float(row.get("body_center_y", 0.0)) * scale_y + offset_y
    changed["body_center_delta"] = _safe_float(row.get("body_center_delta", 0.0)) * scale_y
    changed["vertical_velocity"] = _safe_float(row.get("vertical_velocity", 0.0)) * scale_y
    changed["center_drop"] = _safe_float(row.get("center_drop", 0.0)) * scale_y
    changed["body_width"] = _safe_float(row.get("body_width", 0.0)) * scale_x
    changed["body_height"] = _safe_float(row.get("body_height", 0.0)) * scale_y
    changed["aspect_ratio"] = _safe_float(row.get("aspect_ratio", 0.0)) * scale_x / max(scale_y, 1e-6)
    return changed


if __name__ == "__main__":
    main()
