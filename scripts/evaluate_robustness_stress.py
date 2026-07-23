"""Evaluate a robust artifact under deterministic partial-pose stress patterns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.ml_features import compute_window_accel_features, flatten_window
from fall_prediction.ml_predictor import MachineLearningFallPredictor
from fall_prediction.ml_predictor import DEFAULT_PARTIAL_POSE_GRACE_FRAMES
from fall_prediction.robustness import apply_partial_pose_dropout, calibrate_feature_rows
from fall_prediction.train_model import build_validation_metrics
from fall_prediction.window_dataset import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    _label_for_window,
    _row_frame,
    _video_key,
    infer_label_from_filename,
    load_feature_rows,
    load_label_intervals,
)


PATTERNS = ("original", "torso", "center", "bbox", "temporal", "lower_body", "upper_body")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo_tail60_prefall_accel_robust_classifier.joblib")
    parser.add_argument("--annotations", default="data/ur_up_train_drop60f_15pct_annotations.csv")
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=None,
    )
    parser.add_argument("--output", default="reports/current_model_robust_stress_eval.json")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--sensitivity", choices=("high", "medium", "low"), default="high")
    parser.add_argument("--pattern", action="append", choices=PATTERNS, default=None)
    args = parser.parse_args()

    feature_dirs = args.feature_dir or ["outputs/features/urfall_yolo", "outputs/features/upfall_yolo"]
    csv_paths = sorted(
        path
        for directory in feature_dirs
        for path in Path(directory).glob("*.csv")
    )
    intervals = load_label_intervals(args.annotations)
    pattern_reports = {}

    patterns = args.pattern or PATTERNS
    missing_grace_windows = max(
        1,
        (DEFAULT_PARTIAL_POSE_GRACE_FRAMES + args.stride - 1) // args.stride,
    )
    for pattern in patterns:
        predictor = MachineLearningFallPredictor(
            args.model,
            use_hmm=True,
            use_accel=True,
            use_temporal_fall_validation=True,
            temporal_sensitivity=args.sensitivity,
        )
        true_labels: list[str] = []
        raw_predictions: list[str] = []
        runtime_predictions: list[str] = []
        video_runtime_states: dict[str, Counter] = {}

        for csv_path in csv_paths:
            predictor.reset()
            raw_rows = load_feature_rows(csv_path)
            model_rows, _baseline = calibrate_feature_rows(
                raw_rows,
                baseline_frames=predictor.baseline_frames,
                min_visibility=predictor.min_visibility,
            )
            if len(model_rows) != len(raw_rows):
                continue
            video_key = _video_key(csv_path)
            file_label = infer_label_from_filename(csv_path)
            state_counts: Counter = Counter()
            missing_pose_count = 0

            for start in range(0, len(raw_rows) - args.window_size + 1, args.stride):
                raw_window = raw_rows[start : start + args.window_size]
                model_window = model_rows[start : start + args.window_size]
                end_frame = _row_frame(raw_window[-1], start + args.window_size - 1)
                label = _label_for_window(
                    csv_path=csv_path,
                    video_key=video_key,
                    end_frame=end_frame,
                    file_label=file_label,
                    label_mode="annotations",
                    intervals=intervals,
                )
                if label is None:
                    continue

                stressed_model = (
                    list(model_window)
                    if pattern == "original"
                    else apply_partial_pose_dropout(model_window, pattern)
                )
                stressed_raw = (
                    list(raw_window)
                    if pattern == "original"
                    else apply_partial_pose_dropout(raw_window, pattern)
                )
                base_columns = tuple(
                    column
                    for column in predictor.feature_columns
                    if column not in {"torso_angular_accel", "vertical_accel"}
                )
                prepared = compute_window_accel_features(
                    stressed_model,
                    base_feature_columns=base_columns,
                )
                current_has_pose = float(stressed_model[-1].get("has_pose", 0.0)) > 0.0
                missing_pose_count = 0 if current_has_pose else missing_pose_count + 1
                if (
                    not current_has_pose
                    and missing_pose_count > missing_grace_windows
                ):
                    raw_state = "Unknown"
                    runtime_state = "Unknown"
                else:
                    sample = [flatten_window(prepared, predictor.feature_columns)]
                    raw_state, _risk, raw_alert = predictor._predict_sample(sample)
                    runtime_state, _runtime_alert = predictor._apply_temporal_validation(
                        raw_state,
                        raw_alert,
                        prepared,
                        stressed_raw,
                    )

                true_labels.append(label)
                raw_predictions.append(raw_state)
                runtime_predictions.append(runtime_state)
                state_counts[runtime_state] += 1

            video_runtime_states[video_key] = state_counts

        labels = sorted(set(true_labels) | set(raw_predictions) | set(runtime_predictions))
        fall_videos = [key for key in video_runtime_states if key.startswith(("fall", "subject"))]
        adl_videos = [key for key in video_runtime_states if key not in fall_videos]
        detected_fall_videos = [
            key for key in fall_videos if video_runtime_states[key]["Fall"] > 0
        ]
        false_adl_videos = [
            key for key in adl_videos if video_runtime_states[key]["Fall"] > 0
        ]
        pattern_reports[pattern] = {
            "window_count": len(true_labels),
            "raw_metrics": build_validation_metrics(true_labels, raw_predictions, labels),
            "runtime_metrics": build_validation_metrics(true_labels, runtime_predictions, labels),
            "sequence": {
                "fall_videos": len(fall_videos),
                "fall_videos_with_runtime_fall": sum(
                    video_runtime_states[key]["Fall"] > 0 for key in fall_videos
                ),
                "fall_runtime_missed_videos": sorted(set(fall_videos) - set(detected_fall_videos)),
                "adl_videos": len(adl_videos),
                "adl_videos_with_false_runtime_fall": sum(
                    video_runtime_states[key]["Fall"] > 0 for key in adl_videos
                ),
                "adl_false_fall_videos": sorted(false_adl_videos),
            },
        }

    report = {
        "model": args.model,
        "annotations": args.annotations,
        "sensitivity": args.sensitivity,
        "patterns": pattern_reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for pattern, item in pattern_reports.items():
        runtime = item["runtime_metrics"]["classification_report"]
        sequence = item["sequence"]
        print(
            f"{pattern:8s} "
            f"N_rec={runtime.get('Normal', {}).get('recall', 0.0):.3f} "
            f"PF_rec={runtime.get('Pre-fall', {}).get('recall', 0.0):.3f} "
            f"F_rec={runtime.get('Fall', {}).get('recall', 0.0):.3f} "
            f"fall_videos={sequence['fall_videos_with_runtime_fall']}/{sequence['fall_videos']} "
            f"adl_false={sequence['adl_videos_with_false_runtime_fall']}/{sequence['adl_videos']}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
