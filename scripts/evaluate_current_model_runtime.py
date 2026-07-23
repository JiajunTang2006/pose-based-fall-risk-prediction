"""
Evaluate the current trained model on exported feature CSVs.

This is a lightweight post-training sanity check. It does not re-run pose
estimation; it reuses feature CSVs and simulates the runtime ML path:

    model probabilities -> optional HMM -> alert layer -> TemporalSequenceGate

The report includes optimistic all-data window metrics plus sequence-level
checks for ADL false Fall alarms and Fall-video detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.ml_features import compute_window_accel_features, flatten_window
from fall_prediction.ml_predictor import MachineLearningFallPredictor
from fall_prediction.robustness import calibrate_feature_rows
from fall_prediction.skeleton_dataset import index_landmark_csvs, normalize_skeleton_rows
from fall_prediction.train_model import build_validation_metrics
from fall_prediction.window_dataset import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    _annotation_keys,
    _label_for_window,
    _row_frame,
    _video_key,
    infer_label_from_filename,
    load_feature_rows,
    load_label_intervals,
)


DEFAULT_PREFALL_BOUNDARY_RATIO = 0.20
DEFAULT_CONFIRMED_LYING_SEGMENTS = (
    ("adl-10-cam0-rgb", 203, 299),
    ("adl-21-cam0-rgb", 237, 279),
    ("adl-31-cam0-rgb", 229, 249),
    ("adl-36-cam0-rgb", 229, 339),
    ("adl-37-cam0-rgb", 256, 349),
    ("adl-22-cam0-rgb", 171, 239),
    ("adl-23-cam0-rgb", 147, 219),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current runtime model on feature CSVs.")
    parser.add_argument("--model", default="models/yolo_tail60_prefall_accel_robust_classifier.joblib")
    parser.add_argument("--annotations", default="data/ur_up_train_drop60f_15pct_annotations.csv")
    parser.add_argument("--feature-dir", action="append", default=None)
    parser.add_argument("--landmark-dir", action="append", default=None)
    parser.add_argument("--output", default="reports/current_model_runtime_eval.json")
    parser.add_argument(
        "--group-filter-metrics",
        default=None,
        help="Only evaluate validation_groups recorded in this metrics JSON.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--use-hmm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-temporal-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prefall-boundary-ratio",
        type=float,
        default=DEFAULT_PREFALL_BOUNDARY_RATIO,
        help="Report-style lenient Pre-fall boundary ratio. Default: 0.20 means evaluate only the core 60%.",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("high", "medium", "low"),
        default="high",
        help="Runtime temporal gate sensitivity.",
    )
    parser.add_argument(
        "--prefall-alert-threshold",
        type=float,
        default=None,
        help="Override the model artifact Pre-fall alert threshold. Default: use the artifact value.",
    )
    args = parser.parse_args()

    feature_dirs = args.feature_dir or ["outputs/features/urfall_yolo", "outputs/features/upfall_yolo"]
    csv_paths = collect_csv_paths([Path(path) for path in feature_dirs])
    if args.group_filter_metrics:
        split_report = json.loads(Path(args.group_filter_metrics).read_text(encoding="utf-8"))
        allowed_groups = set(split_report["validation_split"]["validation_groups"])
        csv_paths = [path for path in csv_paths if _video_key(path) in allowed_groups]
    intervals = load_label_intervals(args.annotations)
    predictor = MachineLearningFallPredictor(
        args.model,
        prefall_alert_threshold=args.prefall_alert_threshold,
        prefall_alert_consecutive_frames=1,
        use_hmm=args.use_hmm,
        use_accel=True,
        use_temporal_fall_validation=args.use_temporal_validation,
        temporal_sensitivity=args.sensitivity,
    )
    landmark_index = None
    if predictor._requires_skeleton:
        landmark_dirs = args.landmark_dir or [
            "outputs/landmarks_upperbody/urfall_yolo",
            "outputs/landmarks_upperbody/upfall_yolo",
        ]
        landmark_index = index_landmark_csvs(landmark_dirs)

    raw_true: list[str] = []
    raw_pred: list[str] = []
    runtime_true: list[str] = []
    runtime_pred: list[str] = []
    alert_pred: list[str] = []
    lenient_true: list[str] = []
    lenient_raw_pred: list[str] = []
    lenient_runtime_pred: list[str] = []
    lenient_alert_pred: list[str] = []
    prefall_region_events: list[dict] = []
    video_summaries: dict[str, dict] = {}

    for csv_path in csv_paths:
        predictor.reset()
        rows = load_feature_rows(csv_path)
        skeleton_sequence = None
        if landmark_index is not None:
            landmark_path = landmark_index.get(csv_path.stem.lower())
            if landmark_path is None:
                raise FileNotFoundError(f"No landmark CSV matches {csv_path}")
            skeleton_sequence = normalize_skeleton_rows(load_feature_rows(landmark_path))
        model_rows = rows
        if predictor._use_standing_calibration:
            model_rows, _baseline = calibrate_feature_rows(
                rows,
                baseline_frames=predictor.baseline_frames,
                min_visibility=predictor.min_visibility,
            )
        video_key = _video_key(csv_path)
        file_label = infer_label_from_filename(csv_path)
        video_events = []

        if len(rows) < args.window_size or len(model_rows) != len(rows):
            continue

        for start in range(0, len(rows) - args.window_size + 1, args.stride):
            window_rows = rows[start : start + args.window_size]
            model_window_rows = model_rows[start : start + args.window_size]
            end_frame = _row_frame(window_rows[-1], start + args.window_size - 1)
            true_label = _label_for_window(
                csv_path=csv_path,
                video_key=video_key,
                end_frame=end_frame,
                file_label=file_label,
                label_mode="annotations",
                intervals=intervals,
            )
            if true_label is None:
                continue

            if predictor._use_accel:
                base_feature_columns = tuple(
                    column
                    for column in predictor.feature_columns
                    if column not in {"torso_angular_accel", "vertical_accel"}
                )
                window_list = compute_window_accel_features(
                    model_window_rows,
                    base_feature_columns=base_feature_columns,
                )
            else:
                window_list = list(model_window_rows)
            sample = [flatten_window(window_list, predictor.feature_columns)]
            if skeleton_sequence is not None:
                predictor.model.set_skeleton_window(
                    skeleton_sequence[:, start : start + args.window_size, :]
                )
            model_state, risk_score, model_alert_state = predictor._predict_sample(sample)
            runtime_state, runtime_alert = predictor._apply_temporal_validation(
                model_state,
                model_alert_state,
                window_list,
                window_rows,
            )

            raw_true.append(true_label)
            raw_pred.append(model_state)
            runtime_true.append(true_label)
            runtime_pred.append(runtime_state)
            alert_pred.append(runtime_alert)

            region = prefall_region(
                csv_path=csv_path,
                video_key=video_key,
                end_frame=end_frame,
                intervals=intervals,
                boundary_ratio=args.prefall_boundary_ratio,
            )
            if true_label != "Pre-fall" or region == "core":
                lenient_true.append(true_label)
                lenient_raw_pred.append(model_state)
                lenient_runtime_pred.append(runtime_state)
                lenient_alert_pred.append(runtime_alert)
            if true_label == "Pre-fall":
                prefall_region_events.append(
                    {
                        "region": region or "unknown",
                        "raw": model_state,
                        "runtime": runtime_state,
                        "alert": runtime_alert,
                    }
                )

            video_events.append(
                {
                    "end_frame": end_frame,
                    "true": true_label,
                    "raw": model_state,
                    "runtime": runtime_state,
                    "alert": runtime_alert,
                    "risk": risk_score,
                }
            )

        video_summaries[video_key] = summarize_video(video_key, video_events)

    labels = sorted(set(raw_true) | set(raw_pred) | set(runtime_pred) | set(alert_pred))
    report = {
        "model": args.model,
        "annotations": args.annotations,
        "feature_dirs": feature_dirs,
        "group_filter_metrics": args.group_filter_metrics,
        "use_hmm": args.use_hmm,
        "use_temporal_validation": args.use_temporal_validation,
        "sensitivity": args.sensitivity,
        "window_size": args.window_size,
        "stride": args.stride,
        "prefall_boundary_ratio": args.prefall_boundary_ratio,
        "window_count": len(runtime_true),
        "raw_model_metrics_all_windows": build_validation_metrics(raw_true, raw_pred, labels),
        "runtime_state_metrics_all_windows": build_validation_metrics(runtime_true, runtime_pred, labels),
        "alert_state_metrics_all_windows": build_validation_metrics(runtime_true, alert_pred, labels),
        "raw_model_metrics_lenient_prefall_core": build_validation_metrics(lenient_true, lenient_raw_pred, labels),
        "runtime_state_metrics_lenient_prefall_core": build_validation_metrics(
            lenient_true,
            lenient_runtime_pred,
            labels,
        ),
        "alert_state_metrics_lenient_prefall_core": build_validation_metrics(lenient_true, lenient_alert_pred, labels),
        "prefall_region_summary": summarize_prefall_regions(prefall_region_events),
        "sequence_summary": summarize_sequences(video_summaries),
        "confirmed_lying_summary": summarize_confirmed_lying(video_summaries),
        "videos": video_summaries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_human_summary(report)
    print(f"\nWrote {output_path}")


def collect_csv_paths(feature_dirs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for feature_dir in feature_dirs:
        paths.extend(sorted(feature_dir.glob("*.csv")))
    return sorted(paths)


def prefall_region(
    csv_path: Path,
    video_key: str,
    end_frame: int,
    intervals: dict,
    boundary_ratio: float,
) -> str | None:
    """Classify a Pre-fall window as early/core/late using the report-style boundary split."""
    boundary = max(0.0, min(0.49, float(boundary_ratio)))
    for key in _annotation_keys(csv_path, video_key):
        for interval in intervals.get(key, ()):
            if interval.label.strip() != "Pre-fall":
                continue
            if interval.start_frame <= end_frame <= interval.end_frame:
                ratio = (end_frame - interval.start_frame) / max(interval.end_frame - interval.start_frame, 1)
                if ratio < boundary:
                    return "early"
                if ratio > 1.0 - boundary:
                    return "late"
                return "core"
    return None


def summarize_video(video: str, events: list[dict]) -> dict:
    by_true = Counter(event["true"] for event in events)
    by_raw = Counter(event["raw"] for event in events)
    by_runtime = Counter(event["runtime"] for event in events)
    by_alert = Counter(event["alert"] for event in events)
    first_alert = first_event(events, {"Pre-fall", "Fall"}, key="alert")
    first_fall = first_event(events, {"Fall"}, key="runtime")
    return {
        "video_type": "Fall" if video.startswith(("fall", "subject")) else "ADL",
        "windows": len(events),
        "true_counts": dict(by_true),
        "raw_counts": dict(by_raw),
        "runtime_counts": dict(by_runtime),
        "alert_counts": dict(by_alert),
        "has_runtime_fall": by_runtime.get("Fall", 0) > 0,
        "has_alert_prefall_or_fall": by_alert.get("Pre-fall", 0) + by_alert.get("Fall", 0) > 0,
        "first_alert_frame": first_alert["end_frame"] if first_alert else None,
        "first_alert_state": first_alert["alert"] if first_alert else None,
        "first_runtime_fall_frame": first_fall["end_frame"] if first_fall else None,
    }


def first_event(events: list[dict], states: set[str], key: str) -> dict | None:
    for event in events:
        if event[key] in states:
            return event
    return None


def summarize_prefall_regions(events: list[dict]) -> dict:
    summary = {}
    for prediction_key in ("raw", "runtime", "alert"):
        prediction_summary = {}
        for region in ("early", "core", "late", "unknown"):
            region_events = [event for event in events if event["region"] == region]
            total = len(region_events)
            hits = sum(1 for event in region_events if event[prediction_key] == "Pre-fall")
            miss_to_normal = sum(1 for event in region_events if event[prediction_key] == "Normal")
            miss_to_fall = sum(1 for event in region_events if event[prediction_key] == "Fall")
            prediction_summary[region] = {
                "total": total,
                "prefall_hits": hits,
                "recall": hits / total if total else 0.0,
                "miss_to_normal": miss_to_normal,
                "miss_to_fall": miss_to_fall,
            }
        summary[prediction_key] = prediction_summary
    return summary


def summarize_sequences(video_summaries: dict[str, dict]) -> dict:
    fall_videos = {video: summary for video, summary in video_summaries.items() if summary["video_type"] == "Fall"}
    adl_videos = {video: summary for video, summary in video_summaries.items() if summary["video_type"] == "ADL"}
    adl_false_fall = sorted(video for video, summary in adl_videos.items() if summary["has_runtime_fall"])
    fall_runtime_detected = sorted(video for video, summary in fall_videos.items() if summary["has_runtime_fall"])
    fall_alert_detected = sorted(video for video, summary in fall_videos.items() if summary["has_alert_prefall_or_fall"])
    return {
        "fall_videos": len(fall_videos),
        "adl_videos": len(adl_videos),
        "fall_videos_with_runtime_fall": len(fall_runtime_detected),
        "fall_videos_with_alert_prefall_or_fall": len(fall_alert_detected),
        "adl_videos_with_false_runtime_fall": len(adl_false_fall),
        "adl_false_fall_videos": adl_false_fall,
        "fall_runtime_missed_videos": sorted(set(fall_videos) - set(fall_runtime_detected)),
        "fall_alert_missed_videos": sorted(set(fall_videos) - set(fall_alert_detected)),
    }


def summarize_confirmed_lying(video_summaries: dict[str, dict]) -> dict:
    results = {}
    for video, start, end in DEFAULT_CONFIRMED_LYING_SEGMENTS:
        summary = video_summaries.get(video)
        if not summary:
            continue
        results[f"{video}:{start}-{end}"] = {
            "has_runtime_fall_in_video": summary["has_runtime_fall"],
            "runtime_counts": summary["runtime_counts"],
            "alert_counts": summary["alert_counts"],
        }
    return results


def print_human_summary(report: dict) -> None:
    sequence = report["sequence_summary"]
    raw = report["raw_model_metrics_all_windows"]["classification_report"]
    runtime = report["runtime_state_metrics_all_windows"]["classification_report"]
    alert = report["alert_state_metrics_all_windows"]["classification_report"]
    lenient_runtime = report["runtime_state_metrics_lenient_prefall_core"]["classification_report"]
    lenient_alert = report["alert_state_metrics_lenient_prefall_core"]["classification_report"]
    prefall_regions = report["prefall_region_summary"]

    print("Current model runtime evaluation")
    print(f"  windows: {report['window_count']}")
    print(
        f"  use_hmm: {report['use_hmm']}, "
        f"temporal_validation: {report['use_temporal_validation']}, "
        f"sensitivity: {report['sensitivity']}"
    )
    print("\nAll-window metrics (optimistic, evaluated on available annotated data):")
    print_metric_line("raw model", raw)
    print_metric_line("runtime state", runtime)
    print_metric_line("alert state", alert)
    print(
        f"\nReport-style Pre-fall core metrics "
        f"(boundary={report['prefall_boundary_ratio']:.0%}, core={1 - 2 * report['prefall_boundary_ratio']:.0%}):"
    )
    print_metric_line("runtime core", lenient_runtime)
    print_metric_line("alert core", lenient_alert)
    runtime_core = prefall_regions["runtime"]["core"]
    print(
        f"  runtime PF core recall: "
        f"{runtime_core['prefall_hits']}/{runtime_core['total']} = {runtime_core['recall']:.3f}"
    )
    print("\nSequence-level:")
    print(
        f"  Fall videos with runtime Fall: "
        f"{sequence['fall_videos_with_runtime_fall']}/{sequence['fall_videos']}"
    )
    print(
        f"  Fall videos with Pre-fall/Fall alert: "
        f"{sequence['fall_videos_with_alert_prefall_or_fall']}/{sequence['fall_videos']}"
    )
    print(
        f"  ADL videos with false runtime Fall: "
        f"{sequence['adl_videos_with_false_runtime_fall']}/{sequence['adl_videos']}"
    )
    if sequence["adl_false_fall_videos"]:
        print(f"  ADL false Fall videos: {', '.join(sequence['adl_false_fall_videos'])}")
    if sequence["fall_runtime_missed_videos"]:
        print(f"  Runtime Fall missed videos: {', '.join(sequence['fall_runtime_missed_videos'][:20])}")


def print_metric_line(name: str, report: dict) -> None:
    fall = report.get("Fall", {})
    prefall = report.get("Pre-fall", {})
    normal = report.get("Normal", {})
    print(
        f"  {name:<14} "
        f"N_rec={normal.get('recall', 0):.3f} "
        f"PF_rec={prefall.get('recall', 0):.3f} "
        f"F_rec={fall.get('recall', 0):.3f} "
        f"F_prec={fall.get('precision', 0):.3f}"
    )


if __name__ == "__main__":
    main()
