"""
Analyze motion thresholds for runtime Fall confirmation.

The runtime validator should not confirm Fall from posture alone. This script
uses annotated feature CSVs to compare transition-like Fall/Pre-fall windows
against ADL Normal windows, then ranks conservative threshold candidates.

Example:
    python scripts/analyze_fall_motion_thresholds.py \
      --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
      --feature-dir outputs/features/urfall_yolo \
      --feature-dir outputs/features/upfall_yolo \
      --output reports/fall_motion_threshold_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


DEFAULT_WINDOW_SIZE = 15
DEFAULT_STRIDE = 3


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class MotionMetrics:
    label: str
    video: str
    end_frame: int
    max_vertical_velocity: float
    max_vertical_accel: float
    max_angular_velocity: float
    max_angular_accel: float
    center_drop_delta: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank motion thresholds for Fall confirmation.")
    parser.add_argument("--annotations", required=True, help="Annotation CSV with video,start_frame,end_frame,label.")
    parser.add_argument("--feature-dir", action="append", required=True, help="Feature CSV directory. Can repeat.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--fall-positive-frames",
        type=int,
        default=30,
        help="Use only the first N Fall frames as positive transition windows.",
    )
    parser.add_argument(
        "--max-normal-fpr",
        type=float,
        default=0.05,
        help="Keep candidates whose ADL Normal false-positive rate is at or below this value.",
    )
    parser.add_argument("--output", default=None, help="Optional candidate CSV output path.")
    args = parser.parse_args()

    intervals = load_intervals(Path(args.annotations))
    fall_starts = {
        video: min(interval.start for interval in video_intervals if interval.label == "Fall")
        for video, video_intervals in intervals.items()
        if any(interval.label == "Fall" for interval in video_intervals)
    }
    metrics = collect_metrics(
        feature_dirs=[Path(path) for path in args.feature_dir],
        intervals=intervals,
        fall_starts=fall_starts,
        window_size=args.window_size,
        stride=args.stride,
        fall_positive_frames=args.fall_positive_frames,
    )

    positives = [item for item in metrics if item.label in {"Pre-fall", "Fall-transition"}]
    negatives = [item for item in metrics if item.label == "ADL-Normal"]
    if not positives or not negatives:
        raise RuntimeError("Not enough positive/negative windows to analyze thresholds.")

    print_summary(metrics)
    candidates = rank_candidates(positives, negatives, max_normal_fpr=args.max_normal_fpr)
    print_candidates(candidates[:15])

    if args.output:
        write_candidates(Path(args.output), candidates)
        print(f"\nWrote candidates: {args.output}")


def load_intervals(path: Path) -> dict[str, list[Interval]]:
    intervals: dict[str, list[Interval]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"video", "start_frame", "end_frame", "label"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            video = normalize_video_name(row["video"])
            intervals.setdefault(video, []).append(
                Interval(
                    start=int(float(row["start_frame"])),
                    end=int(float(row["end_frame"])),
                    label=row["label"].strip(),
                )
            )
    return intervals


def collect_metrics(
    feature_dirs: list[Path],
    intervals: dict[str, list[Interval]],
    fall_starts: dict[str, int],
    window_size: int,
    stride: int,
    fall_positive_frames: int,
) -> list[MotionMetrics]:
    metrics: list[MotionMetrics] = []
    for feature_dir in feature_dirs:
        for path in sorted(feature_dir.glob("*.csv")):
            video = video_key_from_path(path)
            rows = read_rows(path)
            if len(rows) < window_size:
                continue
            for start in range(0, len(rows) - window_size + 1, stride):
                window = rows[start : start + window_size]
                end_frame = row_frame(window[-1], start + window_size - 1)
                label = label_for_frame(video, end_frame, intervals)
                if label is None:
                    continue
                bucket = analysis_label(video, label, end_frame, fall_starts, fall_positive_frames)
                if bucket is None:
                    continue
                metrics.append(window_metrics(bucket, video, end_frame, window))
    return metrics


def analysis_label(
    video: str,
    label: str,
    end_frame: int,
    fall_starts: dict[str, int],
    fall_positive_frames: int,
) -> str | None:
    if label == "Pre-fall":
        return "Pre-fall"
    if label == "Fall":
        fall_start = fall_starts.get(video)
        if fall_start is not None and end_frame <= fall_start + fall_positive_frames - 1:
            return "Fall-transition"
        return None
    if label == "Normal" and video.startswith("adl"):
        return "ADL-Normal"
    return None


def window_metrics(label: str, video: str, end_frame: int, rows: list[dict[str, str]]) -> MotionMetrics:
    vertical_velocity = [max(0.0, row_float(row, "vertical_velocity")) for row in rows]
    angular_velocity = [abs(row_float(row, "torso_angular_velocity")) for row in rows]
    center_drop = [row_float(row, "center_drop") for row in rows]
    vertical_accel = positive_deltas(rows, "vertical_velocity")
    angular_accel = [abs(value) for value in deltas(rows, "torso_angular_velocity")]
    return MotionMetrics(
        label=label,
        video=video,
        end_frame=end_frame,
        max_vertical_velocity=max(vertical_velocity, default=0.0),
        max_vertical_accel=max(vertical_accel, default=0.0),
        max_angular_velocity=max(angular_velocity, default=0.0),
        max_angular_accel=max(angular_accel, default=0.0),
        center_drop_delta=max(center_drop, default=0.0) - min(center_drop, default=0.0),
    )


def rank_candidates(
    positives: list[MotionMetrics],
    negatives: list[MotionMetrics],
    max_normal_fpr: float,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for vertical_velocity in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5):
        for vertical_accel in (0.2, 0.4, 0.7, 1.0, 1.5, 2.0):
            for angular_velocity in (100.0, 200.0, 400.0, 600.0, 800.0, 1000.0):
                for angular_accel in (200.0, 400.0, 600.0, 800.0, 1200.0):
                    for center_drop_delta in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
                        candidate = evaluate_candidate(
                            positives,
                            negatives,
                            vertical_velocity,
                            vertical_accel,
                            angular_velocity,
                            angular_accel,
                            center_drop_delta,
                        )
                        if candidate["normal_fpr"] <= max_normal_fpr:
                            candidates.append(candidate)
    candidates.sort(key=lambda item: (item["normal_fpr"], -item["recall"], -item["precision"]))
    return candidates


def evaluate_candidate(
    positives: list[MotionMetrics],
    negatives: list[MotionMetrics],
    vertical_velocity: float,
    vertical_accel: float,
    angular_velocity: float,
    angular_accel: float,
    center_drop_delta: float,
) -> dict[str, float]:
    def passes(item: MotionMetrics) -> bool:
        has_fast_descent = item.max_vertical_velocity >= vertical_velocity
        has_center_drop = item.center_drop_delta >= center_drop_delta
        has_impact = (
            item.max_vertical_accel >= vertical_accel
            or item.max_angular_velocity >= angular_velocity
            or item.max_angular_accel >= angular_accel
        )
        return has_fast_descent and has_center_drop and has_impact

    tp = sum(1 for item in positives if passes(item))
    fp = sum(1 for item in negatives if passes(item))
    recall = tp / len(positives)
    normal_fpr = fp / len(negatives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "vertical_velocity": vertical_velocity,
        "vertical_accel": vertical_accel,
        "angular_velocity": angular_velocity,
        "angular_accel": angular_accel,
        "center_drop_delta": center_drop_delta,
        "recall": recall,
        "precision": precision,
        "normal_fpr": normal_fpr,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "positive_windows": float(len(positives)),
        "negative_windows": float(len(negatives)),
    }


def print_summary(metrics: list[MotionMetrics]) -> None:
    print("Window summary:")
    for label in ("ADL-Normal", "Pre-fall", "Fall-transition"):
        group = [item for item in metrics if item.label == label]
        print(f"  {label}: {len(group)} windows")
        if not group:
            continue
        for field in (
            "max_vertical_velocity",
            "max_vertical_accel",
            "max_angular_velocity",
            "max_angular_accel",
            "center_drop_delta",
        ):
            values = [float(getattr(item, field)) for item in group]
            print(
                f"    {field}: median={median(values):.4f}, "
                f"p90={percentile(values, 90):.4f}, p95={percentile(values, 95):.4f}"
            )


def print_candidates(candidates: list[dict[str, float]]) -> None:
    print("\nTop threshold candidates:")
    for item in candidates:
        print(
            "  "
            f"vv>={item['vertical_velocity']:.2f}, "
            f"va>={item['vertical_accel']:.2f}, "
            f"av>={item['angular_velocity']:.0f}, "
            f"aa>={item['angular_accel']:.0f}, "
            f"cd>={item['center_drop_delta']:.2f} "
            f"recall={item['recall']:.3f} "
            f"precision={item['precision']:.3f} "
            f"normal_fpr={item['normal_fpr']:.3f}"
        )


def write_candidates(path: Path, candidates: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical_velocity",
        "vertical_accel",
        "angular_velocity",
        "angular_accel",
        "center_drop_delta",
        "recall",
        "precision",
        "normal_fpr",
        "true_positives",
        "false_positives",
        "positive_windows",
        "negative_windows",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def label_for_frame(video: str, frame: int, intervals: dict[str, list[Interval]]) -> str | None:
    for interval in intervals.get(video, ()):
        if interval.start <= frame <= interval.end:
            return interval.label
    if video.startswith(("adl", "normal")):
        return "Normal"
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def video_key_from_path(path: Path) -> str:
    stem = normalize_video_name(path.stem)
    upfall_match = re.match(r"^(subject\d+activity\d+trial\d+)camera\d+$", stem)
    if upfall_match:
        return upfall_match.group(1)
    return stem


def normalize_video_name(value: str) -> str:
    value = value.replace("\\", "/").strip().lower()
    suffixes = (".csv", ".mp4", ".avi", ".mov", ".mkv")
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def row_frame(row: dict[str, str], fallback: int) -> int:
    try:
        return int(float(row.get("frame", fallback)))
    except (TypeError, ValueError):
        return fallback


def row_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def deltas(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    values = [row_float(row, key) for row in rows]
    if not values:
        return [0.0]
    result = [0.0]
    for index in range(1, len(values)):
        result.append(values[index] - values[index - 1])
    return result


def positive_deltas(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    return [max(0.0, value) for value in deltas(rows, key)]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


if __name__ == "__main__":
    main()
