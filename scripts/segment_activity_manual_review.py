"""
Split ambiguous activity-review intervals into shorter draft segments.

This is meant for rows that were too mixed to label as one activity. It reads
the current manual-review CSV, uses per-frame feature CSVs, and writes a new
review file where a source interval may become several rows.

Output columns include source_start_frame/source_end_frame so
apply_activity_manual_review.py can replace the original interval precisely.

Example:
    python scripts/segment_activity_manual_review.py \
      --manual-review data/ur_up_train_drop60f_15pct_activity_manual_review.csv \
      --output data/ur_up_train_drop60f_15pct_activity_manual_review_segments.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_FEATURE_DIRS = (
    Path("outputs/features/urfall_yolo"),
    Path("outputs/features/upfall_yolo"),
)
LABELS = {"Normal", "Standing", "Walking", "Sitting", "Squatting", "Bending", "Lying"}


@dataclass(frozen=True)
class FrameDecision:
    frame: int
    label: str
    confidence: str
    reason: str
    torso_angle: float
    vertical_velocity: float
    aspect_ratio: float
    body_height: float
    center_drop: float


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    label: str
    confidence: str
    notes: str
    frames: list[FrameDecision]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create split manual-review rows for ambiguous activity labels.")
    parser.add_argument("--manual-review", required=True, help="Manual review CSV to split.")
    parser.add_argument("--output", required=True, help="Segmented manual review CSV output path.")
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=None,
        help="Directory containing feature CSVs. Can be repeated.",
    )
    parser.add_argument("--smooth-window", type=int, default=7, help="Odd frame window for label smoothing.")
    parser.add_argument("--min-segment-frames", type=int, default=12, help="Merge segments shorter than this.")
    args = parser.parse_args()

    feature_dirs = tuple(Path(path) for path in args.feature_dir) if args.feature_dir else DEFAULT_FEATURE_DIRS
    review_rows = read_manual_review(Path(args.manual_review))
    output_rows: list[dict[str, str]] = []

    for row in review_rows:
        current_label = row["review_label"].strip()
        if current_label != "Normal":
            output_rows.append(to_output_row(row, row["start_frame"], row["end_frame"], current_label, "high", "User-reviewed label kept."))
            continue

        segments = segment_interval(
            video=row["video"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            feature_dirs=feature_dirs,
            smooth_window=args.smooth_window,
            min_segment_frames=args.min_segment_frames,
        )
        if not segments:
            output_rows.append(
                to_output_row(
                    row,
                    row["start_frame"],
                    row["end_frame"],
                    "Normal",
                    "low",
                    "No feature rows found; keep for manual review.",
                )
            )
            continue

        for segment in segments:
            output_rows.append(
                to_output_row(
                    row,
                    str(segment.start_frame),
                    str(segment.end_frame),
                    segment.label,
                    segment.confidence,
                    segment.notes,
                )
            )

    write_segmented_review(Path(args.output), output_rows)
    print_summary(output_rows)


def read_manual_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"video", "start_frame", "end_frame", "current_label", "review_label", "notes"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manual review CSV missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def write_segmented_review(path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = [
        "video",
        "source_start_frame",
        "source_end_frame",
        "start_frame",
        "end_frame",
        "current_label",
        "review_label",
        "confidence",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def segment_interval(
    video: str,
    start_frame: int,
    end_frame: int,
    feature_dirs: tuple[Path, ...],
    smooth_window: int,
    min_segment_frames: int,
) -> list[Segment]:
    rows = load_feature_rows(video, start_frame, end_frame, feature_dirs)
    decisions = [classify_frame(row) for row in rows]
    if not decisions:
        return []

    smoothed = smooth_decisions(decisions, smooth_window=max(1, smooth_window | 1))
    segments = build_segments(smoothed)
    segments = merge_short_segments(segments, min_segment_frames=max(1, min_segment_frames))
    return [summarize_segment(segment) for segment in segments]


def load_feature_rows(
    video: str,
    start_frame: int,
    end_frame: int,
    feature_dirs: tuple[Path, ...],
) -> list[dict[str, str]]:
    feature_path = find_feature_csv(video, feature_dirs)
    if feature_path is None:
        return []

    rows: list[dict[str, str]] = []
    with feature_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame = safe_int(row.get("frame"))
            if frame is None or frame < start_frame or frame > end_frame:
                continue
            rows.append(row)
    return rows


def classify_frame(row: dict[str, str]) -> FrameDecision:
    frame = safe_int(row.get("frame"))
    torso = abs(safe_float(row.get("torso_angle")))
    vertical = abs(safe_float(row.get("vertical_velocity")))
    aspect = safe_float(row.get("aspect_ratio"))
    height = safe_float(row.get("body_height"))
    center_drop = safe_float(row.get("center_drop"))

    label = "Normal"
    confidence = "low"
    reason = "ambiguous normal activity transition"

    if safe_float(row.get("has_pose")) <= 0.0:
        return FrameDecision(
            frame=frame if frame is not None else -1,
            label="Normal",
            confidence="low",
            reason="no reliable pose detected",
            torso_angle=torso,
            vertical_velocity=vertical,
            aspect_ratio=aspect,
            body_height=height,
            center_drop=center_drop,
        )

    if (height <= 0.24 and aspect >= 0.55) or (height <= 0.32 and aspect >= 0.85) or (
        center_drop >= 0.24 and aspect >= 0.65
    ):
        label = "Lying"
        confidence = "medium"
        reason = "low horizontal body shape"
    elif height <= 0.38 and center_drop >= 0.10 and aspect <= 0.65:
        label = "Squatting"
        confidence = "medium"
        reason = "low body center with compact leg-supported posture"
    elif torso >= 28.0 and height >= 0.36 and aspect <= 0.68:
        label = "Bending"
        confidence = "medium"
        reason = "forward-bent torso with body still relatively tall"
    elif height <= 0.32 or aspect >= 0.75:
        label = "Sitting"
        confidence = "medium"
        reason = "compact or wide body shape"
    elif height <= 0.38 and center_drop >= 0.10:
        label = "Sitting"
        confidence = "medium"
        reason = "low body center and compact body"
    elif torso <= 12.0 and height >= 0.46 and aspect <= 0.50 and vertical <= 0.065:
        label = "Standing"
        confidence = "medium"
        reason = "upright, tall, low motion"
    elif torso <= 18.0 and height >= 0.42 and 0.055 <= vertical <= 0.22 and aspect <= 0.56:
        label = "Walking"
        confidence = "medium"
        reason = "upright with repeated vertical motion"

    return FrameDecision(
        frame=frame if frame is not None else -1,
        label=label,
        confidence=confidence,
        reason=reason,
        torso_angle=torso,
        vertical_velocity=vertical,
        aspect_ratio=aspect,
        body_height=height,
        center_drop=center_drop,
    )


def smooth_decisions(decisions: list[FrameDecision], smooth_window: int) -> list[FrameDecision]:
    half = smooth_window // 2
    smoothed: list[FrameDecision] = []
    for index, decision in enumerate(decisions):
        start = max(0, index - half)
        end = min(len(decisions), index + half + 1)
        window = decisions[start:end]
        labels = Counter(item.label for item in window)
        top_label, top_count = labels.most_common(1)[0]
        if top_count <= len(window) / 2 and decision.label != top_label:
            smoothed.append(decision)
            continue
        if top_label == decision.label:
            smoothed.append(decision)
            continue
        smoothed.append(
            FrameDecision(
                frame=decision.frame,
                label=top_label,
                confidence="low" if top_label == "Normal" else "medium",
                reason=f"smoothed from local {smooth_window}-frame majority",
                torso_angle=decision.torso_angle,
                vertical_velocity=decision.vertical_velocity,
                aspect_ratio=decision.aspect_ratio,
                body_height=decision.body_height,
                center_drop=decision.center_drop,
            )
        )
    return smoothed


def build_segments(decisions: list[FrameDecision]) -> list[list[FrameDecision]]:
    segments: list[list[FrameDecision]] = []
    current: list[FrameDecision] = []
    current_label: str | None = None
    for decision in decisions:
        if current_label is None or decision.label == current_label:
            current.append(decision)
            current_label = decision.label
            continue
        segments.append(current)
        current = [decision]
        current_label = decision.label
    if current:
        segments.append(current)
    return segments


def merge_short_segments(segments: list[list[FrameDecision]], min_segment_frames: int) -> list[list[FrameDecision]]:
    if len(segments) <= 1:
        return segments

    changed = True
    while changed and len(segments) > 1:
        changed = False
        merged: list[list[FrameDecision]] = []
        index = 0
        while index < len(segments):
            segment = segments[index]
            if len(segment) >= min_segment_frames:
                merged.append(segment)
                index += 1
                continue

            previous_label = merged[-1][0].label if merged else None
            next_label = segments[index + 1][0].label if index + 1 < len(segments) else None
            if previous_label is not None and previous_label == next_label:
                merged[-1].extend(segment)
                merged[-1].extend(segments[index + 1])
                index += 2
            elif previous_label is not None:
                merged[-1].extend(segment)
                index += 1
            elif index + 1 < len(segments):
                segments[index + 1] = segment + segments[index + 1]
                index += 1
            else:
                merged.append(segment)
                index += 1
            changed = True
        segments = merged
    return segments


def summarize_segment(frames: list[FrameDecision]) -> Segment:
    label = Counter(item.label for item in frames).most_common(1)[0][0]
    confidence_counts = Counter(item.confidence for item in frames)
    low_ratio = confidence_counts.get("low", 0) / max(1, len(frames))
    confidence = "low" if label == "Normal" or low_ratio >= 0.35 else "medium"

    torso = mean(item.torso_angle for item in frames)
    vertical = mean(item.vertical_velocity for item in frames)
    aspect = mean(item.aspect_ratio for item in frames)
    height = mean(item.body_height for item in frames)
    drop = mean(item.center_drop for item in frames)
    reasons = Counter(item.reason for item in frames).most_common(2)
    reason_text = "; ".join(f"{reason} ({count}f)" for reason, count in reasons)
    notes = (
        f"{reason_text}; mean torso={torso:.1f}, abs_vvel={vertical:.3f}, "
        f"aspect={aspect:.3f}, height={height:.3f}, drop={drop:.3f}"
    )
    return Segment(
        start_frame=min(item.frame for item in frames),
        end_frame=max(item.frame for item in frames),
        label=label,
        confidence=confidence,
        notes=notes,
        frames=frames,
    )


def to_output_row(
    source_row: dict[str, str],
    start_frame: str,
    end_frame: str,
    review_label: str,
    confidence: str,
    notes: str,
) -> dict[str, str]:
    if review_label not in LABELS:
        raise ValueError(f"Unsupported review label: {review_label}")
    return {
        "video": source_row["video"],
        "source_start_frame": source_row["start_frame"],
        "source_end_frame": source_row["end_frame"],
        "start_frame": str(int(float(start_frame))),
        "end_frame": str(int(float(end_frame))),
        "current_label": source_row["current_label"],
        "review_label": review_label,
        "confidence": confidence,
        "notes": notes,
    }


def print_summary(rows: list[dict[str, str]]) -> None:
    label_counts = Counter(row["review_label"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    print("Segment rows written:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("Confidence:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")
    low_rows = [row for row in rows if row["confidence"] == "low"]
    if low_rows:
        print("Low-confidence rows:")
        for row in low_rows:
            print(f"  {row['video']} {row['start_frame']}-{row['end_frame']} -> {row['review_label']}")


def find_feature_csv(video: str, feature_dirs: tuple[Path, ...]) -> Path | None:
    key = normalize_video_name(video)
    for directory in feature_dirs:
        path = directory / f"{key}.csv"
        if path.exists():
            return path
    return None


def normalize_video_name(value: str) -> str:
    value = value.replace("\\", "/").strip().lower()
    for suffix in (".csv", ".mp4", ".avi", ".mov", ".mkv"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def safe_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def safe_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
