"""
Create a draft annotation file with Normal split into activity subclasses.

The existing training annotations use:

    Normal / Pre-fall / Fall

This script rewrites Normal intervals into safer, more detailed labels when the
source is clear:

    Standing / Walking / Sitting / Squatting / Bending / Lying

It is intentionally conservative. UR Fall ADL videos contain actions such as
bending and crouching, so uncertain intervals are left as Normal unless a manual
mapping CSV overrides them.

Manual mapping CSV format:

    video,label
    adl-01-cam0-rgb,Sitting
    adl-04-cam0-rgb,Standing

Example:
    python scripts/refine_normal_activity_labels.py \
      --annotations data/ur_up_train_drop60f_15pct_annotations.csv \
      --output data/ur_up_train_drop60f_15pct_activity_annotations.csv \
      --review-output data/ur_up_train_drop60f_15pct_activity_review.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ANNOTATION_FIELDS = ["video", "start_frame", "end_frame", "label"]
ACTIVITY_LABELS = {"Standing", "Walking", "Sitting", "Squatting", "Bending", "Lying"}
DEFAULT_FEATURE_DIRS = (
    Path("outputs/features/urfall_yolo"),
    Path("outputs/features/upfall_yolo"),
)


@dataclass(frozen=True)
class IntervalStats:
    pose_frames: int
    mean_torso_angle: float
    mean_abs_vertical_velocity: float
    mean_aspect_ratio: float
    mean_body_height: float
    mean_center_drop: float


@dataclass(frozen=True)
class LabelDecision:
    label: str
    source: str
    confidence: str
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Normal annotations into detailed normal-activity draft labels.")
    parser.add_argument("--annotations", required=True, help="Input annotation CSV with video,start_frame,end_frame,label.")
    parser.add_argument("--output", required=True, help="Output annotation CSV for training.")
    parser.add_argument(
        "--review-output",
        default=None,
        help="Optional review CSV with decision source, confidence, and feature statistics.",
    )
    parser.add_argument(
        "--feature-dir",
        action="append",
        default=None,
        help="Directory containing feature CSV files. Can be repeated.",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional manual mapping CSV with columns video,label.",
    )
    parser.add_argument(
        "--uncertain-label",
        default="Normal",
        choices=("Normal", "Standing", "Walking", "Sitting", "Squatting", "Bending", "Lying"),
        help="Label to use when an ADL interval is ambiguous. Default keeps it as Normal.",
    )
    args = parser.parse_args()

    feature_dirs = tuple(Path(path) for path in args.feature_dir) if args.feature_dir else DEFAULT_FEATURE_DIRS
    manual_mapping = load_manual_mapping(args.mapping)
    rows = read_annotations(Path(args.annotations))

    output_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    for row in rows:
        label = row["label"].strip()
        stats = None
        decision = LabelDecision(label=label, source="original", confidence="high", notes="Non-Normal label kept unchanged.")

        if label == "Normal":
            stats = load_interval_stats(
                video=row["video"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                feature_dirs=feature_dirs,
            )
            decision = decide_normal_activity_label(
                video=row["video"],
                stats=stats,
                manual_mapping=manual_mapping,
                uncertain_label=args.uncertain_label,
            )

        updated = dict(row)
        updated["label"] = decision.label
        output_rows.append(updated)
        review_rows.append(review_row(row, decision, stats))

    write_annotations(Path(args.output), output_rows)
    if args.review_output:
        write_review(Path(args.review_output), review_rows)

    print_summary(output_rows, review_rows)


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(ANNOTATION_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def write_annotations(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in ANNOTATION_FIELDS} for row in rows)


def write_review(path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = [
        "video",
        "start_frame",
        "end_frame",
        "old_label",
        "new_label",
        "source",
        "confidence",
        "notes",
        "pose_frames",
        "mean_torso_angle",
        "mean_abs_vertical_velocity",
        "mean_aspect_ratio",
        "mean_body_height",
        "mean_center_drop",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manual_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"video", "label"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Mapping CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            label = canonical_activity_label(row["label"])
            if label not in ACTIVITY_LABELS:
                raise ValueError(f"Unsupported activity label for {row['video']!r}: {row['label']!r}")
            mapping[normalize_video_name(row["video"])] = label
    return mapping


def canonical_activity_label(label: str) -> str:
    lower = label.strip().lower().replace("_", "-")
    if lower in {"standing", "stand"}:
        return "Standing"
    if lower in {"walking", "walk"}:
        return "Walking"
    if lower in {"sitting", "sit", "seated"}:
        return "Sitting"
    if lower in {"squatting", "squat", "crouching", "crouch"}:
        return "Squatting"
    if lower in {"bending", "bend", "bent", "stooping", "stoop", "spotting", "spot"}:
        return "Bending"
    if lower in {"lying", "lie", "laying", "reclining", "reclined", "prone", "supine"}:
        return "Lying"
    return label.strip()


def decide_normal_activity_label(
    video: str,
    stats: IntervalStats | None,
    manual_mapping: dict[str, str],
    uncertain_label: str,
) -> LabelDecision:
    key = normalize_video_name(video)
    if key in manual_mapping:
        return LabelDecision(
            label=manual_mapping[key],
            source="manual_mapping",
            confidence="high",
            notes="User-provided mapping.",
        )

    if is_fall_sequence(video):
        return LabelDecision(
            label="Standing",
            source="video_name",
            confidence="high",
            notes="Normal interval before a fall event; treated as pre-event standing posture.",
        )

    if stats is None or stats.pose_frames == 0:
        return LabelDecision(
            label=uncertain_label,
            source="missing_features",
            confidence="low",
            notes="No pose feature rows found for this interval.",
        )

    # Strong lying clues: very low and horizontal body shape.
    if (
        (stats.mean_body_height <= 0.24 and stats.mean_aspect_ratio >= 0.55)
        or (stats.mean_body_height <= 0.32 and stats.mean_aspect_ratio >= 0.85)
        or (stats.mean_center_drop >= 0.24 and stats.mean_aspect_ratio >= 0.65)
    ):
        return LabelDecision(
            label="Lying",
            source="feature_rule",
            confidence="medium",
            notes="Very low horizontal body shape suggests lying/reclining; review against fall-like floor posture.",
        )

    # Squatting/crouching is low but usually still compact and leg-supported.
    if stats.mean_body_height <= 0.38 and stats.mean_center_drop >= 0.10 and stats.mean_aspect_ratio <= 0.65:
        return LabelDecision(
            label="Squatting",
            source="feature_rule",
            confidence="medium",
            notes="Low body center with compact shape suggests squatting/crouching.",
        )

    # Bending is a strong torso tilt while the body is not fully low/horizontal.
    if stats.mean_torso_angle >= 28.0 and stats.mean_body_height >= 0.36 and stats.mean_aspect_ratio <= 0.68:
        return LabelDecision(
            label="Bending",
            source="feature_rule",
            confidence="medium",
            notes="Forward-bent torso with body still relatively tall suggests bending.",
        )

    # Strong sitting clues: compact body height or horizontal bounding shape.
    if stats.mean_body_height <= 0.34 or stats.mean_aspect_ratio >= 0.68:
        return LabelDecision(
            label="Sitting",
            source="feature_rule",
            confidence="medium",
            notes="Low body height or wide body aspect suggests sitting/seated posture; review if this is crouching.",
        )

    # Walking is hard without body-center x movement. Only use high vertical motion
    # when the torso is mostly upright and the body is full height.
    if (
        stats.mean_abs_vertical_velocity >= 0.075
        and stats.mean_torso_angle <= 15.0
        and stats.mean_body_height >= 0.44
    ):
        return LabelDecision(
            label="Walking",
            source="feature_rule",
            confidence="medium",
            notes="Upright body with repeated vertical motion suggests walking.",
        )

    if (
        stats.mean_torso_angle <= 12.0
        and stats.mean_body_height >= 0.44
        and stats.mean_aspect_ratio <= 0.48
        and stats.mean_abs_vertical_velocity <= 0.06
    ):
        return LabelDecision(
            label="Standing",
            source="feature_rule",
            confidence="medium",
            notes="Upright, full-height, low-motion posture suggests standing.",
        )

    return LabelDecision(
        label=uncertain_label,
        source="uncertain_feature_rule",
        confidence="low",
        notes="Could be bending, crouching, sitting transition, or walking; needs manual review.",
    )


def load_interval_stats(
    video: str,
    start_frame: int,
    end_frame: int,
    feature_dirs: tuple[Path, ...],
) -> IntervalStats | None:
    feature_path = find_feature_csv(video, feature_dirs)
    if feature_path is None:
        return None

    torso_angles: list[float] = []
    vertical_velocities: list[float] = []
    aspect_ratios: list[float] = []
    body_heights: list[float] = []
    center_drops: list[float] = []
    with feature_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame = safe_int(row.get("frame"))
            if frame is None or frame < start_frame or frame > end_frame:
                continue
            if safe_float(row.get("has_pose")) <= 0.0:
                continue
            torso_angles.append(abs(safe_float(row.get("torso_angle"))))
            vertical_velocities.append(abs(safe_float(row.get("vertical_velocity"))))
            aspect_ratios.append(safe_float(row.get("aspect_ratio")))
            body_heights.append(safe_float(row.get("body_height")))
            center_drops.append(safe_float(row.get("center_drop")))

    if not torso_angles:
        return IntervalStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return IntervalStats(
        pose_frames=len(torso_angles),
        mean_torso_angle=mean(torso_angles),
        mean_abs_vertical_velocity=mean(vertical_velocities),
        mean_aspect_ratio=mean(aspect_ratios),
        mean_body_height=mean(body_heights),
        mean_center_drop=mean(center_drops),
    )


def find_feature_csv(video: str, feature_dirs: tuple[Path, ...]) -> Path | None:
    key = normalize_video_name(video)
    candidates = [video, key]
    for directory in feature_dirs:
        for candidate in candidates:
            path = directory / f"{candidate}.csv"
            if path.exists():
                return path
    return None


def review_row(row: dict[str, str], decision: LabelDecision, stats: IntervalStats | None) -> dict[str, str]:
    stats = stats or IntervalStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return {
        "video": row["video"],
        "start_frame": row["start_frame"],
        "end_frame": row["end_frame"],
        "old_label": row["label"],
        "new_label": decision.label,
        "source": decision.source,
        "confidence": decision.confidence,
        "notes": decision.notes,
        "pose_frames": str(stats.pose_frames),
        "mean_torso_angle": f"{stats.mean_torso_angle:.4f}",
        "mean_abs_vertical_velocity": f"{stats.mean_abs_vertical_velocity:.4f}",
        "mean_aspect_ratio": f"{stats.mean_aspect_ratio:.4f}",
        "mean_body_height": f"{stats.mean_body_height:.4f}",
        "mean_center_drop": f"{stats.mean_center_drop:.4f}",
    }


def print_summary(annotation_rows: list[dict[str, str]], review_rows: list[dict[str, str]]) -> None:
    label_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for row in annotation_rows:
        label = row["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    for row in review_rows:
        if row["old_label"] == "Normal":
            confidence = row["confidence"]
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    print("Label intervals:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("Normal split confidence:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"  {confidence}: {count}")


def is_fall_sequence(video: str) -> bool:
    key = normalize_video_name(video)
    if key.startswith("fall-"):
        return True
    return re.match(r"^subject\d+activity[1-5]trial\d+camera\d+$", key) is not None


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
