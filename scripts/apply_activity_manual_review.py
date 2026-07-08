"""
Apply manually reviewed activity labels back to a training annotation CSV.

The manual review CSV should contain either:

    video,start_frame,end_frame,current_label,review_label,notes

or split-review rows with:

    video,source_start_frame,source_end_frame,start_frame,end_frame,current_label,review_label,notes

Only rows whose review_label is one of:

    Normal, Standing, Walking, Sitting, Squatting, Bending, Lying

will be applied. For split-review rows, the source interval is matched by
video + source_start_frame + source_end_frame and replaced with the reviewed
sub-intervals.

Example:
    python scripts/apply_activity_manual_review.py \
      --annotations data/ur_up_train_drop60f_15pct_activity_annotations.csv \
      --manual-review data/ur_up_train_drop60f_15pct_activity_manual_review.csv \
      --output data/ur_up_train_drop60f_15pct_activity_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ANNOTATION_FIELDS = ["video", "start_frame", "end_frame", "label"]
REVIEW_LABELS = {
    "Normal",
    "Standing",
    "Walking",
    "Sitting",
    "Squatting",
    "Bending",
    "Lying",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual activity-label review to an annotation CSV.")
    parser.add_argument("--annotations", required=True, help="Input annotation CSV to update.")
    parser.add_argument("--manual-review", required=True, help="Manual review CSV with review_label values.")
    parser.add_argument("--output", required=True, help="Updated annotation CSV output path.")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    manual_review_path = Path(args.manual_review)
    output_path = Path(args.output)

    rows = read_annotations(annotations_path)
    updates = read_manual_updates(manual_review_path)
    updated_rows: list[dict[str, str]] = []
    changed_intervals = 0
    already_matching_intervals = 0
    split_intervals = 0

    for row in rows:
        key = interval_key(row)
        if key not in updates:
            updated_rows.append(row)
            continue

        segments = updates[key]
        if len(segments) == 1 and interval_key(segments[0]) == key:
            new_label = segments[0]["label"]
            if row["label"] == new_label:
                already_matching_intervals += 1
                updated_rows.append(row)
            else:
                changed = dict(row)
                changed["label"] = new_label
                updated_rows.append(changed)
                changed_intervals += 1
            continue

        split_intervals += 1
        updated_rows.extend(segments)

    write_annotations(output_path, updated_rows)
    print(f"Updated annotations written: {output_path}")
    print(f"Changed intervals: {changed_intervals}")
    print(f"Already matching intervals: {already_matching_intervals}")
    print(f"Split source intervals: {split_intervals}")


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(ANNOTATION_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def read_manual_updates(path: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"video", "start_frame", "end_frame", "review_label"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manual review CSV missing columns: {', '.join(sorted(missing))}")

        has_source_interval = {"source_start_frame", "source_end_frame"}.issubset(reader.fieldnames or ())
        updates: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in reader:
            label = canonical_review_label(row["review_label"])
            if label not in REVIEW_LABELS:
                raise ValueError(
                    f"Invalid review_label for {row['video']} "
                    f"{row['start_frame']}-{row['end_frame']}: {label!r}"
                )
            source_key = interval_key(
                {
                    "video": row["video"],
                    "start_frame": row["source_start_frame"] if has_source_interval else row["start_frame"],
                    "end_frame": row["source_end_frame"] if has_source_interval else row["end_frame"],
                }
            )
            segment = {
                "video": row["video"].strip(),
                "start_frame": str(int(float(row["start_frame"]))),
                "end_frame": str(int(float(row["end_frame"]))),
                "label": label,
            }
            updates.setdefault(source_key, []).append(segment)
        for key, segments in updates.items():
            segments.sort(key=lambda segment: int(segment["start_frame"]))
            validate_segments(key, segments)
        return updates


def canonical_review_label(label: str) -> str:
    """Normalize common manual-review aliases into the official label set."""
    cleaned = label.strip()
    lower = cleaned.lower().replace("_", "-")
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
    if lower in {"normal", "adl", "nonfall", "non-fall"}:
        return "Normal"
    return cleaned


def write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in ANNOTATION_FIELDS} for row in rows)


def interval_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["video"].strip(),
        str(int(float(row["start_frame"]))),
        str(int(float(row["end_frame"]))),
    )


def validate_segments(source_key: tuple[str, str, str], segments: list[dict[str, str]]) -> None:
    """Validate split rows for one source interval."""
    video, source_start_text, source_end_text = source_key
    source_start = int(source_start_text)
    source_end = int(source_end_text)
    previous_end: int | None = None
    for segment in segments:
        if segment["video"] != video:
            raise ValueError(f"Segment video mismatch for source {source_key}: {segment}")
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start > end:
            raise ValueError(f"Segment start is after end: {segment}")
        if start < source_start or end > source_end:
            raise ValueError(f"Segment outside source interval {source_key}: {segment}")
        if previous_end is not None and start <= previous_end:
            raise ValueError(f"Overlapping split segments for source {source_key}: {segment}")
        previous_end = end


if __name__ == "__main__":
    main()
