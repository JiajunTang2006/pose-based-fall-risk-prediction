

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DraftInterval:


    video: str
    start_frame: int
    end_frame: int
    label: str
    method: str
    notes: str


@dataclass(frozen=True)
class FallStartGuess:


    frame: int
    method: str
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate draft Normal/Pre-fall/Fall annotations from feature CSV files.")
    parser.add_argument("--input-dir", default="outputs/features/urfall_yolo", help="Directory containing feature CSV files.")
    parser.add_argument("--output", default="data/urfall_annotations_draft.csv", help="Output path for the draft annotation CSV.")
    parser.add_argument("--prefall-frames", type=int, default=30, help="Frames before Fall onset to label as Pre-fall.")
    parser.add_argument("--fall-threshold", type=float, default=0.72, help="Risk threshold for Fall onset.")
    parser.add_argument("--prefall-threshold", type=float, default=0.45, help="Risk threshold for Pre-fall onset.")
    args = parser.parse_args()

    csv_paths = sorted(Path(args.input_dir).glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No feature CSV files found in: {args.input_dir}")

    intervals: list[DraftInterval] = []
    for csv_path in csv_paths:
        rows = load_rows(csv_path)
        if not rows:
            continue
        intervals.extend(
            draft_intervals_for_csv(
                csv_path=csv_path,
                rows=rows,
                prefall_frames=args.prefall_frames,
                fall_threshold=args.fall_threshold,
                prefall_threshold=args.prefall_threshold,
            )
        )

    write_intervals(intervals, args.output)
    print(f"Draft annotations written to: {args.output}")
    print(f"Intervals: {len(intervals)}")
    print("Next step: review the Pre-fall/Fall boundary for every fall-* sequence.")


def draft_intervals_for_csv(
    csv_path: str | Path,
    rows: Sequence[Mapping[str, str]],
    prefall_frames: int = 30,
    fall_threshold: float = 0.72,
    prefall_threshold: float = 0.45,
) -> list[DraftInterval]:

    path = Path(csv_path)
    video = path.stem
    last_frame = row_frame(rows[-1], fallback=len(rows) - 1)

    if is_normal_video(video):
        return [
            DraftInterval(
                video=video,
                start_frame=0,
                end_frame=last_frame,
                label="Normal",
                method="filename",
                notes="ADL/normal filename; the full sequence defaults to Normal.",
            )
        ]

    if not is_fall_video(video):
        return []

    guess = guess_fall_start(rows, fall_threshold=fall_threshold, prefall_threshold=prefall_threshold)
    fall_start = clamp_int(guess.frame, 0, last_frame)
    prefall_start = clamp_int(fall_start - max(prefall_frames, 1), 0, last_frame)

    intervals: list[DraftInterval] = []
    if prefall_start > 0:
        intervals.append(
            DraftInterval(
                video=video,
                start_frame=0,
                end_frame=prefall_start - 1,
                label="Normal",
                method="draft",
                notes="Early frames before Fall onset are drafted as Normal.",
            )
        )

    if prefall_start < fall_start:
        intervals.append(
            DraftInterval(
                video=video,
                start_frame=prefall_start,
                end_frame=fall_start - 1,
                label="Pre-fall",
                method="draft",
                notes=f"The {prefall_frames} frames before Fall onset are drafted as Pre-fall.",
            )
        )

    intervals.append(
        DraftInterval(
            video=video,
            start_frame=fall_start,
            end_frame=last_frame,
            label="Fall",
            method=guess.method,
            notes=guess.notes,
        )
    )
    return intervals


def guess_fall_start(
    rows: Sequence[Mapping[str, str]],
    fall_threshold: float,
    prefall_threshold: float,
) -> FallStartGuess:

    for row in rows:
        if row.get("instant_state") == "Fall" and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            return FallStartGuess(frame, "instant_state", "instant_state first reached Fall.")

    for row in rows:
        if safe_float(row.get("smoothed_risk_score")) >= fall_threshold and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            score = safe_float(row.get("smoothed_risk_score"))
            return FallStartGuess(frame, "smoothed_risk", f"smoothed_risk_score first reached {fall_threshold:.2f}; value={score:.3f}.")

    for row in rows:
        if safe_float(row.get("risk_score")) >= fall_threshold and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            score = safe_float(row.get("risk_score"))
            return FallStartGuess(frame, "risk_score", f"risk_score first reached {fall_threshold:.2f}; value={score:.3f}.")

    for row in rows:
        torso = safe_float(row.get("torso_angle"))
        center_drop = safe_float(row.get("center_drop"))
        if torso >= 60.0 and center_drop >= 0.18 and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            return FallStartGuess(frame, "feature_rule", f"torso_angle={torso:.1f}, center_drop={center_drop:.3f}.")

    best_row = max(rows, key=lambda row: safe_float(row.get("smoothed_risk_score")))
    frame = row_frame(best_row)
    score = safe_float(best_row.get("smoothed_risk_score"))
    if score < prefall_threshold:
        return FallStartGuess(
            frame,
            "max_risk_low_confidence",
            f"No clear Fall signal; selected the maximum smoothed_risk_score={score:.3f}. Manual review is required.",
        )
    return FallStartGuess(frame, "max_risk", f"Selected maximum smoothed_risk_score={score:.3f}.")


def load_rows(csv_path: str | Path) -> list[dict[str, str]]:

    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_intervals(intervals: Sequence[DraftInterval], output_path: str | Path) -> None:

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("video", "start_frame", "end_frame", "label", "method", "notes"),
        )
        writer.writeheader()
        for interval in intervals:
            writer.writerow(
                {
                    "video": interval.video,
                    "start_frame": interval.start_frame,
                    "end_frame": interval.end_frame,
                    "label": interval.label,
                    "method": interval.method,
                    "notes": interval.notes,
                }
            )


def is_fall_video(video: str) -> bool:

    lower = video.lower()
    return lower.startswith("fall") or "-fall" in lower or "_fall" in lower


def is_normal_video(video: str) -> bool:

    lower = video.lower()
    return lower.startswith("adl") or lower.startswith("normal") or "nonfall" in lower


def row_frame(row: Mapping[str, str], fallback: int = 0) -> int:

    try:
        return int(float(row.get("frame", fallback)))
    except (TypeError, ValueError):
        return fallback


def safe_float(value: object) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_int(value: int, low: int, high: int) -> int:

    return max(low, min(high, value))


if __name__ == "__main__":
    main()
