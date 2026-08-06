from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


ANNOTATION_FIELDS = ("video", "start_frame", "end_frame", "label")
POSITIVE_LABELS = {"Pre-fall", "Fall"}
NORMAL_CONTEXT_FRAMES = 30
FALL_MAX_FRAMES = 60


@dataclass(frozen=True)
class ExportResult:
    output_directory: Path
    annotations_path: Path
    video_count: int
    annotation_count: int
    skipped_count: int


def export_reviewed_dataset(
    events: Iterable[Mapping[str, object]],
    destination: str | Path,
    *,
    package_name: str | None = None,
) -> ExportResult:
    destination_path = Path(destination).expanduser()
    if not destination_path.is_dir():
        raise ValueError("Export destination must be an existing directory.")

    package = destination_path / (
        package_name or datetime.now().strftime("FallGuard_Dataset_%Y%m%d_%H%M%S")
    )
    package = _available_package_path(package)
    videos_dir = package / "videos"
    videos_dir.mkdir(parents=True)
    annotations_path = package / "fallguard_annotations.csv"

    rows: list[dict[str, object]] = []
    video_count = 0
    skipped_count = 0

    try:
        for event in events:
            label = str(event.get("annotation_label") or "")
            source = Path(str(event.get("video_clip_path") or "")).expanduser()
            if label not in POSITIVE_LABELS or not source.is_file():
                skipped_count += 1
                continue

            fps, frame_count = _video_metadata(
                source,
                fallback_fps=_positive_float(event.get("clip_fps"), 10.0),
                fallback_duration=_positive_float(
                    event.get("clip_duration_seconds"), 0.0
                ),
            )
            if frame_count <= 0:
                skipped_count += 1
                continue

            video_key = f"FallGuard_{str(event.get('id') or video_count + 1)}"
            event_rows = _annotation_rows(event, video_key, fps, frame_count)
            target = videos_dir / f"{video_key}{source.suffix.lower() or '.mp4'}"
            shutil.copy2(source, target)
            rows.extend(event_rows)
            video_count += 1

        if video_count == 0:
            raise ValueError(
                "No reviewed Pre-fall or Fall clips are available to export."
            )

        with annotations_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        shutil.rmtree(package, ignore_errors=True)
        raise

    return ExportResult(
        output_directory=package,
        annotations_path=annotations_path,
        video_count=video_count,
        annotation_count=len(rows),
        skipped_count=skipped_count,
    )


def _annotation_rows(
    event: Mapping[str, object],
    video_key: str,
    fps: float,
    frame_count: int,
) -> list[dict[str, object]]:
    label = str(event.get("annotation_label") or "")
    last_frame = frame_count - 1
    prefall_start = _seconds_to_frame(
        event.get("prefall_start_seconds"), fps, last_frame
    )
    if prefall_start is None:
        raise ValueError(f"Event {event.get('id')} has no Pre-fall start boundary.")

    result: list[dict[str, object]] = []
    if prefall_start > 0:
        normal_start = max(0, prefall_start - NORMAL_CONTEXT_FRAMES)
        result.append(_row(video_key, normal_start, prefall_start - 1, "Normal"))

    if label == "Pre-fall":
        result.append(_row(video_key, prefall_start, last_frame, "Pre-fall"))
        return result

    fall_start = _seconds_to_frame(event.get("fall_start_seconds"), fps, last_frame)
    if fall_start is None or fall_start <= prefall_start:
        raise ValueError(
            f"Event {event.get('id')} has an invalid Fall start boundary."
        )
    result.append(_row(video_key, prefall_start, fall_start - 1, "Pre-fall"))
    fall_end = min(last_frame, fall_start + FALL_MAX_FRAMES - 1)
    result.append(_row(video_key, fall_start, fall_end, "Fall"))
    return result


def _row(video: str, start: int, end: int, label: str) -> dict[str, object]:
    return {
        "video": video,
        "start_frame": start,
        "end_frame": end,
        "label": label,
    }


def _seconds_to_frame(value: object, fps: float, last_frame: int) -> int | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return min(last_frame, max(0, int(round(seconds * fps))))


def _positive_float(value: object, fallback: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _video_metadata(
    path: Path,
    *,
    fallback_fps: float,
    fallback_duration: float,
) -> tuple[float, int]:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if fps > 0 and frame_count > 0:
            return fps, frame_count
    except Exception:
        pass
    return fallback_fps, int(round(fallback_duration * fallback_fps))


def _available_package_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        alternative = candidate.with_name(f"{candidate.name}_{index}")
        if not alternative.exists():
            return alternative
    raise ValueError("Too many dataset exports with the same name.")
