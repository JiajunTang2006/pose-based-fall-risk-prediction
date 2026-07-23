

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_predictor_config
from .video_app import IMAGE_EXTENSIONS, find_image_sequence_files, process_video


VIDEO_PATTERNS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def iter_dataset_sources(input_dir: str | Path, recursive: bool = True) -> list[Path]:

    root = Path(input_dir)
    sources: list[Path] = []


    for pattern in VIDEO_PATTERNS:
        sources.extend(root.rglob(pattern) if recursive else root.glob(pattern))


    candidate_dirs = [root]
    if recursive:
        candidate_dirs.extend(path for path in root.rglob("*") if path.is_dir())
    else:
        candidate_dirs.extend(path for path in root.iterdir() if path.is_dir())

    for directory in candidate_dirs:
        if has_image_sequence(directory):
            sources.append(directory)

    return sorted(set(sources))


def has_image_sequence(directory: str | Path) -> bool:

    path = Path(directory)
    if not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frame-level feature CSV files from a video or image-sequence dataset.")
    parser.add_argument("--input-dir", default="data/videos/urfall", help="Dataset directory containing videos or image sequences.")
    parser.add_argument("--output-dir", default="outputs/features/urfall_yolo", help="Output directory for feature CSV files.")
    parser.add_argument("--annotated-video-dir", default=None, help="Optional directory for videos annotated with poses and states.")
    parser.add_argument(
        "--landmarks-output-dir",
        default=None,
        help="Optionally save complete YOLO/MediaPipe landmark CSV files for every video.",
    )
    parser.add_argument("--pose-model", default=None, help="Optional pose-model path for the MediaPipe Tasks API.")
    parser.add_argument(
        "--pose-backend",
        choices=("mediapipe", "yolo"),
        default="yolo",
        help="Pose backend: MediaPipe or Ultralytics YOLO-pose.",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo26n-pose.pt",
        help="YOLO-pose .pt model loaded when --pose-backend is yolo.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON configuration, such as configs/default.json.")
    parser.add_argument("--image-fps", type=float, default=30.0, help="Frame rate used for image-sequence directories.")
    parser.add_argument("--no-recursive", action="store_true", help="Scan only the top level of the input directory.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose output CSV already exists.")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview while exporting.")
    args = parser.parse_args()
    predictor_config = load_predictor_config(args.config) if args.config else None


    sources = iter_dataset_sources(args.input_dir, recursive=not args.no_recursive)
    if not sources:
        raise RuntimeError(f"No videos or image-sequence directories found in: {args.input_dir}")


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    annotated_dir = Path(args.annotated_video_dir) if args.annotated_video_dir else None
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(args.landmarks_output_dir) if args.landmarks_output_dir else None
    if landmarks_dir:
        landmarks_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(sources, start=1):

        # fall-01-cam0.mp4     -> fall-01-cam0.csv
        # fall-01-cam0-rgb/    -> fall-01-cam0-rgb.csv
        output_csv = output_dir / f"{source_path.stem}.csv"
        if args.skip_existing and output_csv.exists():
            print(f"[{index}/{len(sources)}] skip {source_path}")
            continue

        output_video = None
        if annotated_dir:
            output_video = annotated_dir / f"{source_path.stem}_annotated.mp4"

        source_kind = "images" if source_path.is_dir() else "video"
        if source_path.is_dir():
            frame_count = len(find_image_sequence_files(source_path))
            source_kind = f"images:{frame_count}"
        print(f"[{index}/{len(sources)}] export {source_kind} {source_path} -> {output_csv}")


        process_video(
            source=str(source_path),
            output_csv=output_csv,
            output_video=output_video,
            model_path=args.pose_model,
            pose_backend=args.pose_backend,
            yolo_model_path=args.yolo_model,
            show=args.show,
            predictor_type="rule",
            image_sequence_fps=args.image_fps,
            predictor_config=predictor_config,
            output_landmarks_csv=(
                landmarks_dir / f"{source_path.stem}_landmarks.csv"
                if landmarks_dir
                else None
            ),
        )


if __name__ == "__main__":
    main()
