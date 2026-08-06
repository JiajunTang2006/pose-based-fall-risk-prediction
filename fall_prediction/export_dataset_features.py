"""
批量把数据集视频或图片序列导出为逐帧特征 CSV。

机器学习训练脚本不直接读取视频，而是读取特征 CSV。
所以训练前通常先运行本脚本，把 data/videos/urfall 里的数据统一处理一遍。

UR Fall 有些下载包不是 mp4，而是一帧一张 png 图片。
这种情况下，本脚本会把一个图片目录当成一个“视频序列”：

    data/videos/fall-01-cam0-rgb/
      fall-01-cam0-rgb-001.png
      fall-01-cam0-rgb-002.png
      fall-01-cam0-rgb-003.png

处理过程：
    1. 找到输入目录里的 mp4/avi/mov/mkv 视频，或者包含图片帧的目录；
    2. 对每个视频调用 video_app.process_video；
    3. MediaPipe 或 YOLO-pose 提取人体关键点；
    4. FeatureExtractor 计算每帧运动特征；
    5. 保存 outputs/features/urfall_yolo/xxx.csv。

使用示例：
    python -m fall_prediction.export_dataset_features \
        --input-dir data/videos/urfall \
        --output-dir outputs/features/urfall_yolo \
        --pose-backend yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_predictor_config
from .video_app import IMAGE_EXTENSIONS, find_image_sequence_files, process_video


# 支持的视频扩展名。UR Fall 常用 mp4；这里也兼容其他常见格式。
VIDEO_PATTERNS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def iter_dataset_sources(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    """
    在输入目录中查找可处理的数据源。

    可处理的数据源包括：
    - 视频文件：.mp4 / .avi / .mov / .mkv
    - 图片序列目录：目录里直接包含 .png / .jpg 等图片

    recursive=True 时会递归扫描子目录，方便你按数据集原始结构存放视频。
    返回值排序后去重，这样每次批量导出的顺序更稳定。
    """
    root = Path(input_dir)
    sources: list[Path] = []

    # 1. 查找视频文件。
    for pattern in VIDEO_PATTERNS:
        sources.extend(root.rglob(pattern) if recursive else root.glob(pattern))

    # 2. 查找图片序列目录。
    # 如果 root 本身就是一个图片序列目录，也要把 root 加进去。
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
    """判断一个目录是否可以作为图片序列处理。"""
    path = Path(directory)
    if not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description="批量导出视频或图片序列数据集的逐帧特征 CSV。")
    parser.add_argument("--input-dir", default="data/videos/urfall", help="包含视频文件或图片序列目录的数据集目录。")
    parser.add_argument("--output-dir", default="outputs/features/urfall_yolo", help="特征 CSV 输出目录。")
    parser.add_argument("--annotated-video-dir", default=None, help="可选：保存带骨架和状态文字的标注视频目录。")
    parser.add_argument(
        "--landmarks-output-dir",
        default=None,
        help="可选：同时保存每个视频的完整 YOLO/MediaPipe 关键点 CSV。",
    )
    parser.add_argument("--pose-model", default=None, help="可选：MediaPipe Tasks API 使用的姿态模型路径。")
    parser.add_argument(
        "--pose-backend",
        choices=("mediapipe", "yolo"),
        default="yolo",
        help="姿态估计后端：mediapipe 使用原流程，yolo 使用 Ultralytics YOLO-pose。",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo26n-pose.pt",
        help="当 --pose-backend yolo 时加载的 YOLO-pose .pt 模型路径。",
    )
    parser.add_argument("--config", default=None, help="可选：JSON 配置文件，例如 configs/default.json。")
    parser.add_argument("--image-fps", type=float, default=30.0, help="图片序列目录使用的帧率，默认 30。")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描输入目录第一层，不递归子目录。")
    parser.add_argument("--skip-existing", action="store_true", help="如果目标 CSV 已存在，就跳过对应视频。")
    parser.add_argument("--show", action="store_true", help="导出时显示 OpenCV 预览窗口。")
    args = parser.parse_args()
    predictor_config = load_predictor_config(args.config) if args.config else None

    # 先收集所有要处理的数据源：可能是视频文件，也可能是图片序列目录。
    sources = iter_dataset_sources(args.input_dir, recursive=not args.no_recursive)
    if not sources:
        raise RuntimeError(f"没有在目录中找到视频文件或图片序列目录: {args.input_dir}")

    # 创建输出目录，避免后面写文件时报目录不存在。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # annotated_video_dir 是可选的；如果提供，就额外保存标注视频。
    annotated_dir = Path(args.annotated_video_dir) if args.annotated_video_dir else None
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(args.landmarks_output_dir) if args.landmarks_output_dir else None
    if landmarks_dir:
        landmarks_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(sources, start=1):
        # 每个数据源对应一个同名 CSV：
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

        # 这里复用原来的视频处理入口。导出训练特征时使用 rule predictor，
        # 因为我们只是需要它计算每帧特征和 center_drop，不依赖已经训练好的 ML 模型。
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
