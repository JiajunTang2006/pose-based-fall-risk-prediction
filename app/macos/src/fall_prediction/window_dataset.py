"""
把逐帧特征 CSV 转换成机器学习训练样本。

视频处理程序导出的 CSV 是"一帧一行"，但机器学习模型最好不要只看单帧。
跌倒是一个连续动作，所以这里会把连续 N 帧切成一个"滑动窗口样本"。

例子：
    window_size = 15，stride = 3

    第 1 个样本: 第 0 到 14 帧
    第 2 个样本: 第 3 到 17 帧
    第 3 个样本: 第 6 到 20 帧

每个窗口会被展开成一个长向量：
    [第1帧特征..., 第2帧特征..., ..., 第15帧特征...]

默认标签来自文件名，适合先跑通 UR Fall：
    fall-*.csv -> Fall
    adl-*.csv / normal-*.csv -> Normal

如果要训练真正的 Pre-fall，需要提供更细的帧区间标注文件：
    video,start_frame,end_frame,label
    fall-01-cam0,0,55,Normal
    fall-01-cam0,56,80,Pre-fall
    fall-01-cam0,81,140,Fall

窗口的标签使用"窗口最后一帧"所在的区间。
这么做的含义是：模型看完这一小段历史后，预测当前时刻处于什么状态。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .ml_features import (
    ML_FEATURE_COLUMNS,
    ACCEL_FEATURE_COLUMNS,
    flatten_window,
    make_window_feature_names,
    compute_window_accel_features,
)
from .robustness import (
    ROBUST_ACCEL_FEATURE_COLUMNS,
    ROBUST_ML_FEATURE_COLUMNS,
    UPPER_BODY_ACCEL_FEATURE_COLUMNS,
    UPPER_BODY_ML_FEATURE_COLUMNS,
    apply_partial_pose_dropout,
    calibrate_feature_rows,
)


DEFAULT_WINDOW_SIZE = 15
DEFAULT_STRIDE = 3


@dataclass(frozen=True)
class WindowDataset:
    """
    内存中的窗口数据集。

    X:
        模型输入。每个元素都是一个展开后的窗口特征向量。

    y:
        模型标签。每个元素对应 X 中同位置的样本，例如 Normal / Fall。

    groups:
        每个样本来自哪个视频。训练/验证划分时用它避免数据泄漏。

    feature_names:
        展开后每一列的名字，主要用于调试和后续分析特征重要性。
    """

    X: list[list[float]]
    y: list[str]
    groups: list[str]
    feature_names: list[str]


@dataclass(frozen=True)
class LabelInterval:
    """一个视频中的一段帧区间标签。"""

    video: str
    start_frame: int
    end_frame: int
    label: str


def build_window_dataset(
    csv_paths: Sequence[str | Path],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
    label_mode: str = "filename",
    annotations_path: str | Path | Sequence[str | Path] | None = None,
    use_accel: bool = False,
    use_standing_calibration: bool = False,
    partial_pose_augmentation: bool = False,
    baseline_frames: int = 15,
    use_upper_body_features: bool = False,
) -> WindowDataset:
    """
    读取多个特征 CSV，并生成展开后的滑动窗口样本。

    参数:
        csv_paths:
            特征 CSV 文件列表。通常由 export_dataset_features.py 生成。

        window_size:
            一个训练样本包含多少连续帧。30fps 视频中，15 帧大约是 0.5 秒。

        stride:
            相邻窗口之间相隔多少帧。stride 越小，样本数量越多，但重复越多。

        feature_columns:
            参与训练的数值特征列，以及它们的顺序。
            顺序非常重要：训练和推理必须一致。

        label_mode:
            "filename" 表示从文件名推断标签；
            "annotations" 表示从帧区间标注 CSV 读取标签。

        annotations_path:
            label_mode="annotations" 时使用，列名必须为：
            video,start_frame,end_frame,label。

        use_accel:
            是否使用加速度增强特征。启用后会自动使用 ACCEL_FEATURE_COLUMNS，
            并在每个窗口中计算 torso_angular_accel 和 vertical_accel。
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if label_mode not in {"filename", "annotations"}:
        raise ValueError("label_mode must be 'filename' or 'annotations'")
    if label_mode == "annotations" and annotations_path is None:
        raise ValueError("annotations_path is required when label_mode='annotations'")

    if use_upper_body_features:
        base_feature_columns = UPPER_BODY_ML_FEATURE_COLUMNS
        feature_columns = (
            UPPER_BODY_ACCEL_FEATURE_COLUMNS if use_accel else UPPER_BODY_ML_FEATURE_COLUMNS
        )
    elif use_standing_calibration:
        base_feature_columns = ROBUST_ML_FEATURE_COLUMNS
        feature_columns = ROBUST_ACCEL_FEATURE_COLUMNS if use_accel else ROBUST_ML_FEATURE_COLUMNS
    else:
        base_feature_columns = ML_FEATURE_COLUMNS
        if use_accel:
            feature_columns = ACCEL_FEATURE_COLUMNS

    intervals = load_label_intervals(annotations_path) if annotations_path else {}
    X: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []

    for csv_path in sorted(Path(path) for path in csv_paths):
        # 每个 rows 元素是一帧，字段来自 video_app.py 的 CSV_COLUMNS。
        rows = load_feature_rows(csv_path)
        if use_standing_calibration or use_upper_body_features:
            rows, _baseline = calibrate_feature_rows(rows, baseline_frames=baseline_frames)
        if len(rows) < window_size:
            continue

        video_key = _video_key(csv_path)
        file_label = infer_label_from_filename(csv_path)

        # 从第 start 帧开始切窗口，每次向前移动 stride 帧。
        for start in range(0, len(rows) - window_size + 1, stride):
            window_rows = rows[start : start + window_size]

            # 使用窗口最后一帧决定标签：模型看的是过去 window_size 帧，
            # 输出的是"当前这一刻"的状态。
            end_frame = _row_frame(window_rows[-1], start + window_size - 1)
            label = _label_for_window(
                csv_path=csv_path,
                video_key=video_key,
                end_frame=end_frame,
                file_label=file_label,
                label_mode=label_mode,
                intervals=intervals,
            )
            if label is None:
                continue

            variants: list[Sequence[Mapping[str, object]]] = [window_rows]
            if (use_standing_calibration or use_upper_body_features) and partial_pose_augmentation:
                patterns = ["torso", "center", "bbox", "temporal"]
                if use_upper_body_features:
                    patterns.extend(("lower_body", "upper_body"))
                variants.extend(
                    apply_partial_pose_dropout(window_rows, pattern)
                    for pattern in patterns
                )

            for variant in variants:
                prepared_rows = list(variant)
                if use_accel:
                    prepared_rows = compute_window_accel_features(
                        prepared_rows,
                        base_feature_columns=base_feature_columns,
                    )
                X.append(flatten_window(prepared_rows, feature_columns))
                y.append(label)
                groups.append(video_key)

    return WindowDataset(
        X=X,
        y=y,
        groups=groups,
        feature_names=make_window_feature_names(window_size, feature_columns),
    )


def load_feature_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """读取一个导出的特征 CSV，返回字典列表。"""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def infer_label_from_filename(path: str | Path) -> str | None:
    """
    根据常见跌倒数据集文件名推断粗标签。

    UR Fall 常见命名：
        fall-01-cam0.csv -> Fall
        adl-01-cam0.csv  -> Normal

    这个方法只能做粗粒度分类，不能自动知道哪几帧是 Pre-fall。
    要训练 Pre-fall，请使用 annotations 标注文件。
    """
    stem = Path(path).stem.lower()
    if stem.startswith("fall") or "_fall" in stem or "-fall" in stem:
        return "Fall"
    if stem.startswith("adl") or stem.startswith("normal") or "nonfall" in stem:
        return "Normal"
    return None


def load_label_intervals(
    annotations_path: str | Path | Sequence[str | Path] | None,
) -> dict[str, list[LabelInterval]]:
    """
    读取可选的帧区间标注文件。可以传单个文件，也可以传多个文件。

    返回值按视频名索引：
        {
            "fall-01-cam0": [
                LabelInterval(...),
                LabelInterval(...),
            ]
        }
    """
    if annotations_path is None:
        return {}

    paths: Sequence[str | Path]
    if isinstance(annotations_path, (str, Path)):
        paths = [annotations_path]
    else:
        paths = annotations_path

    intervals: dict[str, list[LabelInterval]] = {}
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"video", "start_frame", "end_frame", "label"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"Annotation CSV is missing columns: {missing_text}")

            for row in reader:
                video = _normalize_video_name(row["video"])
                interval = LabelInterval(
                    video=video,
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    label=row["label"].strip(),
                )
                intervals.setdefault(video, []).append(interval)

    return intervals


def _label_for_window(
    csv_path: Path,
    video_key: str,
    end_frame: int,
    file_label: str | None,
    label_mode: str,
    intervals: Mapping[str, Sequence[LabelInterval]],
) -> str | None:
    """根据当前窗口最后一帧，决定这个窗口的训练标签。"""
    if label_mode == "filename":
        return file_label

    # annotations 模式下，一个标注文件可能写 fall-01-cam0，
    # 也可能写 fall-01-cam0.csv 或带目录名。这里尝试几种常见 key。
    for key in _annotation_keys(csv_path, video_key):
        for interval in intervals.get(key, ()):
            if interval.start_frame <= end_frame <= interval.end_frame:
                return interval.label

    # ADL/normal 视频通常整段都是正常动作，可以不手动标注。
    return "Normal" if file_label == "Normal" else None


def boundary_distance_for_frame(
    csv_path: str | Path,
    video_key: str,
    frame: int,
    intervals: Mapping[str, Sequence[LabelInterval]],
) -> int | None:
    """Return the source-frame distance to the nearest label transition."""
    path = Path(csv_path)
    matched: Sequence[LabelInterval] = ()
    for key in _annotation_keys(path, video_key):
        if intervals.get(key):
            matched = intervals[key]
            break
    if len(matched) < 2:
        return None
    ordered = sorted(matched, key=lambda item: (item.start_frame, item.end_frame))
    boundaries = [
        current.start_frame
        for previous, current in zip(ordered, ordered[1:])
        if previous.label != current.label
    ]
    if not boundaries:
        return None
    return min(abs(int(frame) - boundary) for boundary in boundaries)


def _annotation_keys(csv_path: Path, video_key: str) -> tuple[str, ...]:
    """生成几种可能的标注匹配名称，提高标注文件的容错性。"""
    stem = _normalize_video_name(csv_path.stem)
    name = _normalize_video_name(csv_path.name)
    parent_name = _normalize_video_name(f"{csv_path.parent.name}/{csv_path.stem}")
    return (video_key, stem, name, parent_name)


def _video_key(path: str | Path) -> str:
    """把文件路径转换成稳定的视频 ID。"""
    stem = _normalize_video_name(Path(path).stem)
    upfall_match = re.match(r"^(subject\d+activity\d+trial\d+)camera\d+$", stem)
    if upfall_match:
        return upfall_match.group(1)
    return stem


def _normalize_video_name(value: str) -> str:
    """统一视频名格式：小写、去掉扩展名、兼容不同路径分隔符。"""
    value = value.replace("\\", "/").strip().lower()
    suffixes = (".csv", ".mp4", ".avi", ".mov", ".mkv")
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _row_frame(row: Mapping[str, str], fallback: int) -> int:
    """读取 CSV 行中的 frame 字段；如果缺失或格式异常，就使用 fallback。"""
    try:
        return int(float(row.get("frame", fallback)))
    except (TypeError, ValueError):
        return fallback
