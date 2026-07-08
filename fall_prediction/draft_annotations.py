"""
根据已经导出的特征 CSV 自动生成一版“帧区间标注草稿”。

为什么叫草稿？
    真正的 Pre-fall / Fall 边界最好由人看视频或图片确认。
    但是完全手工从 0 开始标注很慢，所以这里先用规则系统的结果和特征曲线
    自动猜一个大致区间，然后你再人工检查、修改。

输出 CSV 格式：
    video,start_frame,end_frame,label,method,notes

其中前 4 列可以直接被 train_model.py 的 --label-mode annotations 使用：
    video,start_frame,end_frame,label


使用示例：
    python -m fall_prediction.draft_annotations \
      --input-dir outputs/features/urfall_yolo \
      --output data/urfall_annotations_draft.csv \
      --prefall-frames 30
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DraftInterval:
    """一段帧区间标注。"""

    video: str
    start_frame: int
    end_frame: int
    label: str
    method: str
    notes: str


@dataclass(frozen=True)
class FallStartGuess:
    """自动估计出来的 Fall 起始帧，以及采用的依据。"""

    frame: int
    method: str
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="从特征 CSV 自动生成 Normal/Pre-fall/Fall 标注草稿。")
    parser.add_argument("--input-dir", default="outputs/features/urfall_yolo", help="包含特征 CSV 的目录。")
    parser.add_argument("--output", default="data/urfall_annotations_draft.csv", help="输出标注草稿 CSV 路径。")
    parser.add_argument("--prefall-frames", type=int, default=30, help="Fall 起点前多少帧标为 Pre-fall，默认 30 帧。")
    parser.add_argument("--fall-threshold", type=float, default=0.72, help="认为进入 Fall 的风险阈值。")
    parser.add_argument("--prefall-threshold", type=float, default=0.45, help="认为进入 Pre-fall 的风险阈值。")
    args = parser.parse_args()

    csv_paths = sorted(Path(args.input_dir).glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"没有在目录中找到特征 CSV: {args.input_dir}")

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
    print(f"标注草稿已生成: {args.output}")
    print(f"区间数量: {len(intervals)}")
    print("下一步建议：打开这个 CSV，人工检查每个 fall-* 的 Pre-fall/Fall 分界帧。")


def draft_intervals_for_csv(
    csv_path: str | Path,
    rows: Sequence[Mapping[str, str]],
    prefall_frames: int = 30,
    fall_threshold: float = 0.72,
    prefall_threshold: float = 0.45,
) -> list[DraftInterval]:
    """
    为一个视频/图片序列生成标注区间。

    ADL/normal:
        整段标为 Normal。

    Fall:
        先估计 Fall 起点，然后把 Fall 起点前 prefall_frames 帧标为 Pre-fall，
        再之前标为 Normal，Fall 起点之后标为 Fall。
    """
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
                notes="ADL/normal 文件名，整段默认标为 Normal。",
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
                notes="Fall 起点前较早阶段，草稿标为 Normal。",
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
                notes=f"Fall 起点前 {prefall_frames} 帧，草稿标为 Pre-fall。",
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
    """
    根据多种线索估计 Fall 起点。

    优先级从强到弱：
    1. instant_state 已经被规则系统判为 Fall；
    2. smoothed_risk_score 超过 fall_threshold；
    3. risk_score 超过 fall_threshold；
    4. 躯干角度和身体下降量同时明显变大；
    5. 如果都没有，就取 smoothed_risk_score 最高的帧。
    """
    for row in rows:
        if row.get("instant_state") == "Fall" and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            return FallStartGuess(frame, "instant_state", "instant_state 首次达到 Fall。")

    for row in rows:
        if safe_float(row.get("smoothed_risk_score")) >= fall_threshold and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            score = safe_float(row.get("smoothed_risk_score"))
            return FallStartGuess(frame, "smoothed_risk", f"smoothed_risk_score 首次 >= {fall_threshold:.2f}，当前 {score:.3f}。")

    for row in rows:
        if safe_float(row.get("risk_score")) >= fall_threshold and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            score = safe_float(row.get("risk_score"))
            return FallStartGuess(frame, "risk_score", f"risk_score 首次 >= {fall_threshold:.2f}，当前 {score:.3f}。")

    for row in rows:
        torso = safe_float(row.get("torso_angle"))
        center_drop = safe_float(row.get("center_drop"))
        if torso >= 60.0 and center_drop >= 0.18 and safe_float(row.get("has_pose")) > 0.0:
            frame = row_frame(row)
            return FallStartGuess(frame, "feature_rule", f"torso_angle={torso:.1f}, center_drop={center_drop:.3f}。")

    best_row = max(rows, key=lambda row: safe_float(row.get("smoothed_risk_score")))
    frame = row_frame(best_row)
    score = safe_float(best_row.get("smoothed_risk_score"))
    if score < prefall_threshold:
        return FallStartGuess(frame, "max_risk_low_confidence", f"没有明显 Fall 信号，取最高 smoothed_risk_score={score:.3f}，请重点人工检查。")
    return FallStartGuess(frame, "max_risk", f"取最高 smoothed_risk_score={score:.3f}。")


def load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """读取一个特征 CSV。"""
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_intervals(intervals: Sequence[DraftInterval], output_path: str | Path) -> None:
    """把标注区间写入 CSV。"""
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
    """根据文件名判断是否是跌倒序列。"""
    lower = video.lower()
    return lower.startswith("fall") or "-fall" in lower or "_fall" in lower


def is_normal_video(video: str) -> bool:
    """根据文件名判断是否是正常动作序列。"""
    lower = video.lower()
    return lower.startswith("adl") or lower.startswith("normal") or "nonfall" in lower


def row_frame(row: Mapping[str, str], fallback: int = 0) -> int:
    """读取一行中的 frame 字段。"""
    try:
        return int(float(row.get("frame", fallback)))
    except (TypeError, ValueError):
        return fallback


def safe_float(value: object) -> float:
    """安全转换浮点数，转换失败时返回 0。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_int(value: int, low: int, high: int) -> int:
    """把整数限制在 [low, high] 范围内。"""
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
