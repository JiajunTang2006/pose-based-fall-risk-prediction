"""
将 UR Fall 和 UP Fall 的最终标注合并为单一训练用 CSV。

数据来源:
  - UR Fall:  data/urfall_annotations.csv (40 ADL + 30 fall, 已人工审核)
  - UP Fall S1: data/upfall_annotations.csv (24 视频, voted + manual review)
  - UP Fall S2: data/upfall_subject1_2_complete.csv (24 视频, manual_review_final)

处理:
  - UP Fall S2 从边界格式转为段格式 (Normal / Pre-fall / Fall)
  - Fall 段统一截断为最多 60 帧 (fall_tail_cap=60)
  - 输出格式: video, start_frame, end_frame, label, dataset

输出:
  data/ur_up_merged_falltail60_train_annotations.csv

使用方法:
  python scripts/merge_annotations.py

  # 然后用合并后的标注训练模型:
  python -m fall_prediction.train_model \
      outputs/features/urfall_yolo/*.csv outputs/features/upfall_yolo/*.csv \
      --label-mode annotations \
      --annotations data/ur_up_merged_falltail60_train_annotations.csv \
      --output models/my_classifier.joblib
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT_PATH = DATA / "ur_up_merged_falltail60_train_annotations.csv"

FALL_TAIL_CAP = 60  # Fall 段最多保留 60 帧


def load_csv(path: Path) -> list[dict]:
    """读取 CSV，返回 list[dict]."""
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """写入 CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cap_fall_segment(start_frame: int, end_frame: int) -> int:
    """对 Fall 段应用 60 帧截断，返回截断后的 end_frame."""
    length = end_frame - start_frame + 1
    if length > FALL_TAIL_CAP:
        return start_frame + FALL_TAIL_CAP - 1
    return end_frame


def build_urfall_rows() -> list[dict]:
    """
    从 urfall_annotations.csv 读取 UR Fall 标注。
    这是最终审核后的版本，直接可用。
    Fall 段如果超过 60 帧则截断（当前数据都没有超过 60 帧）。
    """
    path = DATA / "urfall_annotations.csv"
    rows = load_csv(path)
    result = []
    for row in rows:
        video = row["video"].strip()
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        label = row["label"].strip()

        if label == "Fall":
            end = cap_fall_segment(start, end)

        result.append({
            "video": video,
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "dataset": "UR",
        })
    return result


def build_upfall_rows() -> list[dict]:
    """
    构建 UP Fall 标注：
    - Subject1: 来自 upfall_annotations.csv（段格式，直接可用）
    - Subject2: 来自 upfall_subject1_2_complete.csv（边界格式，需转换）
    Fall 段超过 60 帧时截断。
    """
    rows: list[dict] = []

    # ---- Subject1: 段格式，直接读取 ----
    s1_path = DATA / "upfall_annotations.csv"
    for row in load_csv(s1_path):
        video = row["video"].strip()
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        label = row["label"].strip()

        if label == "Fall":
            end = cap_fall_segment(start, end)

        rows.append({
            "video": video,
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "dataset": "UP",
        })

    # ---- Subject2: 边界格式，转换为段格式 ----
    s2_path = DATA / "upfall_subject1_2_complete.csv"
    for row in load_csv(s2_path):
        subject = int(row["subject"])
        if subject != 2:
            continue  # Subject1 已经在上面的 upfall_annotations.csv 中覆盖

        video = row["video"].strip()
        normal_start = int(row["normal_start"])
        normal_end = int(row["normal_end"])
        prefall_start = int(row["prefall_start"])
        prefall_end = int(row["prefall_end"])
        fall_start = int(row["fall_start"])
        fall_end = int(row["fall_end"])

        # 截断 Fall 段
        fall_end_capped = cap_fall_segment(fall_start, fall_end)

        # Normal 段
        rows.append({
            "video": video,
            "start_frame": normal_start,
            "end_frame": normal_end,
            "label": "Normal",
            "dataset": "UP",
        })

        # Pre-fall 段
        rows.append({
            "video": video,
            "start_frame": prefall_start,
            "end_frame": prefall_end,
            "label": "Pre-fall",
            "dataset": "UP",
        })

        # Fall 段（截断后）
        rows.append({
            "video": video,
            "start_frame": fall_start,
            "end_frame": fall_end_capped,
            "label": "Fall",
            "dataset": "UP",
        })

    return rows


def validate(rows: list[dict]) -> None:
    """校验合并结果的完整性."""
    from collections import Counter

    ur_videos = sorted({r["video"] for r in rows if r["dataset"] == "UR"})
    up_videos = sorted({r["video"] for r in rows if r["dataset"] == "UP"})

    print(f"UR Fall 视频数: {len(ur_videos)} (期望: 70 = 40 ADL + 30 fall)")
    print(f"UP Fall 视频数: {len(up_videos)} (期望: 48)")

    label_counts = Counter(r["label"] for r in rows)
    print(f"标签分布: {dict(label_counts)}")

    # 每个 UR fall 视频应该有 Normal + Pre-fall + Fall 三段
    ur_fall_videos = [v for v in ur_videos if v.startswith("fall-")]
    for v in ur_fall_videos:
        labels = {r["label"] for r in rows if r["video"] == v}
        expected = {"Normal", "Pre-fall", "Fall"}
        if labels != expected:
            print(f"  ⚠ {v}: 标签 {labels}, 期望 {expected}")

    # 每个 UR ADL 视频应该只有 Normal
    ur_adl_videos = [v for v in ur_videos if v.startswith("adl-")]
    for v in ur_adl_videos:
        labels = {r["label"] for r in rows if r["video"] == v}
        if labels != {"Normal"}:
            print(f"  ⚠ {v}: 标签 {labels}, 期望 {{'Normal'}}")

    # 每个 UP 视频应该有 Normal + Pre-fall + Fall 三段
    for v in up_videos:
        labels = {r["label"] for r in rows if r["video"] == v}
        expected = {"Normal", "Pre-fall", "Fall"}
        if labels != expected:
            print(f"  ⚠ {v}: 标签 {labels}, 期望 {expected}")

    # 检查 Fall 段长度不超过 60
    for r in rows:
        if r["label"] == "Fall":
            length = r["end_frame"] - r["start_frame"] + 1
            if length > FALL_TAIL_CAP:
                print(f"  ⚠ {r['video']}: Fall 段长度 {length} > {FALL_TAIL_CAP}")

    print("校验完成。")


def main() -> None:
    ur_rows = build_urfall_rows()
    up_rows = build_upfall_rows()
    all_rows = ur_rows + up_rows

    print(f"UR Fall 段数: {len(ur_rows)}")
    print(f"UP Fall 段数: {len(up_rows)}")
    print(f"合并总段数: {len(all_rows)}")

    # 按 video, start_frame 排序
    all_rows.sort(key=lambda r: (r["dataset"], r["video"], r["start_frame"]))

    write_csv(
        OUTPUT_PATH,
        all_rows,
        fieldnames=["video", "start_frame", "end_frame", "label", "dataset"],
    )
    print(f"\n合并标注已保存: {OUTPUT_PATH}")

    validate(all_rows)


if __name__ == "__main__":
    main()
