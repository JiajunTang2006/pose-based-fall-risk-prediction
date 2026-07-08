"""
Pre-fall 宽松评估: 区分"边界模糊区"和"核心过渡区"
  - 边界区 (首尾各25%): 和相邻状态模糊，容忍漏检
  - 核心区 (中间50%):  运动学特征明显，评估模型真实能力
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from fall_prediction.ml_features import ML_FEATURE_COLUMNS
from fall_prediction.window_dataset import (
    DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE,
    build_window_dataset, _video_key, _row_frame, _label_for_window,
    infer_label_from_filename, load_feature_rows, load_label_intervals,
)
from fall_prediction.train_model import (
    _group_train_test_split, create_classifier,
    build_sample_weights, build_validation_metrics,
)

ANNOTATIONS = ROOT / "data/ur_up_train_annotations.csv"
TMP_ANNOTATIONS = Path("/tmp/fall_demo/demo_drop_60f_15pct_annotations.csv")
RANDOM_STATE = 42
BOUNDARY_RATIO = 0.25  # 首尾各 25% 视为边界模糊区


def build_end_frame_map(csv_paths, val_videos, window_size, stride, intervals):
    """
    为每个验证视频重建窗口序列，返回 {video: [end_frame, ...]}。
    列表顺序 = 该视频内窗口的生成顺序。
    """
    result: dict[str, list[int]] = defaultdict(list)
    for csv_path in sorted(csv_paths):
        video_key = _video_key(csv_path)
        if video_key not in val_videos:
            continue
        rows = load_feature_rows(csv_path)
        if len(rows) < window_size:
            continue
        file_label = infer_label_from_filename(csv_path)
        for start in range(0, len(rows) - window_size + 1, stride):
            window_rows = rows[start : start + window_size]
            end_frame = _row_frame(window_rows[-1], start + window_size - 1)
            label = _label_for_window(
                csv_path=csv_path, video_key=video_key, end_frame=end_frame,
                file_label=file_label, label_mode="annotations", intervals=intervals,
            )
            if label is not None:
                result[video_key].append(end_frame)
    return dict(result)


def classify_position(end_frame, pf_segments):
    """判断 end_frame 在 Pre-fall 段中的相对位置。"""
    for pf_start, pf_end in pf_segments:
        if pf_start <= end_frame <= pf_end:
            return (end_frame - pf_start) / max(pf_end - pf_start, 1)
    return None


def main():
    csv_paths = sorted((ROOT / "outputs/features/urfall_yolo").glob("*.csv"))
    csv_paths.extend(sorted((ROOT / "outputs/features/upfall_yolo").glob("*.csv")))

    # 1. 构建数据集 & 训练
    dataset = build_window_dataset(
        csv_paths=csv_paths, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode="annotations", annotations_path=str(TMP_ANNOTATIONS),
    )
    _, val_idx = _group_train_test_split(
        y_array=np.asarray(dataset.y), groups_array=np.asarray(dataset.groups),
        test_size=0.25, random_state=RANDOM_STATE,
    )
    val_videos = {dataset.groups[i] for i in val_idx}
    val_idx_set = set(val_idx)

    X_arr = np.asarray(dataset.X, dtype=float)
    y_arr = np.asarray(dataset.y)
    train_idx = [i for i in range(len(y_arr)) if i not in val_idx_set]
    model = create_classifier("hist_gradient_boosting", RANDOM_STATE)
    sw = build_sample_weights(y_arr[train_idx], {"Normal": 1.0, "Fall": 1.0, "Pre-fall": 8.0})
    model.fit(X_arr[train_idx], y_arr[train_idx],
              sample_weight=sw if sw is not None else None)

    # 2. 建立 {dataset_index → end_frame} 映射
    intervals = load_label_intervals(str(TMP_ANNOTATIONS))
    video_end_frames = build_end_frame_map(
        csv_paths, val_videos, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, intervals)

    # 按 video 分组 val_idx，按 dataset 内顺序分配 end_frame
    dsidx_to_endframe: dict[int, int] = {}
    video_val_groups: dict[str, list[int]] = defaultdict(list)
    for di in val_idx:
        video_val_groups[dataset.groups[di]].append(di)

    for video, dis in video_val_groups.items():
        efs = video_end_frames.get(video, [])
        for i, di in enumerate(sorted(dis, key=lambda d: d)):  # 保持 dataset 原始顺序
            if i < len(efs):
                dsidx_to_endframe[di] = efs[i]

    # 3. 读取 Pre-fall 段信息
    pf_segments: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with TMP_ANNOTATIONS.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"].strip() == "Pre-fall":
                pf_segments[row["video"].strip()].append(
                    (int(row["start_frame"]), int(row["end_frame"])))

    # 4. 按位置分层收集预测
    layers = {
        "early (0-25%)":  lambda r: r < BOUNDARY_RATIO,
        "core (25-75%)":  lambda r: BOUNDARY_RATIO <= r <= 1 - BOUNDARY_RATIO,
        "late (75-100%)": lambda r: r > 1 - BOUNDARY_RATIO,
    }
    layer_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "hit": 0, "miss_to_N": 0, "miss_to_F": 0})

    # 严格 vs 宽松评估的 y_true/y_pred
    strict_y_t, strict_y_p = [], []
    lenient_y_t, lenient_y_p = [], []

    for di in val_idx:
        true_label = dataset.y[di]
        pred_label = str(model.predict(X_arr[di:di+1])[0])

        strict_y_t.append(true_label)
        strict_y_p.append(pred_label)

        if true_label == "Pre-fall":
            ef = dsidx_to_endframe.get(di)
            ratio = classify_position(ef, pf_segments.get(dataset.groups[di], [])) if ef else None

            if ratio is not None:
                for layer_name, check in layers.items():
                    if check(ratio):
                        layer_stats[layer_name]["total"] += 1
                        if pred_label == "Pre-fall":
                            layer_stats[layer_name]["hit"] += 1
                        elif pred_label == "Normal":
                            layer_stats[layer_name]["miss_to_N"] += 1
                        elif pred_label == "Fall":
                            layer_stats[layer_name]["miss_to_F"] += 1
                        break

                # 核心段计入宽松评估
                if BOUNDARY_RATIO <= ratio <= 1 - BOUNDARY_RATIO:
                    lenient_y_t.append("Pre-fall")
                    lenient_y_p.append(pred_label)
                # 边界段: 不参与宽松评估（跳过）
            else:
                # 无法定位的 Pre-fall 窗口: 严格评估计入，宽松不计入
                pass
        else:
            lenient_y_t.append(true_label)
            lenient_y_p.append(pred_label)

    # ── 报告 ──
    total_pf = sum(s["total"] for s in layer_stats.values())

    print(f"{'='*70}")
    print(f"  Pre-fall 分层评估")
    print(f"  边界模糊区 = 首尾各 {BOUNDARY_RATIO*100:.0f}%  |  核心区 = 中间 {100-2*BOUNDARY_RATIO*100:.0f}%")
    print(f"{'='*70}")
    print(f"  {'区域':<22s} {'窗口':>5s} {'命中':>5s} {'Recall':>8s}  "
          f"{'漏→Normal':>10s} {'漏→Fall':>8s}")
    print(f"  {'-'*62}")

    for layer_name in ["early (0-25%)", "core (25-75%)", "late (75-100%)"]:
        s = layer_stats[layer_name]
        if s["total"] > 0:
            recall = s["hit"] / s["total"]
            bar = "█" * int(recall * 30)
            print(f"  {layer_name:<22s} {s['total']:>5d} {s['hit']:>5d} {recall:>7.1%}  {bar}")
            print(f"    → 漏检详情: Normal×{s['miss_to_N']}, Fall×{s['miss_to_F']}")

    # ── 严格 vs 宽松 整体指标 ──
    strict_m = build_validation_metrics(strict_y_t, strict_y_p,
                                         sorted(set(strict_y_t) | set(strict_y_p)))
    lenient_m = build_validation_metrics(lenient_y_t, lenient_y_p,
                                          sorted(set(lenient_y_t) | set(lenient_y_p)))

    print(f"\n{'='*70}")
    print(f"  严格 vs 宽松 评估对比")
    print(f"{'='*70}")
    header = (f"  {'':<20s} {'Acc':>7s} {'MacF1':>7s}  "
              f"{'PF_F1':>7s} {'PF_Rec':>7s} {'PF_Pre':>7s}  "
              f"{'F_F1':>7s} {'F_Rec':>7s} {'F_Pre':>7s}")
    print(header)
    print(f"  {'-'*78}")

    for name, m in [("严格(所有PF)", strict_m), ("宽松(仅核心PF)", lenient_m)]:
        r = m["classification_report"]
        pf = r.get("Pre-fall", {})
        f = r.get("Fall", {})
        n = r.get("Normal", {})
        print(f"  {name:<20s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f}  "
              f"{pf.get('f1-score',0):>7.4f} {pf.get('recall',0):>7.4f} {pf.get('precision',0):>7.4f}  "
              f"{f.get('f1-score',0):>7.4f} {f.get('recall',0):>7.4f} {f.get('precision',0):>7.4f}")
        print(f"  {'':>20s}          {'':>7s}  "
              f"(N_F1={n.get('f1-score',0):.4f} N_Rec={n.get('recall',0):.4f} N_Pre={n.get('precision',0):.4f})")

    # ── 核心段漏检详情 ──
    core = layer_stats["core (25-75%)"]
    print(f"\n{'='*70}")
    print(f"  核心 Pre-fall 能力总结")
    print(f"{'='*70}")
    print(f"  核心段窗口数:     {core['total']}")
    print(f"  模型检测命中:     {core['hit']}  →  Recall = {core['hit']/core['total']:.1%}")
    print(f"  漏检为 Normal:    {core['miss_to_N']} ({core['miss_to_N']/core['total']:.1%})")
    print(f"  漏检为 Fall:      {core['miss_to_F']} ({core['miss_to_F']/core['total']:.1%})")
    print(f"")
    print(f"  结论: 排除标注边界歧义后，模型在 Pre-fall 明显过渡阶段")
    print(f"  的真实检测能力约为 {core['hit']/core['total']:.0%}。")
    print(f"  整体 Pre-fall Recall (含边界) 被边界模糊帧拉低了约 "
          f"{core['hit']/core['total'] - (total_pf - sum(s['miss_to_N']+s['miss_to_F'] for s in layer_stats.values()))/total_pf:.0%}。")

    print(f"\n分析完成。")


if __name__ == "__main__":
    main()
