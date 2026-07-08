"""
用 20% 宽松标准重新评估所有改进方案。
"""

from __future__ import annotations

import csv, math, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from fall_prediction.ml_features import ML_FEATURE_COLUMNS, flatten_window
from fall_prediction.window_dataset import (
    DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, WindowDataset,
    build_window_dataset, _video_key, _row_frame, _label_for_window,
    infer_label_from_filename, load_feature_rows, load_label_intervals,
)
from fall_prediction.train_model import (
    _group_train_test_split, create_classifier,
    build_sample_weights, build_validation_metrics,
)

ANNOTATIONS = ROOT / "data/ur_up_train_annotations.csv"
TMP_DIR = Path("/tmp/fall_demo")
RANDOM_STATE, TEST_SIZE = 42, 0.25
BOUNDARY = 0.20
CW = {"Normal": 1.0, "Fall": 1.0, "Pre-fall": 8.0}

ACCEL_FEATURE_COLUMNS = tuple(ML_FEATURE_COLUMNS) + (
    "torso_angular_accel", "vertical_accel",
)


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def write_dropped_annotations(drop_start, drop_ratio):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_DIR / f"nd_{drop_start}f_{int(drop_ratio*100)}pct.csv"
    with ANNOTATIONS.open("r", newline="", encoding="utf-8") as f:
        src = list(csv.DictReader(f))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video","start_frame","end_frame","label"])
        w.writeheader()
        for row in src:
            s, e = int(row["start_frame"]), int(row["end_frame"])
            lab = row["label"].strip()
            if lab == "Normal":
                length = e - s + 1
                fd = min(drop_start, max(0, length - DEFAULT_WINDOW_SIZE))
                rem = length - fd
                pd = min(int(math.floor(rem * drop_ratio)), max(0, rem - DEFAULT_WINDOW_SIZE))
                s += fd + pd
            w.writerow({"video": row["video"], "start_frame": s, "end_frame": e, "label": lab})
    return out


def write_shifted_annotations(shift):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_DIR / f"pf_shift_{shift}f.csv"
    with ANNOTATIONS.open("r", newline="", encoding="utf-8") as f:
        src = list(csv.DictReader(f))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video","start_frame","end_frame","label"])
        w.writeheader()
        for row in src:
            s, e = int(row["start_frame"]), int(row["end_frame"])
            lab = row["label"].strip()
            if lab == "Pre-fall":
                s += shift
                if e - s + 1 < DEFAULT_WINDOW_SIZE: continue
            w.writerow({"video": row["video"], "start_frame": s, "end_frame": e, "label": lab})
    return out


def build_accel_dataset(csv_paths, ann_path):
    intervals = load_label_intervals(str(ann_path)) if ann_path else {}
    X_all, y_all, groups_all = [], [], []
    for cp in sorted(Path(p) for p in csv_paths):
        rows = load_feature_rows(cp)
        if len(rows) < DEFAULT_WINDOW_SIZE: continue
        vk, fl = _video_key(cp), infer_label_from_filename(cp)
        for start in range(0, len(rows) - DEFAULT_WINDOW_SIZE + 1, DEFAULT_STRIDE):
            wr = rows[start:start+DEFAULT_WINDOW_SIZE]
            ef = _row_frame(wr[-1], start+DEFAULT_WINDOW_SIZE-1)
            lab = _label_for_window(csv_path=cp, video_key=vk, end_frame=ef,
                                    file_label=fl, label_mode="annotations", intervals=intervals)
            if lab is None: continue
            erows = []
            for i, row in enumerate(wr):
                er = dict(row)
                if i == 0:
                    er["torso_angular_accel"] = "0.0"
                    er["vertical_accel"] = "0.0"
                else:
                    er["torso_angular_accel"] = str(_sf(row.get("torso_angular_velocity",0)) - _sf(wr[i-1].get("torso_angular_velocity",0)))
                    er["vertical_accel"] = str(_sf(row.get("vertical_velocity",0)) - _sf(wr[i-1].get("vertical_velocity",0)))
                erows.append(er)
            X_all.append(flatten_window(erows, ACCEL_FEATURE_COLUMNS))
            y_all.append(lab)
            groups_all.append(vk)
    return WindowDataset(X=X_all, y=y_all, groups=groups_all,
                         feature_names=[f"f{i}" for i in range(len(X_all[0]) if X_all else 0)])


def _sf(v):
    try: fv=float(v); return fv if math.isfinite(fv) else 0.0
    except: return 0.0


def build_end_frame_map(csv_paths, val_videos, intervals):
    result = defaultdict(list)
    for cp in sorted(csv_paths):
        vk = _video_key(cp)
        if vk not in val_videos: continue
        rows = load_feature_rows(cp)
        if len(rows) < DEFAULT_WINDOW_SIZE: continue
        fl = infer_label_from_filename(cp)
        for start in range(0, len(rows)-DEFAULT_WINDOW_SIZE+1, DEFAULT_STRIDE):
            wr = rows[start:start+DEFAULT_WINDOW_SIZE]
            ef = _row_frame(wr[-1], start+DEFAULT_WINDOW_SIZE-1)
            lab = _label_for_window(csv_path=cp, video_key=vk, end_frame=ef,
                                    file_label=fl, label_mode="annotations", intervals=intervals)
            if lab is not None: result[vk].append(ef)
    return dict(result)


# ═══════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════

def evaluate(dataset, val_videos, class_weights, name, pf_segments, ef_map):
    """训练+严格+宽松双评估。"""
    train_idx = [i for i,g in enumerate(dataset.groups) if g not in val_videos]
    val_idx = [i for i,g in enumerate(dataset.groups) if g in val_videos]

    X_arr = np.asarray(dataset.X, dtype=float)
    y_arr = np.asarray(dataset.y)
    model = create_classifier("hist_gradient_boosting", RANDOM_STATE)
    sw = build_sample_weights(y_arr[train_idx], class_weights)
    model.fit(X_arr[train_idx], y_arr[train_idx], sample_weight=sw if sw is not None else None)

    # 收集预测
    strict_yt, strict_yp = [], []
    lenient_yt, lenient_yp = [], []
    layer_stats = {"early":{"t":0,"h":0,"n":0,"f":0},
                   "core":{"t":0,"h":0,"n":0,"f":0},
                   "late":{"t":0,"h":0,"n":0,"f":0}}

    for di in val_idx:
        tl, pl = dataset.y[di], str(model.predict(X_arr[di:di+1])[0])
        strict_yt.append(tl); strict_yp.append(pl)

        if tl == "Pre-fall":
            ef = ef_map.get(di)
            if ef:
                for ps, pe in pf_segments.get(dataset.groups[di], []):
                    if ps <= ef <= pe:
                        r = (ef-ps)/max(pe-ps, 1)
                        if r < BOUNDARY:
                            layer_stats["early"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["early"]["h"]+=1
                            elif pl=="Normal": layer_stats["early"]["n"]+=1
                            else: layer_stats["early"]["f"]+=1
                        elif r > 1-BOUNDARY:
                            layer_stats["late"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["late"]["h"]+=1
                            elif pl=="Normal": layer_stats["late"]["n"]+=1
                            else: layer_stats["late"]["f"]+=1
                        else:
                            layer_stats["core"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["core"]["h"]+=1
                            elif pl=="Normal": layer_stats["core"]["n"]+=1
                            else: layer_stats["core"]["f"]+=1
                            lenient_yt.append("Pre-fall"); lenient_yp.append(pl)
                        break
        else:
            lenient_yt.append(tl); lenient_yp.append(pl)

    sm = build_validation_metrics(strict_yt, strict_yp, sorted(set(strict_yt)|set(strict_yp)))
    lm = build_validation_metrics(lenient_yt, lenient_yp, sorted(set(lenient_yt)|set(lenient_yp)))

    spr = sm["classification_report"]
    lpr = lm["classification_report"]

    core = layer_stats["core"]
    cr = core["h"]/core["t"]*100 if core["t"] else 0

    print(f"\n  [{name}]")
    print(f"    严格: Acc={sm['accuracy']:.4f} MacF1={sm['macro_f1']:.4f}  "
          f"PF_F1={spr.get('Pre-fall',{}).get('f1-score',0):.4f} PF_Rec={spr.get('Pre-fall',{}).get('recall',0):.4f}  "
          f"F_F1={spr.get('Fall',{}).get('f1-score',0):.4f}")
    print(f"    宽松: Acc={lm['accuracy']:.4f} MacF1={lm['macro_f1']:.4f}  "
          f"PF_F1={lpr.get('Pre-fall',{}).get('f1-score',0):.4f} PF_Rec={lpr.get('Pre-fall',{}).get('recall',0):.4f}  "
          f"F_F1={lpr.get('Fall',{}).get('f1-score',0):.4f}")
    print(f"    Core: {core['h']}/{core['t']}={cr:.1f}%  "
          f"漏N={core['n']} 漏F={core['f']}  "
          f"early={layer_stats['early']['t']}窗(漏{layer_stats['early']['n']+layer_stats['early']['f']})  "
          f"late={layer_stats['late']['t']}窗(漏{layer_stats['late']['n']+layer_stats['late']['f']})")

    return {
        "name": name,
        "strict_acc": sm["accuracy"], "strict_mf1": sm["macro_f1"],
        "strict_pf_f1": spr.get("Pre-fall",{}).get("f1-score",0),
        "strict_pf_rec": spr.get("Pre-fall",{}).get("recall",0),
        "strict_f_f1": spr.get("Fall",{}).get("f1-score",0),
        "lenient_acc": lm["accuracy"], "lenient_mf1": lm["macro_f1"],
        "lenient_pf_f1": lpr.get("Pre-fall",{}).get("f1-score",0),
        "lenient_pf_rec": lpr.get("Pre-fall",{}).get("recall",0),
        "lenient_f_f1": lpr.get("Fall",{}).get("f1-score",0),
        "core_recall": cr, "core_total": core["t"],
        "core_miss_n": core["n"], "core_miss_f": core["f"],
    }


def two_stage_evaluate(dataset, val_videos, class_weights, name, pf_segments, ef_map):
    """二阶段 + 宽松评估"""
    train_idx = [i for i,g in enumerate(dataset.groups) if g not in val_videos]
    val_idx = [i for i,g in enumerate(dataset.groups) if g in val_videos]

    X_arr = np.asarray(dataset.X, dtype=float)
    y_arr = np.asarray(dataset.y)

    # S1: Normal vs Abnormal
    y_bin = np.array(["Abnormal" if l in ("Pre-fall","Fall") else "Normal" for l in y_arr])
    m1 = create_classifier("hist_gradient_boosting", RANDOM_STATE)
    sw1 = build_sample_weights(y_bin[train_idx], {"Normal":1.0,"Abnormal":class_weights.get("Pre-fall",8.0)})
    m1.fit(X_arr[train_idx], y_bin[train_idx], sample_weight=sw1 if sw1 is not None else None)

    # S2: Pre-fall vs Fall
    mask = np.array([l in ("Pre-fall","Fall") for l in y_arr])
    s2_train = [i for i in train_idx if mask[i]]
    s2_val = [i for i in val_idx if mask[i]]
    m2 = create_classifier("hist_gradient_boosting", RANDOM_STATE)
    sw2 = build_sample_weights(y_arr[s2_train], {"Pre-fall":class_weights.get("Pre-fall",8.0),"Fall":1.0})
    m2.fit(X_arr[s2_train], y_arr[s2_train], sample_weight=sw2 if sw2 is not None else None)

    # Eval
    strict_yt, strict_yp = [], []
    lenient_yt, lenient_yp = [], []
    layer_stats = {"early":{"t":0,"h":0,"n":0,"f":0},
                   "core":{"t":0,"h":0,"n":0,"f":0},
                   "late":{"t":0,"h":0,"n":0,"f":0}}

    for di in val_idx:
        tl = dataset.y[di]
        s1 = str(m1.predict(X_arr[di:di+1])[0])
        if s1 == "Normal":
            pl = "Normal"
        else:
            pl = str(m2.predict(X_arr[di:di+1])[0])
        strict_yt.append(tl); strict_yp.append(pl)

        if tl == "Pre-fall":
            ef = ef_map.get(di)
            if ef:
                for ps, pe in pf_segments.get(dataset.groups[di], []):
                    if ps <= ef <= pe:
                        r = (ef-ps)/max(pe-ps, 1)
                        if r < BOUNDARY:
                            layer_stats["early"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["early"]["h"]+=1
                            elif pl=="Normal": layer_stats["early"]["n"]+=1
                            else: layer_stats["early"]["f"]+=1
                        elif r > 1-BOUNDARY:
                            layer_stats["late"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["late"]["h"]+=1
                            elif pl=="Normal": layer_stats["late"]["n"]+=1
                            else: layer_stats["late"]["f"]+=1
                        else:
                            layer_stats["core"]["t"]+=1
                            if pl=="Pre-fall": layer_stats["core"]["h"]+=1
                            elif pl=="Normal": layer_stats["core"]["n"]+=1
                            else: layer_stats["core"]["f"]+=1
                            lenient_yt.append("Pre-fall"); lenient_yp.append(pl)
                        break
        else:
            lenient_yt.append(tl); lenient_yp.append(pl)

    sm = build_validation_metrics(strict_yt, strict_yp, sorted(set(strict_yt)|set(strict_yp)))
    lm = build_validation_metrics(lenient_yt, lenient_yp, sorted(set(lenient_yt)|set(lenient_yp)))

    spr = sm["classification_report"]
    lpr = lm["classification_report"]
    core = layer_stats["core"]
    cr = core["h"]/core["t"]*100 if core["t"] else 0

    print(f"\n  [{name}]")
    print(f"    严格: Acc={sm['accuracy']:.4f} MacF1={sm['macro_f1']:.4f}  "
          f"PF_F1={spr.get('Pre-fall',{}).get('f1-score',0):.4f} PF_Rec={spr.get('Pre-fall',{}).get('recall',0):.4f}  "
          f"F_F1={spr.get('Fall',{}).get('f1-score',0):.4f}")
    print(f"    宽松: Acc={lm['accuracy']:.4f} MacF1={lm['macro_f1']:.4f}  "
          f"PF_F1={lpr.get('Pre-fall',{}).get('f1-score',0):.4f} PF_Rec={lpr.get('Pre-fall',{}).get('recall',0):.4f}  "
          f"F_F1={lpr.get('Fall',{}).get('f1-score',0):.4f}")
    print(f"    Core: {core['h']}/{core['t']}={cr:.1f}%  漏N={core['n']} 漏F={core['f']}")

    return {
        "name": name,
        "strict_acc": sm["accuracy"], "strict_mf1": sm["macro_f1"],
        "strict_pf_f1": spr.get("Pre-fall",{}).get("f1-score",0),
        "strict_pf_rec": spr.get("Pre-fall",{}).get("recall",0),
        "strict_f_f1": spr.get("Fall",{}).get("f1-score",0),
        "lenient_acc": lm["accuracy"], "lenient_mf1": lm["macro_f1"],
        "lenient_pf_f1": lpr.get("Pre-fall",{}).get("f1-score",0),
        "lenient_pf_rec": lpr.get("Pre-fall",{}).get("recall",0),
        "lenient_f_f1": lpr.get("Fall",{}).get("f1-score",0),
        "core_recall": cr, "core_total": core["t"],
        "core_miss_n": core["n"], "core_miss_f": core["f"],
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    csv_paths = sorted((ROOT/"outputs/features/urfall_yolo").glob("*.csv"))
    csv_paths.extend(sorted((ROOT/"outputs/features/upfall_yolo").glob("*.csv")))

    # 固定验证视频
    full_ds = build_window_dataset(
        csv_paths=csv_paths, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode="annotations", annotations_path=str(ANNOTATIONS))
    _, vif = _group_train_test_split(
        y_array=np.asarray(full_ds.y), groups_array=np.asarray(full_ds.groups),
        test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_videos = {full_ds.groups[i] for i in vif}

    results = []

    # 所有实验共享的 end_frame map 和 pf_segments（基于 Row47 标注）
    base_ann = write_dropped_annotations(60, 0.15)
    base_int = load_label_intervals(str(base_ann))
    base_ef = build_end_frame_map(csv_paths, val_videos, base_int)

    pf_segments = defaultdict(list)
    with base_ann.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"].strip() == "Pre-fall":
                pf_segments[row["video"].strip()].append((int(row["start_frame"]), int(row["end_frame"])))

    # 为 base dataset 建立 dsidx→ef 映射
    base_ds = build_window_dataset(
        csv_paths=csv_paths, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode="annotations", annotations_path=str(base_ann))
    val_idx = [i for i,g in enumerate(base_ds.groups) if g in val_videos]
    video_val_groups = defaultdict(list)
    for di in val_idx: video_val_groups[base_ds.groups[di]].append(di)
    dsidx_to_ef = {}
    for video, dis in video_val_groups.items():
        efs = base_ef.get(video, [])
        for i, di in enumerate(sorted(dis)):
            if i < len(efs): dsidx_to_ef[di] = efs[i]

    print(f"{'='*75}")
    print(f"  20% 宽松评估对比 (边界={BOUNDARY*100:.0f}%)")
    print(f"{'='*75}")

    # 基线 Row 47
    r = evaluate(base_ds, val_videos, CW, "基线 Row47 (drop=60f,15%)", pf_segments, dsidx_to_ef)
    results.append(r)

    # 实验A: PF边界后移3/5帧 + Row47 Normal drop
    for shift in [3, 5]:
        sann = write_shifted_annotations(shift)
        sds = build_window_dataset(
            csv_paths=csv_paths, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
            feature_columns=ML_FEATURE_COLUMNS,
            label_mode="annotations", annotations_path=str(sann))
        r = evaluate(sds, val_videos, CW, f"A: PF后移{shift}f", pf_segments, dsidx_to_ef)
        results.append(r)

    # 实验B: 二阶段
    r = two_stage_evaluate(base_ds, val_videos, CW, "B: 二阶段分类", pf_segments, dsidx_to_ef)
    results.append(r)

    # 实验C: +加速度
    acc_ds = build_accel_dataset(csv_paths, base_ann)
    r = evaluate(acc_ds, val_videos, CW, "C: +加速度特征", pf_segments, dsidx_to_ef)
    results.append(r)

    # 实验D: PF后移5f + 加速度
    sann5 = write_shifted_annotations(5)
    combo_ds = build_accel_dataset(csv_paths, sann5)
    r = evaluate(combo_ds, val_videos, CW, "D: PF后移5f+加速度", pf_segments, dsidx_to_ef)
    results.append(r)

    # 实验E: 提高 class_weight (Pre-fall: 16, 24)
    for pf_w in [16.0, 24.0]:
        cw2 = {"Normal": 1.0, "Fall": 1.0, "Pre-fall": pf_w}
        r = evaluate(base_ds, val_videos, cw2, f"E: PF_weight={pf_w:.0f}", pf_segments, dsidx_to_ef)
        results.append(r)

    # ── 汇总 ──
    print(f"\n{'='*90}")
    print(f"  汇总: 20%宽松标准")
    print(f"{'='*90}")
    hdr = (f"{'方案':<28s} {'宽松Acc':>7s} {'宽松MF1':>7s}  "
           f"{'PF_F1':>7s} {'PF_Rec':>7s} {'PF_Pre':>7s}  "
           f"{'F_F1':>7s} {'CoreR':>6s} {'漏N':>4s} {'漏F':>4s}")
    print(hdr)
    print("-"*len(hdr))
    for r in results:
        n = r["name"]
        print(f"{n:<28s} {r['lenient_acc']:>7.4f} {r['lenient_mf1']:>7.4f}  "
              f"{r['lenient_pf_f1']:>7.4f} {r['lenient_pf_rec']:>7.4f} "
              f"{r['strict_pf_f1']:>7.4f}  "  # using strict PF precision as proxy
              f"{r['lenient_f_f1']:>7.4f} {r['core_recall']:>5.1f}% "
              f"{r['core_miss_n']:>4d} {r['core_miss_f']:>4d}")

    # 同时显示严格标准对比
    print(f"\n{'='*90}")
    print(f"  对比: 严格标准")
    print(f"{'='*90}")
    print(f"{'方案':<28s} {'严格Acc':>7s} {'严格MF1':>7s}  "
          f"{'PF_F1':>7s} {'PF_Rec':>7s}  {'F_F1':>7s}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<28s} {r['strict_acc']:>7.4f} {r['strict_mf1']:>7.4f}  "
              f"{r['strict_pf_f1']:>7.4f} {r['strict_pf_rec']:>7.4f}  "
              f"{r['strict_f_f1']:>7.4f}")

    print(f"\n实验完成。")


if __name__ == "__main__":
    main()
