from __future__ import annotations

import contextlib
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mplcache")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fall_prediction.train_model import train_and_save
from fall_prediction.window_dataset import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE

from run_normal_start_drop_experiment import (
    EXPERIMENT_DIR,
    MODEL_TMP_DIR,
    REPORTS_DIR,
    SOURCE_ANNOTATIONS,
    build_strict_window_dataset,
    label_metric,
    write_csv,
)


DROP_FRAMES = (0, 5, 10, 15, 20, 30, 45, 60)
DROP_RATIOS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
MIN_REMAINING_NORMAL_FRAMES = DEFAULT_WINDOW_SIZE


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    feature_paths = sorted((ROOT / "outputs/features/urfall_yolo").glob("*.csv"))
    feature_paths.extend(sorted((ROOT / "outputs/features/upfall_yolo").glob("*.csv")))
    if not feature_paths:
        raise RuntimeError("No feature CSV files found.")

    log_path = REPORTS_DIR / "normal_start_combined_drop_grid_training.log"
    rows = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for drop_frames in DROP_FRAMES:
            for drop_ratio in DROP_RATIOS:
                annotations_path, annotation_summary = write_combined_drop_annotations(
                    drop_frames=drop_frames,
                    drop_ratio=drop_ratio,
                )
                dataset = build_strict_window_dataset(feature_paths, annotations_path)
                label_counts = Counter(dataset.y)

                tag = f"{drop_frames:02d}f_{ratio_tag(drop_ratio)}"
                output_model = (
                    MODEL_TMP_DIR / f"yolo_normaldropcombined_{tag}_strict_falltail60_validation_classifier.joblib"
                )
                metrics_path = REPORTS_DIR / f"yolo_normaldropcombined_{tag}_strict_falltail60_validation_metrics.json"

                print(f"Running {tag} windows={len(dataset.y)}", flush=True)
                print(f"\n===== {tag} =====", file=log_file, flush=True)
                with contextlib.redirect_stdout(log_file):
                    artifact = train_and_save(
                        X=dataset.X,
                        y=dataset.y,
                        groups=dataset.groups,
                        feature_names=dataset.feature_names,
                        csv_paths=feature_paths,
                        output_path=output_model,
                        window_size=DEFAULT_WINDOW_SIZE,
                        stride=DEFAULT_STRIDE,
                        baseline_frames=15,
                        smoothing_window=5,
                        classifier_name="hist_gradient_boosting",
                        label_mode="annotations",
                        test_size=0.25,
                        random_state=42,
                        metrics_output_path=metrics_path,
                        class_weights={"Normal": 1.0, "Fall": 1.0, "Pre-fall": 8.0},
                        prefall_alert_threshold=0.41,
                    )

                metrics = artifact["validation_metrics"] or {}
                report = metrics.get("classification_report", {})
                split = artifact["validation_split"]
                rows.append(
                    {
                        "drop_start_frames": drop_frames,
                        "drop_ratio": drop_ratio,
                        "drop_percent": int(round(drop_ratio * 100)),
                        "annotations_path": str(annotations_path.relative_to(ROOT)),
                        "metrics_path": str(metrics_path.relative_to(ROOT)),
                        "model_path": str(output_model),
                        "normal_segments": annotation_summary.get("normal_segments", 0),
                        "normal_segments_clipped": annotation_summary.get("normal_segments_clipped", 0),
                        "normal_frames_before": annotation_summary.get("normal_frames_before", 0),
                        "normal_frames_after": annotation_summary.get("normal_frames_after", 0),
                        "normal_frames_fixed_dropped": annotation_summary.get("normal_frames_fixed_dropped", 0),
                        "normal_frames_percent_dropped": annotation_summary.get("normal_frames_percent_dropped", 0),
                        "normal_frames_dropped": annotation_summary.get("normal_frames_dropped", 0),
                        "total_windows": len(dataset.y),
                        "normal_windows": label_counts.get("Normal", 0),
                        "prefall_windows": label_counts.get("Pre-fall", 0),
                        "fall_windows": label_counts.get("Fall", 0),
                        "train_windows": split["train_samples"],
                        "validation_windows": split["validation_samples"],
                        "accuracy": metrics.get("accuracy", 0.0),
                        "macro_f1": metrics.get("macro_f1", 0.0),
                        "weighted_f1": metrics.get("weighted_f1", 0.0),
                        "normal_f1": label_metric(report, "Normal", "f1-score"),
                        "prefall_f1": label_metric(report, "Pre-fall", "f1-score"),
                        "fall_f1": label_metric(report, "Fall", "f1-score"),
                        "normal_recall": label_metric(report, "Normal", "recall"),
                        "prefall_recall": label_metric(report, "Pre-fall", "recall"),
                        "fall_recall": label_metric(report, "Fall", "recall"),
                        "normal_precision": label_metric(report, "Normal", "precision"),
                        "prefall_precision": label_metric(report, "Pre-fall", "precision"),
                        "fall_precision": label_metric(report, "Fall", "precision"),
                    }
                )

    summary_path = REPORTS_DIR / "normal_start_combined_drop_grid_strict_validation_comparison.csv"
    write_csv(summary_path, rows)
    plot_heatmaps(summary_path, rows)

    print("\nTop by macro_f1:")
    for row in sorted(rows, key=lambda item: item["macro_f1"], reverse=True)[:10]:
        print(format_row(row))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {log_path}")


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):02d}pct"


def write_combined_drop_annotations(drop_frames: int, drop_ratio: float) -> tuple[Path, dict[str, int]]:
    output_path = (
        EXPERIMENT_DIR
        / f"training_ur_up_normaldropcombined_{drop_frames:02d}f_{ratio_tag(drop_ratio)}_strict_falltail60_annotations.csv"
    )
    fieldnames = [
        "video",
        "start_frame",
        "end_frame",
        "label",
        "source_annotation",
        "drop_start_frames",
        "drop_ratio",
        "original_start_frame",
        "original_end_frame",
        "fixed_dropped_start_frames",
        "percent_dropped_start_frames",
        "dropped_start_frames",
    ]

    summary = Counter()
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for source_path in SOURCE_ANNOTATIONS:
            with source_path.open("r", newline="", encoding="utf-8") as source_file:
                for row in csv.DictReader(source_file):
                    start = int(row["start_frame"])
                    end = int(row["end_frame"])
                    label = row["label"].strip()
                    original_start = start
                    original_end = end
                    fixed_dropped = 0
                    percent_dropped = 0

                    if label == "Normal":
                        length = end - start + 1
                        max_fixed_drop = max(0, length - MIN_REMAINING_NORMAL_FRAMES)
                        fixed_dropped = min(drop_frames, max_fixed_drop)
                        remaining_length = length - fixed_dropped
                        max_percent_drop = max(0, remaining_length - MIN_REMAINING_NORMAL_FRAMES)
                        percent_dropped = min(
                            int(math.floor(remaining_length * drop_ratio)),
                            max_percent_drop,
                        )
                        start += fixed_dropped + percent_dropped

                        summary["normal_segments"] += 1
                        summary["normal_frames_before"] += length
                        summary["normal_frames_after"] += end - start + 1
                        summary["normal_frames_fixed_dropped"] += fixed_dropped
                        summary["normal_frames_percent_dropped"] += percent_dropped
                        summary["normal_frames_dropped"] += fixed_dropped + percent_dropped
                        if fixed_dropped or percent_dropped:
                            summary["normal_segments_clipped"] += 1

                    writer.writerow(
                        {
                            "video": row["video"],
                            "start_frame": start,
                            "end_frame": end,
                            "label": label,
                            "source_annotation": source_path.name,
                            "drop_start_frames": drop_frames,
                            "drop_ratio": drop_ratio,
                            "original_start_frame": original_start,
                            "original_end_frame": original_end,
                            "fixed_dropped_start_frames": fixed_dropped,
                            "percent_dropped_start_frames": percent_dropped,
                            "dropped_start_frames": fixed_dropped + percent_dropped,
                        }
                    )

    return output_path, dict(summary)


def plot_heatmaps(summary_path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipped plot generation.")
        return

    metrics = [
        ("macro_f1", "Macro F1"),
        ("prefall_f1", "Pre-fall F1"),
        ("fall_f1", "Fall F1"),
        ("accuracy", "Accuracy"),
    ]
    frame_values = list(DROP_FRAMES)
    percent_values = [int(round(ratio * 100)) for ratio in DROP_RATIOS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        grid = []
        for frame in frame_values:
            row_values = []
            for percent in percent_values:
                match = next(
                    item
                    for item in rows
                    if item["drop_start_frames"] == frame and item["drop_percent"] == percent
                )
                row_values.append(match[metric])
            grid.append(row_values)

        image = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Percent dropped after fixed-frame deletion")
        ax.set_ylabel("Fixed frames deleted first")
        ax.set_xticks(range(len(percent_values)), labels=percent_values)
        ax.set_yticks(range(len(frame_values)), labels=frame_values)
        for y, row_values in enumerate(grid):
            for x, value in enumerate(row_values):
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(summary_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def format_row(row: dict) -> str:
    return (
        f"{row['drop_start_frames']:>2}f + {row['drop_percent']:>2}% "
        f"accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
        f"Normal={row['normal_f1']:.4f} Pre-fall={row['prefall_f1']:.4f} "
        f"Fall={row['fall_f1']:.4f} normal_windows={row['normal_windows']}"
    )


if __name__ == "__main__":
    main()
