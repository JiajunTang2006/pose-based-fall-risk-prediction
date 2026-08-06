"""Generate README figures from the final cross-validation report."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "dual_model_tuned_static_lying_postprocess_5fold_cv.json"
OUTPUT_DIR = ROOT / "figures" / "readme"
MPL_CACHE = ROOT / "figures" / ".matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "ink": "#17324D",
    "muted": "#61758A",
    "grid": "#DCE5EC",
    "tree": "#D99A3D",
    "fusion": "#3B82A0",
    "confirmed": "#1F5A78",
    "early": "#C5963E",
    "normal": "#4C956C",
    "prefall": "#D7A63E",
    "fall": "#C85C5C",
    "panel": "#F7FAFC",
}

OUTPUTS = [
    ("tree", "Tree classifier", "#9AA5AE"),
    ("fusion_hmm", "Fusion + HMM", "#6F94A3"),
    ("postprocessed_confirmed", "Final confirmed", COLORS["confirmed"]),
    ("postprocessed_early_warning", "Final early warning", COLORS["early"]),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    face: str,
    edge: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(box)
    x, y = xy
    axis.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", weight="bold", color=COLORS["ink"], fontsize=10)
    axis.text(x + width / 2, y + height * 0.31, subtitle, ha="center", va="center", color=COLORS["muted"], fontsize=8, linespacing=1.25)


def add_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#718096") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.25, color=color, connectionstyle="arc3,rad=0")
    axis.add_patch(arrow)


def system_overview() -> None:
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.set_xlim(0, 1.04)
    axis.set_ylim(0, 1)
    axis.axis("off")

    add_box(axis, (0.02, 0.67), 0.13, 0.18, "Video input", "UR Fall + UP-Fall\nframes", "#EEF3F7", "#708090")
    add_box(axis, (0.19, 0.67), 0.13, 0.18, "YOLO Pose", "17 COCO\nkeypoints", "#E6F1F5", COLORS["fusion"])
    add_box(axis, (0.36, 0.67), 0.13, 0.18, "Temporal window", "15 frames\nstride 3", "#EAF3EE", COLORS["normal"])

    add_box(axis, (0.19, 0.30), 0.19, 0.19, "Engineered features", "angle · velocity · drop\naspect ratio · visibility", "#FCF4E7", COLORS["tree"])
    add_box(axis, (0.41, 0.30), 0.16, 0.19, "HistGradientBoosting", "authoritative\nconfirmed class", "#F6E7C9", COLORS["tree"])
    add_box(axis, (0.61, 0.30), 0.16, 0.19, "Skeleton fusion", "ST-GCN + causal TCN\ncalibrated probabilities", "#E4F0F4", COLORS["fusion"])
    add_box(axis, (0.61, 0.67), 0.16, 0.18, "Skeleton tensor", "5 × 15 × 17\n+ 13 temporal features", "#EDF5F8", COLORS["fusion"])

    add_box(axis, (0.60, 0.03), 0.17, 0.16, "Cooperative decision", "tree confirmation · HMM\n3-step fusion Fall gate", "#EFEAF4", "#77648A")
    add_box(axis, (0.81, 0.03), 0.17, 0.16, "Postprocessing", "static-lying ADL filter\nFall latch", "#F7EAEA", "#A96B6B")

    add_box(axis, (0.81, 0.33), 0.17, 0.15, "Final confirmed", "precision-oriented\nofficial state", "#E7F0F4", COLORS["confirmed"])
    add_box(axis, (0.81, 0.60), 0.17, 0.15, "Early warning", "recall-oriented\nadvisory", "#FCF3E3", COLORS["early"])

    add_arrow(axis, (0.15, 0.76), (0.19, 0.76))
    add_arrow(axis, (0.32, 0.76), (0.36, 0.76))
    add_arrow(axis, (0.425, 0.67), (0.285, 0.49), COLORS["tree"])
    add_arrow(axis, (0.49, 0.76), (0.61, 0.76), COLORS["fusion"])
    add_arrow(axis, (0.38, 0.395), (0.41, 0.395), COLORS["tree"])
    add_arrow(axis, (0.69, 0.67), (0.69, 0.49), COLORS["fusion"])
    add_arrow(axis, (0.49, 0.30), (0.64, 0.19), COLORS["tree"])
    add_arrow(axis, (0.69, 0.30), (0.69, 0.19), COLORS["fusion"])
    add_arrow(axis, (0.77, 0.11), (0.81, 0.11), "#77648A")
    add_arrow(axis, (0.87, 0.19), (0.87, 0.33), COLORS["confirmed"])
    axis.plot([0.98, 1.02, 1.02], [0.11, 0.11, 0.67], color=COLORS["early"], linewidth=1.25)
    add_arrow(axis, (1.02, 0.67), (0.98, 0.67), COLORS["early"])

    axis.text(0.02, 0.03, "Output classes", color=COLORS["muted"], weight="bold", fontsize=9)
    axis.text(0.16, 0.03, "Normal", color=COLORS["normal"], weight="bold", fontsize=9)
    axis.text(0.25, 0.03, "Pre-fall", color=COLORS["prefall"], weight="bold", fontsize=9)
    axis.text(0.35, 0.03, "Fall", color=COLORS["fall"], weight="bold", fontsize=9)
    save_figure(figure, "system_overview")


def window_performance(report: dict) -> None:
    aggregate = report["aggregate"]
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("prefall_recall", "Pre-fall recall"),
        ("fall_recall", "Fall recall"),
    ]
    x = np.arange(len(metrics))
    width = 0.19
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for index, (key, label, color) in enumerate(OUTPUTS):
        values = [aggregate[key][metric]["mean"] * 100 for metric, _ in metrics]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.7)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7, color=COLORS["ink"])
    axis.set_ylabel("Five-fold mean (%)")
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 105)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    save_figure(figure, "window_performance")


def confusion_matrices(report: dict) -> None:
    pooled = report["pooled_out_of_fold_metrics"]
    panels = [
        ("postprocessed_confirmed", "Final confirmed"),
        ("postprocessed_early_warning", "Final early warning"),
    ]
    labels = ["Fall", "Normal", "Pre-fall"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=True)
    image = None
    for axis, (key, title) in zip(axes, panels, strict=True):
        counts = np.asarray(pooled[key]["confusion_matrix"], dtype=float)
        shares = counts / counts.sum(axis=1, keepdims=True) * 100
        image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=100)
        for row in range(3):
            for column in range(3):
                color = "white" if shares[row, column] >= 55 else COLORS["ink"]
                axis.text(column, row, f"{int(counts[row, column])}\n{shares[row, column]:.1f}%", ha="center", va="center", color=color, fontsize=10, weight="bold" if row == column else "normal")
        axis.set_title(title, weight="bold", color=COLORS["ink"], pad=12)
        axis.set_xticks(range(3), labels)
        axis.set_yticks(range(3), labels)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.tick_params(length=0)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.03)
    colorbar.set_label("Row-normalized share (%)")
    save_figure(figure, "confusion_matrices")


def sequence_tradeoff(report: dict) -> None:
    summary = report["pooled_sequence_summary"]
    markers = ["o", "D", "s", "^"]
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    annotations = {
        "tree": ((8, 8), "left"),
        "fusion_hmm": ((8, 8), "left"),
        "postprocessed_confirmed": ((-8, 8), "right"),
        "postprocessed_early_warning": ((8, 8), "left"),
    }
    for (key, label, color), marker in zip(OUTPUTS, markers, strict=True):
        item = summary[key]
        warning_rate = item["nonfall_sequences_with_any_warning"] / item["nonfall_sequences"] * 100
        detection_rate = item["fall_sequences_detected"] / item["fall_sequences"] * 100
        axis.scatter(warning_rate, detection_rate, s=105, marker=marker, color=color, edgecolor="white", linewidth=1.2, zorder=3, label=label)
        offset, alignment = annotations[key]
        axis.annotate(label, (warning_rate, detection_rate), xytext=offset, textcoords="offset points", fontsize=9, color=COLORS["ink"], ha=alignment)
    axis.set_xlabel("Non-fall sequences with any warning (%)")
    axis.set_ylabel("Fall-sequence detection (%)")
    axis.set_xlim(15, 66)
    axis.set_ylim(96.5, 99.1)
    axis.grid(color=COLORS["grid"], linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, "sequence_tradeoff")


def main() -> None:
    configure_style()
    with REPORT_PATH.open(encoding="utf-8") as stream:
        report = json.load(stream)
    system_overview()
    window_performance(report)
    confusion_matrices(report)
    sequence_tradeoff(report)


if __name__ == "__main__":
    main()
