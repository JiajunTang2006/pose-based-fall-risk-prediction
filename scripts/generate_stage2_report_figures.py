from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mplcache")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_window_distribution()
    make_strategy_tradeoff()


def make_window_distribution() -> None:
    metrics = json.loads((REPORTS / "yolo_tail60_prefall_accel_metrics.json").read_text())
    counts = metrics["validation_split"]["train_label_counts"]
    labels = ["Normal", "Pre-fall", "Fall"]
    values = [counts.get(label, 0) for label in labels]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    total = sum(values)
    ax.set_title("Final Training Windows by Class")
    ax.set_ylabel("Window count")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        pct = value / total * 100 if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value:,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage2-window-distribution.png", dpi=180)
    plt.close(fig)


def make_strategy_tradeoff() -> None:
    rows = list(csv.DictReader((REPORTS / "normal_start_combined_drop_grid_strict_validation_comparison.csv").open()))
    for row in rows:
        for key in (
            "drop_start_frames",
            "drop_percent",
            "accuracy",
            "macro_f1",
            "prefall_f1",
            "fall_f1",
            "normal_frames_dropped",
        ):
            row[key] = float(row[key])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter = ax.scatter(
        [row["accuracy"] for row in rows],
        [row["prefall_f1"] for row in rows],
        c=[row["normal_frames_dropped"] for row in rows],
        s=80,
        cmap="viridis",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_title("Normal-Trimming Strategy Trade-off")
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Pre-fall F1")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    annotations = {
        (0, 0): ("Baseline\n0f+0%", (8, 8)),
        (20, 10): ("Balanced\n20f+10%", (8, 8)),
        (60, 15): ("PF-focused\n60f+15%", (10, -38)),
    }
    for row in rows:
        key = (int(row["drop_start_frames"]), int(row["drop_percent"]))
        if key in annotations:
            label, offset = annotations[key]
            ax.annotate(
                label,
                (row["accuracy"], row["prefall_f1"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#777777", "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
            )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Normal frames dropped")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage2-strategy-tradeoff.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
