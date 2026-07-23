

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_csv(csv_path: str | Path, output_path: str | Path) -> None:

    import matplotlib


    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")


    time = [float(row["time"]) for row in rows]

    series = {
        "risk_score": [float(row["smoothed_risk_score"]) for row in rows],
        "torso_angle": [float(row["torso_angle"]) for row in rows],
        "vertical_velocity": [float(row["vertical_velocity"]) for row in rows],
        "aspect_ratio": [float(row["aspect_ratio"]) for row in rows],
    }


    fig, axes = plt.subplots(len(series), 1, figsize=(10, 8), sharex=True)
    for axis, (name, values) in zip(axes, series.items(), strict=True):
        axis.plot(time, values, linewidth=1.8)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()


    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Chart saved to {output}")


def main() -> None:

    parser = argparse.ArgumentParser(description="Plot feature curves from fall prediction CSV output.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/feature_curves.png")
    args = parser.parse_args()
    plot_csv(args.csv_path, args.output)


if __name__ == "__main__":
    main()
