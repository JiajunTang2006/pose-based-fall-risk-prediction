from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "FallGuard.png"
SIZE = 1024


def draw_icon() -> None:
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.axis("off")

    # Soft canvas background.
    canvas = np.ones((SIZE, SIZE, 4), dtype=float)
    canvas[..., :3] = np.array([0.96, 0.97, 0.99])
    canvas[..., 3] = 1.0
    ax.imshow(canvas, extent=(0, SIZE, 0, SIZE), origin="lower")

    # Rounded app tile with a subtle vertical gradient.
    tile = FancyBboxPatch(
        (120, 120),
        784,
        784,
        boxstyle="round,pad=0,rounding_size=150",
        linewidth=0,
        facecolor="#132a4a",
    )
    ax.add_patch(tile)

    y = np.linspace(0, 1, SIZE)
    top = np.array([0x23, 0x3c, 0x68]) / 255.0
    bottom = np.array([0x0d, 0x1f, 0x3a]) / 255.0
    grad = bottom[None, :] * (1 - y[:, None]) + top[None, :] * y[:, None]
    grad_img = np.zeros((SIZE, SIZE, 4))
    grad_img[..., :3] = grad[:, None, :]
    grad_img[..., 3] = 1.0
    im = ax.imshow(grad_img, extent=(120, 904, 120, 904), origin="lower")
    im.set_clip_path(tile)

    # Radar rings.
    for radius, alpha in ((150, 0.16), (220, 0.14), (290, 0.12)):
        ax.add_patch(Circle((512, 560), radius, fill=False, linewidth=5, edgecolor=(0.33, 0.91, 0.92, alpha)))

    # Bottom wave and glow.
    xs = np.linspace(120, 904, 300)
    ys = 365 + 38 * np.sin((xs - 120) / 120) + 22 * np.sin((xs - 120) / 80 + 0.9)
    wave_poly = np.column_stack([np.r_[120, xs, 904], np.r_[120, ys, 120]])
    wave = Polygon(wave_poly, closed=True, facecolor="#35b8c6", alpha=0.55, edgecolor="none")
    ax.add_patch(wave)
    wave.set_clip_path(tile)
    ax.add_patch(Circle((512, 345), 150, color="#4ff5eb", alpha=0.23, linewidth=0))

    # Falling person.
    person_color = "#f3fbff"
    shadow = "#a6d7ee"
    ax.plot([405, 505], [560, 620], color=shadow, linewidth=56, solid_capstyle="round", alpha=0.45)
    ax.plot([505, 615], [620, 505], color=shadow, linewidth=56, solid_capstyle="round", alpha=0.45)
    ax.plot([615, 720], [505, 440], color=shadow, linewidth=56, solid_capstyle="round", alpha=0.45)
    ax.plot([515, 430], [590, 405], color=shadow, linewidth=48, solid_capstyle="round", alpha=0.45)

    ax.add_patch(Circle((405, 625), 42, color=person_color, linewidth=0))
    ax.plot([455, 555], [570, 510], color=person_color, linewidth=64, solid_capstyle="round")
    ax.plot([540, 622], [526, 625], color=person_color, linewidth=44, solid_capstyle="round")
    ax.plot([548, 648], [500, 560], color=person_color, linewidth=48, solid_capstyle="round")
    ax.plot([648, 725], [560, 505], color=person_color, linewidth=48, solid_capstyle="round")
    ax.plot([510, 438], [540, 400], color=person_color, linewidth=44, solid_capstyle="round")
    ax.plot([578, 665], [485, 450], color=person_color, linewidth=46, solid_capstyle="round")
    ax.plot([665, 704], [450, 360], color=person_color, linewidth=46, solid_capstyle="round")

    # Alert triangle.
    triangle = Polygon([(740, 735), (805, 625), (675, 625)], closed=True, facecolor="#ff7377", edgecolor="#ffb3ad", linewidth=6)
    ax.add_patch(triangle)
    ax.text(740, 660, "!", ha="center", va="center", color="white", fontsize=80, fontweight="bold")

    # ECG lines.
    ecg_left_x = [170, 225, 245, 258, 280, 295, 310]
    ecg_left_y = [285, 285, 285, 335, 285, 305, 285]
    ecg_right_x = [710, 760, 775, 792, 810, 825, 845]
    ecg_right_y = [285, 285, 285, 335, 285, 305, 285]
    ax.plot(ecg_left_x, ecg_left_y, color="#67f1e7", linewidth=5, alpha=0.85)
    ax.plot(ecg_right_x, ecg_right_y, color="#67f1e7", linewidth=5, alpha=0.85)

    # Brand text.
    ax.text(512, 242, "Fall", ha="right", va="center", color="#f4f8ff", fontsize=72, fontweight="bold")
    ax.text(512, 242, "Guard", ha="left", va="center", color="#66efe7", fontsize=72, fontweight="bold")
    ax.text(512, 170, "Fall Prediction", ha="center", va="center", color="#b8c6d7", fontsize=34)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    draw_icon()
    print(OUTPUT)
