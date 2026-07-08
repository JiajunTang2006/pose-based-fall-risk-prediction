from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "FallGuard.png"
OUTPUT = ROOT / "assets" / "FallGuard.icns"

ICNS_SIZES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
]

CORNER_RADIUS = 224


def macos_icon_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(1, round(size * CORNER_RADIUS / 1024))
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def prepare_icon(image: Image.Image, size: int = 1024) -> Image.Image:
    icon = image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    alpha = icon.getchannel("A")
    alpha = Image.composite(alpha, Image.new("L", (size, size), 0), macos_icon_mask(size))
    icon.putalpha(alpha)
    return icon


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Icon source not found: {SOURCE}")

    image = prepare_icon(Image.open(SOURCE))
    image.save(SOURCE)
    image.save(OUTPUT, sizes=ICNS_SIZES)
    print(OUTPUT)


if __name__ == "__main__":
    main()
