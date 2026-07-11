from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "FallGuard.png"
ICONSET = ROOT / "assets" / "FallGuard.iconset"

ICON_SIZES = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}

CORNER_RADIUS = 224
CONTENT_SCALE = 0.82


def macos_icon_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(1, round(size * CORNER_RADIUS / 1024))
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def prepare_icon(image: Image.Image, size: int) -> Image.Image:
    # Modern macOS icons include transparent optical padding around the
    # rounded-square artwork.  Filling the complete bitmap makes the icon look
    # noticeably larger than neighbouring apps in the Dock.
    source = image.convert("RGBA")
    alpha = source.getchannel("A")
    alpha = Image.composite(
        alpha,
        Image.new("L", source.size, 0),
        macos_icon_mask(source.width),
    )
    source.putalpha(alpha)

    content_size = max(1, round(size * CONTENT_SCALE))
    content = source.resize((content_size, content_size), Image.Resampling.LANCZOS)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = (size - content_size) // 2
    icon.alpha_composite(content, (offset, offset))
    return icon


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Icon source not found: {SOURCE}")

    image = Image.open(SOURCE)
    ICONSET.mkdir(parents=True, exist_ok=True)

    for filename, size in ICON_SIZES.items():
        output = ICONSET / filename
        prepare_icon(image, size).save(output)

    print(ICONSET)


if __name__ == "__main__":
    main()
