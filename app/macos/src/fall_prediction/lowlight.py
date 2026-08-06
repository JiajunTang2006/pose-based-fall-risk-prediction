from __future__ import annotations

DEFAULT_BRIGHTNESS_THRESHOLD = 100.0

DEFAULT_CLIP_LIMIT = 2.0

DEFAULT_TILE_GRID_SIZE = (8, 8)


def enhance_low_light(
    frame,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
):
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    if float(l_channel.mean()) >= brightness_threshold:
        return frame

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
