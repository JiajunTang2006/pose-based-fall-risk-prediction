"""
低光增强：在姿态估计之前，对偏暗的帧做一次轻量提亮。

背景：
    摄像头一旦进入光线不足的环境，画面又暗又噪，YOLO-pose 的关键点
    就会检不准，后面的风险判定全是拿垃圾数据在推断。真正该修的是链路
    最前端的输入，而不是在状态机里打补丁。

方案（最简单且有效）：
    在 LAB 色彩空间的 L（亮度）通道上做 CLAHE（限制对比度自适应直方图
    均衡）。只动亮度、不动色彩，实时、零额外依赖（cv2 本来就打包着）。

亮度阈值：
    不是每帧都增强。画面本来就够亮时做 CLAHE 是白费算力，还可能过曝、
    放大噪点反而害了关键点检出。所以先看这一帧的平均亮度，只有低于阈值
    才增强。阈值判断和增强共用同一次色彩空间转换，省掉一半开销。
"""

from __future__ import annotations

# L 通道（0-255）的平均亮度阈值：低于这个值才认为画面偏暗、需要增强。
# 室内正常照明的 L 均值通常在 110 以上；夜间/背光场景会明显更低。
# 调高 → 更多帧被判为“暗”从而增强；调低 → 只在很暗时才介入。
DEFAULT_BRIGHTNESS_THRESHOLD = 100.0

# CLAHE 的对比度上限。太高会把噪点一起放大，2.0~3.0 比较稳。
DEFAULT_CLIP_LIMIT = 2.0

# CLAHE 分块网格大小。越大越接近全局均衡，越小越强调局部对比。
DEFAULT_TILE_GRID_SIZE = (8, 8)


def enhance_low_light(
    frame,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
):
    """
    对偏暗的 BGR 帧做 CLAHE 提亮；够亮的帧原样返回。

    参数:
        frame:                输入的 BGR 图像（cv2 读到的原始帧）。
        brightness_threshold: L 通道平均亮度阈值，低于它才增强。
        clip_limit:           CLAHE 对比度上限。
        tile_grid_size:       CLAHE 分块网格。

    返回:
        增强后的 BGR 帧；若画面够亮或输入无效，则返回原始帧本身
        （不复制，调用方拿到的就是原对象）。
    """
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    # 转到 LAB，只取亮度通道做判断和增强，避免破坏色彩。
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 够亮就直接放行，不做任何处理。
    if float(l_channel.mean()) >= brightness_threshold:
        return frame

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
