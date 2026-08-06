"""
MediaPipe 人体姿势关键点（Landmark）的编号定义和辅助函数。

MediaPipe 可以从一张图片中检测出 33 个人体关键点，比如鼻子、肩膀、膝盖等。
每个关键点有 (x, y, z) 坐标和 visibility（可见度，0=完全不可见，1=完全可见）。

这个文件把这些编号定义成常量，方便在代码中使用有意义的名字
（比如用 LEFT_SHOULDER 而不是数字 11），同时提供了一些常用的辅助函数。

为什么要把编号定义在这里而不是直接用 MediaPipe 的？
——这样即使在没有安装 MediaPipe 的环境下，也可以对特征提取层进行单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Landmark:
    """
    一个关键点的数据结构。

    属性：
        x, y: 图像中的归一化坐标（0~1），x 从左到右，y 从上到下
        z:    深度坐标（归一化到 0~1），值越大离摄像头越近
        visibility: 该点的可见度/置信度（0~1），值越大越可靠
    """
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


# ============================================================================
# MediaPipe 33 个关键点的编号常量
# 编号 0~32，对应人体的不同部位
# ============================================================================

NOSE = 0             # 鼻子
LEFT_EYE_INNER = 1   # 左眼内角
LEFT_EYE = 2         # 左眼中心
LEFT_EYE_OUTER = 3   # 左眼外角
RIGHT_EYE_INNER = 4  # 右眼内角
RIGHT_EYE = 5        # 右眼中心
RIGHT_EYE_OUTER = 6  # 右眼外角
LEFT_EAR = 7         # 左耳
RIGHT_EAR = 8        # 右耳
MOUTH_LEFT = 9       # 嘴左角
MOUTH_RIGHT = 10     # 嘴右角
LEFT_SHOULDER = 11   # 左肩
RIGHT_SHOULDER = 12  # 右肩
LEFT_ELBOW = 13      # 左肘
RIGHT_ELBOW = 14     # 右肘
LEFT_WRIST = 15      # 左腕
RIGHT_WRIST = 16     # 右腕
LEFT_PINKY = 17      # 左小指
RIGHT_PINKY = 18     # 右小指
LEFT_INDEX = 19      # 左食指
RIGHT_INDEX = 20     # 右食指
LEFT_THUMB = 21      # 左拇指
RIGHT_THUMB = 22     # 右拇指
LEFT_HIP = 23        # 左髋（左胯）
RIGHT_HIP = 24       # 右髋（右胯）
LEFT_KNEE = 25       # 左膝
RIGHT_KNEE = 26      # 右膝
LEFT_ANKLE = 27      # 左脚踝
RIGHT_ANKLE = 28     # 右脚踝
LEFT_HEEL = 29       # 左脚跟
RIGHT_HEEL = 30      # 右脚跟
LEFT_FOOT_INDEX = 31  # 左脚尖
RIGHT_FOOT_INDEX = 32 # 右脚尖

# MediaPipe 总共检测 33 个关键点
LANDMARK_COUNT = 33

# 对跌倒检测最重要的 8 个关键点：双肩、双髋、双膝、双踝
# 这些点构成了人体的核心骨架，是判断姿势和运动的主要依据
IMPORTANT_LANDMARKS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

# 骨骼连接关系：哪些关键点之间应该画线连接
# 比如左肩连右肩、左肩连左肘、左肘连左腕...形成了人体的骨架图
POSE_CONNECTIONS = (
    # 上半身连接
    (LEFT_SHOULDER, RIGHT_SHOULDER),   # 双肩连线
    (LEFT_SHOULDER, LEFT_ELBOW),       # 左大臂
    (LEFT_ELBOW, LEFT_WRIST),          # 左小臂
    (RIGHT_SHOULDER, RIGHT_ELBOW),     # 右大臂
    (RIGHT_ELBOW, RIGHT_WRIST),        # 右小臂
    # 躯干连接
    (LEFT_SHOULDER, LEFT_HIP),         # 左躯干
    (RIGHT_SHOULDER, RIGHT_HIP),       # 右躯干
    (LEFT_HIP, RIGHT_HIP),             # 髋部连线
    # 下半身连接
    (LEFT_HIP, LEFT_KNEE),             # 左大腿
    (LEFT_KNEE, LEFT_ANKLE),           # 左小腿
    (RIGHT_HIP, RIGHT_KNEE),           # 右大腿
    (RIGHT_KNEE, RIGHT_ANKLE),         # 右小腿
    # 脚部连接
    (LEFT_ANKLE, LEFT_HEEL),           # 左脚踝到脚跟
    (LEFT_HEEL, LEFT_FOOT_INDEX),      # 左脚跟到脚尖
    (RIGHT_ANKLE, RIGHT_HEEL),         # 右脚踝到脚跟
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),    # 右脚跟到脚尖
)


def has_landmarks(landmarks: Sequence[Landmark] | None) -> bool:
    """
    检查是否检测到了有效的关键点数据。

    返回 True 的条件：
    1. landmarks 不是 None（确实检测到了人）
    2. 关键点数量 >= 33（数据完整）

    参数:
        landmarks: 关键点列表，可能为 None

    返回:
        bool: 是否有有效的关键点
    """
    return landmarks is not None and len(landmarks) >= LANDMARK_COUNT


def midpoint(first: Landmark, second: Landmark) -> Landmark:
    """
    计算两个关键点的中点。

    常用于求"躯干中心"——计算左肩和右肩的中点得到肩膀中心，
    计算左髋和右髋的中点得到髋部中心。

    参数:
        first:  第一个关键点
        second: 第二个关键点

    返回:
        Landmark: 包含两个点平均值的新关键点
    """
    return Landmark(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        z=(first.z + second.z) / 2.0,
        visibility=(first.visibility + second.visibility) / 2.0,
    )


def mean_visibility(
    landmarks: Sequence[Landmark],
    indices: Iterable[int] = IMPORTANT_LANDMARKS,
) -> float:
    """
    计算指定关键点的平均可见度。

    可见度低说明摄像头看不清这个人（比如被遮挡或光线不好），
    此时检测结果不可靠，风险评分应该降低。

    参数:
        landmarks: 关键点列表
        indices:   要计算的关键点编号（默认使用 IMPORTANT_LANDMARKS）

    返回:
        float: 平均可见度（0~1）
    """
    values = [landmarks[index].visibility for index in indices]
    return sum(values) / len(values) if values else 0.0


def visible_points(
    landmarks: Sequence[Landmark],
    min_visibility: float = 0.2,
) -> list[Landmark]:
    """
    筛选出可见度足够高的关键点。

    参数:
        landmarks:      关键点列表
        min_visibility: 最低可见度阈值（低于此值的点被过滤掉）

    返回:
        list[Landmark]: 可见度达标的点
    """
    return [point for point in landmarks if point.visibility >= min_visibility]
