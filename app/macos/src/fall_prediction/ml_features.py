"""
机器学习分类器共用的特征工具。

规则版预测器已经能把 MediaPipe 的 33 个人体关键点转换成一组可解释特征，
例如躯干角度、垂直速度、身体宽高比等。

机器学习版本先不直接训练原始视频图像，而是复用这些特征：
    1. 单帧特征更少，训练更快；
    2. 特征含义清楚，方便调试；
    3. 对 UR Fall 这种规模不算特别大的数据集更友好。

本文件主要做两件事：
    - 定义机器学习要使用哪些特征列；
    - 把多帧窗口展开成 scikit-learn 可以接收的一维数字向量。
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .features import PoseFeatures


ML_FEATURE_COLUMNS = (
    # 是否检测到人体。模型可以学习"没有姿态时不要乱判跌倒"。
    "has_pose",
    # 躯干相对竖直方向的角度，倒下时通常会变大。
    "torso_angle",
    # 躯干角速度，突然倾倒时会明显增大。
    "torso_angular_velocity",
    # 身体中心 y 坐标，图像坐标中 y 越大表示越靠下。
    "body_center_y",
    # 身体中心相对上一帧的变化量。
    "body_center_delta",
    # 身体中心竖直速度，快速向下时为正且较大。
    "vertical_velocity",
    # 人体包围盒宽高比，倒地时身体通常更"横"。
    "aspect_ratio",
    # 人体包围盒宽度。
    "body_width",
    # 人体包围盒高度。
    "body_height",
    # MediaPipe 对关键点的平均可见度/置信度。
    "visibility_mean",
    # 身体中心相对初始站立基线下降了多少。
    "center_drop",
)

# 加速度增强特征列：在基础特征上增加二阶导数（加速度），
# 帮助模型捕捉运动状态变化的"拐点"，区分 Pre-fall 过渡阶段。
ACCEL_FEATURE_COLUMNS = ML_FEATURE_COLUMNS + (
    "torso_angular_accel",
    "vertical_accel",
)


def pose_features_to_ml_row(features: PoseFeatures, center_drop: float = 0.0) -> dict[str, float]:
    """
    把一帧 PoseFeatures 转成机器学习使用的数值列。

    PoseFeatures 是代码内部的数据结构；训练和推理时更方便使用普通 dict，
    因为 CSV 读出来也是"列名 -> 值"的形式。
    """
    return {
        "has_pose": 1.0 if features.has_pose else 0.0,
        "torso_angle": features.torso_angle_deg,
        "torso_angular_velocity": features.torso_angular_velocity,
        "body_center_y": features.body_center_y,
        "body_center_delta": features.body_center_delta,
        "vertical_velocity": features.vertical_velocity,
        "aspect_ratio": features.aspect_ratio,
        "body_width": features.body_width,
        "body_height": features.body_height,
        "visibility_mean": features.visibility_mean,
        "center_drop": center_drop,
    }


def row_to_feature_values(
    row: Mapping[str, object],
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[float]:
    """
    把一行 CSV 或一帧特征 dict 转成纯数字列表。

    机器学习模型只能吃数字，所以这里会做安全转换：
    - 正常数字字符串，例如 "0.53" -> 0.53
    - 缺失列 -> 0.0
    - 非法值、NaN、inf -> 0.0

    缺失列填 0.0 的好处是：旧版 CSV 即使少了某些新列，也能先用于粗略实验。
    """
    return [_safe_float(row.get(column, 0.0)) for column in feature_columns]


def flatten_window(
    rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[float]:
    """
    把多帧窗口展开成一个机器学习样本。

    假设 window_size=3，每帧有 11 个特征：
        [
          第1帧 11 个特征,
          第2帧 11 个特征,
          第3帧 11 个特征,
        ]

    展开后就是长度 33 的一维列表：
        [第1帧..., 第2帧..., 第3帧...]
    """
    values: list[float] = []
    for row in rows:
        values.extend(row_to_feature_values(row, feature_columns))
    return values


def make_window_feature_names(
    window_size: int,
    feature_columns: Sequence[str] = ML_FEATURE_COLUMNS,
) -> list[str]:
    """
    生成展开后每个特征位置的名字。

    例如 window_size=3 时，时间位置会命名为：
        t-2_xxx, t-1_xxx, t_xxx

    t 表示当前窗口最后一帧；
    t-1 表示上一帧；
    t-2 表示再上一帧。
    """
    names: list[str] = []
    for index in range(window_size):
        relative = index - window_size + 1
        prefix = "t" if relative == 0 else f"t{relative}"
        names.extend(f"{prefix}_{column}" for column in feature_columns)
    return names


def _safe_float(value: object) -> float:
    """把任意输入尽量安全地转换成有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def compute_window_accel_features(
    window_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, float]]:
    """
    为滑动窗口中的每一帧计算加速度特征（二阶导数）。

    对窗口内相邻帧的 velocity 做差分：
      - torso_angular_accel = Δ(torso_angular_velocity)
      - vertical_accel       = Δ(vertical_velocity)

    窗口第一帧的加速度设为 0（无前一帧可参考）。
    返回增强后的行列表，每行包含原始特征 + 加速度特征。
    """
    enhanced: list[dict[str, float]] = []
    for i, row in enumerate(window_rows):
        entry: dict[str, float] = {}
        # 复制原始特征
        for col in ML_FEATURE_COLUMNS:
            entry[col] = _safe_float(row.get(col, 0.0))
        # 计算加速度
        if i == 0:
            entry["torso_angular_accel"] = 0.0
            entry["vertical_accel"] = 0.0
        else:
            prev = window_rows[i - 1]
            entry["torso_angular_accel"] = (
                _safe_float(row.get("torso_angular_velocity", 0.0))
                - _safe_float(prev.get("torso_angular_velocity", 0.0))
            )
            entry["vertical_accel"] = (
                _safe_float(row.get("vertical_velocity", 0.0))
                - _safe_float(prev.get("vertical_velocity", 0.0))
            )
        enhanced.append(entry)
    return enhanced
