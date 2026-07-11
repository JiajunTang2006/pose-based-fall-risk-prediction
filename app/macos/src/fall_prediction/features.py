"""
从 MediaPipe 关键点中提取有物理意义的特征。

这个模块是整个项目的核心——它把原始的 33 个关键点坐标，转换成可以用来判断跌倒的
物理量。就像一个物理学家在看视频时关注的几个关键指标：

1. 躯干倾斜角度 (torso_angle_deg)
   ——身体有多歪？站着是接近 0°，平躺是接近 90°

2. 躯干角速度 (torso_angular_velocity)
   ——身体歪倒的速度有多快？慢慢弯腰 vs 突然摔倒

3. 身体中心 Y 坐标 (body_center_y)
   ——身体在画面中的高度，向下移动说明人在下降

4. 垂直速度 (vertical_velocity)
   ——身体下降有多快？跌倒时通常很快

5. 身体宽高比 (aspect_ratio)
   ——站立时高>宽（值小），倒下时宽>高（值大）

6. 可见度 (visibility_mean)
   ——关键点的平均置信度，太低说明检测不可靠
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .landmarks import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
    has_landmarks,
    mean_visibility,
    midpoint,
    visible_points,
)


UPPER_BODY_LANDMARKS = (
    NOSE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)
LOWER_BODY_LANDMARKS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


@dataclass(frozen=True)
class PoseFeatures:
    """
    单帧图像中提取的姿势特征（不可变数据类）。

    每个字段的含义：
        frame_index:            帧编号（从 0 开始）
        timestamp:              时间戳（秒）
        has_pose:               这一帧是否检测到了人
        torso_angle_deg:        躯干偏离垂直方向的角度（度），0=笔直站立
        torso_angular_velocity: 躯干角度变化率（度/秒），越大说明倒得越快
        body_center_y:          身体中心在画面中的高度（归一化坐标，越小越靠上）
        body_center_delta:      身体中心高度变化量（正=向下移动）
        vertical_velocity:      身体中心下降速度（归一化单位/秒）
        aspect_ratio:           身体包围盒的宽高比（宽/高）
        body_width:             身体包围盒宽度（归一化）
        body_height:            身体包围盒高度（归一化）
        visibility_mean:        重要关键点的平均可见度（0~1）
    """
    frame_index: int
    timestamp: float
    has_pose: bool                       # 是否检测到人体
    torso_angle_deg: float = 0.0         # 躯干倾斜角度（度）
    torso_angular_velocity: float = 0.0  # 躯干角速度（度/秒）
    body_center_y: float = 0.0           # 身体中心 Y 坐标
    body_center_delta: float = 0.0       # 身体中心变化量
    vertical_velocity: float = 0.0       # 垂直速度
    aspect_ratio: float = 0.0            # 宽高比
    body_width: float = 0.0              # 身体宽度
    body_height: float = 0.0             # 身体高度
    visibility_mean: float = 0.0         # 平均可见度
    torso_signed_angle_deg: float = 0.0  # 带方向的躯干角，供相机站立校准使用
    torso_valid: bool = False             # 双肩和双髋是否都可靠
    center_valid: bool = False            # 身体中心/下降特征是否可靠
    bbox_valid: bool = False              # 可见关键点包围盒是否可靠
    shoulder_center_y: float = 0.0        # 双肩中心的垂直位置
    shoulder_center_delta: float = 0.0    # 双肩中心逐帧位移
    shoulder_vertical_velocity: float = 0.0
    shoulder_line_angle_deg: float = 0.0  # 肩线相对水平线的带方向角度
    shoulder_line_angular_velocity: float = 0.0
    upper_body_width: float = 0.0
    upper_body_height: float = 0.0
    upper_body_aspect_ratio: float = 0.0
    upper_body_valid: bool = False
    upper_body_visibility_mean: float = 0.0


class FeatureExtractor:
    """
    特征提取器：负责从每一帧的关键点中提取 PoseFeatures。

    它会记住上一帧的信息，这样才能计算"变化率"（速度、角速度等）。
    比如要知道身体下降有多快，就需要知道"这一帧的身体高度 - 上一帧的身体高度"。

    使用方式：
        extractor = FeatureExtractor()
        features = extractor.extract(landmarks, frame_index=0, timestamp=0.0)
        # features.torso_angle_deg 就是这一帧的躯干倾斜角度
    """

    def __init__(self, min_visibility: float = 0.2) -> None:
        """
        参数:
            min_visibility: 关键点最低可见度阈值，低于此值的点不参与计算
        """
        self.min_visibility = min_visibility
        # 保存上一帧的信息，用于计算速度
        self._previous_center_y: float | None = None       # 上一帧的身体中心 Y
        self._previous_torso_angle: float | None = None    # 上一帧的躯干角度
        self._previous_timestamp: float | None = None      # 上一帧的时间戳
        self._previous_shoulder_center_y: float | None = None
        self._previous_shoulder_angle: float | None = None
        self._previous_shoulder_timestamp: float | None = None

    def extract(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> PoseFeatures:
        """
        从一帧的关键点数据中提取特征。

        这是最主要的函数，每处理一帧图像就调用一次。

        参数:
            landmarks:   33 个关键点的列表（或 None 表示没检测到人）
            frame_index: 帧编号
            timestamp:   当前时间（秒）

        返回:
            PoseFeatures: 提取出的特征数据
        """
        # 如果没有检测到人，返回一个"空"的特征（has_pose=False）
        if not has_landmarks(landmarks):
            return PoseFeatures(frame_index=frame_index, timestamp=timestamp, has_pose=False)

        assert landmarks is not None  # 到这里 landmarks 一定不为 None

        shoulders_valid = self._points_visible(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER))
        hips_valid = self._points_visible(landmarks, (LEFT_HIP, RIGHT_HIP))
        torso_valid = shoulders_valid and hips_valid
        center_valid = torso_valid

        body_center_y = 0.0
        torso_angle = 0.0
        torso_signed_angle = 0.0
        shoulder_center_y = 0.0
        shoulder_center_delta = 0.0
        shoulder_vertical_velocity = 0.0
        shoulder_line_angle = 0.0
        shoulder_line_angular_velocity = 0.0
        if torso_valid:
            shoulder_mid = midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
            hip_mid = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
            body_center_y = (shoulder_mid.y + hip_mid.y) / 2.0
            torso_signed_angle = self._signed_torso_angle_from_vertical(shoulder_mid, hip_mid)
            torso_angle = abs(torso_signed_angle)

        if shoulders_valid:
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            shoulder_mid = midpoint(left_shoulder, right_shoulder)
            shoulder_center_y = shoulder_mid.y
            shoulder_line_angle = math.degrees(
                math.atan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x,
                )
            )
            if self._previous_shoulder_center_y is not None and self._previous_shoulder_timestamp is not None:
                shoulder_dt = max(timestamp - self._previous_shoulder_timestamp, 1e-6)
                shoulder_center_delta = shoulder_center_y - self._previous_shoulder_center_y
                shoulder_vertical_velocity = shoulder_center_delta / shoulder_dt
                if self._previous_shoulder_angle is not None:
                    shoulder_line_angular_velocity = (
                        shoulder_line_angle - self._previous_shoulder_angle
                    ) / shoulder_dt
            self._previous_shoulder_center_y = shoulder_center_y
            self._previous_shoulder_angle = shoulder_line_angle
            self._previous_shoulder_timestamp = timestamp

        # ---- 计算身体包围盒（最小外接矩形）----
        # 宽高比 = 宽度/高度，站立时窄高（值小），倒下时矮宽（值大）
        body_width, body_height, aspect_ratio = self._body_box(landmarks)
        visible_lower_body_points = sum(
            landmarks[index].visibility >= self.min_visibility
            for index in LOWER_BODY_LANDMARKS
        )
        bbox_valid = (
            body_width > 1e-6
            and body_height > 1e-6
            and visible_lower_body_points >= 2
        )
        upper_body_width, upper_body_height, upper_body_aspect_ratio = self._body_box(
            landmarks,
            indices=UPPER_BODY_LANDMARKS,
        )
        upper_body_valid = (
            shoulders_valid
            and upper_body_width > 1e-6
            and upper_body_height > 1e-6
        )

        # ---- 计算关键点平均可见度 ----
        visibility = mean_visibility(landmarks)
        upper_body_visibility = mean_visibility(landmarks, indices=UPPER_BODY_LANDMARKS)

        # ---- 计算速度（需要和上一帧对比）----
        dt = self._delta_time(timestamp)  # 两帧之间的时间间隔
        center_delta = 0.0       # 身体中心变化量
        vertical_velocity = 0.0   # 垂直速度
        angular_velocity = 0.0    # 角速度

        if center_valid and self._previous_center_y is not None:
            # 身体中心的变化量（正数 = 身体在画面中向下移动）
            center_delta = body_center_y - self._previous_center_y
            # 垂直速度 = 变化量 / 时间间隔
            vertical_velocity = center_delta / dt

        if torso_valid and self._previous_torso_angle is not None:
            # 角速度 = 角度变化量 / 时间间隔（度/秒）
            angular_velocity = (torso_angle - self._previous_torso_angle) / dt

        # ---- 保存当前帧信息，供下一帧计算速度时使用 ----
        if center_valid:
            self._previous_center_y = body_center_y
        if torso_valid:
            self._previous_torso_angle = torso_angle
        # Do not advance the motion clock on a bbox-only/fully missing frame.
        # When torso points return, velocity is then divided by the whole gap
        # duration instead of being exaggerated as a one-frame jump.
        if center_valid or torso_valid:
            self._previous_timestamp = timestamp

        return PoseFeatures(
            frame_index=frame_index,
            timestamp=timestamp,
            has_pose=True,
            torso_angle_deg=torso_angle,
            torso_angular_velocity=angular_velocity,
            body_center_y=body_center_y,
            body_center_delta=center_delta,
            vertical_velocity=vertical_velocity,
            aspect_ratio=aspect_ratio,
            body_width=body_width,
            body_height=body_height,
            visibility_mean=visibility,
            torso_signed_angle_deg=torso_signed_angle,
            torso_valid=torso_valid,
            center_valid=center_valid,
            bbox_valid=bbox_valid,
            shoulder_center_y=shoulder_center_y,
            shoulder_center_delta=shoulder_center_delta,
            shoulder_vertical_velocity=shoulder_vertical_velocity,
            shoulder_line_angle_deg=shoulder_line_angle,
            shoulder_line_angular_velocity=shoulder_line_angular_velocity,
            upper_body_width=upper_body_width,
            upper_body_height=upper_body_height,
            upper_body_aspect_ratio=upper_body_aspect_ratio,
            upper_body_valid=upper_body_valid,
            upper_body_visibility_mean=upper_body_visibility,
        )

    def reset(self) -> None:
        """
        重置状态（清空上一帧的记录）。

        切换到新的视频源时应该调用，避免把上一个视频的最后一帧
        和下一个视频的第一帧之间错误地计算速度。
        """
        self._previous_center_y = None
        self._previous_torso_angle = None
        self._previous_timestamp = None
        self._previous_shoulder_center_y = None
        self._previous_shoulder_angle = None
        self._previous_shoulder_timestamp = None

    def _delta_time(self, timestamp: float) -> float:
        """
        计算两帧之间的时间间隔。

        如果是第一帧（没有上一帧），默认假设帧率为 30fps，即间隔约 0.033 秒。
        实际使用中取 max(dt, 1e-6) 是为了防止除零错误。
        """
        if self._previous_timestamp is None:
            return 1.0 / 30.0  # 默认假设 30fps
        return max(timestamp - self._previous_timestamp, 1e-6)  # 1e-6 是防止除零的安全值

    def _body_box(
        self,
        landmarks: Sequence[Landmark],
        indices: Sequence[int] | None = None,
    ) -> tuple[float, float, float]:
        """
        计算人体可见关键点的包围盒（bounding box）。

        返回:
            (width, height, aspect_ratio): 宽度、高度、宽高比
            宽高比 = 宽度 / 高度
            站立时宽高比 < 1（窄高），跌倒时宽高比接近或 > 1（矮宽）
        """
        # 只使用可见度足够的点
        points = (
            visible_points(landmarks, self.min_visibility)
            if indices is None
            else [
                landmarks[index]
                for index in indices
                if landmarks[index].visibility >= self.min_visibility
            ]
        )
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        # 找到所有可见点的最小/最大坐标
        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        min_y = min(point.y for point in points)
        max_y = max(point.y for point in points)

        width = max_x - min_x
        height = max_y - min_y
        # 宽高比，分母加 1e-6 防止除零
        aspect_ratio = width / max(height, 1e-6)
        return width, height, aspect_ratio

    def _points_visible(self, landmarks: Sequence[Landmark], indices: Sequence[int]) -> bool:
        return all(landmarks[index].visibility >= self.min_visibility for index in indices)

    @staticmethod
    def _torso_angle_from_vertical(shoulder_mid: Landmark, hip_mid: Landmark) -> float:
        """
        计算躯干偏离垂直方向的角度。

        原理：
        1. 计算肩膀中心到髋部中心的水平距离 (dx) 和垂直距离 (dy)
        2. 用 atan2 算出角度（弧度），再转换为度数
        3. 笔直站立时 dx≈0，角度≈0°
           身体倾斜时 dx 增大，角度也增大
           完全水平时 dx>0 且 dy≈0，角度≈90°

        注意：这里取 dx 的绝对值，因为我们只关心倾斜的"程度"，
        不关心是向左还是向右倒。
        """
        dx = shoulder_mid.x - hip_mid.x  # 水平偏移
        dy = shoulder_mid.y - hip_mid.y  # 垂直偏移
        # atan2: 根据 dx 和 dy 计算角度，degrees: 弧度转度数
        return math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))

    @staticmethod
    def _signed_torso_angle_from_vertical(shoulder_mid: Landmark, hip_mid: Landmark) -> float:
        """Signed tilt around the image vertical; standing is approximately 0°."""
        dx = shoulder_mid.x - hip_mid.x
        upward_dy = hip_mid.y - shoulder_mid.y
        return math.degrees(math.atan2(dx, max(upward_dy, 1e-6)))
