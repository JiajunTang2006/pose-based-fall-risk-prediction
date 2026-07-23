"""
特征提取器（FeatureExtractor）的单元测试。

测试内容：
1. 正常站立的躯干角度应该接近 0°
2. 身体向下移动时，垂直速度应该为正值
3. 身体倾斜时，躯干角度应该显著增大

使用方式：
    python -m pytest tests/test_features.py
    或
    python -m unittest tests.test_features
"""

import unittest

from fall_prediction.features import FeatureExtractor
from fall_prediction.landmarks import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    Landmark,
)


def make_landmarks(
    left_shoulder=(0.45, 0.25),
    right_shoulder=(0.55, 0.25),
    left_hip=(0.46, 0.55),
    right_hip=(0.54, 0.55),
    left_knee=(0.46, 0.75),
    right_knee=(0.54, 0.75),
    left_ankle=(0.46, 0.95),
    right_ankle=(0.54, 0.95),
):
    """
    构造一组模拟的人体关键点数据。

    这个函数创建 33 个 Landmark，其中 8 个重要关键点（双肩、双髋、双膝、双踝）
    使用传入的坐标，其余 25 个填充默认值。这样可以方便地模拟不同的姿势。

    坐标说明：
        x: 水平位置（0~1，0.5=画面中间）
        y: 垂直位置（0~1，0=顶部，1=底部）

    默认参数模拟的是"正常站立"姿势：
        - 肩膀在上方（y≈0.25）
        - 髋部在中间（y≈0.55）
        - 膝盖在下方（y≈0.75）
        - 脚踝在底部（y≈0.95）
    """
    # 先创建 33 个默认关键点（所有点都在画面中央，可见度 0.9）
    landmarks = [Landmark(0.5, 0.5, visibility=0.9) for _ in range(LANDMARK_COUNT)]

    # 用传入的坐标覆盖 8 个重要关键点
    points = {
        LEFT_SHOULDER: left_shoulder,
        RIGHT_SHOULDER: right_shoulder,
        LEFT_HIP: left_hip,
        RIGHT_HIP: right_hip,
        LEFT_KNEE: left_knee,
        RIGHT_KNEE: right_knee,
        LEFT_ANKLE: left_ankle,
        RIGHT_ANKLE: right_ankle,
    }
    for index, (x, y) in points.items():
        landmarks[index] = Landmark(x, y, visibility=0.95)  # 可见度设为 0.95（高质量）
    return landmarks


class FeatureExtractorTest(unittest.TestCase):
    """
    测试 FeatureExtractor 的各个功能。

    每个 test_ 开头的方法都是一个独立的测试用例。
    """

    def test_standing_torso_angle_is_close_to_vertical(self):
        """
        测试 1：正常站立时，躯干倾斜角度应该接近 0°（垂直）。

        原理：正常站立的人，肩膀中心和髋部中心几乎在一条垂直线上，
        所以偏离垂直方向的角度应该很小（< 5°）。
        """
        extractor = FeatureExtractor()
        features = extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)

        self.assertTrue(features.has_pose)          # 应该检测到人
        self.assertLess(features.torso_angle_deg, 5.0)  # 躯干角度应小于 5°
        self.assertGreater(features.visibility_mean, 0.9)  # 可见度应大于 0.9

    def test_downward_motion_has_positive_vertical_velocity(self):
        """
        测试 2：身体向下移动时，垂直速度应该为正值。

        原理：在图像坐标中，y 轴向下增长。身体中心从 y1 移动到更大的 y2，
        说明人在画面中向下移动了。vertical_velocity = (y2 - y1) / dt > 0。
        """
        extractor = FeatureExtractor()
        # 第一帧：正常站立
        extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)
        # 第二帧：整个身体向下移动（所有 y 坐标都增大）
        lower_pose = make_landmarks(
            left_shoulder=(0.45, 0.35),   # y 从 0.25 → 0.35
            right_shoulder=(0.55, 0.35),
            left_hip=(0.46, 0.65),        # y 从 0.55 → 0.65
            right_hip=(0.54, 0.65),
            left_knee=(0.46, 0.82),       # y 从 0.75 → 0.82
            right_knee=(0.54, 0.82),
            left_ankle=(0.46, 0.98),      # y 从 0.95 → 0.98
            right_ankle=(0.54, 0.98),
        )
        features = extractor.extract(lower_pose, frame_index=1, timestamp=0.1)

        self.assertGreater(features.vertical_velocity, 0.0)  # 垂直速度应为正

    def test_tilted_pose_has_large_torso_angle(self):
        """
        测试 3：身体倾斜时，躯干角度应该显著增大。

        原理：当身体向一侧倾斜时，肩膀中心会偏离髋部中心的正上方，
        导致躯干偏离垂直方向的角度增大。
        """
        extractor = FeatureExtractor()
        # 构造一个倾斜的姿势：肩膀偏左，髋部偏右
        tilted = make_landmarks(
            left_shoulder=(0.24, 0.36),
            right_shoulder=(0.34, 0.36),
            left_hip=(0.50, 0.56),
            right_hip=(0.60, 0.56),
            left_knee=(0.62, 0.70),
            right_knee=(0.72, 0.70),
            left_ankle=(0.78, 0.84),
            right_ankle=(0.88, 0.84),
        )
        features = extractor.extract(tilted, frame_index=0, timestamp=0.0)

        self.assertGreater(features.torso_angle_deg, 45.0)  # 倾斜角度应大于 45°
        self.assertGreater(features.aspect_ratio, 0.5)       # 宽高比也应增大

    def test_missing_hips_keeps_bbox_but_marks_torso_and_center_invalid(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        landmarks = make_landmarks()
        landmarks[LEFT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        landmarks[RIGHT_HIP] = Landmark(0.0, 0.0, visibility=0.0)

        features = extractor.extract(landmarks, frame_index=0, timestamp=0.0)

        self.assertTrue(features.has_pose)
        self.assertFalse(features.torso_valid)
        self.assertFalse(features.center_valid)
        self.assertTrue(features.bbox_valid)
        self.assertTrue(features.upper_body_valid)
        self.assertEqual(features.torso_angle_deg, 0.0)

    def test_motion_after_partial_gap_uses_full_elapsed_time(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        extractor.extract(make_landmarks(), frame_index=0, timestamp=0.0)
        partial = make_landmarks()
        partial[LEFT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        partial[RIGHT_HIP] = Landmark(0.0, 0.0, visibility=0.0)
        extractor.extract(partial, frame_index=1, timestamp=0.1)
        lower = make_landmarks(
            left_shoulder=(0.45, 0.35),
            right_shoulder=(0.55, 0.35),
            left_hip=(0.46, 0.65),
            right_hip=(0.54, 0.65),
        )

        features = extractor.extract(lower, frame_index=2, timestamp=0.2)

        self.assertAlmostEqual(features.vertical_velocity, 0.5, places=5)

    def test_upper_body_only_pose_does_not_claim_a_full_body_bbox(self):
        extractor = FeatureExtractor(min_visibility=0.2)
        landmarks = make_landmarks()
        for index in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
            landmarks[index] = Landmark(0.0, 0.0, visibility=0.0)

        features = extractor.extract(landmarks, frame_index=0, timestamp=0.0)

        self.assertTrue(features.upper_body_valid)
        self.assertFalse(features.torso_valid)
        self.assertFalse(features.center_valid)
        self.assertFalse(features.bbox_valid)


if __name__ == "__main__":
    unittest.main()
