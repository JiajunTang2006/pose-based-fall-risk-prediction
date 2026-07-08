"""
跌倒预测器（FallPredictor）的单元测试。

测试内容：
用模拟的人体关键点构造一个"从站立到跌倒"的过程，
验证 FallPredictor 能否正确检测到 Pre-fall 或 Fall 状态。

使用方式：
    python -m pytest tests/test_predictor.py
    或
    python -m unittest tests.test_predictor
"""

import unittest

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
from fall_prediction.predictor import FallPredictor, PredictorConfig


def pose_from_points(shoulder_mid, hip_mid, knee_y, ankle_y, spread=0.05):
    """
    根据关键坐标构造一组模拟的人体关键点。

    这个函数比 test_features.py 中的 make_landmarks 更简洁，
    只需要提供几个关键参数就能生成完整的 33 个关键点。

    参数:
        shoulder_mid: 肩膀中心的 (x, y) 坐标
        hip_mid:      髋部中心的 (x, y) 坐标
        knee_y:       膝盖的 y 坐标（x 自动推导）
        ankle_y:      脚踝的 y 坐标（x 自动推导）
        spread:       左右关键点的水平间距（默认 0.05，越大身体越宽）

    返回:
        list[Landmark]: 33 个关键点的列表
    """
    sx, sy = shoulder_mid  # 肩膀中心
    hx, hy = hip_mid       # 髋部中心

    # 先创建 33 个默认关键点
    landmarks = [Landmark(0.5, 0.5, visibility=0.9) for _ in range(LANDMARK_COUNT)]

    # 设置左右对称的关键点
    # 左肩 = (肩膀中心 x - spread, 肩膀中心 y)
    landmarks[LEFT_SHOULDER] = Landmark(sx - spread, sy, visibility=0.95)
    landmarks[RIGHT_SHOULDER] = Landmark(sx + spread, sy, visibility=0.95)
    # 左髋 = (髋部中心 x - spread, 髋部中心 y)
    landmarks[LEFT_HIP] = Landmark(hx - spread, hy, visibility=0.95)
    landmarks[RIGHT_HIP] = Landmark(hx + spread, hy, visibility=0.95)
    # 膝盖和脚踝使用髋部中心的 x 坐标，y 坐标独立指定
    landmarks[LEFT_KNEE] = Landmark(hx - spread, knee_y, visibility=0.95)
    landmarks[RIGHT_KNEE] = Landmark(hx + spread, knee_y, visibility=0.95)
    landmarks[LEFT_ANKLE] = Landmark(hx - spread, ankle_y, visibility=0.95)
    landmarks[RIGHT_ANKLE] = Landmark(hx + spread, ankle_y, visibility=0.95)
    return landmarks


class FallPredictorTest(unittest.TestCase):
    """
    测试 FallPredictor 的核心功能。
    """

    def test_synthetic_fall_sequence_reaches_prefall_or_fall(self):
        """
        测试：模拟从站立到跌倒的过程，验证预测器能否正确响应。

        测试流程：
        1. 前 4 帧：正常站立（让预测器建立基线）
        2. 后 4 帧：逐渐倾斜并向下移动（模拟跌倒过程）
        3. 检查：后面的帧是否出现 Pre-fall 或 Fall 状态

        这里使用了较短的配置参数（baseline_frames=3, consecutive_frames=2），
        这样用少量帧就能完成测试，不需要跑几百帧。
        """
        # 创建预测器，使用"加速"配置（适合测试用小数据集）
        predictor = FallPredictor(
            PredictorConfig(
                baseline_frames=3,               # 只需 3 帧建立基线
                smoothing_window=3,              # 平滑窗口 3 帧
                prefall_consecutive_frames=2,    # 连续 2 帧预跌倒即判定
                fall_consecutive_frames=2,       # 连续 2 帧跌倒即判定
            )
        )

        predictions = []

        # ---- 阶段 1：正常站立（4 帧）----
        # 肩膀在 y=0.25、髋部在 y=0.55、膝盖在 y=0.75、脚踝在 y=0.95
        # 这是一个标准的人体站姿比例
        for frame in range(4):
            landmarks = pose_from_points((0.50, 0.25), (0.50, 0.55), 0.75, 0.95)
            predictions.append(predictor.predict(landmarks, frame, frame / 10.0))

        # ---- 阶段 2：模拟跌倒（4 帧，每帧更倾斜、更低）----
        # 帧 4：稍微倾斜
        # 帧 5：倾斜加重，身体下降
        # 帧 6：更倾斜，继续下降
        # 帧 7：接近平躺的姿势
        falling_poses = [
            pose_from_points((0.42, 0.38), (0.56, 0.58), 0.72, 0.86, spread=0.08),
            pose_from_points((0.34, 0.50), (0.62, 0.66), 0.72, 0.82, spread=0.10),
            pose_from_points((0.28, 0.64), (0.70, 0.74), 0.78, 0.84, spread=0.12),
            pose_from_points((0.22, 0.76), (0.76, 0.80), 0.82, 0.86, spread=0.14),
        ]
        for offset, landmarks in enumerate(falling_poses, start=4):
            predictions.append(predictor.predict(landmarks, offset, offset / 10.0))

        # ---- 验证结果 ----
        states = [prediction.state for prediction in predictions]
        # 至少有一帧应该被判定为 Pre-fall 或 Fall
        self.assertTrue(any(state in {"Pre-fall", "Fall"} for state in states))
        # 最后一帧的风险分数应该比较高（> 0.45）
        self.assertGreater(predictions[-1].smoothed_risk_score, 0.45)


if __name__ == "__main__":
    unittest.main()
