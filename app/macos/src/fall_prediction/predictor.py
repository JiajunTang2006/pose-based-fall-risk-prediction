"""
时间序列跌倒预测逻辑。

这个模块是整个系统的"大脑"，负责把逐帧的特征分析整合成可靠的预测结果。

核心思想：只看一帧的结果不可靠（可能误判），需要结合时间信息：
1. 先收集一些帧来建立"基线"（baseline）——人在正常站立时身体中心在哪里
2. 对风险分数做平滑处理（取最近几帧的平均值），减少单帧噪声
3. 要求连续多帧都判定为"跌倒"才算跌倒，避免单帧误判

处理流程：
  视频帧 → 特征提取 → 风险评分 → 时间平滑 → 状态判定
  (pose.py) (features.py) (risk.py)   (本模块)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .features import FeatureExtractor, PoseFeatures
from .landmarks import Landmark
from .risk import RiskBreakdown, RiskConfig, RiskScorer


@dataclass(frozen=True)
class PredictorConfig:
    """
    预测器的配置参数。

    参数说明:
        baseline_frames:           收集多少帧来建立"站立基线"（默认 15 帧）
                                  前 15 帧假设人是正常站立的，记录身体中心位置
        smoothing_window:          平滑窗口大小（默认 5 帧）
                                  取最近 5 帧风险分数的平均值，减少抖动
        prefall_consecutive_frames: 连续多少帧预跌倒才正式判定为 Pre-fall（默认 3）
        fall_consecutive_frames:    连续多少帧跌倒才正式判定为 Fall（默认 3）
        risk:                       风险评分配置（来自 RiskConfig）
    """
    baseline_frames: int = 15
    smoothing_window: int = 5
    prefall_consecutive_frames: int = 3
    fall_consecutive_frames: int = 3
    risk: RiskConfig = RiskConfig()


@dataclass(frozen=True)
class Prediction:
    """
    单帧的完整预测结果。

    字段:
        frame_index:        帧编号
        timestamp:          时间戳（秒）
        state:              经过时间平滑后的状态（"Normal"/"Pre-fall"/"Fall"/"Unknown"）
        instant_state:      只看当前帧的瞬时状态（没有平滑，反应更快但更不稳定）
        risk_score:         当前帧的瞬时风险分数（0~1）
        smoothed_risk_score: 平滑后的风险分数（最近几帧的平均值）
        features:           当前帧的特征数据
        breakdown:          风险分数的详细分解
        baseline_center_y:  站立时的身体中心 Y 坐标（用于计算身体下降量）
        alert_state:        推理阶段报警状态；可以比分类状态更敏感
    """
    frame_index: int
    timestamp: float
    state: str
    instant_state: str
    risk_score: float
    smoothed_risk_score: float
    features: PoseFeatures
    breakdown: RiskBreakdown
    baseline_center_y: float | None
    alert_state: str | None = None
    system_status: str | None = None
    advisory_state: str | None = None
    decision_tier: str | None = None


class FallPredictor:
    """
    跌倒预测器：整合特征提取、风险评分和时间平滑。

    这是最终对外暴露的核心类，用户只需要：
    1. 创建 FallPredictor()
    2. 每来一帧图像，调用 predictor.predict(landmarks, frame_index, timestamp)
    3. 从返回的 Prediction 中获取当前状态和风险分数

    工作原理：
        前 N 帧（baseline_frames）→ 收集身体中心位置，建立"正常站立"的基线
        之后每帧 → 计算特征 → 评分 → 平滑 → 判定状态

    使用示例：
        predictor = FallPredictor()
        for frame in video:
            landmarks = pose_estimator.process_bgr(frame)
            result = predictor.predict(landmarks, frame_idx, timestamp)
            if result.state == "Fall":
                print("检测到跌倒！")
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        self.config = config or PredictorConfig()
        # 特征提取器（记住上一帧信息，计算速度）
        self.extractor = FeatureExtractor(min_visibility=self.config.risk.min_visibility)
        # 风险评分器
        self.scorer = RiskScorer(self.config.risk)
        # 基线收集：前几帧的身体中心 Y 坐标存到这里，用于计算"正常站立"的平均位置
        self._baseline_samples: list[float] = []
        # 建立好的基线值
        self._baseline_center_y: float | None = None
        # 风险分数历史（用于平滑），是一个固定长度的双端队列
        self._risk_history: deque[float] = deque(maxlen=self.config.smoothing_window)
        # 连续计数器：记录"预跌倒"或"跌倒"状态连续出现了多少帧
        self._prefall_count = 0
        self._fall_count = 0

    @property
    def baseline_center_y(self) -> float | None:
        """返回已建立的站立基线（身体中心 Y 坐标）。"""
        return self._baseline_center_y

    def predict(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> Prediction:
        """
        处理一帧数据并返回预测结果。

        这是最主要的方法。每处理一帧视频就调用一次。

        参数:
            landmarks:   当前帧的关键点（33 个，或 None）
            frame_index: 帧编号
            timestamp:   当前时间（秒）

        返回:
            Prediction: 完整的预测结果
        """
        # ---- 第一步：提取特征 ----
        features = self.extractor.extract(landmarks, frame_index, timestamp)

        # ---- 第二步：建立基线（仅在前 N 帧进行）----
        # 在收集够 baseline_frames 帧之前，不计算 center_drop
        if features.center_valid and self._baseline_center_y is None:
            self._baseline_samples.append(features.body_center_y)
            if len(self._baseline_samples) >= self.config.baseline_frames:
                # 收集够了！计算这些帧的身体中心平均值作为"正常站立"的基线
                self._baseline_center_y = mean(self._baseline_samples)

        # 如果在建立基线之前就需要基线值，用已收集的样本临时计算
        fallback_baseline = self._baseline_center_y
        if fallback_baseline is None and self._baseline_samples:
            fallback_baseline = mean(self._baseline_samples)

        # ---- 第三步：风险评分 ----
        breakdown = self.scorer.score(features, fallback_baseline)
        # 当前帧的瞬时状态（不平滑）
        instant_state = self.scorer.state_from_score(breakdown.risk_score)

        # ---- 第四步：时间平滑 ----
        # 把当前帧的风险分数加入历史队列，取最近 N 帧的平均值
        self._risk_history.append(breakdown.risk_score)
        smoothed_risk = mean(self._risk_history)  # 平滑后的风险分数

        # ---- 第五步：时序判定 ----
        # 基于平滑分数 + 连续帧计数器，判定最终状态
        state = self._temporal_state(smoothed_risk, features)

        return Prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,                          # 平滑后的最终状态
            instant_state=instant_state,          # 瞬时状态（未平滑）
            risk_score=breakdown.risk_score,      # 瞬时风险分数
            smoothed_risk_score=smoothed_risk,    # 平滑后的风险分数
            features=features,
            breakdown=breakdown,
            baseline_center_y=fallback_baseline,
        )

    def reset(self) -> None:
        """
        重置预测器状态，切换到新视频时调用。

        会清空基线、平滑历史、连续计数器等所有状态。
        """
        self.extractor.reset()
        self._baseline_samples.clear()
        self._baseline_center_y = None
        self._risk_history.clear()
        self._prefall_count = 0
        self._fall_count = 0

    def _temporal_state(self, smoothed_risk: float, features: PoseFeatures) -> str:
        """
        基于平滑分数和连续帧计数判断最终状态。

        判定逻辑：
        1. 如果没检测到人，或可见度太低 → 返回 "Unknown"（未知）
        2. 如果平滑分数 ≥ fall_threshold → Fall 计数器 +1
        3. 如果平滑分数 ≥ prefall_threshold → Pre-fall 计数器 +1
        4. 如果分数不够，清空计数器
        5. 连续 Fall 帧数 ≥ fall_consecutive_frames → 正式判定为 "Fall"
        6. 连续 Pre-fall 帧数 ≥ prefall_consecutive_frames → 正式判定为 "Pre-fall"

        这种"连续帧确认"机制避免了单帧误判：
        ——不会因为一帧的检测误差就报"跌倒"，必须连续多帧都是跌倒才算。

        阈值参考（默认值）：
        - 风险分数 ≥ 0.72，连续 3 帧 → Fall（跌倒）
        - 风险分数 ≥ 0.45，连续 3 帧 → Pre-fall（预跌倒）
        """
        cfg = self.config.risk

        # 没有检测到人，或可见度太低 → 状态未知，计数器归零
        if not features.has_pose or features.visibility_mean < cfg.min_visibility:
            self._prefall_count = 0
            self._fall_count = 0
            return "Unknown"

        # 检查是否达到跌倒阈值
        if smoothed_risk >= cfg.fall_threshold:
            self._fall_count += 1    # 连续跌倒计数 +1
        else:
            self._fall_count = 0     # 中断，重置计数

        # 检查是否达到预跌倒阈值
        if smoothed_risk >= cfg.prefall_threshold:
            self._prefall_count += 1  # 连续预跌倒计数 +1
        else:
            self._prefall_count = 0   # 中断，重置计数

        # 优先判断跌倒（跌倒比预跌倒更严重）
        if self._fall_count >= self.config.fall_consecutive_frames:
            return "Fall"
        if self._prefall_count >= self.config.prefall_consecutive_frames:
            return "Pre-fall"
        return "Normal"
