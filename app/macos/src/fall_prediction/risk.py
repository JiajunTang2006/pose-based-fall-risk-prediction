"""
风险评分规则：根据特征数据计算跌倒风险分数。

这个模块的核心思路是：
1. 对每个特征（躯干角度、角速度、垂直速度等），根据它的值计算一个 0~1 的子分数
2. 用加权求和的方式把所有子分数合并成一个"原始风险分数"
3. 用可见度对原始分数进行修正（看不清时降低分数，避免误报）
4. 根据最终分数判断状态：Normal（正常）、Pre-fall（预跌倒）、Fall（跌倒）

核心函数 ramp 的作用：
    把特征值线性映射到 0~1 区间。
    比如躯干角度：低于 25° → 0 分（正常），高于 75° → 1 分（危险），
    在 25°~75° 之间按比例计算。

    分数
    1.0 ┤         ╱
        │       ╱
    0.5 ┤     ╱
        │   ╱
    0.0 ┤─╱─────┼────┼──→ 特征值
          low    25°  high
                        75°
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import PoseFeatures


@dataclass(frozen=True)
class RiskConfig:
    """
    风险评分配置参数（所有阈值和权重都可以调整）。

    这个配置分为三组：

    【状态判定阈值】（risk_score 达到多少算跌倒/预跌倒）
        prefall_threshold:  ≥ 0.45 判定为"预跌倒"（即将跌倒，需要预警）
        fall_threshold:     ≥ 0.72 判定为"跌倒"（已经跌倒）

    【各特征的 ramp 阈值】（特征值超过 high 时该特征贡献满分 1.0）
        torso_warn_deg = 25.0        躯干倾斜 25° 开始预警
        torso_fall_deg = 75.0        躯干倾斜 75° 达到最大危险
        angular_velocity_warn = 25.0  角速度 25°/s 开始预警
        angular_velocity_fall = 120.0 角速度 120°/s 达到最大危险
        vertical_velocity_warn = 0.22 垂直速度 0.22/s 开始预警
        vertical_velocity_fall = 0.85 垂直速度 0.85/s 达到最大危险
        center_drop_warn = 0.06      身体下降 0.06 开始预警
        center_drop_fall = 0.22      身体下降 0.22 达到最大危险
        aspect_ratio_warn = 0.55     宽高比 0.55 开始预警
        aspect_ratio_fall = 1.15     宽高比 1.15 达到最大危险

    【各特征的权重】（决定每个特征对最终分数的重要程度，总和=1.0）
        torso_weight = 0.22             躯干倾斜占 22%
        angular_velocity_weight = 0.12  角速度占 12%
        vertical_velocity_weight = 0.34 垂直速度占 34%（最重要！）
        center_drop_weight = 0.16       身体下降占 16%
        aspect_ratio_weight = 0.16      宽高比占 16%

    为什么垂直速度权重最高？因为跌倒最明显的特征就是身体快速向下移动。
    """
    # 状态判定阈值
    prefall_threshold: float = 0.45
    fall_threshold: float = 0.72

    # 可见度最低要求
    min_visibility: float = 0.35

    # 躯干倾斜角度阈值
    torso_warn_deg: float = 25.0
    torso_fall_deg: float = 75.0
    # 躯干角速度阈值
    angular_velocity_warn: float = 25.0
    angular_velocity_fall: float = 120.0
    # 垂直速度阈值
    vertical_velocity_warn: float = 0.22
    vertical_velocity_fall: float = 0.85
    # 身体中心下降阈值
    center_drop_warn: float = 0.06
    center_drop_fall: float = 0.22
    # 宽高比阈值
    aspect_ratio_warn: float = 0.55
    aspect_ratio_fall: float = 1.15

    # 各特征权重
    torso_weight: float = 0.22
    angular_velocity_weight: float = 0.12
    vertical_velocity_weight: float = 0.34
    center_drop_weight: float = 0.16
    aspect_ratio_weight: float = 0.16


@dataclass(frozen=True)
class RiskBreakdown:
    """
    风险分数的详细分解（让用户知道每个特征分别贡献了多少分）。

    字段:
        risk_score:           最终风险分数（0~1），考虑了可见度修正
        torso_score:          躯干倾斜子分数（0~1）
        angular_velocity_score: 角速度子分数（0~1）
        vertical_velocity_score: 垂直速度子分数（0~1）
        center_drop_score:    身体下降子分数（0~1）
        aspect_ratio_score:   宽高比子分数（0~1）
        visibility_factor:    可见度修正因子（0~1），低可见度会降低总分
        center_drop:          身体中心相对基线的下降量
    """
    risk_score: float
    torso_score: float
    angular_velocity_score: float
    vertical_velocity_score: float
    center_drop_score: float
    aspect_ratio_score: float
    visibility_factor: float
    center_drop: float


class RiskScorer:
    """
    风险评分器：根据 PoseFeatures 计算跌倒风险分数。

    使用方式：
        scorer = RiskScorer()
        breakdown = scorer.score(features, baseline_center_y=0.5)
        print(f"风险分数: {breakdown.risk_score:.2f}")
        print(f"当前状态: {scorer.state_from_score(breakdown.risk_score)}")
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """如果不传配置，使用默认的 RiskConfig。"""
        self.config = config or RiskConfig()

    def score(
        self,
        features: PoseFeatures,
        baseline_center_y: float | None,
    ) -> RiskBreakdown:
        """
        计算一帧特征的风险分数。

        参数:
            features:           当前帧的特征数据
            baseline_center_y:  站立时的身体中心 Y 坐标（基线）
                                如果不提供，center_drop 为 0

        返回:
            RiskBreakdown: 包含各项子分数和总分的详细结果
        """
        # 如果这一帧没检测到人，所有分数都是 0
        if not features.has_pose:
            return RiskBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        cfg = self.config

        # ---- 计算身体下降量 ----
        # center_drop = 当前身体中心 Y - 站立时的基线 Y
        # 正常站立时 center_drop ≈ 0；跌倒时身体向下移动，center_drop > 0
        center_drop = 0.0
        if baseline_center_y is not None:
            center_drop = max(0.0, features.body_center_y - baseline_center_y)

        # ---- 计算每个特征的子分数 ----
        # 每个子分数都是通过 ramp 函数计算的，范围 [0, 1]
        # 例：躯干倾斜 40° → ramp(40, 25, 75) = (40-25)/(75-25) = 15/50 = 0.3
        torso_score = ramp(features.torso_angle_deg, cfg.torso_warn_deg, cfg.torso_fall_deg)
        angular_velocity_score = ramp(
            max(0.0, features.torso_angular_velocity),  # 只关心正向角速度（倾斜加剧）
            cfg.angular_velocity_warn,
            cfg.angular_velocity_fall,
        )
        vertical_velocity_score = ramp(
            max(0.0, features.vertical_velocity),  # 只关心向下速度
            cfg.vertical_velocity_warn,
            cfg.vertical_velocity_fall,
        )
        center_drop_score = ramp(center_drop, cfg.center_drop_warn, cfg.center_drop_fall)
        aspect_ratio_score = ramp(
            features.aspect_ratio,
            cfg.aspect_ratio_warn,
            cfg.aspect_ratio_fall,
        )

        # ---- 加权求和，计算原始风险分数 ----
        raw_score = (
            cfg.torso_weight * torso_score                           # 躯干倾斜 × 22%
            + cfg.angular_velocity_weight * angular_velocity_score   # 角速度 × 12%
            + cfg.vertical_velocity_weight * vertical_velocity_score # 垂直速度 × 34%
            + cfg.center_drop_weight * center_drop_score             # 身体下降 × 16%
            + cfg.aspect_ratio_weight * aspect_ratio_score           # 宽高比 × 16%
        )

        # ---- 可见度修正 ----
        # 如果关键点可见度很低（如被遮挡），说明检测不可靠，应该降低风险分数
        # visibility_factor 在 [0, 1] 之间
        # 最终分数 = 原始分数 × (0.35 + 0.65 × 可见度因子)
        #   可见度=1.0 → 修正系数=1.0（不降低）
        #   可见度=0.35 → 修正系数=0.35（大幅降低）
        #   可见度=0.0  → 修正系数=0.35（最低只降到 35%，保留一定分数）
        visibility_factor = ramp(features.visibility_mean, cfg.min_visibility, 0.75)
        risk_score = clamp(raw_score * (0.35 + 0.65 * visibility_factor), 0.0, 1.0)

        return RiskBreakdown(
            risk_score=risk_score,
            torso_score=torso_score,
            angular_velocity_score=angular_velocity_score,
            vertical_velocity_score=vertical_velocity_score,
            center_drop_score=center_drop_score,
            aspect_ratio_score=aspect_ratio_score,
            visibility_factor=visibility_factor,
            center_drop=center_drop,
        )

    def state_from_score(self, risk_score: float) -> str:
        """
        根据风险分数判断当前状态。

        返回:
            "Normal"   → 正常（risk < 0.45）
            "Pre-fall" → 预跌倒/即将跌倒（0.45 ≤ risk < 0.72）
            "Fall"     → 已跌倒（risk ≥ 0.72）
        """
        if risk_score >= self.config.fall_threshold:
            return "Fall"
        if risk_score >= self.config.prefall_threshold:
            return "Pre-fall"
        return "Normal"


def ramp(value: float, low: float, high: float) -> float:
    """
    线性映射函数：把特征值映射到 [0, 1] 区间。

    公式: (value - low) / (high - low)，然后限制在 [0, 1]

    参数:
        value: 当前特征值
        low:   预警阈值（低于此值 → 0 分）
        high:  危险阈值（高于此值 → 1 分）

    返回:
        float: 0~1 之间的分数

    示例:
        ramp(50, 25, 75)  → (50-25)/(75-25) = 0.5
        ramp(10, 25, 75)  → clamp(-0.3, 0, 1) = 0.0
        ramp(90, 25, 75)  → clamp(1.3, 0, 1) = 1.0
    """
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def clamp(value: float, low: float, high: float) -> float:
    """
    限幅函数：把 value 限制在 [low, high] 范围内。

    示例:
        clamp(1.5, 0, 1) → 1.0
        clamp(-0.3, 0, 1) → 0.0
        clamp(0.6, 0, 1) → 0.6
    """
    return max(low, min(high, value))
