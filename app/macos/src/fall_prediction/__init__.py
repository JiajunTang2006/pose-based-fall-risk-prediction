"""
基于姿态估计的跌倒检测与跌倒预测原型系统。

这个项目使用摄像头或视频作为输入，通过 MediaPipe 或 YOLO-pose 识别人体姿势关键点，
然后提取物理特征（如躯干倾斜角度、垂直速度等）来判断一个人是否即将跌倒或已经跌倒。

主要模块：
- landmarks:  定义人体关键点（肩膀、髋部、膝盖等）的编号和辅助函数
- pose:       调用 MediaPipe 或 YOLO-pose 进行姿势估计
- features:   从关键点中提取有物理意义的特征（角度、速度、宽高比等）
- risk:       根据特征计算风险分数（0~1），判断状态（正常/预跌倒/跌倒）
- predictor:  综合多帧信息，进行时间平滑，给出最终的预测结果
- ml_predictor: 使用训练好的机器学习分类器进行窗口预测
- video_app:  视频/摄像头入口程序，可视化输出
- plot_features: 将输出 CSV 中的特征曲线绘制成图表
"""

from .config import load_predictor_config
from .features import FeatureExtractor, PoseFeatures
from .predictor import FallPredictor, Prediction, PredictorConfig
from .risk import RiskConfig, RiskScorer
from .sensitivity import (
    DEFAULT_SENSITIVITY,
    SENSITIVITY_LEVELS,
    ml_config_for_sensitivity,
    normalize_sensitivity,
    predictor_config_for_sensitivity,
    sensitivity_profile,
    sensitivity_thresholds,
)

__all__ = [
    "FallPredictor",    # 跌倒预测器（核心类）
    "MachineLearningFallPredictor",  # 机器学习窗口预测器
    "DualModelFallPredictor",  # 树模型正式判断 + 深度融合辅助判断
    "FeatureExtractor", # 特征提取器
    "PoseFeatures",     # 单帧姿势特征数据结构
    "Prediction",       # 单帧预测结果数据结构
    "PredictorConfig",  # 预测器配置参数
    "RiskConfig",       # 风险评分配置参数
    "RiskScorer",       # 风险评分器
    "load_predictor_config",  # 从 JSON 加载运行配置
    "DEFAULT_SENSITIVITY",
    "SENSITIVITY_LEVELS",
    "ml_config_for_sensitivity",
    "normalize_sensitivity",
    "predictor_config_for_sensitivity",
    "sensitivity_profile",
    "sensitivity_thresholds",
]


def __getattr__(name: str):
    if name == "MachineLearningFallPredictor":
        from .ml_predictor import MachineLearningFallPredictor

        return MachineLearningFallPredictor
    if name == "DualModelFallPredictor":
        from .ensemble_predictor import DualModelFallPredictor

        return DualModelFallPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
