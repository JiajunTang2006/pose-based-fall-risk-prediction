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
    "FallPredictor",
    "MachineLearningFallPredictor",
    "DualModelFallPredictor",
    "FeatureExtractor",
    "PoseFeatures",
    "Prediction",
    "PredictorConfig",
    "RiskConfig",
    "RiskScorer",
    "load_predictor_config",
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
