"""
使用训练好的机器学习模型进行逐帧预测。

这个文件负责"模型上线推理"，和 train_model.py 正好对应：

    train_model.py:
        读取很多 CSV -> 切窗口 -> 训练模型 -> 保存 joblib 文件

    ml_predictor.py:
        读取一帧视频 -> 提取特征 -> 放进最近 N 帧窗口
        -> 窗口满了之后调用模型预测 -> 返回 Prediction

为了让 video_app.py 不需要关心背后是规则系统还是 ML 系统，
MachineLearningFallPredictor 的 predict() 方法返回的也是 Prediction 对象。

从 Stage 2 开始，增加了可选的 HMM Viterbi 时序平滑层。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .features import FeatureExtractor, PoseFeatures
from .landmarks import Landmark
from .ml_features import (
    ML_FEATURE_COLUMNS,
    ACCEL_FEATURE_COLUMNS,
    flatten_window,
    pose_features_to_ml_row,
    compute_window_accel_features,
)
from .predictor import Prediction, PredictorConfig
from .risk import RiskBreakdown
from .window_dataset import DEFAULT_WINDOW_SIZE


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()
DEFAULT_PREFALL_ALERT_THRESHOLD = 0.25
DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES = 1
NORMAL_STATES = {"Normal"}

# HMM 默认参数
DEFAULT_HMM_BUFFER_SIZE = 25  # 用于 Viterbi 解码的概率历史长度

# 时序 Fall 确认层默认阈值。它只用于运行时减少“稳定躺着被判 Fall”的误报。
DEFAULT_FALL_VALIDATION_HISTORY = 30
DEFAULT_FALL_PREFALL_MEMORY = 12
DEFAULT_FALL_HOLD_FRAMES = 45
DEFAULT_FALL_AFTER_PREFALL_CONFIRM_FRAMES = 2
DEFAULT_FALL_VERTICAL_VELOCITY = 0.60
DEFAULT_FALL_VERTICAL_ACCEL = 0.20
DEFAULT_FALL_ANGULAR_VELOCITY = 200.0
DEFAULT_FALL_ANGULAR_ACCEL = 400.0
DEFAULT_FALL_CENTER_DROP_DELTA = 0.10


# ═══════════════════════════════════════════════════════════════════════
# HMM Viterbi 时序平滑层
# ═══════════════════════════════════════════════════════════════════════

def build_hmm_transition_matrix(
    stay_normal: float = 0.92,
    stay_prefall: float = 0.80,
    stay_fall: float = 0.88,
    normal_to_prefall: float = 0.07,
    prefall_to_fall: float = 0.12,
    prefall_to_normal: float = 0.08,
    fall_to_prefall: float = 0.10,
) -> np.ndarray:
    """
    构建 3×3 转移矩阵 T[i][j] = P(state_j at t+1 | state_i at t).

    状态顺序: 0=Normal, 1=Pre-fall, 2=Fall

    物理约束:
      - Normal → Fall   概率极低 (≤0.01), 不能跳过 Pre-fall 直接倒地
      - Fall   → Normal 概率极低 (≤0.02), 不能瞬间恢复站立
      - Normal → Pre-fall 允许 (0.07), 对应开始失衡
      - Pre-fall → Fall 允许 (0.12), 对应过渡完成
    """
    T = np.zeros((3, 3))
    T[0] = [stay_normal, normal_to_prefall, max(0.0, 1.0 - stay_normal - normal_to_prefall)]
    T[1] = [prefall_to_normal, stay_prefall, prefall_to_fall]
    T[2] = [max(0.0, 1.0 - stay_fall - fall_to_prefall), fall_to_prefall, stay_fall]
    # 确保行归一化
    T = T / T.sum(axis=1, keepdims=True)
    return T


class HMMStateSmoother:
    """
    基于 Viterbi 解码的时序状态平滑器。

    维护一个概率历史缓冲区，当缓冲区满时运行 Viterbi 解码
    找到最可能的状态序列，输出平滑后的最后一帧状态。

    这可以消除模型独立 argmax 产生的物理不可能跳变
    （如 Normal→Fall→Normal），并减少单帧误判导致的假警报。
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_HMM_BUFFER_SIZE,
        transition_matrix: np.ndarray | None = None,
        initial_probs: list[float] | None = None,
    ) -> None:
        self._buffer_size = max(3, int(buffer_size))
        self._transition = (
            transition_matrix
            if transition_matrix is not None
            else build_hmm_transition_matrix()
        )
        self._initial_probs = (
            initial_probs if initial_probs is not None else [0.98, 0.02, 0.0]
        )
        # 概率缓冲区: 每个元素是 [P(Normal), P(Pre-fall), P(Fall)]
        self._prob_buffer: deque[list[float]] = deque(maxlen=self._buffer_size)
        self._state_idx = {0: "Normal", 1: "Pre-fall", 2: "Fall"}

    def reset(self) -> None:
        """清空概率历史。"""
        self._prob_buffer.clear()

    def smooth(self, probabilities: list[float]) -> str:
        """
        输入当前帧的三类概率，返回平滑后的状态。

        probabilities: [P(Normal), P(Pre-fall), P(Fall)]

        当缓冲区未满时，直接返回 argmax 状态；
        缓冲区满后，运行 Viterbi 解码返回最优路径的最后一帧状态。
        """
        self._prob_buffer.append(list(probabilities))

        if len(self._prob_buffer) < self._buffer_size:
            # 缓冲未满: 退化到 argmax
            idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
            return self._state_idx[idx]

        # 运行 Viterbi
        obs = list(self._prob_buffer)
        state_seq = self._viterbi(obs)
        return self._state_idx[state_seq[-1]]

    def _viterbi(self, observations: list[list[float]]) -> list[int]:
        """
        对概率观测序列做 Viterbi 解码。

        在对数空间中计算，避免浮点下溢。
        """
        n_states = 3
        T_len = len(observations)
        if T_len == 0:
            return []

        log_T = np.log(np.maximum(self._transition, 1e-12))
        log_init = np.log(np.maximum(self._initial_probs, 1e-12))

        dp = np.full((T_len, n_states), -np.inf)
        bp = np.zeros((T_len, n_states), dtype=int)

        # 初始化
        for j in range(n_states):
            obs_prob = max(observations[0][j], 1e-12)
            dp[0][j] = log_init[j] + np.log(obs_prob)

        # 递推
        for t in range(1, T_len):
            for j in range(n_states):
                scores = dp[t - 1] + log_T[:, j]
                best_i = int(np.argmax(scores))
                dp[t][j] = scores[best_i] + np.log(max(observations[t][j], 1e-12))
                bp[t][j] = best_i

        # 回溯
        states = [0] * T_len
        states[T_len - 1] = int(np.argmax(dp[T_len - 1]))
        for t in range(T_len - 2, -1, -1):
            states[t] = bp[t + 1][states[t + 1]]
        return states


@dataclass(frozen=True)
class FallMotionEvidence:
    """最近窗口里用于确认 Fall 的动态证据。"""

    max_vertical_velocity: float
    max_vertical_accel: float
    max_angular_velocity: float
    max_angular_accel: float
    center_drop_delta: float
    has_fast_descent: bool
    has_impact_or_rotation: bool
    has_center_drop_change: bool


class TemporalFallValidator:
    """
    运行时 Fall 时序确认层。

    三分类模型仍然输出 Normal / Pre-fall / Fall，但 Fall 需要满足更保守的
    时序 + 动态组合证据才确认：
    - Pre-fall 可以保持敏感，用于提前提醒；
    - Fall 前必须在近期看到 Pre-fall 证据，最好是 Normal -> Pre-fall 的状态过渡；
    - 当前窗口必须已经被模型/HMM 判为 Fall；
    - Fall 还必须看到快速下落和冲击/下降证据，或者在 Pre-fall 后连续多个窗口被判为 Fall；
    - 未确认的 Fall 会按证据强弱降级成 Pre-fall 或 Normal；
    - Fall 一旦确认，会保持一小段时间，减少 Fall/Normal 抖动。

    如果一个人一开始就稳定躺着，模型可能因为静态姿态像倒地而输出 Fall。
    这时窗口里通常没有“下落/失衡过程”，这里会把 Fall 降回 Normal。
    """

    def __init__(
        self,
        history_size: int = DEFAULT_FALL_VALIDATION_HISTORY,
        prefall_memory: int = DEFAULT_FALL_PREFALL_MEMORY,
        fall_hold_frames: int = DEFAULT_FALL_HOLD_FRAMES,
        fall_after_prefall_confirm_frames: int = DEFAULT_FALL_AFTER_PREFALL_CONFIRM_FRAMES,
        vertical_velocity_threshold: float = DEFAULT_FALL_VERTICAL_VELOCITY,
        vertical_accel_threshold: float = DEFAULT_FALL_VERTICAL_ACCEL,
        angular_velocity_threshold: float = DEFAULT_FALL_ANGULAR_VELOCITY,
        angular_accel_threshold: float = DEFAULT_FALL_ANGULAR_ACCEL,
        center_drop_delta_threshold: float = DEFAULT_FALL_CENTER_DROP_DELTA,
    ) -> None:
        self._state_history: deque[str] = deque(maxlen=max(1, int(history_size)))
        self._prefall_memory = max(1, int(prefall_memory))
        self._fall_hold_frames = max(0, int(fall_hold_frames))
        self._fall_hold_remaining = 0
        self._fall_after_prefall_confirm_frames = max(1, int(fall_after_prefall_confirm_frames))
        self._fall_after_prefall_count = 0
        self.vertical_velocity_threshold = max(0.0, float(vertical_velocity_threshold))
        self.vertical_accel_threshold = max(0.0, float(vertical_accel_threshold))
        self.angular_velocity_threshold = max(0.0, float(angular_velocity_threshold))
        self.angular_accel_threshold = max(0.0, float(angular_accel_threshold))
        self.center_drop_delta_threshold = max(0.0, float(center_drop_delta_threshold))

    def reset(self) -> None:
        self._state_history.clear()
        self._fall_hold_remaining = 0
        self._fall_after_prefall_count = 0

    def validate(
        self,
        state: str,
        alert_state: str,
        window_rows: Sequence[dict[str, float]],
    ) -> tuple[str, str]:
        if self._fall_hold_remaining > 0:
            self._fall_hold_remaining -= 1
            self._state_history.append("Fall")
            return "Fall", "Fall"

        if state != "Fall":
            self._fall_after_prefall_count = 0
            filtered_alert = "Pre-fall" if alert_state == "Fall" else alert_state
            self._state_history.append(self._history_state(state, filtered_alert))
            return state, filtered_alert

        evidence = self._motion_evidence(window_rows)
        has_prefall_evidence = self._has_recent_prefall_evidence()
        if has_prefall_evidence:
            self._fall_after_prefall_count += 1
        else:
            self._fall_after_prefall_count = 0

        confirmed = self._confirms_fall(evidence, has_prefall_evidence)
        if confirmed:
            self._fall_hold_remaining = self._fall_hold_frames
            self._state_history.append("Fall")
            return "Fall", "Fall"

        filtered_state, filtered_alert = self._unconfirmed_fall_state(alert_state, evidence)
        self._state_history.append(self._history_state(filtered_state, filtered_alert))
        return filtered_state, filtered_alert

    def _has_recent_prefall_evidence(self) -> bool:
        recent_states = list(self._state_history)[-self._prefall_memory :]
        return "Pre-fall" in recent_states

    @staticmethod
    def _history_state(state: str, alert_state: str) -> str:
        return "Pre-fall" if alert_state == "Pre-fall" else state

    def _confirms_fall(self, evidence: FallMotionEvidence, has_prefall_evidence: bool) -> bool:
        if not has_prefall_evidence:
            return False
        if evidence.has_fast_descent and (
            evidence.has_center_drop_change or evidence.has_impact_or_rotation
        ):
            return True
        return self._fall_after_prefall_count >= self._fall_after_prefall_confirm_frames

    def _unconfirmed_fall_state(
        self,
        alert_state: str,
        evidence: FallMotionEvidence,
    ) -> tuple[str, str]:
        suspicious_but_unconfirmed = (
            alert_state == "Pre-fall"
            or self._has_recent_prefall_evidence()
            or evidence.has_fast_descent
            or evidence.has_impact_or_rotation
        )
        if suspicious_but_unconfirmed:
            return "Pre-fall", "Pre-fall"
        return "Normal", "Normal"

    def _motion_evidence(self, window_rows: Sequence[dict[str, float]]) -> FallMotionEvidence:
        if not window_rows:
            return FallMotionEvidence(
                max_vertical_velocity=0.0,
                max_vertical_accel=0.0,
                max_angular_velocity=0.0,
                max_angular_accel=0.0,
                center_drop_delta=0.0,
                has_fast_descent=False,
                has_impact_or_rotation=False,
                has_center_drop_change=False,
            )

        max_vertical_velocity = max(
            max(0.0, _row_float(row, "vertical_velocity"))
            for row in window_rows
        )
        max_vertical_accel = max(_positive_delta(window_rows, "vertical_velocity"))

        max_angular_velocity = max(
            abs(_row_float(row, "torso_angular_velocity"))
            for row in window_rows
        )
        max_angular_accel = max(abs(value) for value in _delta(window_rows, "torso_angular_velocity"))

        center_drop_values = [_row_float(row, "center_drop") for row in window_rows]
        center_drop_delta = max(center_drop_values) - min(center_drop_values) if center_drop_values else 0.0

        has_fast_descent = max_vertical_velocity >= self.vertical_velocity_threshold
        has_impact_or_rotation = (
            max_vertical_accel >= self.vertical_accel_threshold
            or max_angular_velocity >= self.angular_velocity_threshold
            or max_angular_accel >= self.angular_accel_threshold
        )
        has_center_drop_change = center_drop_delta >= self.center_drop_delta_threshold

        return FallMotionEvidence(
            max_vertical_velocity=max_vertical_velocity,
            max_vertical_accel=max_vertical_accel,
            max_angular_velocity=max_angular_velocity,
            max_angular_accel=max_angular_accel,
            center_drop_delta=center_drop_delta,
            has_fast_descent=has_fast_descent,
            has_impact_or_rotation=has_impact_or_rotation,
            has_center_drop_change=has_center_drop_change,
        )


class MachineLearningFallPredictor:
    """
    基于滑动窗口的机器学习跌倒预测器。

    model_path 指向 train_model.py 保存的 joblib 文件。
    这个 joblib 不是只有模型本身，还包含一些推理时必须一致的元数据：

    - model: scikit-learn 分类器
    - window_size: 训练时每个样本用了多少帧
    - feature_columns: 训练时使用了哪些特征列，以及列顺序
    - baseline_frames: 计算 center_drop 时使用多少帧建立初始站立基线
    """

    def __init__(
        self,
        model_path: str | Path,
        baseline_frames: int | None = None,
        smoothing_window: int | None = None,
        min_visibility: float = 0.35,
        prefall_alert_threshold: float | None = None,
        prefall_alert_consecutive_frames: int | None = None,
        use_hmm: bool = False,
        hmm_buffer_size: int = DEFAULT_HMM_BUFFER_SIZE,
        use_accel: bool | None = None,
        use_temporal_fall_validation: bool = True,
        fall_validator_settings: Mapping[str, float | int] | None = None,
    ) -> None:
        # 加载训练脚本保存下来的 artifact。
        # 这里会延迟导入 joblib，避免规则版预测也强制依赖 sklearn/joblib。
        artifact = load_model_artifact(model_path)

        # 这些设置必须和训练时一致，否则模型输入的含义会错位。
        self.model = artifact["model"]
        self.window_size = int(artifact.get("window_size", DEFAULT_WINDOW_SIZE))
        self.feature_columns = tuple(artifact.get("feature_columns", ML_FEATURE_COLUMNS))
        self.baseline_frames = _resolve_positive_int_setting(
            artifact=artifact,
            name="baseline_frames",
            explicit_value=baseline_frames,
            default_value=DEFAULT_PREDICTOR_CONFIG.baseline_frames,
        )
        self.smoothing_window = _resolve_positive_int_setting(
            artifact=artifact,
            name="smoothing_window",
            explicit_value=smoothing_window,
            default_value=DEFAULT_PREDICTOR_CONFIG.smoothing_window,
        )
        self.prefall_alert_threshold = _resolve_probability_setting(
            artifact=artifact,
            name="prefall_alert_threshold",
            explicit_value=prefall_alert_threshold,
            default_value=DEFAULT_PREFALL_ALERT_THRESHOLD,
        )
        self.prefall_alert_consecutive_frames = _resolve_positive_int_setting(
            artifact=artifact,
            name="prefall_alert_consecutive_frames",
            explicit_value=prefall_alert_consecutive_frames,
            default_value=DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
        )
        self.min_visibility = min_visibility

        # FeatureExtractor 和规则版一样：负责从 MediaPipe 关键点计算单帧运动特征。
        self.extractor = FeatureExtractor(min_visibility=min_visibility)

        # 保存最近 N 帧的特征。deque(maxlen=N) 会自动丢掉最旧的一帧，
        # 非常适合做实时滑动窗口。
        self._window: deque[dict[str, float]] = deque(maxlen=self.window_size)

        # risk_history 只是为了输出 smoothed_risk_score，便于 CSV 和画图保持一致。
        self._risk_history: deque[float] = deque(maxlen=self.smoothing_window)

        # center_drop 需要知道"正常站立时身体中心大概在哪里"。
        # 所以前几帧会先收集 baseline，再计算后续身体下降量。
        self._baseline_samples: list[float] = []
        self._baseline_center_y: float | None = None

        # 报警策略单独计数：分类 state 保持模型 argmax，alert_state 可以更早提醒。
        self._prefall_alert_count = 0

        # HMM 时序平滑（可选）
        self._use_hmm = bool(use_hmm)
        self._hmm: HMMStateSmoother | None = None
        if self._use_hmm:
            self._hmm = HMMStateSmoother(buffer_size=hmm_buffer_size)
            # 如果模型已保存 prefall_alert_threshold，可适当降低 alert 阈值
            # 因为 HMM 已经抑制了假阳

        # 加速度特征：显式参数优先，未提供时从 artifact 自动检测
        if use_accel is not None:
            self._use_accel = bool(use_accel)
        else:
            self._use_accel = bool(artifact.get("use_accel", False))

        self._use_temporal_fall_validation = bool(use_temporal_fall_validation)
        self._temporal_fall_validator = TemporalFallValidator(
            **dict(fall_validator_settings or {})
        )

    @property
    def baseline_center_y(self) -> float | None:
        """返回当前已经建立好的身体中心基线；未建立好时返回 None。"""
        return self._baseline_center_y

    def predict(
        self,
        landmarks: Sequence[Landmark] | None,
        frame_index: int,
        timestamp: float,
    ) -> Prediction:
        # 1. 把当前帧关键点转换成可解释的数值特征。
        features = self.extractor.extract(landmarks, frame_index, timestamp)

        # 2. 更新站立基线，并计算当前身体中心相对基线下降了多少。
        center_drop = self._update_baseline_and_center_drop(features)

        # 3. 把当前帧特征加入滑动窗口。
        self._window.append(pose_features_to_ml_row(features, center_drop))

        # 如果当前帧没有可靠人体姿态，直接返回 Unknown。
        # 这样模型不会在"看不清人"的情况下硬给一个 Normal/Fall。
        if not features.has_pose or features.visibility_mean < self.min_visibility:
            self._prefall_alert_count = 0
            self._temporal_fall_validator.reset()
            return self._prediction(
                frame_index=frame_index,
                timestamp=timestamp,
                state="Unknown",
                instant_state="Unknown",
                risk_score=0.0,
                features=features,
                center_drop=center_drop,
                alert_state="Unknown",
            )

        # 窗口还没有收集满时，模型没有足够历史信息可看。
        # 例如 window_size=15，则前 14 帧先输出 Normal。
        if len(self._window) < self.window_size:
            self._prefall_alert_count = 0
            return self._prediction(
                frame_index=frame_index,
                timestamp=timestamp,
                state="Normal",
                instant_state="Normal",
                risk_score=0.0,
                features=features,
                center_drop=center_drop,
                alert_state="Normal",
            )

        # 4. 窗口满了以后，把最近 N 帧展开成一个模型输入样本。
        # scikit-learn 的 predict/predict_proba 期望输入是二维结构：
        # [样本1, 样本2, ...]，所以这里外面再套一层 list。
        window_list = list(self._window)
        if self._use_accel:
            window_list = compute_window_accel_features(window_list)
            feature_cols = ACCEL_FEATURE_COLUMNS
        else:
            feature_cols = self.feature_columns
        sample = [flatten_window(window_list, feature_cols)]
        model_state, risk_score, model_alert_state = self._predict_sample(sample)
        if self._use_temporal_fall_validation:
            state, alert_state = self._temporal_fall_validator.validate(
                model_state,
                model_alert_state,
                window_list,
            )
        else:
            state, alert_state = model_state, model_alert_state
        return self._prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            instant_state=model_state,
            risk_score=risk_score,
            features=features,
            center_drop=center_drop,
            alert_state=alert_state,
        )

    def reset(self) -> None:
        """切换新视频或重新开始时，清空所有时序状态。"""
        self.extractor.reset()
        self._window.clear()
        self._risk_history.clear()
        self._baseline_samples.clear()
        self._baseline_center_y = None
        self._prefall_alert_count = 0
        if self._hmm is not None:
            self._hmm.reset()
        self._temporal_fall_validator.reset()

    def _update_baseline_and_center_drop(self, features: PoseFeatures) -> float:
        """
        更新站立基线，并返回身体下降量 center_drop。

        body_center_y 是归一化图像坐标，y 越大表示越靠下。
        如果当前身体中心比初始站立基线更靠下，就认为出现了下降。
        """
        if features.has_pose and self._baseline_center_y is None:
            self._baseline_samples.append(features.body_center_y)
            if len(self._baseline_samples) >= self.baseline_frames:
                self._baseline_center_y = mean(self._baseline_samples)

        baseline = self._baseline_center_y
        if baseline is None and self._baseline_samples:
            baseline = mean(self._baseline_samples)
        if baseline is None:
            return 0.0
        return max(0.0, features.body_center_y - baseline)

    def _predict_sample(self, sample: list[list[float]]) -> tuple[str, float, str]:
        """
        调用 scikit-learn 模型预测一个窗口样本。

        返回:
            state:
                Normal / Pre-fall / Fall 等状态字符串。
                如果启用了 HMM，这是经过 Viterbi 平滑后的状态。

            risk_score:
                用概率近似出来的风险值，主要用于画图和 CSV 分析。

            alert_state:
                运行时报警状态。它允许比 state 更敏感，但不会改掉模型原始分类。
                HMM 启用时，alert_state 也从平滑后的概率计算。
        """
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            classes = [str(label) for label in self.model.classes_]

            # 如果启用了 HMM，用 Viterbi 平滑后的状态替代独立 argmax
            if self._use_hmm and self._hmm is not None:
                # 将概率按 [Normal, Pre-fall, Fall] 顺序传入 HMM。
                # 当前主流程只保留三分类输出。
                hmm_probs = [
                    _normal_probability(classes, probabilities),
                    _state_probability(classes, probabilities, "Pre-fall"),
                    _state_probability(classes, probabilities, "Fall"),
                ]
                state = self._hmm.smooth(hmm_probs)
                # 用平滑后的状态重新计算风险分数和报警
                risk_score = _fall_probability(classes, probabilities)
                alert_state = self._alert_state_from_probabilities(state, classes, probabilities)
                return state, risk_score, alert_state

            # 没有 HMM：独立 argmax
            best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
            state = normalize_state(classes[best_index])

            # 风险分数不一定等于最终类别，而是把和跌倒相关的概率合成一个 0~1 值。
            risk_score = _fall_probability(classes, probabilities)
            alert_state = self._alert_state_from_probabilities(state, classes, probabilities)
            return state, risk_score, alert_state

        label = self.model.predict(sample)[0]
        state = normalize_state(str(label))
        risk_score = 1.0 if state in {"Pre-fall", "Fall"} else 0.0
        self._prefall_alert_count = 0
        return state, risk_score, state

    def _alert_state_from_probabilities(
        self,
        state: str,
        classes: Sequence[str],
        probabilities: Sequence[float],
    ) -> str:
        """
        从模型概率里生成更适合实时显示的报警状态。

        state 仍然是模型概率最高的类别；alert_state 只是预警层：
        - Fall / Pre-fall 已经是最高概率时，直接报警；
        - Normal-like 安全子类最高但 Pre-fall 概率持续偏高时，提前显示 Pre-fall；
        - 其他情况保持原状态。
        """
        if state == "Fall":
            self._prefall_alert_count = 0
            return "Fall"
        if state == "Pre-fall":
            self._prefall_alert_count = self.prefall_alert_consecutive_frames
            return "Pre-fall"

        prefall_probability = _state_probability(classes, probabilities, "Pre-fall")
        if prefall_probability >= self.prefall_alert_threshold:
            self._prefall_alert_count += 1
        else:
            self._prefall_alert_count = 0

        if self._prefall_alert_count >= self.prefall_alert_consecutive_frames:
            return "Pre-fall"
        return state

    def _prediction(
        self,
        frame_index: int,
        timestamp: float,
        state: str,
        instant_state: str,
        risk_score: float,
        features: PoseFeatures,
        center_drop: float,
        alert_state: str | None = None,
    ) -> Prediction:
        """
        把 ML 预测结果包装成项目统一的 Prediction 对象。

        video_app.py 后续会用 Prediction 来：
        - 写 CSV
        - 绘制状态文字
        - 绘制风险曲线

        因为 ML 模型没有规则系统那种 torso_score、center_drop_score 等子分数，
        所以 RiskBreakdown 里的子分数字段填 0，只保留总风险和 center_drop。
        """
        self._risk_history.append(risk_score)
        smoothed_risk = mean(self._risk_history) if self._risk_history else 0.0
        return Prediction(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            instant_state=instant_state,
            risk_score=risk_score,
            smoothed_risk_score=smoothed_risk,
            features=features,
            breakdown=RiskBreakdown(
                risk_score=risk_score,
                torso_score=0.0,
                angular_velocity_score=0.0,
                vertical_velocity_score=0.0,
                center_drop_score=0.0,
                aspect_ratio_score=0.0,
                visibility_factor=1.0 if features.has_pose else 0.0,
                center_drop=center_drop,
            ),
            baseline_center_y=self._baseline_center_y,
            alert_state=alert_state or state,
        )


def load_model_artifact(model_path: str | Path) -> dict:
    """
    加载 joblib 模型文件。

    为了兼容以后你可能保存"裸模型"的情况：
    - 如果加载出来是 dict，就按 train_model.py 保存的 artifact 使用；
    - 如果加载出来不是 dict，就当作旧格式裸模型，并补默认元数据。
    """
    artifact = _load_model_artifact_cached(str(Path(model_path).expanduser().resolve()))
    if "model" not in artifact:
        raise RuntimeError(f"模型文件中缺少 'model' 字段: {model_path}")
    return artifact


@lru_cache(maxsize=4)
def _load_model_artifact_cached(model_path: str) -> dict:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "ML 预测需要 joblib。请先运行：python -m pip install -r requirements.txt"
        ) from exc

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        artifact = {
            "model": artifact,
            "window_size": DEFAULT_WINDOW_SIZE,
            "feature_columns": ML_FEATURE_COLUMNS,
            "baseline_frames": DEFAULT_PREDICTOR_CONFIG.baseline_frames,
            "smoothing_window": DEFAULT_PREDICTOR_CONFIG.smoothing_window,
            "prefall_alert_threshold": DEFAULT_PREFALL_ALERT_THRESHOLD,
            "prefall_alert_consecutive_frames": DEFAULT_PREFALL_ALERT_CONSECUTIVE_FRAMES,
        }
    if "model" not in artifact:
        raise RuntimeError(f"模型文件中缺少 'model' 字段: {model_path}")
    return artifact


def normalize_state(label: str) -> str:
    """
    把不同数据集/模型可能输出的标签统一成应用内部状态。

    例如：
        1、fall、Fall       -> Fall
        0、normal、adl      -> Normal
        prefall、pre-fall   -> Pre-fall
    """
    cleaned = label.strip()
    lower = cleaned.lower().replace("_", "-")
    if lower in {"1", "fall", "fallen"}:
        return "Fall"
    if lower in {"pre-fall", "prefall", "pre fall"}:
        return "Pre-fall"
    if lower in {
        "0",
        "normal",
        "adl",
        "nonfall",
        "non-fall",
        "no-fall",
        "standing",
        "stand",
        "normal-standing",
        "normal-stand",
        "walking",
        "walk",
        "normal-walking",
        "normal-walk",
        "sitting",
        "sit",
        "seated",
        "normal-sitting",
        "normal-sit",
        "squatting",
        "squat",
        "crouching",
        "crouch",
        "normal-squatting",
        "normal-squat",
        "normal-crouching",
        "normal-crouch",
        "bending",
        "bend",
        "bent",
        "stooping",
        "stoop",
        "spotting",
        "spot",
        "bowing",
        "bow",
        "normal-bending",
        "normal-bend",
        "normal-stooping",
        "normal-stoop",
        "normal-spotting",
        "normal-spot",
        "lying",
        "lie",
        "laying",
        "laid",
        "reclining",
        "reclined",
        "prone",
        "supine",
        "horizontal",
        "normal-lying",
        "normal-lie",
        "normal-laying",
    }:
        return "Normal"
    if "pre" in lower and "fall" in lower:
        return "Pre-fall"
    if "fall" in lower:
        return "Fall"
    return cleaned


def _resolve_positive_int_setting(
    artifact: dict,
    name: str,
    explicit_value: int | None,
    default_value: int,
) -> int:
    """
    Resolve integer settings with a clear priority:
    explicit constructor argument > saved model artifact > project default.
    """
    value = explicit_value if explicit_value is not None else artifact.get(name, default_value)
    return max(1, int(value))


def _resolve_probability_setting(
    artifact: dict,
    name: str,
    explicit_value: float | None,
    default_value: float,
) -> float:
    """
    Resolve probability-like settings and keep them inside [0, 1].
    """
    value = explicit_value if explicit_value is not None else artifact.get(name, default_value)
    return max(0.0, min(1.0, float(value)))


def _fall_probability(classes: Sequence[str], probabilities: Sequence[float]) -> float:
    """
    根据分类概率估算一个 0~1 的跌倒风险分数。

    如果类别是 Fall，完整计入风险；
    如果类别是 Pre-fall，只按 0.5 权重计入，因为它表示风险升高但还未倒地。
    """
    fall_prob = 0.0
    for label, probability in zip(classes, probabilities):
        state = normalize_state(label)
        if state == "Fall":
            fall_prob += float(probability)
        elif state == "Pre-fall":
            fall_prob += float(probability) * 0.5
    return max(0.0, min(1.0, fall_prob))


def _state_probability(classes: Sequence[str], probabilities: Sequence[float], target_state: str) -> float:
    """Return the summed probability for one normalized state."""
    total = 0.0
    for label, probability in zip(classes, probabilities):
        if normalize_state(label) == target_state:
            total += float(probability)
    return max(0.0, min(1.0, total))


def _normal_probability(classes: Sequence[str], probabilities: Sequence[float]) -> float:
    """Return summed probability for normalized Normal states."""
    total = 0.0
    for label, probability in zip(classes, probabilities):
        if normalize_state(label) in NORMAL_STATES:
            total += float(probability)
    return max(0.0, min(1.0, total))


def _delta(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Return frame-to-frame deltas for one numeric row field."""
    if not rows:
        return [0.0]

    values = [_row_float(row, key) for row in rows]
    deltas = [0.0]
    for index in range(1, len(values)):
        deltas.append(values[index] - values[index - 1])
    return deltas


def _positive_delta(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Return positive frame-to-frame deltas for one numeric row field."""
    return [max(0.0, value) for value in _delta(rows, key)]


def _row_float(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value
