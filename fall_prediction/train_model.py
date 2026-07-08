"""
训练机器学习跌倒分类器。

这个脚本接收的不是原始视频，而是前一步导出的"逐帧特征 CSV"。
也就是说，完整流程是：

    原始视频
      -> MediaPipe 提取人体关键点
      -> features.py 计算每一帧的身体运动特征
      -> export_dataset_features.py 保存 CSV
      -> 本脚本把连续多帧切成一个训练样本
      -> scikit-learn 训练分类器
      -> 保存 models/yolo_tail60_prefall_accel_classifier.joblib

为什么要把"连续多帧"作为一个样本？
跌倒不是一张静态图片能可靠判断的事件，它有时间过程：
站立 -> 身体快速下降/倾斜 -> 接近地面 -> 倒地。
所以模型需要看到一小段时间窗口。当前默认使用 15 帧窗口，约等于 0.5 秒视频；
它比 30 帧窗口更偏向提前识别 Pre-fall 过渡阶段。

使用示例：
    python -m fall_prediction.train_model outputs/features/urfall_yolo/*.csv \
        --output models/yolo_tail60_prefall_accel_classifier.joblib \
        --window-size 15 \
        --stride 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .ml_features import ACCEL_FEATURE_COLUMNS, ML_FEATURE_COLUMNS
from .predictor import PredictorConfig
from .window_dataset import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, build_window_dataset


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="从特征 CSV 训练机器学习跌倒分类器。")
    parser.add_argument("csv_paths", nargs="*", help="特征 CSV 文件路径，可以直接写多个文件。")
    parser.add_argument("--input-dir", default=None, help="可选：包含特征 CSV 的目录，会递归读取其中的 .csv 文件。")
    parser.add_argument(
        "--output",
        default="models/yolo_tail60_prefall_accel_classifier.joblib",
        help="训练完成后保存的 joblib 模型路径。",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="验证指标 JSON 输出路径；默认保存到模型旁边的 .metrics.json 文件。",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="每个训练样本包含多少帧；15 帧约等于 0.5 秒，更偏向 Pre-fall 提前预警。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="滑动窗口每次向前移动多少帧；越小样本越多，但重复也越多。",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=DEFAULT_PREDICTOR_CONFIG.baseline_frames,
        help="推理时建立身体中心基线使用多少帧，默认 15。",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_PREDICTOR_CONFIG.smoothing_window,
        help="推理时输出 smoothed_risk_score 的平滑窗口，默认 5。",
    )
    parser.add_argument(
        "--classifier",
        choices=("random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"),
        default="random_forest",
        help="使用哪种 scikit-learn 分类器。默认 random_forest。",
    )
    parser.add_argument(
        "--label-mode",
        choices=("filename", "annotations"),
        default="filename",
        help="标签来源：filename 根据文件名推断；annotations 使用帧区间标注文件。",
    )
    parser.add_argument(
        "--annotations",
        action="append",
        default=None,
        help="帧区间标注 CSV，列名必须是 video,start_frame,end_frame,label。可重复传入多个文件。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="验证集比例，按视频分组划分，默认 25%%；设为 0 时使用全部数据训练并跳过验证。",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子；固定后每次划分和训练结果更容易复现。")
    parser.add_argument("--normal-weight", type=float, default=1.0, help="Normal 样本训练权重，默认 1.0。")
    parser.add_argument("--fall-weight", type=float, default=1.0, help="Fall 样本训练权重，默认 1.0。")
    parser.add_argument("--prefall-weight", type=float, default=1.0, help="Pre-fall 样本训练权重，默认 1.0。")
    parser.add_argument(
        "--tune-prefall-alert-threshold",
        action="store_true",
        help="在验证集上搜索 Pre-fall 报警阈值，并保存到模型 artifact。",
    )
    parser.add_argument(
        "--prefall-threshold-beta",
        type=float,
        default=1.5,
        help="搜索 Pre-fall 报警阈值时使用的 F-beta beta 值，默认 1.5，更偏召回。",
    )
    parser.add_argument(
        "--prefall-alert-threshold",
        type=float,
        default=None,
        help="直接写入模型的 Pre-fall 报警概率阈值；适合全量训练时沿用验证实验得到的阈值。",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="启用加速度增强特征 (torso_angular_accel, vertical_accel)。",
    )
    args = parser.parse_args()

    # 收集训练用 CSV。可以来自命令行显式传入，也可以来自 --input-dir。
    csv_paths = collect_csv_paths(args.csv_paths, args.input_dir)
    if not csv_paths:
        raise RuntimeError("没有找到特征 CSV 文件。请先运行 export_dataset_features.py 导出特征。")

    # 把逐帧 CSV 切成"滑动窗口样本"：
    # X 是模型输入，每一项是一段窗口展开后的数字特征；
    # y 是标签，例如 Normal / Fall / Pre-fall；
    # groups 记录样本来自哪个视频，用于按视频划分训练集和验证集。
    dataset = build_window_dataset(
        csv_paths=csv_paths,
        window_size=args.window_size,
        stride=args.stride,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode=args.label_mode,
        annotations_path=args.annotations,
        use_accel=args.use_accel,
    )
    if not dataset.X:
        raise RuntimeError("没有生成训练窗口。请检查标签、窗口大小和 CSV 文件内容。")

    # 真正训练模型，并把模型与必要元数据一起保存。
    train_and_save(
        X=dataset.X,
        y=dataset.y,
        groups=dataset.groups,
        feature_names=dataset.feature_names,
        csv_paths=csv_paths,
        output_path=args.output,
        window_size=args.window_size,
        stride=args.stride,
        baseline_frames=args.baseline_frames,
        smoothing_window=args.smoothing_window,
        classifier_name=args.classifier,
        label_mode=args.label_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        metrics_output_path=args.metrics_output,
        class_weights={
            "Normal": args.normal_weight,
            "Fall": args.fall_weight,
            "Pre-fall": args.prefall_weight,
        },
        tune_prefall_alert_threshold=args.tune_prefall_alert_threshold,
        prefall_threshold_beta=args.prefall_threshold_beta,
        prefall_alert_threshold=args.prefall_alert_threshold,
        use_accel=args.use_accel,
    )


def collect_csv_paths(paths: list[str], input_dir: str | None) -> list[Path]:
    """
    收集所有训练 CSV 文件。

    paths:
        命令行中直接写出的文件路径，例如 outputs/features/urfall_yolo/fall-01-cam0.csv。

    input_dir:
        一个目录路径。如果提供，就递归读取里面所有 .csv 文件。

    返回值去重并排序，是为了让同一批数据每次训练时顺序稳定。
    """
    csv_paths = [Path(path) for path in paths]
    if input_dir:
        csv_paths.extend(sorted(Path(input_dir).rglob("*.csv")))
    return sorted(set(csv_paths))


def train_and_save(
    X: list[list[float]],
    y: list[str],
    groups: list[str],
    feature_names: list[str],
    csv_paths: list[Path],
    output_path: str | Path,
    window_size: int,
    stride: int,
    baseline_frames: int,
    smoothing_window: int,
    classifier_name: str,
    label_mode: str,
    test_size: float,
    random_state: int,
    metrics_output_path: str | Path | None = None,
    class_weights: dict[str, float] | None = None,
    tune_prefall_alert_threshold: bool = False,
    prefall_threshold_beta: float = 1.5,
    prefall_alert_threshold: float | None = None,
    use_accel: bool = False,
) -> dict[str, Any]:
    """
    训练分类器并保存模型文件。

    这里先使用 scikit-learn 的传统机器学习模型，而不是一上来就用 LSTM/Transformer，原因是：
    1. 当前输入是手工提取的数值特征，树模型很适合这种表格数据；
    2. UR Fall 这类数据集规模不算特别大，复杂深度模型容易过拟合；
    3. 这些模型训练快，适合先验证整条流程是否跑通。

    当前可选分类器：
    - random_forest: 随机森林，稳定、抗噪声，默认选择
    - extra_trees: 极端随机树，随机性更强，有时泛化更好
    - gradient_boosting: 梯度提升树，逐步修正错误，常常能挤出一点准确率
    - hist_gradient_boosting: 直方图梯度提升，速度快，适合更大数据

    保存的不是裸模型，而是一个 artifact 字典，里面包含：
    - model: 训练好的 scikit-learn 模型
    - window_size: 推理时必须使用同样长度的窗口
    - feature_columns: 推理时必须使用同样顺序的特征列
    - training_videos: 记录训练用过哪些 CSV，方便以后排查
    """
    # 这些依赖只在训练时需要，所以放在函数内部导入。
    # 这样用户只是运行规则版预测时，不会因为没装 sklearn 就导入失败。
    try:
        import joblib
        import numpy as np
        from sklearn.metrics import classification_report
    except ImportError as exc:
        raise RuntimeError(
            "训练机器学习模型需要 numpy、scikit-learn 和 joblib。"
            "请先运行：python -m pip install -r requirements.txt"
        ) from exc

    # scikit-learn 通常使用 numpy 数组作为输入。
    # X_array 的形状大致是：[样本数, 每个窗口展开后的特征数]。
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)

    # 注意：验证集按"视频"划分，而不是按"窗口"随机划分。
    # 如果同一个视频的相邻窗口同时出现在训练集和验证集，模型会看到高度相似的数据，
    # 验证分数会虚高，看起来很准，实际换新视频可能不准。
    train_index, test_index = _group_train_test_split(
        y_array=y_array,
        groups_array=groups_array,
        test_size=test_size,
        random_state=random_state,
    )

    # 根据命令行参数创建模型。把这一步单独抽出来，方便后续比较不同模型。
    model = create_classifier(classifier_name, random_state)
    sample_weight = build_sample_weights(y_array[train_index], class_weights)
    if sample_weight is None:
        model.fit(X_array[train_index], y_array[train_index])
    else:
        model.fit(X_array[train_index], y_array[train_index], sample_weight=sample_weight)

    print(f"训练样本数: {len(train_index)}")
    print(f"分类器: {classifier_name}")
    if sample_weight is not None:
        print(f"样本权重: {normalized_class_weights(class_weights)}")
    print(f"类别: {', '.join(str(label) for label in model.classes_)}")

    validation_metrics = None
    prefall_alert_threshold_search = None
    if len(test_index) > 0:
        predictions = model.predict(X_array[test_index])
        validation_metrics = build_validation_metrics(
            y_true=y_array[test_index],
            y_pred=predictions,
            labels=model.classes_,
        )
        if tune_prefall_alert_threshold and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_array[test_index])
            prefall_alert_threshold_search = tune_prefall_alert_threshold_on_validation(
                y_true=y_array[test_index],
                y_pred=predictions,
                classes=model.classes_,
                probabilities=probabilities,
                beta=prefall_threshold_beta,
            )
        print("\n验证集报告:")
        print(classification_report(y_array[test_index], predictions, zero_division=0))
        if prefall_alert_threshold_search is not None and prefall_alert_threshold_search.get("best") is not None:
            best = prefall_alert_threshold_search["best"]
            print(
                "\nPre-fall 报警阈值搜索:"
                f" threshold={best['threshold']:.2f}, precision={best['precision']:.3f},"
                f" recall={best['recall']:.3f}, f_beta={best['f_beta']:.3f}"
            )
    else:
        print("\n跳过验证：视频数量或类别数量不足，无法按视频分组划分验证集。")

    created_at = datetime.now().isoformat(timespec="seconds")
    validation_split = build_validation_split_summary(
        y_array=y_array,
        groups_array=groups_array,
        train_index=train_index,
        test_index=test_index,
    )

    # 把训练时的关键设置一起保存。推理时必须保持特征顺序、窗口长度一致，
    # 否则模型接收到的数字含义会错位，预测结果就没有意义。
    saved_feature_columns = ACCEL_FEATURE_COLUMNS if use_accel else ML_FEATURE_COLUMNS
    artifact = {
        "model": model,
        "window_size": window_size,
        "stride": stride,
        "classifier": classifier_name,
        "feature_columns": list(saved_feature_columns),
        "feature_names": feature_names,
        "baseline_frames": max(1, int(baseline_frames)),
        "smoothing_window": max(1, int(smoothing_window)),
        "label_mode": label_mode,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "created_at": created_at,
        "training_samples": int(X_array.shape[0]),
        "training_videos": [str(path) for path in csv_paths],
        "class_weights": normalized_class_weights(class_weights),
        "validation_split": validation_split,
        "validation_metrics": validation_metrics,
        "prefall_alert_threshold_search": prefall_alert_threshold_search,
        "use_accel": bool(use_accel),
    }
    explicit_prefall_alert_threshold = normalize_probability_threshold(prefall_alert_threshold)
    if prefall_alert_threshold_search is not None and prefall_alert_threshold_search.get("best") is not None:
        artifact["prefall_alert_threshold"] = prefall_alert_threshold_search["best"]["threshold"]
        artifact["prefall_alert_threshold_metric"] = prefall_alert_threshold_search["best"]
    elif explicit_prefall_alert_threshold is not None:
        artifact["prefall_alert_threshold"] = explicit_prefall_alert_threshold

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    print(f"\n模型已保存: {output}")

    metrics_output = Path(metrics_output_path) if metrics_output_path else default_metrics_output_path(output)
    write_metrics_report(
        metrics_output,
        build_metrics_report(
            created_at=created_at,
            classifier_name=classifier_name,
            label_mode=label_mode,
            window_size=window_size,
            stride=stride,
            baseline_frames=baseline_frames,
            smoothing_window=smoothing_window,
            test_size=test_size,
            random_state=random_state,
            csv_paths=csv_paths,
            total_samples=int(X_array.shape[0]),
            validation_split=validation_split,
            validation_metrics=validation_metrics,
            class_weights=normalized_class_weights(class_weights),
            prefall_alert_threshold_search=prefall_alert_threshold_search,
            prefall_alert_threshold=artifact.get("prefall_alert_threshold"),
            use_accel=use_accel,
        ),
    )
    print(f"验证指标已保存: {metrics_output}")
    return artifact


def build_validation_metrics(y_true, y_pred, labels) -> dict[str, Any]:
    """Build a JSON-serializable validation metrics summary."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_true_names = [str(label) for label in y_true]
    y_pred_names = [str(label) for label in y_pred]
    label_names = sorted({str(label) for label in labels} | set(y_true_names) | set(y_pred_names))
    report = classification_report(
        y_true_names,
        y_pred_names,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_names, y_pred_names, labels=label_names)

    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    return {
        "labels": label_names,
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(macro_avg.get("f1-score", 0.0)),
        "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        "classification_report": json_ready(report),
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
    }


def build_sample_weights(labels, class_weights: dict[str, float] | None) -> list[float] | None:
    """Build per-sample weights, returning None when all weights are neutral."""
    weights = normalized_class_weights(class_weights)
    if not weights or all(abs(weight - 1.0) <= 1e-9 for weight in weights.values()):
        return None
    return [weights.get(str(label), 1.0) for label in labels]


def normalized_class_weights(class_weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize and validate class weights for artifact/metrics output."""
    if not class_weights:
        return {}
    weights = {str(label): float(weight) for label, weight in class_weights.items()}
    for label, weight in weights.items():
        if weight <= 0:
            raise ValueError(f"Class weight for {label!r} must be positive")
    return dict(sorted(weights.items()))


def normalize_probability_threshold(value: float | None) -> float | None:
    """Validate a user-provided probability threshold."""
    if value is None:
        return None
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Pre-fall alert threshold must be between 0 and 1")
    return threshold


def tune_prefall_alert_threshold_on_validation(
    y_true,
    y_pred,
    classes,
    probabilities,
    beta: float = 1.5,
) -> dict[str, Any]:
    """Search a Pre-fall probability threshold for the alert layer."""
    beta = max(float(beta), 1e-6)
    class_names = [str(label) for label in classes]
    if "Pre-fall" not in class_names:
        return {
            "beta": beta,
            "best": None,
            "candidates": [],
            "alert_validation_metrics": None,
        }

    candidates = []
    for threshold_index in range(5, 96):
        threshold = threshold_index / 100.0
        alert_predictions = prefall_alert_predictions(
            y_pred=y_pred,
            classes=class_names,
            probabilities=probabilities,
            threshold=threshold,
        )
        metrics = prefall_binary_metrics(y_true, alert_predictions, beta=beta)
        candidates.append({"threshold": threshold, **metrics})

    best = max(
        candidates,
        key=lambda item: (
            item["f_beta"],
            item["recall"],
            item["precision"],
            -item["threshold"],
        ),
    )
    best_alert_predictions = prefall_alert_predictions(
        y_pred=y_pred,
        classes=class_names,
        probabilities=probabilities,
        threshold=best["threshold"],
    )
    return {
        "beta": beta,
        "best": best,
        "candidates": candidates,
        "alert_validation_metrics": build_validation_metrics(
            y_true=y_true,
            y_pred=best_alert_predictions,
            labels=class_names,
        ),
    }


def prefall_alert_predictions(y_pred, classes, probabilities, threshold: float) -> list[str]:
    """Apply the runtime-style Pre-fall alert threshold to validation predictions."""
    prefall_index = list(classes).index("Pre-fall")
    alert_predictions: list[str] = []
    for label, probability_row in zip(y_pred, probabilities):
        state = str(label)
        if state in {"Fall", "Pre-fall"}:
            alert_predictions.append(state)
            continue
        if float(probability_row[prefall_index]) >= threshold:
            alert_predictions.append("Pre-fall")
        else:
            alert_predictions.append(state)
    return alert_predictions


def prefall_binary_metrics(y_true, y_pred, beta: float) -> dict[str, Any]:
    """Compute binary Pre-fall precision/recall/F-beta for threshold search."""
    true_labels = [str(label) for label in y_true]
    pred_labels = [str(label) for label in y_pred]
    true_positive = sum(1 for true, pred in zip(true_labels, pred_labels) if true == "Pre-fall" and pred == "Pre-fall")
    false_positive = sum(1 for true, pred in zip(true_labels, pred_labels) if true != "Pre-fall" and pred == "Pre-fall")
    false_negative = sum(1 for true, pred in zip(true_labels, pred_labels) if true == "Pre-fall" and pred != "Pre-fall")

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    f_beta = (1 + beta_squared) * precision * recall / denominator if denominator else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def build_validation_split_summary(y_array, groups_array, train_index, test_index) -> dict[str, Any]:
    """Summarize the grouped train/validation split for reproducibility."""
    return {
        "train_samples": int(len(train_index)),
        "validation_samples": int(len(test_index)),
        "train_groups": sorted({str(group) for group in groups_array[train_index]}),
        "validation_groups": sorted({str(group) for group in groups_array[test_index]}),
        "train_label_counts": label_counts(y_array[train_index]),
        "validation_label_counts": label_counts(y_array[test_index]),
    }


def label_counts(labels) -> dict[str, int]:
    """Return stable string label counts for JSON output."""
    return dict(sorted(Counter(str(label) for label in labels).items()))


def default_metrics_output_path(model_output: Path) -> Path:
    """Derive the default metrics file name from the model artifact path."""
    return model_output.with_suffix(".metrics.json")


def build_metrics_report(
    created_at: str,
    classifier_name: str,
    label_mode: str,
    window_size: int,
    stride: int,
    baseline_frames: int,
    smoothing_window: int,
    test_size: float,
    random_state: int,
    csv_paths: list[Path],
    total_samples: int,
    validation_split: dict[str, Any],
    validation_metrics: dict[str, Any] | None,
    class_weights: dict[str, float],
    prefall_alert_threshold_search: dict[str, Any] | None,
    prefall_alert_threshold: float | None = None,
    use_accel: bool = False,
) -> dict[str, Any]:
    """Create the standalone metrics report written next to the model."""
    return {
        "created_at": created_at,
        "classifier": classifier_name,
        "label_mode": label_mode,
        "window_size": int(window_size),
        "stride": int(stride),
        "baseline_frames": max(1, int(baseline_frames)),
        "smoothing_window": max(1, int(smoothing_window)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "total_samples": int(total_samples),
        "training_videos": [str(path) for path in csv_paths],
        "class_weights": class_weights,
        "use_accel": bool(use_accel),
        "validation_split": validation_split,
        "validation_metrics": validation_metrics,
        "prefall_alert_threshold_search": prefall_alert_threshold_search,
        "prefall_alert_threshold": prefall_alert_threshold,
    }


def write_metrics_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write validation metrics as stable, UTF-8 JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    """Convert common numpy/scikit-learn values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def create_classifier(classifier_name: str, random_state: int):
    """
    根据名称创建 scikit-learn 分类器。

    这里统一设置 random_state，是为了让同一份数据、同一组参数下的结果尽量可复现。
    """
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    if classifier_name == "random_forest":
        # 随机森林由很多棵决策树投票组成。class_weight="balanced" 用来缓解类别不平衡。
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_name == "extra_trees":
        # ExtraTrees 比随机森林更随机，训练也很快，有时能降低过拟合。
        return ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_name == "gradient_boosting":
        # GradientBoosting 会一轮轮修正前一轮的错误，通常比随机森林更"精细"，但训练更慢。
        return GradientBoostingClassifier(random_state=random_state)

    if classifier_name == "hist_gradient_boosting":
        # 直方图版本的梯度提升，适合数据量更大时使用。
        return HistGradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"未知分类器: {classifier_name}")


def _group_train_test_split(y_array, groups_array, test_size: float, random_state: int):
    """
    按视频分组划分训练集和验证集。

    y_array:
        每个窗口样本的标签。

    groups_array:
        每个窗口来自哪个视频。例如同一个 fall-01-cam0.csv 切出的所有窗口，
        group 都是 fall-01-cam0。

    为什么要分组？
        一个视频切出的连续窗口非常相似。如果随机按窗口切分，训练集可能包含
        第 0-29 帧，验证集包含第 5-34 帧，这等于让模型在验证时见过几乎一样的
        动作片段，评估结果会过于乐观。
    """
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    all_indices = np.arange(len(y_array))
    if test_size <= 0:
        return all_indices, np.asarray([], dtype=int)
    if test_size >= 1:
        raise ValueError("test_size must be smaller than 1")

    unique_groups = np.unique(groups_array)
    unique_labels = np.unique(y_array)

    # 至少需要 2 个视频、2 个类别，才有意义划分训练/验证。
    # 数据太少时就全部用于训练，先把流程跑通。
    if len(unique_groups) < 2 or len(unique_labels) < 2:
        return all_indices, np.asarray([], dtype=int)

    # 多尝试几种随机分组，优先选择训练集和验证集都包含全部类别的划分。
    # 这对小数据集很重要：例如现在只有 2 个 Fall 视频、4 个 ADL 视频，
    # 如果只随机一次，验证集可能刚好全是 Normal，看不到模型对 Fall 的效果。
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    fallback_split = None
    for train_index, test_index in splitter.split(all_indices, y_array, groups_array):
        train_labels = np.unique(y_array[train_index])
        test_labels = np.unique(y_array[test_index])
        if fallback_split is None and len(train_labels) >= 2:
            fallback_split = (train_index, test_index)
        if set(train_labels) == set(unique_labels) and set(test_labels) == set(unique_labels):
            return train_index, test_index

    # 如果数据太少，实在找不到"训练/验证都包含全部类别"的划分，
    # 就退回到至少训练集包含多个类别的划分；否则全部用于训练，不做验证。
    if fallback_split is not None:
        return fallback_split
    return all_indices, np.asarray([], dtype=int)


if __name__ == "__main__":
    main()
