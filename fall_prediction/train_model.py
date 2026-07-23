

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .ml_features import ACCEL_FEATURE_COLUMNS, ML_FEATURE_COLUMNS
from .predictor import PredictorConfig
from .robustness import (
    ROBUST_ACCEL_FEATURE_COLUMNS,
    ROBUST_ML_FEATURE_COLUMNS,
    UPPER_BODY_ACCEL_FEATURE_COLUMNS,
    UPPER_BODY_ML_FEATURE_COLUMNS,
)
from .window_dataset import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, build_window_dataset


DEFAULT_PREDICTOR_CONFIG = PredictorConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a machine-learning fall classifier from feature CSV files.")
    parser.add_argument("csv_paths", nargs="*", help="One or more feature CSV paths.")
    parser.add_argument("--input-dir", default=None, help="Optional directory scanned recursively for feature CSV files.")
    parser.add_argument(
        "--output",
        default="models/yolo_tail60_prefall_accel_classifier.joblib",
        help="Output path for the trained joblib model.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Validation metrics JSON path; defaults to a .metrics.json file beside the model.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Frames per training sample; 15 frames is about 0.5 seconds.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Sliding-window step in frames.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=DEFAULT_PREDICTOR_CONFIG.baseline_frames,
        help="Frames used to establish the body-center baseline during inference.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_PREDICTOR_CONFIG.smoothing_window,
        help="Smoothing window for smoothed_risk_score during inference.",
    )
    parser.add_argument(
        "--classifier",
        choices=("random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"),
        default="random_forest",
        help="scikit-learn classifier to train.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("filename", "annotations"),
        default="filename",
        help="Label source: infer from filenames or use frame-interval annotations.",
    )
    parser.add_argument(
        "--annotations",
        action="append",
        default=None,
        help="Frame-interval annotation CSV with video,start_frame,end_frame,label columns. May be repeated.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Group-wise validation fraction. Set to 0 to train on all data without validation.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible splitting and training.")
    parser.add_argument("--normal-weight", type=float, default=1.0, help="Training weight for Normal samples.")
    parser.add_argument("--fall-weight", type=float, default=1.0, help="Training weight for Fall samples.")
    parser.add_argument("--prefall-weight", type=float, default=1.0, help="Training weight for Pre-fall samples.")
    parser.add_argument(
        "--tune-prefall-alert-threshold",
        action="store_true",
        help="Tune the Pre-fall alert threshold on validation data and store it in the model artifact.",
    )
    parser.add_argument(
        "--prefall-threshold-beta",
        type=float,
        default=1.5,
        help="F-beta beta used to tune the Pre-fall alert threshold.",
    )
    parser.add_argument(
        "--prefall-alert-threshold",
        type=float,
        default=None,
        help="Store an explicit Pre-fall probability threshold when training on the full dataset.",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="Enable acceleration features (torso_angular_accel and vertical_accel).",
    )
    parser.add_argument(
        "--use-standing-calibration",
        action="store_true",
        help="Calibrate angle, scale, and motion features against each sequence's initial standing pose.",
    )
    parser.add_argument(
        "--partial-pose-augmentation",
        action="store_true",
        help="Simulate torso, center, bounding-box, and short temporal occlusions during training.",
    )
    parser.add_argument(
        "--use-upper-body-features",
        action="store_true",
        help="Add shoulder-center, shoulder-rotation, and upper-body bounding-box features.",
    )
    args = parser.parse_args()
    if args.partial_pose_augmentation and not args.use_standing_calibration:
        parser.error("--partial-pose-augmentation requires --use-standing-calibration")
    if args.use_upper_body_features and not args.use_standing_calibration:
        parser.error("--use-upper-body-features requires --use-standing-calibration")


    csv_paths = collect_csv_paths(args.csv_paths, args.input_dir)
    if not csv_paths:
        raise RuntimeError("No feature CSV files found. Run export_dataset_features.py first.")


    dataset = build_window_dataset(
        csv_paths=csv_paths,
        window_size=args.window_size,
        stride=args.stride,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode=args.label_mode,
        annotations_path=args.annotations,
        use_accel=args.use_accel,
        use_standing_calibration=args.use_standing_calibration,
        partial_pose_augmentation=args.partial_pose_augmentation,
        baseline_frames=args.baseline_frames,
        use_upper_body_features=args.use_upper_body_features,
    )
    if not dataset.X:
        raise RuntimeError("No training windows were generated. Check the labels, window size, and CSV contents.")


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
        saved_feature_columns=(
            UPPER_BODY_ACCEL_FEATURE_COLUMNS
            if args.use_upper_body_features and args.use_accel
            else UPPER_BODY_ML_FEATURE_COLUMNS
            if args.use_upper_body_features
            else ROBUST_ACCEL_FEATURE_COLUMNS
            if args.use_standing_calibration and args.use_accel
            else ROBUST_ML_FEATURE_COLUMNS
            if args.use_standing_calibration
            else ACCEL_FEATURE_COLUMNS
            if args.use_accel
            else ML_FEATURE_COLUMNS
        ),
        use_standing_calibration=args.use_standing_calibration,
        partial_pose_augmentation=args.partial_pose_augmentation,
        use_upper_body_features=args.use_upper_body_features,
    )


def collect_csv_paths(paths: list[str], input_dir: str | None) -> list[Path]:

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
    saved_feature_columns: Sequence[str] | None = None,
    use_standing_calibration: bool = False,
    partial_pose_augmentation: bool = False,
    use_upper_body_features: bool = False,
) -> dict[str, Any]:


    try:
        import joblib
        import numpy as np
        from sklearn.metrics import classification_report
    except ImportError as exc:
        raise RuntimeError(
            "Training requires numpy, scikit-learn, and joblib. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc


    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)


    # Split by video group so overlapping windows cannot leak across train and validation.
    train_index, test_index = _group_train_test_split(
        y_array=y_array,
        groups_array=groups_array,
        test_size=test_size,
        random_state=random_state,
    )


    model = create_classifier(classifier_name, random_state)
    sample_weight = build_sample_weights(y_array[train_index], class_weights)
    if sample_weight is None:
        model.fit(X_array[train_index], y_array[train_index])
    else:
        model.fit(X_array[train_index], y_array[train_index], sample_weight=sample_weight)

    print(f"Training samples: {len(train_index)}")
    print(f"Classifier: {classifier_name}")
    if sample_weight is not None:
        print(f"Sample weights: {normalized_class_weights(class_weights)}")
    print(f"Classes: {', '.join(str(label) for label in model.classes_)}")

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
        print("\nValidation report:")
        print(classification_report(y_array[test_index], predictions, zero_division=0))
        if prefall_alert_threshold_search is not None and prefall_alert_threshold_search.get("best") is not None:
            best = prefall_alert_threshold_search["best"]
            print(
                "\nPre-fall alert threshold search:"
                f" threshold={best['threshold']:.2f}, precision={best['precision']:.3f},"
                f" recall={best['recall']:.3f}, f_beta={best['f_beta']:.3f}"
            )
    else:
        print("\nValidation skipped: too few videos or classes for a group-wise split.")

    created_at = datetime.now().isoformat(timespec="seconds")
    validation_split = build_validation_split_summary(
        y_array=y_array,
        groups_array=groups_array,
        train_index=train_index,
        test_index=test_index,
    )


    if saved_feature_columns is None:
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
        "use_standing_calibration": bool(use_standing_calibration),
        "partial_pose_augmentation": bool(partial_pose_augmentation),
        "use_upper_body_features": bool(use_upper_body_features),
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
    print(f"\nModel saved to: {output}")

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
            use_standing_calibration=use_standing_calibration,
            partial_pose_augmentation=partial_pose_augmentation,
            use_upper_body_features=use_upper_body_features,
        ),
    )
    print(f"Validation metrics saved to: {metrics_output}")
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
    use_standing_calibration: bool = False,
    partial_pose_augmentation: bool = False,
    use_upper_body_features: bool = False,
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
        "use_standing_calibration": bool(use_standing_calibration),
        "partial_pose_augmentation": bool(partial_pose_augmentation),
        "use_upper_body_features": bool(use_upper_body_features),
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

    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    if classifier_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_name == "extra_trees":

        return ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_name == "gradient_boosting":

        return GradientBoostingClassifier(random_state=random_state)

    if classifier_name == "hist_gradient_boosting":

        return HistGradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"Unknown classifier: {classifier_name}")


def _group_train_test_split(y_array, groups_array, test_size: float, random_state: int):

    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    all_indices = np.arange(len(y_array))
    if test_size <= 0:
        return all_indices, np.asarray([], dtype=int)
    if test_size >= 1:
        raise ValueError("test_size must be smaller than 1")

    unique_groups = np.unique(groups_array)
    unique_labels = np.unique(y_array)


    if len(unique_groups) < 2 or len(unique_labels) < 2:
        return all_indices, np.asarray([], dtype=int)


    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    fallback_split = None
    for train_index, test_index in splitter.split(all_indices, y_array, groups_array):
        train_labels = np.unique(y_array[train_index])
        test_labels = np.unique(y_array[test_index])
        if fallback_split is None and len(train_labels) >= 2:
            fallback_split = (train_index, test_index)
        if set(train_labels) == set(unique_labels) and set(test_labels) == set(unique_labels):
            return train_index, test_index


    if fallback_split is not None:
        return fallback_split
    return all_indices, np.asarray([], dtype=int)


if __name__ == "__main__":
    main()
