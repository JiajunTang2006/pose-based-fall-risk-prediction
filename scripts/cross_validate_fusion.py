"""Five-fold grouped cross-validation for tree and skeleton-feature fusion models."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.deep_dataset import normalize_temporal_features
from fall_prediction.fusion_model import build_skeleton_feature_fusion_net
from fall_prediction.ml_predictor import HMMStateSmoother
from fall_prediction.skeleton_dataset import (
    build_paired_temporal_dataset,
    normalize_skeleton_windows,
)
from fall_prediction.train_deep_model import LABELS
from fall_prediction.train_fusion_model import train_and_save_fusion
from fall_prediction.train_model import (
    build_sample_weights,
    build_validation_metrics,
    collect_csv_paths,
    create_classifier,
    json_ready,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped cross-validation for fall prediction.")
    parser.add_argument("--input-dir", default="outputs/features")
    parser.add_argument("--landmark-dir", action="append", default=None)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--output", default="reports/fusion_grouped_5fold_cv.json")
    parser.add_argument("--model-dir", default="models/cross_validation")
    parser.add_argument("--fold-report-dir", default="reports/cross_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--inner-validation-size", type=float, default=0.20)
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    feature_paths = collect_csv_paths([], args.input_dir)
    landmark_dirs = args.landmark_dir or [
        "outputs/landmarks_upperbody/urfall_yolo",
        "outputs/landmarks_upperbody/upfall_yolo",
    ]
    dataset = build_paired_temporal_dataset(
        feature_csv_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        annotations_path=args.annotations,
        window_size=args.window_size,
        stride=args.stride,
        use_accel=True,
    )

    splits = grouped_stratified_splits(
        dataset.y,
        dataset.groups,
        n_splits=args.folds,
        random_state=args.random_state,
    )
    model_dir = Path(args.model_dir)
    fold_report_dir = Path(args.fold_report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_report_dir.mkdir(parents=True, exist_ok=True)

    fold_reports = []
    pooled_true: list[str] = []
    pooled_tree: list[str] = []
    pooled_fusion: list[str] = []
    pooled_fusion_hmm: list[str] = []

    for fold_number, (outer_train, outer_test) in enumerate(splits, start=1):
        fold_seed = args.random_state + fold_number
        print(
            f"\n=== fold {fold_number}/{args.folds}: "
            f"outer train={len(outer_train)}, test={len(outer_test)} ===",
            flush=True,
        )
        fold_model = model_dir / f"fusion_fold_{fold_number}.pt"
        inner_report_path = fold_report_dir / f"fusion_fold_{fold_number}_inner.json"
        inner_report = train_and_save_fusion(
            features=dataset.features[outer_train],
            skeletons=dataset.skeletons[outer_train],
            y=dataset.y[outer_train],
            groups=dataset.groups[outer_train],
            feature_columns=dataset.feature_columns,
            feature_csv_paths=feature_paths,
            landmark_dirs=landmark_dirs,
            output_path=fold_model,
            metrics_output_path=inner_report_path,
            mode="fusion",
            window_size=args.window_size,
            stride=args.stride,
            test_size=args.inner_validation_size,
            random_state=fold_seed,
            graph_channels=(16, 32, 32),
            temporal_channels=(32, 32),
            dropout=0.30,
            batch_size=64,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=8e-4,
            weight_decay=2e-4,
            class_weights={"Normal": 1.0, "Pre-fall": 5.0, "Fall": 1.0},
        )
        probabilities = predict_fusion_artifact(
            fold_model,
            dataset.features[outer_test],
            dataset.skeletons[outer_test],
        )
        fusion_predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        fusion_hmm_predictions = apply_hmm_by_sequence(
            probabilities,
            dataset.sequences[outer_test],
        )
        tree_predictions = train_and_predict_tree(
            dataset.features[outer_train],
            dataset.y[outer_train],
            dataset.features[outer_test],
            random_state=fold_seed,
        )
        true_labels = dataset.y[outer_test]
        tree_metrics = build_validation_metrics(true_labels, tree_predictions, LABELS)
        fusion_metrics = build_validation_metrics(true_labels, fusion_predictions, LABELS)
        fusion_hmm_metrics = build_validation_metrics(
            true_labels, fusion_hmm_predictions, LABELS
        )
        fold_report = {
            "fold": fold_number,
            "seed": fold_seed,
            "outer_train_samples": int(len(outer_train)),
            "outer_test_samples": int(len(outer_test)),
            "outer_train_groups": sorted(
                {str(group) for group in dataset.groups[outer_train]}
            ),
            "outer_test_groups": sorted(
                {str(group) for group in dataset.groups[outer_test]}
            ),
            "outer_test_label_counts": dict(
                sorted(Counter(str(label) for label in true_labels).items())
            ),
            "inner_best_epoch": inner_report["best_epoch"],
            "inner_validation_metrics": inner_report["validation_metrics"],
            "tree_metrics": tree_metrics,
            "fusion_metrics": fusion_metrics,
            "fusion_hmm_metrics": fusion_hmm_metrics,
        }
        fold_reports.append(fold_report)
        pooled_true.extend(str(value) for value in true_labels)
        pooled_tree.extend(str(value) for value in tree_predictions)
        pooled_fusion.extend(str(value) for value in fusion_predictions)
        pooled_fusion_hmm.extend(str(value) for value in fusion_hmm_predictions)
        print_fold_summary(fold_report)

    report = {
        "method": "grouped_stratified_cross_validation_with_inner_grouped_early_stopping",
        "folds": args.folds,
        "random_state": args.random_state,
        "window_size": args.window_size,
        "stride": args.stride,
        "sample_count": int(len(dataset.y)),
        "group_count": int(len(set(dataset.groups))),
        "sequence_count": int(len(set(dataset.sequences))),
        "leakage_controls": {
            "outer_split_unit": "video/trial group",
            "upfall_cameras_same_outer_fold": True,
            "outer_test_used_for_early_stopping": False,
            "inner_validation_size": args.inner_validation_size,
            "note": (
                "Fusion folds are trained on the inner-training subset and selected on inner validation; "
                "the untouched outer fold is used only once for final evaluation."
            ),
        },
        "fold_reports": fold_reports,
        "aggregate": {
            "tree": aggregate_fold_metrics(fold_reports, "tree_metrics"),
            "fusion": aggregate_fold_metrics(fold_reports, "fusion_metrics"),
            "fusion_hmm": aggregate_fold_metrics(fold_reports, "fusion_hmm_metrics"),
        },
        "pooled_out_of_fold_metrics": {
            "tree": build_validation_metrics(pooled_true, pooled_tree, LABELS),
            "fusion": build_validation_metrics(pooled_true, pooled_fusion, LABELS),
            "fusion_hmm": build_validation_metrics(
                pooled_true, pooled_fusion_hmm, LABELS
            ),
        },
        "fold_win_counts": fold_win_counts(fold_reports),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== aggregate ===")
    print_aggregate(report)
    print(f"Wrote {output}")


def grouped_stratified_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder = np.zeros((len(labels), 1), dtype=np.float32)
    splits = list(splitter.split(placeholder, labels, groups))
    all_labels = set(str(label) for label in labels)
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        test_labels = set(str(label) for label in labels[test_index])
        if test_labels != all_labels:
            raise RuntimeError(
                f"Outer fold {fold} does not contain all labels: {sorted(test_labels)}"
            )
        if set(groups[train_index]) & set(groups[test_index]):
            raise RuntimeError(f"Group leakage detected in fold {fold}")
    return splits


def predict_fusion_artifact(
    model_path: str | Path,
    features: np.ndarray,
    skeletons: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    import torch

    try:
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location="cpu")
    feature_mean = np.asarray(artifact["feature_normalizer_mean"], dtype=np.float32)
    feature_std = np.asarray(artifact["feature_normalizer_std"], dtype=np.float32)
    skeleton_mean = np.asarray(artifact["skeleton_normalizer_mean"], dtype=np.float32)
    skeleton_std = np.asarray(artifact["skeleton_normalizer_std"], dtype=np.float32)
    normalized_features = normalize_temporal_features(features, feature_mean, feature_std)
    normalized_skeletons = normalize_skeleton_windows(skeletons, skeleton_mean, skeleton_std)
    model = build_skeleton_feature_fusion_net(**artifact["model_config"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    probability_temperature = float(artifact.get("probability_temperature", 1.0))
    if probability_temperature <= 0.0:
        raise ValueError("probability_temperature must be positive")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = model(
                torch.from_numpy(normalized_skeletons[start : start + batch_size]),
                torch.from_numpy(normalized_features[start : start + batch_size]),
            )
            batches.append(
                torch.softmax(logits / probability_temperature, dim=1).numpy()
            )
    return np.concatenate(batches, axis=0)


def train_and_predict_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = create_classifier("hist_gradient_boosting", random_state)
    flattened_train = train_features.reshape(len(train_features), -1)
    flattened_test = test_features.reshape(len(test_features), -1)
    weights = build_sample_weights(
        train_labels,
        {"Normal": 1.0, "Pre-fall": 8.0, "Fall": 1.0},
    )
    model.fit(flattened_train, train_labels, sample_weight=weights)
    return model.predict(flattened_test)


def apply_hmm_by_sequence(
    probabilities: np.ndarray,
    sequences: np.ndarray,
) -> list[str]:
    smoothers: dict[str, HMMStateSmoother] = {}
    predictions = []
    for probability_row, sequence in zip(probabilities, sequences):
        key = str(sequence)
        smoother = smoothers.setdefault(key, HMMStateSmoother())
        predictions.append(smoother.smooth(probability_row.tolist()))
    return predictions


def aggregate_fold_metrics(
    fold_reports: Sequence[dict[str, Any]],
    metrics_key: str,
) -> dict[str, Any]:
    paths = {
        "accuracy": ("accuracy",),
        "macro_f1": ("macro_f1",),
        "normal_recall": ("classification_report", "Normal", "recall"),
        "prefall_precision": ("classification_report", "Pre-fall", "precision"),
        "prefall_recall": ("classification_report", "Pre-fall", "recall"),
        "prefall_f1": ("classification_report", "Pre-fall", "f1-score"),
        "fall_precision": ("classification_report", "Fall", "precision"),
        "fall_recall": ("classification_report", "Fall", "recall"),
        "fall_f1": ("classification_report", "Fall", "f1-score"),
    }
    result = {}
    for name, path in paths.items():
        values = []
        for fold in fold_reports:
            value: Any = fold[metrics_key]
            for key in path:
                value = value[key]
            values.append(float(value))
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "values": values,
        }
    return result


def fold_win_counts(fold_reports: Sequence[dict[str, Any]]) -> dict[str, int]:
    result = {"fusion_vs_tree_macro_f1": 0, "fusion_hmm_vs_tree_macro_f1": 0}
    for fold in fold_reports:
        tree = fold["tree_metrics"]["macro_f1"]
        if fold["fusion_metrics"]["macro_f1"] > tree:
            result["fusion_vs_tree_macro_f1"] += 1
        if fold["fusion_hmm_metrics"]["macro_f1"] > tree:
            result["fusion_hmm_vs_tree_macro_f1"] += 1
    return result


def print_fold_summary(fold: dict[str, Any]) -> None:
    for label, key in (
        ("tree", "tree_metrics"),
        ("fusion", "fusion_metrics"),
        ("fusion+hmm", "fusion_hmm_metrics"),
    ):
        metrics = fold[key]
        prefall = metrics["classification_report"]["Pre-fall"]
        print(
            f"  {label:10s} acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"prefall_recall={prefall['recall']:.4f}",
            flush=True,
        )


def print_aggregate(report: dict[str, Any]) -> None:
    for model_name in ("tree", "fusion", "fusion_hmm"):
        summary = report["aggregate"][model_name]
        print(
            f"{model_name:10s} "
            f"acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f} "
            f"macro_f1={summary['macro_f1']['mean']:.4f}±{summary['macro_f1']['std']:.4f} "
            f"prefall_recall={summary['prefall_recall']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
