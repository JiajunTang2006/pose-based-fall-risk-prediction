"""Train a compact skeleton ST-GCN with optional engineered-feature fusion."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .deep_dataset import fit_feature_normalizer, normalize_temporal_features
from .fusion_model import build_skeleton_feature_fusion_net
from .probability_calibration import (
    apply_temperature_scaling,
    fit_temperature_scaling,
    tune_prefall_alert_threshold_with_recall_floor,
)
from .skeleton_dataset import (
    SKELETON_CHANNELS,
    build_paired_temporal_dataset,
    fit_skeleton_normalizer,
    normalize_skeleton_windows,
)
from .train_deep_model import LABELS
from .train_model import (
    _group_train_test_split,
    build_validation_metrics,
    collect_csv_paths,
    json_ready,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a skeleton ST-GCN or ST-GCN+TCN fusion model.")
    parser.add_argument("csv_paths", nargs="*", help="Feature CSV paths.")
    parser.add_argument("--input-dir", default="outputs/features")
    parser.add_argument(
        "--landmark-dir",
        action="append",
        default=None,
        help="Directory containing *_landmarks.csv; repeat for multiple directories.",
    )
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--output", default="models/skeleton_feature_fusion_holdout.pt")
    parser.add_argument("--metrics-output", default="reports/skeleton_feature_fusion_holdout.json")
    parser.add_argument("--mode", choices=("skeleton", "fusion"), default="fusion")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--graph-channels", default="16,32,32")
    parser.add_argument("--temporal-channels", default="32,32")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--prefall-weight", type=float, default=5.0)
    parser.add_argument("--fall-weight", type=float, default=1.0)
    parser.add_argument("--prefall-alert-threshold", type=float, default=None)
    parser.add_argument(
        "--prefall-recall-floor",
        type=float,
        default=0.80,
        help="校准报警阈值时要求的最低Pre-fall召回率，默认0.80。",
    )
    parser.add_argument(
        "--probability-temperature",
        type=float,
        default=None,
        help="显式概率温度；有验证集时默认自动拟合。",
    )
    args = parser.parse_args()

    feature_paths = collect_csv_paths(args.csv_paths, args.input_dir)
    landmark_dirs = args.landmark_dir or [
        "outputs/landmarks_upperbody/urfall_yolo",
        "outputs/landmarks_upperbody/upfall_yolo",
    ]
    graph_channels = _parse_channels(args.graph_channels, "--graph-channels")
    temporal_channels = _parse_channels(args.temporal_channels, "--temporal-channels")
    dataset = build_paired_temporal_dataset(
        feature_csv_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        annotations_path=args.annotations,
        window_size=args.window_size,
        stride=args.stride,
        use_accel=True,
    )
    report = train_and_save_fusion(
        features=dataset.features,
        skeletons=dataset.skeletons,
        y=dataset.y,
        groups=dataset.groups,
        feature_columns=dataset.feature_columns,
        feature_csv_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        output_path=args.output,
        metrics_output_path=args.metrics_output,
        mode=args.mode,
        window_size=args.window_size,
        stride=args.stride,
        test_size=args.test_size,
        random_state=args.random_state,
        graph_channels=graph_channels,
        temporal_channels=temporal_channels,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights={
            "Normal": args.normal_weight,
            "Pre-fall": args.prefall_weight,
            "Fall": args.fall_weight,
        },
        prefall_alert_threshold=args.prefall_alert_threshold,
        prefall_recall_floor=args.prefall_recall_floor,
        probability_temperature=args.probability_temperature,
    )
    metrics = report.get("validation_metrics")
    if metrics:
        print(
            f"Saved {args.output}; accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    else:
        print(f"Saved full-data fusion model {args.output}; no holdout metrics were computed")


def train_and_save_fusion(
    features: np.ndarray,
    skeletons: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_columns: Sequence[str],
    feature_csv_paths: Sequence[Path],
    landmark_dirs: Sequence[str | Path],
    output_path: str | Path,
    metrics_output_path: str | Path,
    mode: str,
    window_size: int,
    stride: int,
    test_size: float,
    random_state: int,
    graph_channels: Sequence[int],
    temporal_channels: Sequence[int],
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: dict[str, float],
    prefall_alert_threshold: float | None = None,
    prefall_recall_floor: float = 0.80,
    probability_temperature: float | None = None,
    sample_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for skeleton-fusion training.") from exc

    _seed_everything(random_state, torch)
    train_index, validation_index = _group_train_test_split(y, groups, test_size, random_state)
    feature_mean, feature_std = fit_feature_normalizer(features[train_index])
    skeleton_mean, skeleton_std = fit_skeleton_normalizer(skeletons[train_index])
    normalized_features = normalize_temporal_features(features, feature_mean, feature_std)
    normalized_skeletons = normalize_skeleton_windows(skeletons, skeleton_mean, skeleton_std)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    y_indices = np.asarray([label_to_index[str(label)] for label in y], dtype=np.int64)
    normalized_sample_weights = _normalize_sample_weights(sample_weights, len(y))

    train_dataset = TensorDataset(
        torch.from_numpy(normalized_skeletons[train_index]),
        torch.from_numpy(normalized_features[train_index]),
        torch.from_numpy(y_indices[train_index]),
        torch.from_numpy(normalized_sample_weights[train_index]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
        num_workers=0,
    )
    has_validation = len(validation_index) > 0
    validation_skeletons = (
        torch.from_numpy(normalized_skeletons[validation_index]) if has_validation else None
    )
    validation_features = (
        torch.from_numpy(normalized_features[validation_index]) if has_validation else None
    )
    validation_y = y[validation_index] if has_validation else None

    model_config = {
        "feature_count": int(features.shape[2]),
        "skeleton_channels": int(skeletons.shape[1]),
        "num_classes": len(LABELS),
        "graph_channels": [int(value) for value in graph_channels],
        "temporal_channels": [int(value) for value in temporal_channels],
        "dropout": float(dropout),
        "mode": mode,
    }
    model = build_skeleton_feature_fusion_net(**model_config)
    class_weight_tensor = torch.tensor(
        [float(class_weights.get(label, 1.0)) for label in LABELS], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] | None = None
    best_probabilities: np.ndarray | None = None
    best_epoch = 0
    best_score = float("-inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_skeletons, batch_features, batch_y, batch_sample_weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch_skeletons,
                batch_features if mode == "fusion" else None,
            )
            per_sample_loss = criterion(logits, batch_y)
            effective_weight_sum = (
                class_weight_tensor[batch_y] * batch_sample_weights
            ).sum().clamp_min(1e-12)
            loss = (per_sample_loss * batch_sample_weights).sum() / effective_weight_sum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            sample_count += len(batch_y)

        if not has_validation:
            summary = {
                "epoch": epoch,
                "train_loss": total_loss / max(sample_count, 1),
            }
            history.append(summary)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch={epoch:03d} loss={summary['train_loss']:.4f}")
            continue

        probabilities = _predict_probabilities(
            model,
            validation_skeletons,
            validation_features,
            mode,
            batch_size,
            torch,
        )
        predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        metrics = build_validation_metrics(validation_y, predictions, LABELS)
        prefall = metrics["classification_report"]["Pre-fall"]
        summary = {
            "epoch": epoch,
            "train_loss": total_loss / max(sample_count, 1),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_prefall_f1": prefall["f1-score"],
            "validation_prefall_recall": prefall["recall"],
        }
        history.append(summary)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} loss={summary['train_loss']:.4f} "
                f"macro_f1={summary['validation_macro_f1']:.4f} "
                f"prefall_recall={summary['validation_prefall_recall']:.4f}"
            )

        score = float(metrics["macro_f1"])
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_probabilities = probabilities.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    threshold_search = None
    probability_calibration = None
    if has_validation:
        if best_metrics is None or best_probabilities is None:
            raise RuntimeError("Fusion training did not produce validation metrics.")
        if probability_temperature is None:
            probability_calibration = fit_temperature_scaling(
                best_probabilities,
                validation_y,
                LABELS,
            )
            probability_temperature = float(
                probability_calibration["temperature"]
            )
        calibrated_probabilities = apply_temperature_scaling(
            best_probabilities,
            probability_temperature,
        )
        best_predictions = np.asarray(LABELS)[best_probabilities.argmax(axis=1)]
        threshold_search = tune_prefall_alert_threshold_with_recall_floor(
            y_true=validation_y,
            base_predictions=best_predictions,
            classes=LABELS,
            probabilities=calibrated_probabilities,
            recall_floor=prefall_recall_floor,
        )
        best_threshold = threshold_search.get("best")
        if best_threshold:
            prefall_alert_threshold = float(best_threshold["threshold"])
    if probability_temperature is None:
        probability_temperature = 1.0
    if probability_temperature <= 0.0:
        raise ValueError("probability_temperature must be positive")
    if prefall_alert_threshold is None:
        prefall_alert_threshold = 0.25
    if not 0.0 <= prefall_alert_threshold <= 1.0:
        raise ValueError("prefall_alert_threshold must be between 0 and 1")
    created_at = datetime.now().isoformat(timespec="seconds")
    artifact = {
        "artifact_type": "pytorch_skeleton_fusion",
        "format_version": 2,
        "state_dict": best_state,
        "model_config": model_config,
        "classes": list(LABELS),
        "window_size": int(window_size),
        "stride": int(stride),
        "feature_columns": list(feature_columns),
        "skeleton_channels": list(SKELETON_CHANNELS),
        "feature_normalizer_mean": feature_mean.tolist(),
        "feature_normalizer_std": feature_std.tolist(),
        "skeleton_normalizer_mean": skeleton_mean.tolist(),
        "skeleton_normalizer_std": skeleton_std.tolist(),
        "prefall_alert_threshold": prefall_alert_threshold,
        "probability_temperature": float(probability_temperature),
        "requires_skeleton": True,
        "use_accel": True,
        "created_at": created_at,
        "best_epoch": int(best_epoch),
        "validation_metrics": best_metrics,
        "probability_calibration": probability_calibration,
        "training_sample_weight_summary": _sample_weight_summary(
            normalized_sample_weights
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    split_summary = {
        "train_samples": int(len(train_index)),
        "validation_samples": int(len(validation_index)),
        "train_groups": sorted({str(group) for group in groups[train_index]}),
        "validation_groups": sorted({str(group) for group in groups[validation_index]}),
        "train_label_counts": dict(sorted(Counter(str(value) for value in y[train_index]).items())),
        "validation_label_counts": dict(
            sorted(Counter(str(value) for value in y[validation_index]).items())
        ),
    }
    report = {
        "created_at": created_at,
        "model": "small_stgcn_feature_tcn" if mode == "fusion" else "small_stgcn",
        "model_config": model_config,
        "window_size": int(window_size),
        "stride": int(stride),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "total_samples": int(len(y)),
        "feature_csv_paths": [str(path) for path in feature_csv_paths],
        "landmark_dirs": [str(path) for path in landmark_dirs],
        "class_weights": class_weights,
        "best_epoch": int(best_epoch),
        "training_history": history,
        "validation_split": split_summary,
        "validation_metrics": best_metrics,
        "prefall_alert_threshold_search": threshold_search,
        "prefall_alert_threshold": prefall_alert_threshold,
        "prefall_recall_floor": float(prefall_recall_floor),
        "probability_temperature": float(probability_temperature),
        "probability_calibration": probability_calibration,
        "training_sample_weight_summary": _sample_weight_summary(
            normalized_sample_weights
        ),
        "comparisons": _load_comparisons(best_metrics) if best_metrics else None,
    }
    metrics_output = Path(metrics_output_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _predict_probabilities(
    model,
    skeletons,
    features,
    mode: str,
    batch_size: int,
    torch_module,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch_module.inference_mode():
        for start in range(0, len(skeletons), batch_size):
            logits = model(
                skeletons[start : start + batch_size],
                features[start : start + batch_size] if mode == "fusion" else None,
            )
            batches.append(torch_module.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _parse_channels(value: str, option: str) -> tuple[int, ...]:
    try:
        channels = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated integers") from exc
    if not channels or any(channel <= 0 for channel in channels):
        raise ValueError(f"{option} must contain positive integers")
    return channels


def _normalize_sample_weights(
    sample_weights: Sequence[float] | None,
    sample_count: int,
) -> np.ndarray:
    if sample_weights is None:
        return np.ones(sample_count, dtype=np.float32)
    values = np.asarray(sample_weights, dtype=np.float32)
    if values.shape != (sample_count,):
        raise ValueError("sample_weights must have one value per training sample")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("sample_weights must contain finite positive values")
    return values


def _sample_weight_summary(sample_weights: np.ndarray) -> dict[str, Any]:
    counts = Counter(f"{float(value):.2f}" for value in sample_weights)
    return {
        "count": int(len(sample_weights)),
        "min": float(sample_weights.min()) if len(sample_weights) else 0.0,
        "max": float(sample_weights.max()) if len(sample_weights) else 0.0,
        "mean": float(sample_weights.mean()) if len(sample_weights) else 0.0,
        "counts": dict(sorted(counts.items())),
    }


def _seed_everything(random_state: int, torch_module) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch_module.manual_seed(random_state)
    try:
        torch_module.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def _load_comparisons(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    baselines = {
        "tree": Path("reports/current_model_holdout_metrics.json"),
        "feature_tcn": Path("reports/tcn_prefall_holdout_hmm_runtime_eval.json"),
    }
    for name, path in baselines.items():
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        baseline = report.get("validation_metrics")
        if baseline is None:
            baseline = report.get("runtime_state_metrics_all_windows")
        if not baseline:
            continue
        current_prefall = metrics["classification_report"]["Pre-fall"]
        baseline_prefall = baseline["classification_report"]["Pre-fall"]
        comparisons[name] = {
            "path": str(path),
            "accuracy_delta": metrics["accuracy"] - baseline["accuracy"],
            "macro_f1_delta": metrics["macro_f1"] - baseline["macro_f1"],
            "prefall_f1_delta": (
                current_prefall["f1-score"] - baseline_prefall["f1-score"]
            ),
            "prefall_recall_delta": current_prefall["recall"] - baseline_prefall["recall"],
        }
    return comparisons


if __name__ == "__main__":
    main()
