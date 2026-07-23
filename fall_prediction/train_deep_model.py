"""Train a lightweight causal TCN on the existing fall-prediction windows."""

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

from .deep_dataset import (
    fit_feature_normalizer,
    normalize_temporal_features,
    preserve_temporal_shape,
)
from .deep_model import build_temporal_conv_net
from .ml_features import ACCEL_FEATURE_COLUMNS, ML_FEATURE_COLUMNS
from .train_model import (
    _group_train_test_split,
    build_validation_metrics,
    collect_csv_paths,
    json_ready,
    tune_prefall_alert_threshold_on_validation,
)
from .window_dataset import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, build_window_dataset


LABELS = ("Normal", "Pre-fall", "Fall")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal TCN fall-state classifier.")
    parser.add_argument("csv_paths", nargs="*", help="Feature CSV paths.")
    parser.add_argument("--input-dir", default=None, help="Directory recursively containing feature CSVs.")
    parser.add_argument("--output", default="models/tcn_prefall_classifier.pt")
    parser.add_argument("--metrics-output", default="reports/tcn_prefall_holdout_metrics.json")
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--channels", default="32,32,32")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--pooling", choices=("last", "last_mean_max"), default="last")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--prefall-weight", type=float, default=8.0)
    parser.add_argument("--fall-weight", type=float, default=1.0)
    parser.add_argument("--prefall-threshold-beta", type=float, default=1.5)
    parser.add_argument("--prefall-alert-threshold", type=float, default=None)
    parser.add_argument("--baseline-metrics", default="reports/current_model_holdout_metrics.json")
    parser.add_argument("--use-accel", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    csv_paths = collect_csv_paths(args.csv_paths, args.input_dir)
    if not csv_paths:
        raise RuntimeError("No feature CSV files were found.")
    channels = tuple(int(value) for value in args.channels.split(",") if value.strip())
    if not channels or any(value <= 0 for value in channels):
        parser.error("--channels must contain positive comma-separated integers")

    feature_columns = ACCEL_FEATURE_COLUMNS if args.use_accel else ML_FEATURE_COLUMNS
    flat_dataset = build_window_dataset(
        csv_paths=csv_paths,
        window_size=args.window_size,
        stride=args.stride,
        feature_columns=ML_FEATURE_COLUMNS,
        label_mode="annotations",
        annotations_path=args.annotations,
        use_accel=args.use_accel,
    )
    temporal_dataset = preserve_temporal_shape(flat_dataset, args.window_size, feature_columns)
    result = train_and_save_tcn(
        X=temporal_dataset.X,
        y=temporal_dataset.y,
        groups=temporal_dataset.groups,
        csv_paths=csv_paths,
        feature_columns=feature_columns,
        output_path=args.output,
        metrics_output_path=args.metrics_output,
        window_size=args.window_size,
        stride=args.stride,
        test_size=args.test_size,
        random_state=args.random_state,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        pooling=args.pooling,
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
        prefall_threshold_beta=args.prefall_threshold_beta,
        prefall_alert_threshold=args.prefall_alert_threshold,
        baseline_metrics_path=args.baseline_metrics,
        use_accel=args.use_accel,
    )
    metrics = result.get("validation_metrics") or {}
    if metrics:
        print(
            f"Saved {args.output}; validation accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    else:
        print(f"Saved full-data model {args.output}; no holdout metrics were computed")


def train_and_save_tcn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    csv_paths: Sequence[Path],
    feature_columns: Sequence[str],
    output_path: str | Path,
    metrics_output_path: str | Path,
    window_size: int,
    stride: int,
    test_size: float,
    random_state: int,
    channels: Sequence[int],
    kernel_size: int,
    dropout: float,
    pooling: str,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: dict[str, float],
    prefall_threshold_beta: float,
    prefall_alert_threshold: float | None = None,
    baseline_metrics_path: str | Path | None = None,
    use_accel: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to train the TCN model.") from exc

    _seed_everything(random_state, torch)
    train_index, validation_index = _group_train_test_split(y, groups, test_size, random_state)
    mean, std = fit_feature_normalizer(X[train_index])
    X_normalized = normalize_temporal_features(X, mean, std)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    unknown_labels = sorted(set(str(label) for label in y) - set(label_to_index))
    if unknown_labels:
        raise ValueError(f"Unsupported labels for the three-state TCN: {unknown_labels}")
    y_indices = np.asarray([label_to_index[str(label)] for label in y], dtype=np.int64)

    train_dataset = TensorDataset(
        torch.from_numpy(X_normalized[train_index]),
        torch.from_numpy(y_indices[train_index]),
    )
    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    has_validation = len(validation_index) > 0
    validation_X = torch.from_numpy(X_normalized[validation_index]) if has_validation else None
    validation_y = y[validation_index] if has_validation else None

    model_config = {
        "num_features": int(X.shape[2]),
        "num_classes": len(LABELS),
        "channels": [int(value) for value in channels],
        "kernel_size": int(kernel_size),
        "dropout": float(dropout),
        "pooling": str(pooling),
    }
    model = build_temporal_conv_net(**model_config)
    weight_tensor = torch.tensor(
        [float(class_weights.get(label, 1.0)) for label in LABELS], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] | None = None
    best_probabilities: np.ndarray | None = None
    best_epoch = 0
    best_score = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_X)
            sample_count += len(batch_X)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": loss_sum / max(sample_count, 1),
        }
        if not has_validation:
            history.append(epoch_summary)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch={epoch:03d} loss={epoch_summary['train_loss']:.4f}")
            continue

        probabilities = _predict_probabilities(model, validation_X, batch_size, torch)
        predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        metrics = build_validation_metrics(validation_y, predictions, LABELS)
        score = float(metrics["macro_f1"])
        epoch_summary.update(
            {
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": score,
                "validation_prefall_f1": metrics["classification_report"]["Pre-fall"]["f1-score"],
                "validation_prefall_recall": metrics["classification_report"]["Pre-fall"]["recall"],
            }
        )
        history.append(epoch_summary)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} loss={epoch_summary['train_loss']:.4f} "
                f"macro_f1={score:.4f} "
                f"prefall_recall={epoch_summary['validation_prefall_recall']:.4f}"
            )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_probabilities = probabilities.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    threshold_search = None
    if has_validation:
        if best_metrics is None or best_probabilities is None:
            raise RuntimeError("TCN training did not produce validation metrics.")
        best_predictions = np.asarray(LABELS)[best_probabilities.argmax(axis=1)]
        threshold_search = tune_prefall_alert_threshold_on_validation(
            y_true=validation_y,
            y_pred=best_predictions,
            classes=LABELS,
            probabilities=best_probabilities,
            beta=prefall_threshold_beta,
        )
        best_threshold = threshold_search.get("best")
        if best_threshold:
            prefall_alert_threshold = float(best_threshold["threshold"])
    if prefall_alert_threshold is None:
        prefall_alert_threshold = 0.25
    if not 0.0 <= prefall_alert_threshold <= 1.0:
        raise ValueError("prefall_alert_threshold must be between 0 and 1")

    created_at = datetime.now().isoformat(timespec="seconds")
    artifact = {
        "artifact_type": "pytorch_tcn",
        "format_version": 1,
        "state_dict": best_state,
        "model_config": model_config,
        "classes": list(LABELS),
        "window_size": int(window_size),
        "stride": int(stride),
        "feature_columns": list(feature_columns),
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "baseline_frames": 15,
        "smoothing_window": 5,
        "prefall_alert_threshold": prefall_alert_threshold,
        "prefall_alert_consecutive_frames": 1,
        "use_accel": bool(use_accel),
        "use_standing_calibration": False,
        "use_upper_body_features": False,
        "created_at": created_at,
        "best_epoch": int(best_epoch),
        "validation_metrics": best_metrics,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    split_summary = {
        "train_samples": int(len(train_index)),
        "validation_samples": int(len(validation_index)),
        "train_groups": sorted({str(group) for group in groups[train_index]}),
        "validation_groups": sorted({str(group) for group in groups[validation_index]}),
        "train_label_counts": dict(sorted(Counter(str(label) for label in y[train_index]).items())),
        "validation_label_counts": dict(
            sorted(Counter(str(label) for label in y[validation_index]).items())
        ),
    }
    report = {
        "created_at": created_at,
        "model": "causal_tcn",
        "model_config": model_config,
        "window_size": int(window_size),
        "stride": int(stride),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "total_samples": int(len(X)),
        "training_videos": [str(path) for path in csv_paths],
        "class_weights": class_weights,
        "best_epoch": int(best_epoch),
        "training_history": history,
        "validation_split": split_summary,
        "validation_metrics": best_metrics,
        "prefall_alert_threshold_search": threshold_search,
        "prefall_alert_threshold": prefall_alert_threshold,
        "baseline_comparison": (
            _baseline_comparison(best_metrics, baseline_metrics_path) if best_metrics else None
        ),
    }
    metrics_output = Path(metrics_output_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _predict_probabilities(model, values, batch_size: int, torch_module) -> np.ndarray:
    model.eval()
    batches = []
    with torch_module.inference_mode():
        for start in range(0, len(values), batch_size):
            logits = model(values[start : start + batch_size])
            batches.append(torch_module.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _seed_everything(random_state: int, torch_module) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch_module.manual_seed(random_state)
    try:
        torch_module.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def _baseline_comparison(
    metrics: dict[str, Any],
    baseline_metrics_path: str | Path | None,
) -> dict[str, Any] | None:
    if baseline_metrics_path is None or not Path(baseline_metrics_path).exists():
        return None
    baseline_report = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    baseline = baseline_report.get("validation_metrics")
    if not baseline:
        return None
    tcn_prefall = metrics["classification_report"].get("Pre-fall", {})
    baseline_prefall = baseline["classification_report"].get("Pre-fall", {})
    return {
        "baseline_path": str(baseline_metrics_path),
        "accuracy_delta": float(metrics["accuracy"] - baseline["accuracy"]),
        "macro_f1_delta": float(metrics["macro_f1"] - baseline["macro_f1"]),
        "prefall_f1_delta": float(
            tcn_prefall.get("f1-score", 0.0) - baseline_prefall.get("f1-score", 0.0)
        ),
        "prefall_recall_delta": float(
            tcn_prefall.get("recall", 0.0) - baseline_prefall.get("recall", 0.0)
        ),
    }


if __name__ == "__main__":
    main()
