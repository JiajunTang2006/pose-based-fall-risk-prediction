"""Nested grouped CV for fusion class weight and probability calibration."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.probability_calibration import prefall_alert_predictions
from fall_prediction.skeleton_dataset import build_paired_temporal_dataset
from fall_prediction.train_fusion_model import train_and_save_fusion
from fall_prediction.train_model import build_validation_metrics, collect_csv_paths, json_ready
from scripts.cross_validate_fusion import (
    LABELS,
    aggregate_fold_metrics,
    apply_hmm_by_sequence,
    grouped_stratified_splits,
    predict_fusion_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested grouped selection of fusion Pre-fall weight and calibration."
    )
    parser.add_argument(
        "--source-report", default="reports/fusion_grouped_5fold_cv_full_outer.json"
    )
    parser.add_argument("--input-dir", default="outputs/features")
    parser.add_argument("--landmark-dir", action="append", default=None)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument(
        "--output", default="reports/fusion_nested_weight_calibration_5fold_cv.json"
    )
    parser.add_argument(
        "--final-model", default="models/skeleton_feature_fusion_tuned.pt"
    )
    parser.add_argument(
        "--final-report", default="reports/skeleton_feature_fusion_tuned_training.json"
    )
    parser.add_argument("--model-dir", default="models/cross_validation_tuned")
    parser.add_argument("--fold-report-dir", default="reports/cross_validation_tuned")
    parser.add_argument("--prefall-weights", default="2,3,4,5")
    parser.add_argument("--prefall-recall-floor", type=float, default=0.80)
    parser.add_argument("--inner-validation-size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed candidate/final fold artifacts when present.",
    )
    args = parser.parse_args()
    weights = parse_weights(args.prefall_weights)
    if not 0.0 <= args.prefall_recall_floor <= 1.0:
        parser.error("--prefall-recall-floor must be between 0 and 1")

    source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    feature_paths = collect_csv_paths([], args.input_dir)
    landmark_dirs = args.landmark_dir or [
        "outputs/landmarks_upperbody/urfall_yolo",
        "outputs/landmarks_upperbody/upfall_yolo",
    ]
    dataset = build_paired_temporal_dataset(
        feature_csv_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        annotations_path=args.annotations,
        window_size=int(source["window_size"]),
        stride=int(source["stride"]),
        use_accel=True,
    )
    splits = grouped_stratified_splits(
        dataset.y,
        dataset.groups,
        n_splits=int(source["folds"]),
        random_state=int(source["random_state"]),
    )
    model_dir = Path(args.model_dir)
    report_dir = Path(args.fold_report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, Any]] = []
    pooled = {"true": [], "raw": [], "hmm": [], "alert": []}
    selected_settings: list[dict[str, float | int]] = []
    for source_fold, (outer_train, outer_test) in zip(source["fold_reports"], splits):
        fold_number = int(source_fold["fold"])
        fold_seed = int(source_fold["seed"])
        print(
            f"\n=== outer fold {fold_number}/{source['folds']}: "
            f"train={len(outer_train)}, test={len(outer_test)} ===",
            flush=True,
        )
        candidates = []
        for weight in weights:
            weight_name = format_weight(weight)
            candidate_model = model_dir / f"fold_{fold_number}_weight_{weight_name}_inner.pt"
            candidate_report_path = (
                report_dir / f"fold_{fold_number}_weight_{weight_name}_inner.json"
            )
            if args.resume and candidate_model.exists() and candidate_report_path.exists():
                candidate_report = json.loads(
                    candidate_report_path.read_text(encoding="utf-8")
                )
                print(f"reuse weight={weight:g}: {candidate_report_path}", flush=True)
            else:
                print(f"train inner candidate weight={weight:g}", flush=True)
                candidate_report = train_and_save_fusion(
                    features=dataset.features[outer_train],
                    skeletons=dataset.skeletons[outer_train],
                    y=dataset.y[outer_train],
                    groups=dataset.groups[outer_train],
                    feature_columns=dataset.feature_columns,
                    feature_csv_paths=feature_paths,
                    landmark_dirs=landmark_dirs,
                    output_path=candidate_model,
                    metrics_output_path=candidate_report_path,
                    mode="fusion",
                    window_size=int(source["window_size"]),
                    stride=int(source["stride"]),
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
                    class_weights={"Normal": 1.0, "Pre-fall": weight, "Fall": 1.0},
                    prefall_recall_floor=args.prefall_recall_floor,
                )
            candidates.append(candidate_summary(weight, candidate_report))
        selected = select_candidate(candidates, args.prefall_recall_floor)
        selected_settings.append(
            {
                "weight": float(selected["weight"]),
                "epoch": int(selected["best_epoch"]),
                "temperature": float(selected["probability_temperature"]),
                "threshold": float(selected["prefall_alert_threshold"]),
            }
        )
        print(
            f"selected weight={selected['weight']:g}, epoch={selected['best_epoch']}, "
            f"T={selected['probability_temperature']:.3f}, "
            f"threshold={selected['prefall_alert_threshold']:.2f}",
            flush=True,
        )

        final_fold_model = model_dir / f"fold_{fold_number}_selected_full_outer.pt"
        final_fold_report = report_dir / f"fold_{fold_number}_selected_full_outer.json"
        if not (args.resume and final_fold_model.exists() and final_fold_report.exists()):
            train_and_save_fusion(
                features=dataset.features[outer_train],
                skeletons=dataset.skeletons[outer_train],
                y=dataset.y[outer_train],
                groups=dataset.groups[outer_train],
                feature_columns=dataset.feature_columns,
                feature_csv_paths=feature_paths,
                landmark_dirs=landmark_dirs,
                output_path=final_fold_model,
                metrics_output_path=final_fold_report,
                mode="fusion",
                window_size=int(source["window_size"]),
                stride=int(source["stride"]),
                test_size=0.0,
                random_state=fold_seed,
                graph_channels=(16, 32, 32),
                temporal_channels=(32, 32),
                dropout=0.30,
                batch_size=64,
                epochs=int(selected["best_epoch"]),
                patience=int(selected["best_epoch"]),
                learning_rate=8e-4,
                weight_decay=2e-4,
                class_weights={
                    "Normal": 1.0,
                    "Pre-fall": float(selected["weight"]),
                    "Fall": 1.0,
                },
                prefall_alert_threshold=float(selected["prefall_alert_threshold"]),
                prefall_recall_floor=args.prefall_recall_floor,
                probability_temperature=float(selected["probability_temperature"]),
            )

        probabilities = predict_fusion_artifact(
            final_fold_model,
            dataset.features[outer_test],
            dataset.skeletons[outer_test],
        )
        raw_predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        hmm_predictions = apply_hmm_by_sequence(
            probabilities, dataset.sequences[outer_test]
        )
        alert_predictions = prefall_alert_predictions(
            hmm_predictions,
            LABELS,
            probabilities,
            float(selected["prefall_alert_threshold"]),
        )
        true_labels = dataset.y[outer_test]
        fold_report = {
            "fold": fold_number,
            "seed": fold_seed,
            "outer_train_samples": int(len(outer_train)),
            "outer_test_samples": int(len(outer_test)),
            "candidate_summaries": candidates,
            "selected": selected,
            "raw_metrics": build_validation_metrics(true_labels, raw_predictions, LABELS),
            "hmm_metrics": build_validation_metrics(true_labels, hmm_predictions, LABELS),
            "calibrated_alert_metrics": build_validation_metrics(
                true_labels, alert_predictions, LABELS
            ),
            "previous_weight5_hmm_metrics": source_fold["fusion_hmm_metrics"],
        }
        fold_reports.append(fold_report)
        pooled["true"].extend(str(value) for value in true_labels)
        pooled["raw"].extend(str(value) for value in raw_predictions)
        pooled["hmm"].extend(str(value) for value in hmm_predictions)
        pooled["alert"].extend(str(value) for value in alert_predictions)
        print_fold_summary(fold_report)

    final_settings = aggregate_final_settings(selected_settings)
    train_final_full_data_model(
        dataset=dataset,
        feature_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        source=source,
        settings=final_settings,
        output_path=Path(args.final_model),
        report_path=Path(args.final_report),
        recall_floor=args.prefall_recall_floor,
        resume=args.resume,
    )
    report = {
        "method": "nested_grouped_5fold_weight_selection_temperature_calibration",
        "source_report": args.source_report,
        "sample_count": int(len(dataset.y)),
        "group_count": int(len(set(dataset.groups))),
        "sequence_count": int(len(set(dataset.sequences))),
        "folds": int(source["folds"]),
        "random_state": int(source["random_state"]),
        "window_size": int(source["window_size"]),
        "stride": int(source["stride"]),
        "candidate_prefall_weights": weights,
        "prefall_recall_floor": args.prefall_recall_floor,
        "selection_rule": (
            "Within each outer fold, choose the candidate with the highest calibrated-alert "
            "macro F1 among candidates meeting the Pre-fall recall floor; outer test labels are "
            "not used for weight, epoch, temperature, or threshold selection."
        ),
        "evaluation_layers": {
            "raw": "calibrated softmax argmax without sequence rules",
            "hmm": "calibrated probabilities with the light HMM",
            "calibrated_alert": "HMM plus inner-selected Pre-fall advisory threshold",
            "adl_lying_postprocessing_applied": False,
            "note": (
                "ADL lying/static-posture product rules are intentionally evaluated separately; "
                "these metrics describe the deep component, not the final product output."
            ),
        },
        "fold_reports": fold_reports,
        "aggregate": {
            "raw": aggregate_fold_metrics(fold_reports, "raw_metrics"),
            "hmm": aggregate_fold_metrics(fold_reports, "hmm_metrics"),
            "calibrated_alert": aggregate_fold_metrics(
                fold_reports, "calibrated_alert_metrics"
            ),
            "previous_weight5_hmm": source["aggregate"]["fusion_hmm"],
        },
        "pooled_out_of_fold_metrics": {
            name: build_validation_metrics(pooled["true"], pooled[name], LABELS)
            for name in ("raw", "hmm", "alert")
        },
        "selected_settings_by_fold": selected_settings,
        "final_full_data_settings": final_settings,
        "final_model": args.final_model,
        "final_training_report": args.final_report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== nested CV aggregate ===")
    print_aggregate(report)
    print(f"Wrote {output}")


def candidate_summary(weight: float, report: dict[str, Any]) -> dict[str, Any]:
    threshold_search = report.get("prefall_alert_threshold_search") or {}
    alert_metrics = threshold_search.get("alert_validation_metrics")
    if alert_metrics is None:
        raise RuntimeError(f"Candidate weight={weight:g} has no calibrated alert metrics")
    prefall = alert_metrics["classification_report"]["Pre-fall"]
    calibration = report.get("probability_calibration") or {}
    return {
        "weight": float(weight),
        "best_epoch": int(report["best_epoch"]),
        "probability_temperature": float(report["probability_temperature"]),
        "prefall_alert_threshold": float(report["prefall_alert_threshold"]),
        "recall_floor_satisfied": bool(threshold_search.get("recall_floor_satisfied")),
        "validation_macro_f1": float(alert_metrics["macro_f1"]),
        "validation_accuracy": float(alert_metrics["accuracy"]),
        "validation_prefall_precision": float(prefall["precision"]),
        "validation_prefall_recall": float(prefall["recall"]),
        "validation_prefall_f1": float(prefall["f1-score"]),
        "calibration_before_nll": calibration.get("before_negative_log_likelihood"),
        "calibration_after_nll": calibration.get("after_negative_log_likelihood"),
        "calibration_before_ece": calibration.get("before_expected_calibration_error"),
        "calibration_after_ece": calibration.get("after_expected_calibration_error"),
    }


def select_candidate(
    candidates: Sequence[dict[str, Any]], recall_floor: float
) -> dict[str, Any]:
    eligible = [
        item
        for item in candidates
        if item["validation_prefall_recall"] >= recall_floor
    ]
    pool = eligible or list(candidates)
    return max(
        pool,
        key=lambda item: (
            item["validation_macro_f1"],
            item["validation_prefall_precision"],
            item["validation_prefall_recall"],
            -item["weight"],
        ),
    )


def aggregate_final_settings(
    selected: Sequence[dict[str, float | int]],
) -> dict[str, float | int | dict[str, int]]:
    weight_counts = Counter(float(item["weight"]) for item in selected)
    weight = max(weight_counts, key=lambda value: (weight_counts[value], -value))
    matching = [item for item in selected if float(item["weight"]) == weight]
    source = matching or list(selected)
    return {
        "prefall_weight": float(weight),
        "selected_weight_counts": {
            format_weight(value): int(count) for value, count in sorted(weight_counts.items())
        },
        "epochs": max(1, round(statistics.median(int(item["epoch"]) for item in source))),
        "probability_temperature": float(
            statistics.median(float(item["temperature"]) for item in source)
        ),
        "prefall_alert_threshold": float(
            statistics.median(float(item["threshold"]) for item in source)
        ),
    }


def train_final_full_data_model(
    *,
    dataset,
    feature_paths: Sequence[Path],
    landmark_dirs: Sequence[str | Path],
    source: dict[str, Any],
    settings: dict[str, Any],
    output_path: Path,
    report_path: Path,
    recall_floor: float,
    resume: bool,
) -> None:
    if resume and output_path.exists() and report_path.exists():
        print(f"reuse final full-data model: {output_path}", flush=True)
        return
    print(f"\ntrain final full-data tuned model: {settings}", flush=True)
    train_and_save_fusion(
        features=dataset.features,
        skeletons=dataset.skeletons,
        y=dataset.y,
        groups=dataset.groups,
        feature_columns=dataset.feature_columns,
        feature_csv_paths=feature_paths,
        landmark_dirs=landmark_dirs,
        output_path=output_path,
        metrics_output_path=report_path,
        mode="fusion",
        window_size=int(source["window_size"]),
        stride=int(source["stride"]),
        test_size=0.0,
        random_state=int(source["random_state"]),
        graph_channels=(16, 32, 32),
        temporal_channels=(32, 32),
        dropout=0.30,
        batch_size=64,
        epochs=int(settings["epochs"]),
        patience=int(settings["epochs"]),
        learning_rate=8e-4,
        weight_decay=2e-4,
        class_weights={
            "Normal": 1.0,
            "Pre-fall": float(settings["prefall_weight"]),
            "Fall": 1.0,
        },
        prefall_alert_threshold=float(settings["prefall_alert_threshold"]),
        prefall_recall_floor=recall_floor,
        probability_temperature=float(settings["probability_temperature"]),
    )


def parse_weights(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--prefall-weights must contain positive comma-separated numbers")
    return values


def format_weight(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def print_fold_summary(fold: dict[str, Any]) -> None:
    selected = fold["selected"]
    for label, key in (
        ("raw", "raw_metrics"),
        ("hmm", "hmm_metrics"),
        ("alert", "calibrated_alert_metrics"),
    ):
        metrics = fold[key]
        prefall = metrics["classification_report"]["Pre-fall"]
        print(
            f"fold {fold['fold']} {label:5s}: weight={selected['weight']:g} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"PF P/R={prefall['precision']:.4f}/{prefall['recall']:.4f}",
            flush=True,
        )


def print_aggregate(report: dict[str, Any]) -> None:
    for name in ("raw", "hmm", "calibrated_alert", "previous_weight5_hmm"):
        metrics = report["aggregate"][name]
        print(
            f"{name:22s} "
            f"macro_f1={metrics['macro_f1']['mean']:.4f}±{metrics['macro_f1']['std']:.4f} "
            f"PF P/R={metrics['prefall_precision']['mean']:.4f}/"
            f"{metrics['prefall_recall']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
