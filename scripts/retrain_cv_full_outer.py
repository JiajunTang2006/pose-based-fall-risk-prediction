"""Retrain each CV fusion fold on its complete outer-training set after epoch selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fall_prediction.skeleton_dataset import build_paired_temporal_dataset
from fall_prediction.train_fusion_model import train_and_save_fusion
from fall_prediction.train_model import build_validation_metrics, collect_csv_paths, json_ready
from scripts.cross_validate_fusion import (
    LABELS,
    aggregate_fold_metrics,
    apply_hmm_by_sequence,
    fold_win_counts,
    grouped_stratified_splits,
    predict_fusion_artifact,
    print_aggregate,
    print_fold_summary,
    train_and_predict_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain complete outer folds using selected epochs.")
    parser.add_argument("--source-report", default="reports/fusion_grouped_5fold_cv.json")
    parser.add_argument("--input-dir", default="outputs/features")
    parser.add_argument("--landmark-dir", action="append", default=None)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--output", default="reports/fusion_grouped_5fold_cv_full_outer.json")
    parser.add_argument("--model-dir", default="models/cross_validation")
    parser.add_argument("--fold-report-dir", default="reports/cross_validation")
    args = parser.parse_args()

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
    fold_report_dir = Path(args.fold_report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_report_dir.mkdir(parents=True, exist_ok=True)

    fold_reports = []
    pooled = {"true": [], "tree": [], "fusion": [], "fusion_hmm": []}
    for source_fold, (outer_train, outer_test) in zip(source["fold_reports"], splits):
        fold_number = int(source_fold["fold"])
        fold_seed = int(source_fold["seed"])
        selected_epochs = int(source_fold["inner_best_epoch"])
        print(
            f"\n=== fold {fold_number}/{source['folds']}: retrain {len(outer_train)} samples "
            f"for {selected_epochs} epochs ===",
            flush=True,
        )
        model_path = model_dir / f"fusion_fold_{fold_number}_full_outer.pt"
        training_report_path = (
            fold_report_dir / f"fusion_fold_{fold_number}_full_outer_training.json"
        )
        train_and_save_fusion(
            features=dataset.features[outer_train],
            skeletons=dataset.skeletons[outer_train],
            y=dataset.y[outer_train],
            groups=dataset.groups[outer_train],
            feature_columns=dataset.feature_columns,
            feature_csv_paths=feature_paths,
            landmark_dirs=landmark_dirs,
            output_path=model_path,
            metrics_output_path=training_report_path,
            mode="fusion",
            window_size=int(source["window_size"]),
            stride=int(source["stride"]),
            test_size=0.0,
            random_state=fold_seed,
            graph_channels=(16, 32, 32),
            temporal_channels=(32, 32),
            dropout=0.30,
            batch_size=64,
            epochs=selected_epochs,
            patience=selected_epochs,
            learning_rate=8e-4,
            weight_decay=2e-4,
            class_weights={"Normal": 1.0, "Pre-fall": 5.0, "Fall": 1.0},
            prefall_alert_threshold=0.25,
        )
        probabilities = predict_fusion_artifact(
            model_path,
            dataset.features[outer_test],
            dataset.skeletons[outer_test],
        )
        fusion_predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        fusion_hmm_predictions = apply_hmm_by_sequence(
            probabilities, dataset.sequences[outer_test]
        )
        tree_predictions = train_and_predict_tree(
            dataset.features[outer_train],
            dataset.y[outer_train],
            dataset.features[outer_test],
            random_state=fold_seed,
        )
        true_labels = dataset.y[outer_test]
        fold_report = {
            "fold": fold_number,
            "seed": fold_seed,
            "selected_epochs_from_inner_validation": selected_epochs,
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
            "tree_metrics": build_validation_metrics(
                true_labels, tree_predictions, LABELS
            ),
            "fusion_metrics": build_validation_metrics(
                true_labels, fusion_predictions, LABELS
            ),
            "fusion_hmm_metrics": build_validation_metrics(
                true_labels, fusion_hmm_predictions, LABELS
            ),
        }
        fold_reports.append(fold_report)
        pooled["true"].extend(str(value) for value in true_labels)
        pooled["tree"].extend(str(value) for value in tree_predictions)
        pooled["fusion"].extend(str(value) for value in fusion_predictions)
        pooled["fusion_hmm"].extend(str(value) for value in fusion_hmm_predictions)
        print_fold_summary(fold_report)

    report = {
        "method": "grouped_5fold_cv_inner_epoch_selection_then_full_outer_retraining",
        "source_inner_report": args.source_report,
        "folds": source["folds"],
        "random_state": source["random_state"],
        "window_size": source["window_size"],
        "stride": source["stride"],
        "sample_count": int(len(dataset.y)),
        "group_count": int(len(set(dataset.groups))),
        "sequence_count": int(len(set(dataset.sequences))),
        "leakage_controls": {
            "outer_split_unit": "video/trial group",
            "upfall_cameras_same_outer_fold": True,
            "outer_test_used_for_epoch_selection": False,
            "complete_outer_training_set_used_for_final_fold_model": True,
        },
        "fold_reports": fold_reports,
        "aggregate": {
            "tree": aggregate_fold_metrics(fold_reports, "tree_metrics"),
            "fusion": aggregate_fold_metrics(fold_reports, "fusion_metrics"),
            "fusion_hmm": aggregate_fold_metrics(fold_reports, "fusion_hmm_metrics"),
        },
        "pooled_out_of_fold_metrics": {
            name: build_validation_metrics(pooled["true"], pooled[name], LABELS)
            for name in ("tree", "fusion", "fusion_hmm")
        },
        "fold_win_counts": fold_win_counts(fold_reports),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== final aggregate ===")
    print_aggregate(report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
