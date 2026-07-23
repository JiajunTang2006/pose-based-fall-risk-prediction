"""Evaluate the cooperative tree + fusion decision layer on saved CV folds."""

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

from fall_prediction.ensemble_predictor import DualModelDecisionEngine
from fall_prediction.lying_adl_filter import StaticLyingADLFilter
from fall_prediction.skeleton_dataset import build_paired_temporal_dataset
from fall_prediction.train_model import build_validation_metrics, collect_csv_paths, json_ready
from scripts.cross_validate_fusion import (
    LABELS,
    aggregate_fold_metrics,
    apply_hmm_by_sequence,
    grouped_stratified_splits,
    predict_fusion_artifact,
    train_and_predict_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cooperative tree-confirmation + fusion-warning decisions."
    )
    parser.add_argument(
        "--source-report", default="reports/fusion_grouped_5fold_cv_full_outer.json"
    )
    parser.add_argument("--input-dir", default="outputs/features")
    parser.add_argument("--landmark-dir", action="append", default=None)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--model-dir", default="models/cross_validation")
    parser.add_argument(
        "--fusion-model-pattern",
        default="fusion_fold_{fold}_full_outer.pt",
        help="Fold model filename pattern; {fold} is replaced by the fold number.",
    )
    parser.add_argument(
        "--output", default="reports/dual_model_grouped_5fold_cv.json"
    )
    parser.add_argument("--fusion-fall-confirmation-steps", type=int, default=3)
    args = parser.parse_args()
    if args.fusion_fall_confirmation_steps < 1:
        parser.error("--fusion-fall-confirmation-steps must be at least 1")

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

    fold_reports: list[dict[str, Any]] = []
    pooled = {
        "true": [],
        "tree": [],
        "fusion_hmm": [],
        "dual_confirmed": [],
        "dual_alert": [],
        "dual_early_warning": [],
        "postprocessed_confirmed": [],
        "postprocessed_alert": [],
        "postprocessed_early_warning": [],
        "sequence": [],
    }
    for source_fold, (outer_train, outer_test) in zip(source["fold_reports"], splits):
        fold_number = int(source_fold["fold"])
        fold_seed = int(source_fold["seed"])
        fusion_model_path = Path(args.model_dir) / args.fusion_model_pattern.format(
            fold=fold_number
        )
        if not fusion_model_path.exists():
            raise FileNotFoundError(f"Missing saved fold model: {fusion_model_path}")

        fusion_probabilities = predict_fusion_artifact(
            fusion_model_path,
            dataset.features[outer_test],
            dataset.skeletons[outer_test],
        )
        fusion_states = apply_hmm_by_sequence(
            fusion_probabilities, dataset.sequences[outer_test]
        )
        tree_states = train_and_predict_tree(
            dataset.features[outer_train],
            dataset.y[outer_train],
            dataset.features[outer_test],
            random_state=fold_seed,
        )
        dual_states, dual_alerts, dual_early_warnings, tier_counts = apply_dual_by_sequence(
            tree_states,
            fusion_states,
            dataset.sequences[outer_test],
            fusion_fall_confirmation_steps=args.fusion_fall_confirmation_steps,
        )
        (
            postprocessed_states,
            postprocessed_alerts,
            postprocessed_early_warnings,
            lying_filter_summary,
        ) = apply_static_lying_filter_by_sequence(
            dual_states,
            dual_alerts,
            dual_early_warnings,
            dataset.features[outer_test],
            dataset.sequences[outer_test],
            dataset.feature_columns,
        )
        true_labels = dataset.y[outer_test]
        fold_report = {
            "fold": fold_number,
            "seed": fold_seed,
            "outer_train_samples": int(len(outer_train)),
            "outer_test_samples": int(len(outer_test)),
            "tree_metrics": build_validation_metrics(true_labels, tree_states, LABELS),
            "fusion_hmm_metrics": build_validation_metrics(
                true_labels, fusion_states, LABELS
            ),
            "dual_confirmed_metrics": build_validation_metrics(
                true_labels, dual_states, LABELS
            ),
            "dual_alert_metrics": build_validation_metrics(
                true_labels, dual_alerts, LABELS
            ),
            "dual_early_warning_metrics": build_validation_metrics(
                true_labels, dual_early_warnings, LABELS
            ),
            "postprocessed_confirmed_metrics": build_validation_metrics(
                true_labels, postprocessed_states, LABELS
            ),
            "postprocessed_alert_metrics": build_validation_metrics(
                true_labels, postprocessed_alerts, LABELS
            ),
            "postprocessed_early_warning_metrics": build_validation_metrics(
                true_labels, postprocessed_early_warnings, LABELS
            ),
            "decision_tier_counts": tier_counts,
            "static_lying_filter_summary": lying_filter_summary,
            "sequence_summary_confirmed": summarize_sequence_events(
                true_labels, dual_states, dataset.sequences[outer_test]
            ),
            "sequence_summary_alert": summarize_sequence_events(
                true_labels, dual_alerts, dataset.sequences[outer_test]
            ),
            "sequence_summary_early_warning": summarize_sequence_events(
                true_labels, dual_early_warnings, dataset.sequences[outer_test]
            ),
            "sequence_summary_postprocessed_confirmed": summarize_sequence_events(
                true_labels, postprocessed_states, dataset.sequences[outer_test]
            ),
            "sequence_summary_postprocessed_alert": summarize_sequence_events(
                true_labels, postprocessed_alerts, dataset.sequences[outer_test]
            ),
            "sequence_summary_postprocessed_early_warning": summarize_sequence_events(
                true_labels, postprocessed_early_warnings, dataset.sequences[outer_test]
            ),
        }
        fold_reports.append(fold_report)
        for value in true_labels:
            pooled["true"].append(str(value))
        for name, values in (
            ("tree", tree_states),
            ("fusion_hmm", fusion_states),
            ("dual_confirmed", dual_states),
            ("dual_alert", dual_alerts),
            ("dual_early_warning", dual_early_warnings),
            ("postprocessed_confirmed", postprocessed_states),
            ("postprocessed_alert", postprocessed_alerts),
            ("postprocessed_early_warning", postprocessed_early_warnings),
        ):
            pooled[name].extend(str(value) for value in values)
        pooled["sequence"].extend(str(value) for value in dataset.sequences[outer_test])
        print_fold_summary(fold_report)

    report = {
        "method": (
            "grouped_5fold_cv_saved_fusion_folds_with_cooperative_decision_and_"
            "static_lying_adl_postprocessing"
        ),
        "source_report": args.source_report,
        "folds": int(source["folds"]),
        "random_state": int(source["random_state"]),
        "window_size": int(source["window_size"]),
        "stride": int(source["stride"]),
        "sample_count": int(len(dataset.y)),
        "group_count": int(len(set(dataset.groups))),
        "sequence_count": int(len(set(dataset.sequences))),
        "fusion_fall_confirmation_steps": args.fusion_fall_confirmation_steps,
        "fusion_model_pattern": args.fusion_model_pattern,
        "decision_policy": {
            "tree_prefall": "confirmed Pre-fall",
            "fusion_only_prefall": (
                "low-level Pre-fall advisory while confirmed and formal alert states remain Normal"
            ),
            "tree_fall": "immediate confirmed Fall",
            "fusion_only_fall": (
                "Pre-fall alert while confirming, then Fall alert after consecutive fusion "
                "outputs; confirmed state remains authoritative tree output"
            ),
            "strict_legacy_temporal_gate": False,
            "fusion_hmm": True,
            "static_lying_adl_filter": (
                "enabled after dual-model decision; only static low posture without recent "
                "fall-like motion is changed to Normal"
            ),
            "fall_latch": (
                "confirmed dynamic or non-static Fall remains Fall until acknowledgement"
            ),
        },
        "interpretation_note": (
            "This is an exploratory comparison on the same outer folds previously used to inspect "
            "the component models. It measures the proposed product logic but is not a new, fully "
            "untouched external validation."
        ),
        "fold_reports": fold_reports,
        "aggregate": {
            "tree": aggregate_fold_metrics(fold_reports, "tree_metrics"),
            "fusion_hmm": aggregate_fold_metrics(fold_reports, "fusion_hmm_metrics"),
            "dual_confirmed": aggregate_fold_metrics(
                fold_reports, "dual_confirmed_metrics"
            ),
            "dual_alert": aggregate_fold_metrics(fold_reports, "dual_alert_metrics"),
            "dual_early_warning": aggregate_fold_metrics(
                fold_reports, "dual_early_warning_metrics"
            ),
            "postprocessed_confirmed": aggregate_fold_metrics(
                fold_reports, "postprocessed_confirmed_metrics"
            ),
            "postprocessed_alert": aggregate_fold_metrics(
                fold_reports, "postprocessed_alert_metrics"
            ),
            "postprocessed_early_warning": aggregate_fold_metrics(
                fold_reports, "postprocessed_early_warning_metrics"
            ),
        },
        "pooled_out_of_fold_metrics": {
            name: build_validation_metrics(pooled["true"], pooled[name], LABELS)
            for name in (
                "tree",
                "fusion_hmm",
                "dual_confirmed",
                "dual_alert",
                "dual_early_warning",
                "postprocessed_confirmed",
                "postprocessed_alert",
                "postprocessed_early_warning",
            )
        },
        "pooled_sequence_summary": {
            name: summarize_sequence_events(
                pooled["true"], pooled[name], pooled["sequence"]
            )
            for name in (
                "tree",
                "fusion_hmm",
                "dual_confirmed",
                "dual_alert",
                "dual_early_warning",
                "postprocessed_confirmed",
                "postprocessed_alert",
                "postprocessed_early_warning",
            )
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== aggregate mean ± sample std ===")
    print_aggregate(report)
    print(f"Wrote {output}")


def apply_dual_by_sequence(
    tree_states: Sequence[str],
    fusion_states: Sequence[str],
    sequences: Sequence[str],
    *,
    fusion_fall_confirmation_steps: int,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    if not (len(tree_states) == len(fusion_states) == len(sequences)):
        raise ValueError("tree_states, fusion_states, and sequences must have equal length")
    engines: dict[str, DualModelDecisionEngine] = {}
    states: list[str] = []
    alerts: list[str] = []
    early_warnings: list[str] = []
    tiers: Counter[str] = Counter()
    for tree_state, fusion_state, sequence in zip(tree_states, fusion_states, sequences):
        key = str(sequence)
        engine = engines.setdefault(
            key,
            DualModelDecisionEngine(fusion_fall_confirmation_steps),
        )
        decision = engine.decide(str(tree_state), str(fusion_state))
        states.append(decision.state)
        alerts.append(decision.alert_state)
        early_warnings.append(decision.advisory_state or decision.alert_state)
        tiers[decision.tier] += 1
    return states, alerts, early_warnings, dict(sorted(tiers.items()))


def apply_static_lying_filter_by_sequence(
    confirmed_states: Sequence[str],
    alert_states: Sequence[str],
    early_warning_states: Sequence[str],
    feature_windows: np.ndarray,
    sequences: Sequence[str],
    feature_columns: Sequence[str],
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Apply the runtime static-lying rule independently to each video."""
    lengths = {
        len(confirmed_states),
        len(alert_states),
        len(early_warning_states),
        len(feature_windows),
        len(sequences),
    }
    if len(lengths) != 1:
        raise ValueError("states, feature_windows, and sequences must have equal length")

    filters: dict[str, StaticLyingADLFilter] = {}
    states: list[str] = []
    alerts: list[str] = []
    early_warnings: list[str] = []
    counts: Counter[str] = Counter()
    for state, alert, early, window, sequence in zip(
        confirmed_states,
        alert_states,
        early_warning_states,
        feature_windows,
        sequences,
    ):
        key = str(sequence)
        rule = filters.setdefault(key, StaticLyingADLFilter())
        advisory = None if str(early) == str(alert) else str(early)
        rows = [
            {
                column: float(value)
                for column, value in zip(feature_columns, frame_values)
            }
            for frame_values in window
        ]
        was_latched = rule.fall_latched
        decision = rule.process(str(state), str(alert), advisory, rows)
        states.append(decision.state)
        alerts.append(decision.alert_state)
        early_warnings.append(decision.advisory_state or decision.alert_state)
        counts["windows"] += 1
        counts["static_low_posture_windows"] += int(decision.is_static_low_posture)
        counts["static_lying_posture_windows"] += int(
            decision.is_static_lying_posture
        )
        counts["windows_changed_by_filter"] += int(decision.filtered)
        counts["lying_adl_override_windows"] += int(
            decision.tier in {"lying-adl-watch", "lying-adl-normal"}
        )
        counts["settled_lying_normal_windows"] += int(
            decision.tier == "lying-adl-normal"
        )
        counts["confirmed_fall_windows_suppressed"] += int(
            str(state) == "Fall" and decision.state != "Fall"
        )
        counts["fall_alert_windows_suppressed"] += int(
            str(alert) == "Fall" and decision.alert_state != "Fall"
        )
        counts["fall_latch_events"] += int(not was_latched and decision.fall_latched)
        counts["fall_latched_windows"] += int(decision.fall_latched)
    return states, alerts, early_warnings, dict(sorted(counts.items()))


def summarize_sequence_events(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    sequences: Sequence[str],
) -> dict[str, int]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for truth, prediction, sequence in zip(true_labels, predicted_labels, sequences):
        record = grouped.setdefault(str(sequence), {"truth": [], "prediction": []})
        record["truth"].append(str(truth))
        record["prediction"].append(str(prediction))
    result = {
        "fall_sequences": 0,
        "fall_sequences_detected": 0,
        "fall_sequences_missed": 0,
        "nonfall_sequences": 0,
        "nonfall_sequences_with_false_fall": 0,
        "nonfall_sequences_with_any_warning": 0,
    }
    for record in grouped.values():
        truth_has_fall = "Fall" in record["truth"]
        prediction_has_fall = "Fall" in record["prediction"]
        prediction_has_warning = any(
            value in {"Pre-fall", "Fall"} for value in record["prediction"]
        )
        if truth_has_fall:
            result["fall_sequences"] += 1
            if prediction_has_fall:
                result["fall_sequences_detected"] += 1
            else:
                result["fall_sequences_missed"] += 1
        else:
            result["nonfall_sequences"] += 1
            result["nonfall_sequences_with_false_fall"] += int(prediction_has_fall)
            result["nonfall_sequences_with_any_warning"] += int(prediction_has_warning)
    return result


def print_fold_summary(fold: dict[str, Any]) -> None:
    items = []
    for label, key in (
        ("tree", "tree_metrics"),
        ("fusion+hmm", "fusion_hmm_metrics"),
        ("dual-confirmed", "dual_confirmed_metrics"),
        ("dual-alert", "dual_alert_metrics"),
        ("dual-early", "dual_early_warning_metrics"),
        ("post-confirmed", "postprocessed_confirmed_metrics"),
        ("post-alert", "postprocessed_alert_metrics"),
        ("post-early", "postprocessed_early_warning_metrics"),
    ):
        metrics = fold[key]
        prefall = metrics["classification_report"]["Pre-fall"]
        items.append(
            f"{label}: macro_f1={metrics['macro_f1']:.4f}, "
            f"PF P/R={prefall['precision']:.4f}/{prefall['recall']:.4f}"
        )
    print(f"fold {fold['fold']}: " + " | ".join(items), flush=True)


def print_aggregate(report: dict[str, Any]) -> None:
    for name in (
        "tree",
        "fusion_hmm",
        "dual_confirmed",
        "dual_alert",
        "dual_early_warning",
        "postprocessed_confirmed",
        "postprocessed_alert",
        "postprocessed_early_warning",
    ):
        summary = report["aggregate"][name]
        print(
            f"{name:15s} "
            f"acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f} "
            f"macro_f1={summary['macro_f1']['mean']:.4f}±{summary['macro_f1']['std']:.4f} "
            f"PF P/R={summary['prefall_precision']['mean']:.4f}/"
            f"{summary['prefall_recall']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
