"""Small-validation-set probability calibration for fall classifiers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def apply_temperature_scaling(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling when only softmax probabilities remain."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    clipped = np.clip(values, 1e-9, 1.0)
    scaled_logits = np.log(clipped) / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled_logits)
    calibrated = exponentials / exponentials.sum(axis=1, keepdims=True)
    return calibrated.astype(np.float32)


def fit_temperature_scaling(
    probabilities: np.ndarray,
    y_true: Sequence[str],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Choose one temperature by validation negative log likelihood."""
    values = np.asarray(probabilities, dtype=np.float64)
    class_names = [str(label) for label in classes]
    label_to_index = {label: index for index, label in enumerate(class_names)}
    try:
        targets = np.asarray([label_to_index[str(label)] for label in y_true], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown calibration label: {exc.args[0]}") from exc
    if len(values) != len(targets):
        raise ValueError("probabilities and y_true must have equal length")
    if not len(values):
        raise ValueError("at least one calibration sample is required")

    coarse = np.geomspace(0.25, 4.0, num=97)
    candidates = []
    for temperature in coarse:
        calibrated = apply_temperature_scaling(values, float(temperature))
        candidates.append(
            {
                "temperature": float(temperature),
                "negative_log_likelihood": multiclass_negative_log_likelihood(
                    calibrated, targets
                ),
            }
        )
    best_coarse = min(candidates, key=lambda item: item["negative_log_likelihood"])
    center = float(best_coarse["temperature"])
    lower = max(0.10, center / 1.08)
    upper = min(10.0, center * 1.08)
    for temperature in np.linspace(lower, upper, num=81):
        calibrated = apply_temperature_scaling(values, float(temperature))
        candidates.append(
            {
                "temperature": float(temperature),
                "negative_log_likelihood": multiclass_negative_log_likelihood(
                    calibrated, targets
                ),
            }
        )
    best = min(candidates, key=lambda item: item["negative_log_likelihood"])
    calibrated = apply_temperature_scaling(values, float(best["temperature"]))
    return {
        "temperature": float(best["temperature"]),
        "before_negative_log_likelihood": multiclass_negative_log_likelihood(
            values, targets
        ),
        "after_negative_log_likelihood": float(best["negative_log_likelihood"]),
        "before_expected_calibration_error": expected_calibration_error(values, targets),
        "after_expected_calibration_error": expected_calibration_error(
            calibrated, targets
        ),
    }


def tune_prefall_alert_threshold_with_recall_floor(
    y_true: Sequence[str],
    base_predictions: Sequence[str],
    classes: Sequence[str],
    probabilities: np.ndarray,
    *,
    recall_floor: float = 0.80,
) -> dict[str, Any]:
    """Find the most precise early-alert threshold that preserves target recall.

    Existing Pre-fall and Fall states are retained. The threshold can only
    promote a Normal state to a low-level Pre-fall alert, matching runtime use.
    """
    from .train_model import build_validation_metrics, prefall_binary_metrics

    recall_floor = float(recall_floor)
    if not 0.0 <= recall_floor <= 1.0:
        raise ValueError("recall_floor must be between 0 and 1")
    class_names = [str(label) for label in classes]
    if "Pre-fall" not in class_names:
        raise ValueError("classes must include Pre-fall")
    candidates: list[dict[str, Any]] = []
    for threshold_index in range(5, 96):
        threshold = threshold_index / 100.0
        predictions = prefall_alert_predictions(
            base_predictions,
            class_names,
            probabilities,
            threshold,
        )
        binary = prefall_binary_metrics(y_true, predictions, beta=1.0)
        metrics = build_validation_metrics(y_true, predictions, class_names)
        candidates.append(
            {
                "threshold": threshold,
                **binary,
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    eligible = [item for item in candidates if item["recall"] >= recall_floor]
    if eligible:
        best = max(
            eligible,
            key=lambda item: (
                item["precision"],
                item["macro_f1"],
                item["threshold"],
            ),
        )
        floor_satisfied = True
    else:
        best = max(
            candidates,
            key=lambda item: (
                item["recall"],
                item["precision"],
                item["macro_f1"],
            ),
        )
        floor_satisfied = False
    best_predictions = prefall_alert_predictions(
        base_predictions,
        class_names,
        probabilities,
        float(best["threshold"]),
    )
    return {
        "recall_floor": recall_floor,
        "recall_floor_satisfied": floor_satisfied,
        "best": best,
        "candidates": candidates,
        "alert_validation_metrics": build_validation_metrics(
            y_true, best_predictions, class_names
        ),
    }


def prefall_alert_predictions(
    base_predictions: Sequence[str],
    classes: Sequence[str],
    probabilities: np.ndarray,
    threshold: float,
) -> list[str]:
    """Promote Normal to Pre-fall when calibrated probability crosses a threshold."""
    class_names = [str(label) for label in classes]
    prefall_index = class_names.index("Pre-fall")
    values = np.asarray(probabilities)
    if len(base_predictions) != len(values):
        raise ValueError("base_predictions and probabilities must have equal length")
    predictions: list[str] = []
    for label, probability_row in zip(base_predictions, values):
        state = str(label)
        if state in {"Fall", "Pre-fall"}:
            predictions.append(state)
        elif float(probability_row[prefall_index]) >= float(threshold):
            predictions.append("Pre-fall")
        else:
            predictions.append(state)
    return predictions


def multiclass_negative_log_likelihood(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(values[np.arange(len(values)), targets], 1e-9, 1.0)
    return float(-np.log(clipped).mean())


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    correct = predictions == targets
    total = max(len(values), 1)
    error = 0.0
    for start in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        end = start + 1.0 / bins
        mask = (confidence >= start) & (
            confidence <= end if end >= 1.0 else confidence < end
        )
        count = int(mask.sum())
        if count:
            error += count / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)
