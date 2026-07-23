"""Create and validate a reproducible trial-group holdout split."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Sequence

import numpy as np


DEFAULT_HOLDOUT_RANDOM_STATE = 20260721
DEFAULT_HOLDOUT_FRACTION = 0.20


def create_fixed_group_holdout_manifest(
    labels: Sequence[str],
    groups: Sequence[str],
    sequences: Sequence[str],
    *,
    test_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    random_state: int = DEFAULT_HOLDOUT_RANDOM_STATE,
) -> dict[str, Any]:
    """Randomly select complete groups while approximately stratifying labels."""
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if not (len(labels) == len(groups) == len(sequences)):
        raise ValueError("labels, groups, and sequences must have equal length")
    from sklearn.model_selection import StratifiedGroupKFold

    labels_array = np.asarray([str(value) for value in labels])
    groups_array = np.asarray([str(value) for value in groups])
    sequences_array = np.asarray([str(value) for value in sequences])
    n_splits = max(2, int(round(1.0 / test_fraction)))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    placeholder = np.zeros((len(labels_array), 1), dtype=np.float32)
    candidates = list(splitter.split(placeholder, labels_array, groups_array))
    all_labels = set(labels_array)
    valid = [
        split
        for split in candidates
        if set(labels_array[split[1]]) == all_labels
    ]
    if not valid:
        raise RuntimeError("No grouped random holdout contains all labels")
    train_index, test_index = min(
        valid,
        key=lambda split: abs(len(split[1]) / len(labels_array) - test_fraction),
    )
    if set(groups_array[train_index]) & set(groups_array[test_index]):
        raise RuntimeError("Group leakage detected in fixed holdout")

    train_groups = sorted(set(groups_array[train_index]))
    test_groups = sorted(set(groups_array[test_index]))
    return {
        "format_version": 1,
        "method": "seeded_stratified_group_random_holdout",
        "random_state": int(random_state),
        "requested_test_fraction": float(test_fraction),
        "actual_test_sample_fraction": float(len(test_index) / len(labels_array)),
        "dataset_fingerprint": dataset_fingerprint(
            labels_array, groups_array, sequences_array
        ),
        "sample_count": int(len(labels_array)),
        "group_count": int(len(set(groups_array))),
        "sequence_count": int(len(set(sequences_array))),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train_sequences": sorted(set(sequences_array[train_index])),
        "test_sequences": sorted(set(sequences_array[test_index])),
        "train_label_counts": dict(sorted(Counter(labels_array[train_index]).items())),
        "test_label_counts": dict(sorted(Counter(labels_array[test_index]).items())),
    }


def indices_from_fixed_manifest(
    manifest: dict[str, Any],
    labels: Sequence[str],
    groups: Sequence[str],
    sequences: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve saved group names and reject a silently changed dataset."""
    fingerprint = dataset_fingerprint(labels, groups, sequences)
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise ValueError(
            "Fixed holdout manifest does not match the current dataset; "
            "create a new explicitly versioned manifest instead of silently changing it"
        )
    groups_array = np.asarray([str(value) for value in groups])
    test_groups = set(str(value) for value in manifest["test_groups"])
    test_mask = np.asarray([value in test_groups for value in groups_array])
    train_index = np.flatnonzero(~test_mask)
    test_index = np.flatnonzero(test_mask)
    if not len(train_index) or not len(test_index):
        raise ValueError("Fixed holdout produced an empty train or test partition")
    if set(groups_array[train_index]) & set(groups_array[test_index]):
        raise RuntimeError("Group leakage detected while loading fixed holdout")
    return train_index, test_index


def dataset_fingerprint(
    labels: Sequence[str],
    groups: Sequence[str],
    sequences: Sequence[str],
) -> str:
    if not (len(labels) == len(groups) == len(sequences)):
        raise ValueError("labels, groups, and sequences must have equal length")
    records = sorted(
        f"{group}\t{sequence}\t{label}"
        for label, group, sequence in zip(labels, groups, sequences)
    )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
