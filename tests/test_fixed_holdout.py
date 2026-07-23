import unittest

from fall_prediction.fixed_holdout import (
    create_fixed_group_holdout_manifest,
    indices_from_fixed_manifest,
)


class FixedHoldoutTest(unittest.TestCase):
    def test_fixed_holdout_is_reproducible_and_has_no_group_leakage(self):
        labels = []
        groups = []
        sequences = []
        for group_index in range(15):
            group = f"g{group_index}"
            sequence = f"s{group_index}"
            group_labels = (
                ["Normal"] * 4
                if group_index % 3 == 0
                else ["Normal", "Pre-fall", "Fall", "Fall"]
            )
            for label in group_labels:
                labels.append(label)
                groups.append(group)
                sequences.append(sequence)

        first = create_fixed_group_holdout_manifest(
            labels, groups, sequences, random_state=7
        )
        second = create_fixed_group_holdout_manifest(
            labels, groups, sequences, random_state=7
        )
        train_index, test_index = indices_from_fixed_manifest(
            first, labels, groups, sequences
        )

        self.assertEqual(first["test_groups"], second["test_groups"])
        self.assertFalse(
            {groups[index] for index in train_index}
            & {groups[index] for index in test_index}
        )
        self.assertEqual(set(first["test_label_counts"]), {"Normal", "Pre-fall", "Fall"})


if __name__ == "__main__":
    unittest.main()
