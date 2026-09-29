from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pitalpha.data import CausalSequenceStore


class TemporalSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = pd.bdate_range("2025-01-02", periods=5)
        rows: list[dict[str, object]] = []
        for position, date in enumerate(self.calendar):
            rows.append(
                {
                    "datetime": date,
                    "instrument": "A",
                    "f1": float(position + 1),
                    "f2": np.nan if position == 2 else float(10 + position),
                    "label": 999.0,
                }
            )
        for position in range(2, 5):
            rows.append(
                {
                    "datetime": self.calendar[position],
                    "instrument": "B",
                    "f1": float(100 + position),
                    "f2": float(200 + position),
                    "label": 999.0,
                }
            )
        self.panel = pd.DataFrame(rows)

    def test_sequence_ends_at_decision_and_distinguishes_missingness(self) -> None:
        store = CausalSequenceStore.from_panel(
            self.panel,
            ["f1", "f2"],
            calendar=self.calendar,
        )
        decisions = pd.DataFrame(
            {
                "datetime": [self.calendar[3], self.calendar[2]],
                "instrument": ["A", "B"],
            },
            index=[41, 99],
        )
        batch = store.build_batch(decisions, lookback_sessions=3)

        np.testing.assert_array_equal(batch.values[0, :, 0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(batch.row_mask[0], [True, True, True])
        np.testing.assert_array_equal(batch.observed_mask[0, :, 1], [True, False, True])
        self.assertEqual(batch.values[0, 1, 1], 0.0)
        np.testing.assert_array_equal(batch.row_mask[1], [False, False, True])
        self.assertEqual(batch.source_index, (41, 99))
        np.testing.assert_array_equal(batch.session_dates[:, -1], batch.decision_dates)
        self.assertFalse(batch.values.flags.writeable)

    def test_calendar_boundary_is_left_padded_without_future_rows(self) -> None:
        store = CausalSequenceStore.from_panel(
            self.panel,
            ["f1", "f2"],
            calendar=self.calendar,
        )
        decisions = pd.DataFrame(
            {"datetime": [self.calendar[0]], "instrument": ["A"]}
        )
        batch = store.build_batch(decisions, lookback_sessions=3)

        self.assertTrue(np.isnat(batch.session_dates[0, 0]))
        self.assertTrue(np.isnat(batch.session_dates[0, 1]))
        self.assertEqual(batch.session_dates[0, 2], self.calendar[0].to_datetime64())
        np.testing.assert_array_equal(batch.row_mask[0], [False, False, True])

    def test_future_feature_perturbation_cannot_change_past_sequence(self) -> None:
        decision = pd.DataFrame(
            {"datetime": [self.calendar[3]], "instrument": ["A"]}
        )
        baseline = CausalSequenceStore.from_panel(
            self.panel, ["f1", "f2"], calendar=self.calendar
        ).build_batch(decision, lookback_sessions=4)
        perturbed_panel = self.panel.copy()
        future = perturbed_panel["datetime"] > self.calendar[3]
        perturbed_panel.loc[future, ["f1", "f2"]] = 1_000_000.0
        perturbed = CausalSequenceStore.from_panel(
            perturbed_panel, ["f1", "f2"], calendar=self.calendar
        ).build_batch(decision, lookback_sessions=4)

        np.testing.assert_array_equal(baseline.values, perturbed.values)
        np.testing.assert_array_equal(baseline.observed_mask, perturbed.observed_mask)

    def test_missing_decision_key_fails_closed(self) -> None:
        store = CausalSequenceStore.from_panel(
            self.panel, ["f1", "f2"], calendar=self.calendar
        )
        decision = pd.DataFrame(
            {"datetime": [self.calendar[1]], "instrument": ["B"]}
        )
        with self.assertRaisesRegex(ValueError, "no panel row for decision key"):
            store.build_batch(decision, lookback_sessions=2)

    def test_duplicate_keys_and_outcome_features_are_rejected(self) -> None:
        duplicate = pd.concat([self.panel, self.panel.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            CausalSequenceStore.from_panel(duplicate, ["f1"], calendar=self.calendar)
        with self.assertRaisesRegex(ValueError, "future or outcome"):
            CausalSequenceStore.from_panel(self.panel, ["f1", "label"])


if __name__ == "__main__":
    unittest.main()
