import unittest

import numpy as np
import pandas as pd

from pitalpha.splits.validation import purged_validation_split


class InnerValidationTests(unittest.TestCase):
    def test_label_endpoint_precedes_validation_and_missing_labels_keep_calendar(self):
        dates = pd.bdate_range("2024-01-01", periods=40)
        frame = pd.DataFrame({"datetime": dates, "label": 0.01})
        frame.loc[31:33, "label"] = np.nan
        subtrain, validation, metadata = purged_validation_split(frame, validation_dates=5)
        self.assertEqual(subtrain["datetime"].max(), dates[28])
        self.assertEqual(validation["datetime"].min(), dates[35])
        # open(t+6) is strictly before the validation decision date.
        self.assertLess(dates[28 + 6], validation["datetime"].min())
        self.assertEqual(metadata["purged_rows"], 6)

    def test_context_only_tail_does_not_consume_validation_dates(self):
        dates = pd.bdate_range("2024-01-01", periods=40)
        frame = pd.DataFrame({"datetime": dates, "label": 0.01})
        frame.loc[34:, "label"] = np.nan
        _, validation, _ = purged_validation_split(frame, validation_dates=5)
        self.assertEqual(len(validation), 5)
        self.assertEqual(validation["datetime"].min(), dates[29])

    def test_insufficient_history_fails(self):
        frame = pd.DataFrame({"datetime": pd.bdate_range("2024-01-01", periods=10), "label": 0.01})
        with self.assertRaisesRegex(ValueError, "lacks dates"):
            purged_validation_split(frame, validation_dates=5)
