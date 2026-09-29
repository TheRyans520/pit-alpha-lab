from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pitalpha.config import load_config
from pitalpha.data import FEATURE_COLUMNS, generate_synthetic_panel


ROOT = Path(__file__).resolve().parents[1]


class SyntheticPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        cls.panel = generate_synthetic_panel(cls.config)

    def test_daily_point_in_time_membership_is_exactly_thirty(self) -> None:
        counts = self.panel.groupby("datetime")["instrument"].nunique()
        self.assertTrue(counts.eq(30).all())
        self.assertGreater(self.panel["instrument"].nunique(), 30)

    def test_observations_respect_membership_intervals(self) -> None:
        self.assertTrue((self.panel["datetime"] >= self.panel["membership_start"]).all())
        self.assertTrue((self.panel["datetime"] <= self.panel["membership_end"]).all())

    def test_future_fields_are_missing_at_the_terminal_boundary(self) -> None:
        final_date = self.panel["datetime"].max()
        final_rows = self.panel[self.panel["datetime"] == final_date]
        self.assertTrue(final_rows["label"].isna().all())
        self.assertTrue(final_rows["realized_return_1d"].isna().all())

    def test_fixture_contains_missing_features_for_imputation(self) -> None:
        missing = self.panel[list(FEATURE_COLUMNS)].isna().sum().sum()
        self.assertGreater(int(missing), 0)
        self.assertTrue(np.isfinite(self.panel["realized_return_1d"].dropna()).all())


if __name__ == "__main__":
    unittest.main()
