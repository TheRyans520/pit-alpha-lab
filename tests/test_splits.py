from __future__ import annotations

import unittest
from pathlib import Path

from pitalpha.config import load_config
from pitalpha.data import generate_synthetic_panel
from pitalpha.splits import annual_walk_forward_folds


ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def test_annual_folds_are_disjoint_and_embargoed(self) -> None:
        config = load_config(ROOT / "configs" / "demo_synthetic.yaml")
        panel = generate_synthetic_panel(config)
        calendar = panel["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
        folds = annual_walk_forward_folds(calendar, config)
        self.assertEqual([fold.name for fold in folds], ["2023", "2024", "2025"])
        for fold in folds:
            train_end_position = int(calendar[calendar == fold.train_end].index[0])
            test_start_position = int(calendar[calendar == fold.test_start].index[0])
            self.assertEqual(test_start_position - train_end_position - 1, fold.embargo_trading_days)
            self.assertLess(fold.train_end, fold.test_start)


if __name__ == "__main__":
    unittest.main()
