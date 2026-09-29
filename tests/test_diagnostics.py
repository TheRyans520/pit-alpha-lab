from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pitalpha.signals import coverage_diagnostics, quantile_return_diagnostics, signal_decay_diagnostics
from pitalpha.statistics import moving_block_mean_interval


class DiagnosticTests(unittest.TestCase):
    def test_quantiles_recover_monotonic_order(self) -> None:
        rows = []
        for day in pd.date_range("2026-01-01", periods=4):
            for value in range(10):
                rows.append({"datetime": day, "score": float(value), "label": float(value) / 100.0})
        summary, statistics = quantile_return_diagnostics(pd.DataFrame(rows), quantiles=5)
        self.assertEqual(summary["quantile"].tolist(), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(statistics["quantile_monotonicity_spearman"], 1.0)
        self.assertGreater(statistics["top_minus_bottom_mean_daily_return"], 0)

    def test_signal_decay_handles_unavailable_horizon(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-01"] * 3),
                "score": [1.0, 2.0, 3.0],
                "forward_return_1d": [0.01, 0.02, 0.03],
            }
        )
        result = signal_decay_diagnostics(frame, [1, 5])
        self.assertAlmostEqual(float(result.loc[result["horizon_days"] == 1, "mean_rank_ic"].iloc[0]), 1.0)
        self.assertEqual(int(result.loc[result["horizon_days"] == 5, "days"].iloc[0]), 0)

    def test_coverage_is_explicit(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "instrument": ["A", "B"],
                "score": [1.0, np.nan],
                "label": [0.1, 0.2],
                "realized_return_1d": [0.01, np.nan],
            }
        )
        result = coverage_diagnostics(frame)
        self.assertEqual(result["score_coverage"], 0.5)
        self.assertEqual(result["next_return_coverage"], 0.5)

    def test_constant_bootstrap_interval_is_exact(self) -> None:
        interval = moving_block_mean_interval(
            np.ones(50) * 0.25, block_length=10, replications=100, seed=17
        )
        self.assertEqual(interval["point_estimate"], 0.25)
        self.assertEqual(interval["ci_low"], 0.25)
        self.assertEqual(interval["ci_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
