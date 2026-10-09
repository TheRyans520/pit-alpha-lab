from __future__ import annotations

import unittest

import pandas as pd

from pitalpha.metrics import breakeven_cost_bps, return_metrics


class ReturnMetricTests(unittest.TestCase):
    def test_missing_or_impossible_return_is_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                return_metrics(pd.Series([0.01, value]))

    def test_first_day_loss_is_a_drawdown_from_initial_capital(self) -> None:
        metrics = return_metrics(pd.Series([-0.10]))
        self.assertAlmostEqual(metrics["cumulative_return"], -0.10)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.10)

    def test_drawdown_uses_compounded_wealth(self) -> None:
        metrics = return_metrics(pd.Series([0.10, -0.20, 0.05]))
        expected = (1.10 * 0.80 / 1.10) - 1.0
        self.assertAlmostEqual(metrics["maximum_drawdown"], expected)

    def test_risk_diagnostics_are_explicit_losses(self) -> None:
        metrics = return_metrics(pd.Series([-0.10, 0.02, 0.03]))
        self.assertAlmostEqual(metrics["worst_day"], -0.10)
        self.assertGreater(metrics["historical_var_95_loss"], 0.0)
        self.assertGreaterEqual(metrics["historical_cvar_95_loss"], metrics["historical_var_95_loss"])
        self.assertGreater(metrics["annualized_volatility"], 0.0)

    def test_breakeven_cost_matches_single_period_arithmetic(self) -> None:
        daily = pd.DataFrame({"gross_return": [0.01], "turnover": [1.0]})
        self.assertAlmostEqual(breakeven_cost_bps(daily), 100.0, places=8)

    def test_unprofitable_gross_strategy_has_zero_breakeven_cost(self) -> None:
        daily = pd.DataFrame({"gross_return": [-0.01], "turnover": [1.0]})
        self.assertEqual(breakeven_cost_bps(daily), 0.0)


if __name__ == "__main__":
    unittest.main()
