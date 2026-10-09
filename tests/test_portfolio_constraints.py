from __future__ import annotations

import unittest

from pitalpha.portfolio import (
    ExecutionConstraints,
    enforce_execution_constraints,
    project_bounded_long_only,
)


class PortfolioConstraintTests(unittest.TestCase):
    def test_projection_solves_original_weights_not_preclipped_weights(self):
        result = project_bounded_long_only({"A": 1., "B": .3}, max_weight=.4, max_gross_exposure=.5)
        self.assertAlmostEqual(result["A"], .4)
        self.assertAlmostEqual(result["B"], .1)

    def test_nonfinite_notional_and_ambiguous_masks_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                ExecutionConstraints(max_participation_rate=.1, portfolio_notional=value)
        for value in (float("nan"), "false", 1):
            with self.assertRaisesRegex(ValueError, "boolean"):
                enforce_execution_constraints({}, {"A": 1.}, can_buy={"A": value}, can_sell={},
                                              constraints=ExecutionConstraints())
    def test_invalid_participation_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            ExecutionConstraints(max_participation_rate=0.05)

    def test_bounded_projection_never_creates_leverage(self) -> None:
        projected = project_bounded_long_only(
            {"A": 0.8, "B": 0.8, "C": 0.1},
            max_weight=0.4,
            max_gross_exposure=0.75,
        )
        self.assertLessEqual(sum(projected.values()), 0.75 + 1e-12)
        self.assertLessEqual(max(projected.values()), 0.4)
        self.assertTrue(all(weight >= 0.0 for weight in projected.values()))

    def test_blocked_sell_has_precedence_and_buys_use_available_cash(self) -> None:
        target, diagnostics = enforce_execution_constraints(
            {"A": 0.5, "B": 0.5},
            {"C": 1.0},
            can_buy={"C": True},
            can_sell={"A": False, "B": True},
            constraints=ExecutionConstraints(),
        )
        self.assertEqual(target, {"A": 0.5, "C": 0.5})
        self.assertEqual(diagnostics.blocked_sells, ("A",))
        self.assertAlmostEqual(diagnostics.buy_scale, 0.5)
        self.assertAlmostEqual(diagnostics.invested_weight, 1.0)

    def test_turnover_cap_scales_the_self_financing_trade(self) -> None:
        target, diagnostics = enforce_execution_constraints(
            {"A": 1.0},
            {"B": 1.0},
            can_buy={"B": True},
            can_sell={"A": True},
            constraints=ExecutionConstraints(max_turnover=0.4),
        )
        self.assertAlmostEqual(target["A"], 0.8)
        self.assertAlmostEqual(target["B"], 0.2)
        self.assertAlmostEqual(diagnostics.requested_turnover, 2.0)
        self.assertAlmostEqual(diagnostics.executed_turnover, 0.4)
        self.assertAlmostEqual(diagnostics.turnover_scale, 0.2)

    def test_participation_cap_fails_closed_on_missing_adv(self) -> None:
        target, diagnostics = enforce_execution_constraints(
            {},
            {"A": 0.6, "B": 0.4, "C": 0.2},
            can_buy={"A": True, "B": True, "C": True},
            can_sell={},
            average_daily_value={"A": 100.0, "B": 50.0},
            constraints=ExecutionConstraints(
                max_participation_rate=0.10,
                portfolio_notional=1_000.0,
            ),
        )
        self.assertAlmostEqual(target["A"], 0.01)
        self.assertAlmostEqual(target["B"], 0.005)
        self.assertNotIn("C", target)
        self.assertEqual(diagnostics.participation_clips, ("A", "B", "C"))
        self.assertAlmostEqual(diagnostics.cash_weight, 0.985)

    def test_missing_tradeability_mask_fails_closed(self) -> None:
        target, diagnostics = enforce_execution_constraints(
            {},
            {"A": 1.0},
            can_buy={},
            can_sell={},
            constraints=ExecutionConstraints(),
        )
        self.assertEqual(target, {})
        self.assertEqual(diagnostics.blocked_buys, ("A",))


if __name__ == "__main__":
    unittest.main()
