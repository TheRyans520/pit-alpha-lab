from __future__ import annotations

import unittest

import pandas as pd

from pitalpha.backtest import build_buffered_ranked_portfolio, build_ranked_portfolio


class BacktestTests(unittest.TestCase):
    def test_initial_entry_is_charged_and_non_rebalance_day_is_carried(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-05"] * 3 + ["2026-01-06"] * 3),
                "instrument": ["A", "B", "C"] * 2,
                "score": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
                "label": [0.0] * 6,
                "realized_return_1d": [0.01, 0.00, -0.01, 0.00, 0.01, -0.01],
            }
        )
        daily, holdings = build_ranked_portfolio(frame, top_k=2, cost_bps=[0, 10])
        self.assertAlmostEqual(float(daily.loc[0, "turnover"]), 1.0)
        self.assertAlmostEqual(float(daily.loc[0, "gross_return"]), 0.005)
        self.assertAlmostEqual(float(daily.loc[0, "net_return_10bps"]), 0.004)
        self.assertFalse(bool(daily.loc[1, "rebalance"]))
        self.assertAlmostEqual(float(daily.loc[1, "turnover"]), 0.0)
        self.assertEqual(len(holdings), 4)
        self.assertEqual(len(holdings[holdings["rebalance"]]), 2)

    def test_retention_band_avoids_unnecessary_weekly_replacement(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-05"] * 4 + ["2026-01-12"] * 4),
                "instrument": ["A", "B", "C", "D"] * 2,
                "score": [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
                "label": [0.0] * 8,
                "realized_return_1d": [0.0] * 8,
            }
        )
        ranked, _ = build_ranked_portfolio(frame, top_k=2, cost_bps=[0])
        buffered, holdings = build_buffered_ranked_portfolio(
            frame, top_k=2, retention_rank=3, cost_bps=[0]
        )
        self.assertAlmostEqual(float(ranked.loc[1, "turnover"]), 1.0)
        self.assertAlmostEqual(float(buffered.loc[1, "turnover"]), 0.0)
        second_week = holdings[holdings["datetime"] == pd.Timestamp("2026-01-12")]
        self.assertEqual(set(second_week["instrument"]), {"A", "B"})

    def test_selected_missing_future_return_fails_explicitly(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                "instrument": ["A", "B"],
                "score": [2.0, 1.0],
                "realized_return_1d": [float("nan"), 0.01],
            }
        )
        with self.assertRaisesRegex(ValueError, "unavailable next-period returns"):
            build_ranked_portfolio(frame, top_k=1, cost_bps=[10])

    def test_missing_unselected_outcome_does_not_change_selection(self) -> None:
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2026-01-05"] * 2),
            "instrument": ["A", "B"], "score": [2., 1.],
            "realized_return_1d": [0.02, float("nan")],
        })
        daily, holdings = build_ranked_portfolio(frame, top_k=1, cost_bps=[10])
        self.assertEqual(holdings["instrument"].tolist(), ["A"])
        self.assertAlmostEqual(daily.iloc[0]["net_return_10bps"], 0.019)

    def test_carried_missing_outcome_and_disappearance_fail(self) -> None:
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2026-01-05"] * 2 + ["2026-01-06"] * 2),
            "instrument": ["A", "B"] * 2, "score": [2., 1., 0., 10.],
            "realized_return_1d": [0., 0., float("nan"), 0.01],
        })
        for factory in (build_ranked_portfolio,
                        lambda frame, **kw: build_buffered_ranked_portfolio(frame, retention_rank=2, **kw)):
            with self.subTest(factory=factory):
                with self.assertRaisesRegex(ValueError, "unavailable next-period"):
                    factory(frame, top_k=1, cost_bps=[0])
                with self.assertRaisesRegex(ValueError, "held instruments disappeared"):
                    factory(frame.drop(index=2), top_k=1, cost_bps=[0])
        daily, holdings = build_ranked_portfolio(
            frame.drop(index=2), top_k=1, cost_bps=[10],
            universe_exit_policy="synthetic_liquidate_at_last_mark",
        )
        self.assertAlmostEqual(daily.iloc[1]["turnover"], 1.)
        self.assertAlmostEqual(daily.iloc[1]["cash_weight"], 1.)
        self.assertTrue(holdings.iloc[-1]["assumed_universe_exit"])
        self.assertEqual(holdings.iloc[-1]["target_weight"], 0.)

    def test_rebalance_charges_both_sale_and_purchase(self) -> None:
        frame = pd.DataFrame({
            "datetime": pd.to_datetime(["2026-01-05"] * 2 + ["2026-01-12"] * 2),
            "instrument": ["A", "B"] * 2, "score": [2., 1., 1., 2.],
            "realized_return_1d": [0., 0., 0., 0.],
        })
        daily, holdings = build_ranked_portfolio(frame, top_k=1, cost_bps=iter([10]))
        self.assertAlmostEqual(daily.iloc[1]["turnover"], 2.)
        self.assertAlmostEqual(daily.iloc[1]["net_return_10bps"], -0.002)
        self.assertEqual(len(holdings[holdings["target_weight"] == 0]), 1)

    def test_retention_rank_cannot_be_smaller_than_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least top_k"):
            build_buffered_ranked_portfolio(pd.DataFrame(), top_k=3, retention_rank=2, cost_bps=[0])


if __name__ == "__main__":
    unittest.main()
