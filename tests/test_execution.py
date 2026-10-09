from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pitalpha.backtest.execution import Decision, MarketEvent, replay_execution
from pitalpha.backtest.execution_audit import reconcile_execution
from pitalpha.execution_demo import execution_fixture, run_execution_demo
from pitalpha.portfolio import ExecutionConstraints


def event(day, prices, weights=None, buys=None, sells=None):
    at = pd.Timestamp(f"2026-01-{day:02d}T01:30:00Z")
    return MarketEvent(at, prices, buys or {}, sells or {},
                       Decision(at - pd.Timedelta(hours=17), weights) if weights is not None else None)


class ExecutionTests(unittest.TestCase):
    def test_entry_and_exit_pay_fees_from_cash_with_known_share_arithmetic(self):
        events = [event(5, {"A": 10.}, {"A": 1.}, {"A": True}),
                  event(6, {"A": 12.}, {}, sells={"A": True})]
        result = replay_execution(events, initial_cash=100., fee_bps=100.)
        quantity = 100. / (10. * 1.01)
        self.assertAlmostEqual(result.positions.iloc[0].quantity, quantity)
        self.assertAlmostEqual(result.daily.iloc[0].cash, 0.)
        self.assertAlmostEqual(result.daily.iloc[-1].cash, quantity * 12. * .99)
        self.assertAlmostEqual(result.daily.iloc[-1].invested_value, 0.)
        audit = reconcile_execution(events, result.daily, result.positions, initial_cash=100.,
                                    fee_bps=100., constraints=ExecutionConstraints())
        self.assertEqual(audit["status"], "pass")

    def test_blocked_sell_is_retained_and_cannot_fund_a_purchase(self):
        events = [event(5, {"A": 10.}, {"A": 1.}, {"A": True}),
                  event(6, {"A": 11., "B": 20.}, {"B": 1.}, {"B": True}, {"A": False})]
        result = replay_execution(events, initial_cash=100., fee_bps=10.)
        self.assertEqual(result.daily.iloc[-1].blocked_sells, "A")
        self.assertAlmostEqual(result.daily.iloc[-1].buy_notional, 0.)
        self.assertAlmostEqual(result.daily.iloc[-1].mark_pnl, 100. / 1.001 * .1)
        self.assertAlmostEqual(result.daily.iloc[-1].fees, 0.)

    def test_missing_held_price_cannot_be_a_free_exit(self):
        events = [event(5, {"A": 10.}, {"A": 1.}, {"A": True}), event(6, {}, {})]
        with self.assertRaisesRegex(ValueError, "valuation for A"):
            replay_execution(events, initial_cash=100., fee_bps=0.)

    def test_unknown_side_blocks_buy_and_invalid_side_is_rejected(self):
        result = replay_execution([event(5, {"A": 10.}, {"A": 1.})], initial_cash=100., fee_bps=10.)
        self.assertEqual(result.daily.iloc[0].cash, 100.)
        with self.assertRaisesRegex(ValueError, "boolean"):
            replay_execution([event(5, {"A": 10.}, {"A": 1.}, {"A": float("nan")})],
                             initial_cash=100., fee_bps=10.)

    def test_future_decision_or_adv_and_reordered_events_are_rejected(self):
        events, limits = execution_fixture()
        bad_decision = replace(events[0], decision=Decision(events[0].at, {"A": 1.}))
        with self.assertRaisesRegex(ValueError, "precede"):
            replay_execution([bad_decision], initial_cash=100000., fee_bps=10.)
        bad_adv = replace(events[0], adv_asof=events[0].at)
        with self.assertRaisesRegex(ValueError, "ADV timestamp"):
            replay_execution([bad_adv], initial_cash=100000., fee_bps=10., constraints=limits)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replay_execution([events[0], events[0]], initial_cash=100000., fee_bps=10.)

    def test_fixture_clips_adv_and_independent_audit_detects_tampering(self):
        events, limits = execution_fixture()
        result = replay_execution(events, initial_cash=100000., fee_bps=10., constraints=limits)
        self.assertEqual(result.daily.iloc[2].participation_clips, "A,B")
        self.assertAlmostEqual(result.daily.iloc[2].buy_notional, 8000.)
        self.assertAlmostEqual(result.daily.iloc[2].sell_notional, 10000.)
        self.assertEqual(reconcile_execution(events, result.daily, result.positions, initial_cash=100000.,
                                            fee_bps=10., constraints=limits)["status"], "pass")
        changed = result.daily.copy()
        changed.loc[0, "cash"] += 1.
        with self.assertRaisesRegex(ValueError, "closing cash"):
            reconcile_execution(events, changed, result.positions, initial_cash=100000., fee_bps=10., constraints=limits)

    def test_future_event_changes_do_not_change_past_positions(self):
        events, limits = execution_fixture()
        first = replay_execution(events, initial_cash=100000., fee_bps=10., constraints=limits)
        altered = [*events[:-1], replace(events[-1], prices={"A": 100., "B": 200.})]
        second = replay_execution(altered, initial_cash=100000., fee_bps=10., constraints=limits)
        pd.testing.assert_frame_equal(first.daily.iloc[:-1], second.daily.iloc[:-1])
        pd.testing.assert_frame_equal(first.positions[first.positions.datetime < events[-1].at],
                                      second.positions[second.positions.datetime < events[-1].at])

    def test_seeded_paths_preserve_cash_inventory_and_turnover_limits(self):
        rng = np.random.default_rng(17)
        for seed in range(8):
            with self.subTest(seed=seed):
                prices = {"A": 10., "B": 20., "C": 30.}
                events = []
                for day in range(5, 20):
                    prices = {name: price * float(np.exp(rng.normal(0, .03))) for name, price in prices.items()}
                    weights = dict(zip(prices, rng.dirichlet([1., 1., 1.]) * .9))
                    masks = {name: bool(rng.random() > .2) for name in prices}
                    events.append(event(day, dict(prices), weights, masks, masks))
                limits = ExecutionConstraints(max_turnover=.4)
                result = replay_execution(events, initial_cash=100000., fee_bps=20., constraints=limits)
                self.assertTrue((result.daily.cash >= 0.).all())
                self.assertTrue((result.daily.turnover <= .4 + 1e-9).all())
                reconcile_execution(events, result.daily, result.positions, initial_cash=100000., fee_bps=20., constraints=limits)

    def test_exported_demo_has_explicit_l0_identity_and_matching_hashes(self):
        from pitalpha.artifacts import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            result = run_execution_demo(Path(directory))
            root = Path(result["run_directory"])
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["kind"], "execution_sandbox")
            self.assertEqual(manifest["evidence_level"], "L0_synthetic")
            self.assertEqual(result["reconciliation"]["status"], "pass")
            for artifact in manifest["artifacts"]:
                self.assertEqual(sha256_file(root / artifact["path"]), artifact["sha256"])
