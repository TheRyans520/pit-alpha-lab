"""Export an independently reconciled, explicitly synthetic execution case study."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pitalpha.artifacts import sha256_file, write_frame_atomic, write_json_atomic, write_text_atomic
from pitalpha.artifacts.manifest import source_identity
from pitalpha.backtest.execution import Decision, MarketEvent, replay_execution
from pitalpha.backtest.execution_audit import reconcile_execution
from pitalpha.environment import repository_root
from pitalpha.portfolio import ExecutionConstraints


def execution_fixture() -> tuple[list[MarketEvent], ExecutionConstraints]:
    events = []
    scenarios = [
        ({"A": 10., "B": 20.}, {"A": .5, "B": .5}, {"A": True, "B": False}, {"A": True, "B": True}, {"A": 2e6, "B": 2e6}),
        ({"A": 11., "B": 19.}, {"B": 1.}, {"A": True, "B": True}, {"A": False, "B": True}, {"A": 2e6, "B": 2e6}),
        ({"A": 10.5, "B": 21.}, {"B": 1.}, {"A": True, "B": True}, {"A": True, "B": True}, {"A": 100000., "B": 80000.}),
        ({"A": 10.7, "B": 20.}, None, {}, {}, {}),
        ({"A": 10.8, "B": 20.5}, {}, {"A": True, "B": True}, {"A": True, "B": True}, {"A": 2e6, "B": 2e6}),
    ]
    for index, (prices, weights, buys, sells, adv) in enumerate(scenarios):
        at = pd.Timestamp("2026-01-05T01:30:00Z") + pd.offsets.BDay(index)
        decision_at = at - pd.Timedelta(hours=17)
        events.append(MarketEvent(at=at, prices=prices, can_buy=buys, can_sell=sells,
                                 decision=Decision(decision_at, weights) if weights is not None else None,
                                 average_daily_value=adv, adv_asof=decision_at))
    return events, ExecutionConstraints(max_participation_rate=.1, portfolio_notional=100000.)


def run_execution_demo(output_root: Path | None = None) -> dict:
    events, constraints = execution_fixture()
    cash, cost = 100000., 10.
    result = replay_execution(events, initial_cash=cash, fee_bps=cost, constraints=constraints)
    audit = reconcile_execution(events, result.daily, result.positions, initial_cash=cash,
                                fee_bps=cost, constraints=constraints)
    run_id = "execution-synthetic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (output_root if output_root is not None else repository_root() / "artifacts" / "execution-demo") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {"evidence_level": "L0_synthetic", "initial_cash": cash, "fee_bps": cost,
              "constraints": asdict(constraints), "events": [asdict(event) for event in events]}
    # Dataclass timestamps are serialized explicitly; no source market data enters this demo.
    from pitalpha.artifacts import json_ready
    files = [write_json_atomic(directory / "inputs.json", json_ready(inputs)),
             write_frame_atomic(result.daily, directory / "execution_daily.csv"),
             write_frame_atomic(result.positions, directory / "execution_positions.csv"),
             write_json_atomic(directory / "reconciliation.json", audit)]
    report = f"""# Synthetic cash-funded execution demo

Five events cover initial entry with a blocked buy, a blocked sell, ADV-clipped
orders, mark-only valuation and fee-charged liquidation. Initial capital: {cash:.2f};
one-way proportional fees: {cost:g} bps. Final NAV: {audit['final_nav']:.6f}.
Total fees: {audit['fees_paid']:.6f}. Independent reconciliation: {audit['status']}.

This is a deterministic accounting fixture, not an alpha backtest. Fractional
shares, supplied marks/masks and immediate settlement are assumed. There is no
order book, slippage, market impact, lot sizing, calibrated capacity or corporate
action model. Position and participation caps use pre-trade NAV. Fees can change
post-trade weights. The separate historical CSI300 cost-overlay study is unchanged.
"""
    files.append(write_text_atomic(report, directory / "report.md"))
    manifest = {"schema_version": 1, "kind": "execution_sandbox", "status": "completed",
                "run_id": run_id, "evidence_level": "L0_synthetic", "source": source_identity(repository_root()),
                "artifacts": [{"path": f.name, "sha256": sha256_file(f), "bytes": f.stat().st_size} for f in files]}
    write_json_atomic(directory / "run_manifest.json", manifest)
    return {"run_id": run_id, "run_directory": str(directory.resolve()), "reconciliation": audit}
