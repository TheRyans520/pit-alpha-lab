"""Independent inventory/cash reconciliation, without calling execution helpers."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    from pitalpha.backtest.execution import MarketEvent
    from pitalpha.portfolio import ExecutionConstraints


def reconcile_execution(
    events: Sequence[MarketEvent], daily: pd.DataFrame, positions: pd.DataFrame,
    *, initial_cash: float, fee_bps: float, constraints: ExecutionConstraints,
) -> dict[str, object]:
    def equal(actual, expected, field):
        if not math.isfinite(float(actual)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"execution reconciliation failed: {field}")

    if len(daily) != len(events):
        raise ValueError("execution reconciliation failed: event count")
    if not positions.empty and positions.duplicated(["datetime", "instrument"]).any():
        raise ValueError("execution reconciliation failed: duplicate positions")
    quantities, cash, previous_nav = {}, initial_cash, initial_cash
    seen_rows = 0
    for event, day in zip(events, daily.itertuples(), strict=True):
        if pd.Timestamp(day.datetime) != pd.Timestamp(event.at):
            raise ValueError("execution reconciliation failed: event timestamp")
        rows = positions[positions["datetime"] == day.datetime] if not positions.empty else positions
        seen_rows += len(rows)
        marks = event.prices
        before = cash + sum(quantity * marks[name] for name, quantity in quantities.items())
        equal(day.opening_cash, cash, "opening cash")
        equal(day.nav_before, before, "pre-trade NAV")
        equal(day.mark_pnl, before - previous_nav, "mark P&L")
        expected_names = set(quantities) | set(event.decision.weights if event.decision else {})
        if set(rows["instrument"] if not rows.empty else []) != expected_names:
            raise ValueError("execution reconciliation failed: missing or extra inventory rows")
        trades, new_quantities, total_fees = [], {}, 0.
        for row in rows.itertuples():
            name = row.instrument
            old = quantities.get(name, 0.)
            equal(row.previous_quantity, old, "carried inventory")
            equal(row.price, marks[name], "source mark")
            if not math.isfinite(row.quantity) or row.quantity < 0:
                raise ValueError("execution reconciliation failed: invalid inventory")
            trade = (row.quantity - old) * marks[name]
            equal(row.trade_notional, trade, "executed notional")
            equal(row.fee, abs(trade) * fee_bps / 10000., "fee")
            equal(row.market_value, row.quantity * marks[name], "position value")
            if abs(trade) > 1e-8:
                mask = event.can_buy if trade > 0 else event.can_sell
                if event.decision is None or mask.get(name) is not True:
                    raise ValueError("execution reconciliation failed: blocked order was filled")
                if constraints.max_participation_rate is not None:
                    adv = (event.average_daily_value or {}).get(name, 0.)
                    if not math.isfinite(adv) or abs(trade) > constraints.max_participation_rate * adv + 1e-8:
                        raise ValueError("execution reconciliation failed: participation limit")
            trades.append(trade)
            total_fees += abs(trade) * fee_bps / 10000.
            if row.quantity > 1e-12:
                new_quantities[name] = row.quantity
        cash -= sum(trades) + total_fees
        if cash < -1e-8:
            raise ValueError("execution reconciliation failed: negative cash")
        invested = sum(quantity * marks[name] for name, quantity in new_quantities.items())
        after = cash + invested
        equal(day.cash, cash, "closing cash")
        equal(day.fees, total_fees, "total fees")
        equal(day.invested_value, invested, "invested value")
        equal(day.nav_after, after, "post-trade NAV")
        equal(after, before - total_fees, "capital conservation")
        equal(day.net_return, after / previous_nav - 1., "net return")
        equal(day.turnover, sum(abs(value) for value in trades) / before, "turnover")
        equal(day.buy_notional, sum(max(value, 0.) for value in trades), "buys")
        equal(day.sell_notional, -sum(min(value, 0.) for value in trades), "sells")
        if constraints.max_turnover is not None and day.turnover > constraints.max_turnover + 1e-9:
            raise ValueError("execution reconciliation failed: turnover cap")
        quantities, previous_nav = new_quantities, after
    if seen_rows != len(positions):
        raise ValueError("execution reconciliation failed: unmatched position dates")
    return {"status": "pass", "events": len(events), "position_rows": len(positions),
            "final_nav": previous_nav, "fees_paid": float(daily["fees"].sum())}
