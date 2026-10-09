"""Fractional-share execution simulator with fees paid from account cash.

This is an L0 accounting engine, not an exchange fill or settlement model.
Target/participation limits are measured against pre-trade NAV. Prices and
side masks are supplied observations; unknown held valuations stop the replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping, Sequence

import pandas as pd

from pitalpha.portfolio import ExecutionConstraints, enforce_execution_constraints


@dataclass(frozen=True)
class Decision:
    at: datetime
    weights: Mapping[str, float]


@dataclass(frozen=True)
class MarketEvent:
    at: datetime
    prices: Mapping[str, float]
    can_buy: Mapping[str, bool]
    can_sell: Mapping[str, bool]
    decision: Decision | None = None
    average_daily_value: Mapping[str, float] | None = None
    adv_asof: datetime | None = None


@dataclass(frozen=True)
class ExecutionResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


def _timestamp(value: datetime) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("event and decision timestamps must be timezone-aware")
    return result.tz_convert("UTC")


def replay_execution(
    events: Sequence[MarketEvent], *, initial_cash: float, fee_bps: float,
    constraints: ExecutionConstraints | None = None,
) -> ExecutionResult:
    """Mark holdings, project orders, sell, then cash/fee-limit purchases.

    A decision must precede the execution event. ADV must be available by the
    decision timestamp. Missing side-mask entries block that side. Each event
    must value every carried holding; a market disappearance is not a free sale.
    No interest, borrow, market impact, board lots or settlement queues are modeled.
    """
    if not events or not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("nonempty events and finite positive initial cash required")
    if not math.isfinite(fee_bps) or not 0 <= fee_bps < 10000:
        raise ValueError("fee_bps must be finite and in [0, 10000)")
    limits = constraints or ExecutionConstraints()
    rate = fee_bps / 10000.
    cash = initial_cash
    quantities: dict[str, float] = {}
    previous_nav = initial_cash
    previous_time = None
    days, positions = [], []
    for event in events:
        at = _timestamp(event.at)
        if previous_time is not None and at <= previous_time:
            raise ValueError("execution events must be strictly increasing")
        if event.decision is not None and _timestamp(event.decision.at) >= at:
            raise ValueError("decision must precede execution")
        if event.decision is not None and limits.max_participation_rate is not None:
            if event.adv_asof is None or _timestamp(event.adv_asof) > _timestamp(event.decision.at):
                raise ValueError("ADV timestamp must be available by decision time")
        desired = event.decision.weights if event.decision else {}
        names = sorted(set(quantities) | set(desired))
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("instrument names must be nonempty strings")
        marks = {}
        for name in names:
            value = float(event.prices.get(name, math.nan))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"missing or invalid execution valuation for {name} at {at}")
            marks[name] = value
        opening_cash = cash
        nav_before = cash + math.fsum(quantities.get(name, 0.) * marks[name] for name in names)
        prior = {name: quantities[name] * marks[name] / nav_before for name in quantities}
        diagnostics = None
        projected = prior
        if event.decision is not None:
            current_limits = replace(limits, portfolio_notional=nav_before) if limits.max_participation_rate is not None else limits
            projected, diagnostics = enforce_execution_constraints(
                prior, desired, can_buy=event.can_buy, can_sell=event.can_sell,
                average_daily_value=event.average_daily_value, constraints=current_limits,
            )
        deltas = {name: (projected.get(name, 0.) - prior.get(name, 0.)) * nav_before for name in names}
        sold = -math.fsum(min(value, 0.) for value in deltas.values())
        cash_after_sells = cash + sold * (1. - rate)
        requested_buys = math.fsum(max(value, 0.) for value in deltas.values())
        fee_buy_scale = min(1., max(0., cash_after_sells) / (requested_buys * (1. + rate))) if requested_buys else 1.
        trades = {name: value * fee_buy_scale if value > 0 else value for name, value in deltas.items()}
        fees = rate * math.fsum(abs(value) for value in trades.values())
        cash = cash - math.fsum(trades.values()) - fees
        tolerance = max(1., nav_before) * 1e-10
        if cash < -tolerance:
            raise RuntimeError("cash conservation violated")
        cash = max(0., cash)
        next_quantities = {}
        for name in names:
            old = quantities.get(name, 0.)
            new = old + trades[name] / marks[name]
            if new < -tolerance / marks[name]:
                raise RuntimeError("negative inventory created")
            new = max(0., new)
            if new > 1e-12:
                next_quantities[name] = new
            positions.append({
                "datetime": at, "instrument": name, "price": marks[name],
                "previous_quantity": old, "quantity": new, "trade_notional": trades[name],
                "fee": abs(trades[name]) * rate, "market_value": new * marks[name],
                "desired_weight": desired.get(name, 0.) if event.decision else prior.get(name, 0.),
                "projected_weight": projected.get(name, 0.),
                "can_buy": event.can_buy.get(name, False), "can_sell": event.can_sell.get(name, False),
            })
        invested = math.fsum(next_quantities[name] * marks[name] for name in next_quantities)
        nav_after = cash + invested
        if not math.isclose(nav_after, nav_before - fees, rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError("NAV conservation violated")
        days.append({
            "datetime": at, "opening_cash": opening_cash, "cash": cash,
            "nav_before": nav_before, "nav_after": nav_after, "invested_value": invested,
            "mark_pnl": nav_before - previous_nav, "fees": fees,
            "buy_notional": math.fsum(max(v, 0.) for v in trades.values()), "sell_notional": sold,
            "turnover": math.fsum(abs(v) for v in trades.values()) / nav_before,
            "net_return": nav_after / previous_nav - 1., "fee_buy_scale": fee_buy_scale,
            "blocked_buys": ",".join(diagnostics.blocked_buys) if diagnostics else "",
            "blocked_sells": ",".join(diagnostics.blocked_sells) if diagnostics else "",
            "participation_clips": ",".join(diagnostics.participation_clips) if diagnostics else "",
        })
        cash, quantities, previous_nav, previous_time = cash, next_quantities, nav_after, at
    return ExecutionResult(pd.DataFrame(days), pd.DataFrame(positions))
