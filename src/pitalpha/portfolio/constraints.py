"""Deterministic market-feasibility constraints for target portfolio weights.

The model proposes desired weights. This module applies higher-priority execution and
capital constraints without consulting future returns. It intentionally keeps the
constraint engine independent from any neural architecture.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


def _validated_weights(weights: Mapping[str, float], name: str) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for instrument, raw_weight in weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name}[{instrument!r}] must be finite and non-negative")
        if weight > 0.0:
            cleaned[str(instrument)] = weight
    return cleaned


@dataclass(frozen=True)
class ExecutionConstraints:
    """Hard limits applied after a model or selector proposes target weights."""

    max_weight: float = 1.0
    max_gross_exposure: float = 1.0
    max_turnover: float | None = None
    max_participation_rate: float | None = None
    portfolio_notional: float | None = None
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0.0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")
        if self.max_turnover is not None and not 0.0 <= self.max_turnover <= 2.0:
            raise ValueError("max_turnover must be in [0, 2]")
        participation_fields = (self.max_participation_rate, self.portfolio_notional)
        if (participation_fields[0] is None) != (participation_fields[1] is None):
            raise ValueError(
                "max_participation_rate and portfolio_notional must be configured together"
            )
        if self.max_participation_rate is not None and not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if self.portfolio_notional is not None and (
            not math.isfinite(self.portfolio_notional) or self.portfolio_notional <= 0.0
        ):
            raise ValueError("portfolio_notional must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")


@dataclass(frozen=True)
class ConstraintDiagnostics:
    """Auditable explanation of how requested weights became executable weights."""

    requested_turnover: float
    executed_turnover: float
    invested_weight: float
    cash_weight: float
    blocked_buys: tuple[str, ...]
    blocked_sells: tuple[str, ...]
    participation_clips: tuple[str, ...]
    grandfathered_overweights: tuple[str, ...]
    turnover_scale: float
    buy_scale: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def project_bounded_long_only(
    weights: Mapping[str, float],
    *,
    max_weight: float,
    max_gross_exposure: float,
) -> dict[str, float]:
    """Project non-negative desired weights into a capped long-only simplex.

    The projection never creates leverage. If the position cap prevents full
    investment, the residual is cash instead of being silently redistributed.
    """

    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if not 0.0 < max_gross_exposure <= 1.0:
        raise ValueError("max_gross_exposure must be in (0, 1]")
    desired = _validated_weights(weights, "weights")
    clipped = {name: min(weight, max_weight) for name, weight in desired.items()}
    total = math.fsum(clipped.values())
    if total <= max_gross_exposure:
        return clipped

    # Euclidean projection onto sum(w) <= gross with box constraints. The
    # monotone threshold is solved deterministically by bisection.
    values = list(desired.values())
    lower = min(value - max_weight for value in values)
    upper = max(values)
    for _ in range(100):
        threshold = (lower + upper) / 2.0
        projected_total = math.fsum(min(max(value - threshold, 0.0), max_weight) for value in values)
        if projected_total > max_gross_exposure:
            lower = threshold
        else:
            upper = threshold
    threshold = upper
    projected = {
        name: min(max(weight - threshold, 0.0), max_weight)
        for name, weight in desired.items()
    }
    return {name: weight for name, weight in projected.items() if weight > 0.0}


def enforce_execution_constraints(
    previous_weights: Mapping[str, float],
    desired_weights: Mapping[str, float],
    *,
    can_buy: Mapping[str, bool],
    can_sell: Mapping[str, bool],
    constraints: ExecutionConstraints,
    average_daily_value: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], ConstraintDiagnostics]:
    """Convert desired weights into a self-financing executable target.

    ``can_buy`` and ``can_sell`` are point-in-time execution masks. They can
    encode suspension, price-limit and settlement rules such as T+1. Missing
    mask entries fail closed. When participation limits are enabled, missing or
    non-positive ADV also prevents the requested trade.
    """

    previous = _validated_weights(previous_weights, "previous_weights")
    for mask in (can_buy, can_sell):
        if any(type(value) is not bool for value in mask.values()):
            raise ValueError("tradeability masks must contain explicit boolean values")
    if math.fsum(previous.values()) > constraints.max_gross_exposure + constraints.tolerance:
        raise ValueError("previous_weights exceed max_gross_exposure")
    desired = project_bounded_long_only(
        desired_weights,
        max_weight=constraints.max_weight,
        max_gross_exposure=constraints.max_gross_exposure,
    )
    names = sorted(set(previous) | set(desired))
    requested_deltas = {name: desired.get(name, 0.0) - previous.get(name, 0.0) for name in names}
    requested_turnover = float(math.fsum(abs(delta) for delta in requested_deltas.values()))

    blocked_buys: list[str] = []
    blocked_sells: list[str] = []
    participation_clips: list[str] = []
    executable_deltas: dict[str, float] = {}
    for name in names:
        delta = requested_deltas[name]
        if delta > constraints.tolerance and not bool(can_buy.get(name, False)):
            blocked_buys.append(name)
            delta = 0.0
        elif delta < -constraints.tolerance and not bool(can_sell.get(name, False)):
            blocked_sells.append(name)
            delta = 0.0

        if abs(delta) > constraints.tolerance and constraints.max_participation_rate is not None:
            adv = float((average_daily_value or {}).get(name, math.nan))
            if not math.isfinite(adv) or adv <= 0.0:
                allowed_weight = 0.0
            else:
                allowed_weight = (
                    constraints.max_participation_rate * adv / float(constraints.portfolio_notional)
                )
            if abs(delta) > allowed_weight + constraints.tolerance:
                delta = math.copysign(allowed_weight, delta) if allowed_weight > 0.0 else 0.0
                participation_clips.append(name)
        executable_deltas[name] = delta

    constrained_turnover = float(math.fsum(abs(delta) for delta in executable_deltas.values()))
    turnover_scale = 1.0
    if (
        constraints.max_turnover is not None
        and constrained_turnover > constraints.max_turnover + constraints.tolerance
    ):
        turnover_scale = constraints.max_turnover / constrained_turnover
        executable_deltas = {
            name: delta * turnover_scale for name, delta in executable_deltas.items()
        }

    # Execute sells before buys. Buys are scaled to available cash so blocked
    # sells can never create implicit leverage.
    target = dict(previous)
    for name, delta in executable_deltas.items():
        if delta < 0.0:
            target[name] = max(0.0, previous.get(name, 0.0) + delta)
    invested_after_sells = math.fsum(target.values())
    available_cash = max(0.0, constraints.max_gross_exposure - invested_after_sells)
    requested_buys = math.fsum(max(delta, 0.0) for delta in executable_deltas.values())
    buy_scale = min(1.0, available_cash / requested_buys) if requested_buys > 0.0 else 1.0
    for name, delta in executable_deltas.items():
        if delta > 0.0:
            target[name] = previous.get(name, 0.0) + delta * buy_scale

    target = {
        name: weight
        for name, weight in target.items()
        if weight > constraints.tolerance
    }
    invested_weight = float(math.fsum(target.values()))
    if invested_weight > constraints.max_gross_exposure + 10.0 * constraints.tolerance:
        raise RuntimeError("execution constraints created leverage")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in target.values()):
        raise RuntimeError("execution constraints created invalid weights")

    executed_turnover = float(
        math.fsum(abs(target.get(name, 0.0) - previous.get(name, 0.0)) for name in names)
    )
    grandfathered = tuple(
        sorted(name for name, weight in target.items() if weight > constraints.max_weight + constraints.tolerance)
    )
    diagnostics = ConstraintDiagnostics(
        requested_turnover=requested_turnover,
        executed_turnover=executed_turnover,
        invested_weight=invested_weight,
        cash_weight=max(0.0, 1.0 - invested_weight),
        blocked_buys=tuple(blocked_buys),
        blocked_sells=tuple(blocked_sells),
        participation_clips=tuple(participation_clips),
        grandfathered_overweights=grandfathered,
        turnover_scale=turnover_scale,
        buy_scale=buy_scale,
    )
    return target, diagnostics
