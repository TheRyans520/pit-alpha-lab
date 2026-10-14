"""Portfolio return and risk summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def return_metrics(returns: pd.Series) -> dict[str, float]:
    values = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -1.0):
        raise ValueError("returns must be finite and at least -100%; missing outcomes need valuation")
    if len(values) == 0:
        raise ValueError("return series is empty")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    cumulative = float(wealth[-1] - 1.0)
    annualized = float(wealth[-1] ** (252.0 / len(values)) - 1.0) if wealth[-1] > 0 else np.nan
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    sharpe = float(np.mean(values) / standard_deviation * np.sqrt(252.0)) if standard_deviation > 0 else np.nan
    annualized_volatility = standard_deviation * np.sqrt(252.0)
    downside = np.minimum(values, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(252.0))
    fifth_percentile = float(np.quantile(values, 0.05))
    tail = values[values <= fifth_percentile]
    running_peak = np.maximum.accumulate(wealth)
    maximum_drawdown = float(np.min(wealth / running_peak - 1.0))
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "annualized_volatility": annualized_volatility,
        "downside_deviation": downside_deviation,
        "worst_day": float(np.min(values)),
        "historical_var_95_loss": max(0.0, -fifth_percentile),
        "historical_cvar_95_loss": max(0.0, -float(np.mean(tail))),
        "maximum_drawdown": maximum_drawdown,
    }


def breakeven_cost_bps(daily: pd.DataFrame, *, upper_bound_bps: float = 1_000.0) -> float:
    """Return the one-way cost that reduces terminal cumulative return to zero.

    A result of zero means the gross strategy did not compound positively. ``NaN``
    means it remained profitable even at the deliberately conservative upper bound.
    """

    gross = daily["gross_return"].to_numpy(dtype=np.float64)
    turnover = daily["turnover"].to_numpy(dtype=np.float64)

    def terminal_log_wealth(basis_points: float) -> float:
        net = gross - turnover * basis_points / 10_000.0
        if np.any(net <= -1.0):
            return -np.inf
        return float(np.log1p(net).sum())

    if terminal_log_wealth(0.0) <= 0.0:
        return 0.0
    positive_turnover = turnover > 0.0
    if not np.any(positive_turnover):
        return np.nan
    solvency_limit = float(np.min((1.0 + gross[positive_turnover]) / turnover[positive_turnover]) * 10_000.0)
    high = min(float(upper_bound_bps), solvency_limit * (1.0 - 1e-12))
    if terminal_log_wealth(high) > 0.0:
        return np.nan
    low = 0.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if terminal_log_wealth(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
    return float((low + high) / 2.0)


def portfolio_summary(
    daily: pd.DataFrame,
    *,
    strategy: str,
    cost_bps: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    windows: list[tuple[str, pd.DataFrame]] = [
        (str(year), frame) for year, frame in daily.groupby(daily["datetime"].dt.year, sort=True)
    ]
    windows.append(("ALL", daily))
    for window, frame in windows:
        for basis_points in cost_bps:
            metrics = return_metrics(frame[f"net_return_{basis_points}bps"])
            rows.append(
                {
                    "strategy": strategy,
                    "window": window,
                    "cost_bps": basis_points,
                    "days": len(frame),
                    **metrics,
                    "mean_turnover": float(frame["turnover"].mean()),
                    "total_turnover": float(frame["turnover"].sum()),
                    "maximum_missing_return_weight": float(frame["missing_return_weight"].max()),
                    "mean_weight_hhi": float(frame["weight_hhi"].mean()),
                    "mean_effective_positions": float(frame["effective_positions"].mean()),
                }
            )
    return pd.DataFrame(rows)
