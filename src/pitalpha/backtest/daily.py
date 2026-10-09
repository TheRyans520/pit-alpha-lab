"""Transparent daily decision-to-return portfolio ledger."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _week_key(timestamp: pd.Timestamp) -> tuple[int, int]:
    iso = timestamp.isocalendar()
    return int(iso.year), int(iso.week)


def _next_weights(weights: dict[str, float], returns: dict[str, float], gross_return: float) -> dict[str, float]:
    denominator = 1.0 + gross_return
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("cannot drift an insolvent or unvalued portfolio")
    updated = {}
    for instrument, weight in weights.items():
        realized = returns.get(instrument, np.nan)
        if not np.isfinite(realized):
            raise ValueError(f"cannot drift holding without a finite return: {instrument}")
        asset_return = float(realized)
        value = weight * (1.0 + asset_return) / denominator
        if value > 0:
            updated[instrument] = value
    return updated


def _ledger(
    predictions: pd.DataFrame,
    cost_bps: Iterable[int],
    selector,
    *,
    universe_exit_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if universe_exit_policy not in {"error", "synthetic_liquidate_at_last_mark"}:
        raise ValueError("unsupported universe exit policy")
    required = {"datetime", "instrument", "score", "realized_return_1d"}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("portfolio input must contain nonempty dated predictions and returns")
    if predictions[["datetime", "instrument"]].isna().any().any():
        raise ValueError("portfolio keys must not be missing")
    if predictions.duplicated(["datetime", "instrument"]).any():
        raise ValueError("duplicate instrument-date portfolio keys")
    cost_bps = tuple(cost_bps)
    if not cost_bps or any(not np.isfinite(cost) or cost < 0 for cost in cost_bps):
        raise ValueError("costs must be finite and non-negative")
    previous_weights: dict[str, float] = {}
    previous_week: tuple[int, int] | None = None
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []

    for timestamp, group in predictions.groupby("datetime", sort=True):
        timestamp = pd.Timestamp(timestamp)
        group = group.sort_values("instrument", kind="stable")
        names = set(group["instrument"])
        disappeared = sorted(set(previous_weights) - names)
        if disappeared and universe_exit_policy == "error":
            raise ValueError(
                f"held instruments disappeared at {timestamp.date()}: {disappeared}; "
                "verified exit valuation and tradeability evidence is required"
            )
        carried = {name: weight for name, weight in previous_weights.items() if name in names}
        week = _week_key(timestamp)
        rebalance = week != previous_week
        target = selector(group, carried) if rebalance else carried
        union = sorted(set(previous_weights) | set(target))
        turnover = float(
            math.fsum(abs(target.get(name, 0.0) - previous_weights.get(name, 0.0)) for name in union)
        )

        realized = dict(zip(group["instrument"], group["realized_return_1d"], strict=True))
        missing_return_weight = float(
            math.fsum(
                target[name]
                for name in sorted(target)
                if not np.isfinite(realized.get(name, np.nan))
            )
        )
        if missing_return_weight > 0.0:
            raise ValueError(
                f"selected holdings have unavailable next-period returns at {timestamp.date()}: "
                f"{sorted(name for name in target if not np.isfinite(realized.get(name, np.nan)))}; "
                "do not infer trade eligibility from future outcome availability"
            )
        if any(float(realized[name]) < -1.0 for name in target):
            raise ValueError("asset return below -100% is invalid for the long-only ledger")
        gross_return = float(
            math.fsum(
                target[name] * float(realized[name])
                for name in sorted(target)
                if np.isfinite(realized.get(name, np.nan))
            )
        )
        if gross_return <= -1.0:
            raise ValueError("portfolio is insolvent; cannot continue the ledger")
        weight_hhi = float(math.fsum(weight**2 for weight in target.values()))
        row: dict[str, object] = {
            "datetime": timestamp,
            "rebalance": rebalance,
            "selected_count": len(target),
            "invested_weight": float(math.fsum(target.values())),
            "cash_weight": max(0.0, 1.0 - float(math.fsum(target.values()))),
            "assumed_exit_count": len(disappeared),
            "missing_return_weight": missing_return_weight,
            "turnover": turnover,
            "gross_return": gross_return,
            "weight_hhi": weight_hhi,
            "effective_positions": float(1.0 / weight_hhi) if weight_hhi > 0.0 else 0.0,
        }
        for basis_points in cost_bps:
            net = gross_return - turnover * basis_points / 10_000.0
            if net <= -1.0:
                raise ValueError("portfolio is insolvent after transaction costs")
            row[f"net_return_{basis_points}bps"] = net
        daily_rows.append(row)

        scores = dict(zip(group["instrument"], group["score"], strict=True))
        for name in union:
            weight = target.get(name, 0.0)
            asset_return = float(realized[name]) if weight > 0.0 else 0.0
            holding_rows.append(
                {
                    "datetime": timestamp,
                    "instrument": name,
                    "rebalance": rebalance,
                    "previous_weight": previous_weights.get(name, 0.0),
                    "target_weight": weight,
                    "asset_return": asset_return,
                    "gross_contribution": weight * asset_return,
                    "turnover_contribution": abs(weight - previous_weights.get(name, 0.0)),
                    "assumed_universe_exit": name in disappeared,
                    "score": scores.get(name, np.nan),
                }
            )

        previous_weights = _next_weights(target, realized, gross_return)
        previous_week = week

    return pd.DataFrame(daily_rows), pd.DataFrame(holding_rows)


def build_ranked_portfolio(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    cost_bps: Iterable[int],
    universe_exit_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    def selector(group: pd.DataFrame, previous: dict[str, float]) -> dict[str, float]:
        del previous
        usable = group[np.isfinite(group["score"])].nlargest(top_k, "score", keep="first")
        if usable.empty:
            return {}
        weight = 1.0 / len(usable)
        return {str(name): weight for name in usable["instrument"]}

    return _ledger(predictions, cost_bps, selector, universe_exit_policy=universe_exit_policy)


def build_buffered_ranked_portfolio(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    retention_rank: int,
    cost_bps: Iterable[int],
    universe_exit_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal-weight Top-K holdings with a transparent rank retention band.

    Existing names are retained while their current score rank is no worse than
    ``retention_rank``. Vacancies are filled from the best-ranked eligible names.
    This changes holdings only on scheduled rebalance dates and never uses returns.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if retention_rank < top_k:
        raise ValueError("retention_rank must be at least top_k")

    def selector(group: pd.DataFrame, previous: dict[str, float]) -> dict[str, float]:
        ranked = group[np.isfinite(group["score"])].sort_values(
            ["score", "instrument"], ascending=[False, True], kind="stable"
        )
        ordered = [str(name) for name in ranked["instrument"]]
        rank = {name: position for position, name in enumerate(ordered, start=1)}
        retained = sorted(
            (name for name in previous if rank.get(name, retention_rank + 1) <= retention_rank),
            key=rank.__getitem__,
        )[:top_k]
        selected = list(retained)
        for name in ordered:
            if len(selected) >= top_k:
                break
            if name not in selected:
                selected.append(name)
        if not selected:
            return {}
        weight = 1.0 / len(selected)
        return {name: weight for name in selected}

    return _ledger(predictions, cost_bps, selector, universe_exit_policy=universe_exit_policy)


def build_equal_weight_benchmark(
    predictions: pd.DataFrame,
    *,
    cost_bps: Iterable[int],
    universe_exit_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def selector(group: pd.DataFrame, previous: dict[str, float]) -> dict[str, float]:
        del previous
        names = sorted(str(name) for name in group["instrument"].unique())
        weight = 1.0 / len(names)
        return {name: weight for name in names}

    return _ledger(predictions, cost_bps, selector, universe_exit_policy=universe_exit_policy)
