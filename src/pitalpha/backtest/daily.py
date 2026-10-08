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
        return {}
    updated = {}
    for instrument, weight in weights.items():
        realized = returns.get(instrument, np.nan)
        asset_return = float(realized) if np.isfinite(realized) else 0.0
        value = weight * (1.0 + asset_return) / denominator
        if value > 0:
            updated[instrument] = value
    return updated


def _ledger(
    predictions: pd.DataFrame,
    cost_bps: Iterable[int],
    selector,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_weights: dict[str, float] = {}
    previous_week: tuple[int, int] | None = None
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []

    for timestamp, group in predictions.groupby("datetime", sort=True):
        timestamp = pd.Timestamp(timestamp)
        group = group.sort_values("instrument", kind="stable")
        names = set(group["instrument"])
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
                f"selected holdings have unavailable next-period returns at {timestamp.date()}; "
                "do not infer trade eligibility from future outcome availability"
            )
        gross_return = float(
            math.fsum(
                target[name] * float(realized[name])
                for name in sorted(target)
                if np.isfinite(realized.get(name, np.nan))
            )
        )
        weight_hhi = float(math.fsum(weight**2 for weight in target.values()))
        row: dict[str, object] = {
            "datetime": timestamp,
            "rebalance": rebalance,
            "selected_count": len(target),
            "invested_weight": float(math.fsum(target.values())),
            "missing_return_weight": missing_return_weight,
            "turnover": turnover,
            "gross_return": gross_return,
            "weight_hhi": weight_hhi,
            "effective_positions": float(1.0 / weight_hhi) if weight_hhi > 0.0 else 0.0,
        }
        for basis_points in cost_bps:
            row[f"net_return_{basis_points}bps"] = gross_return - turnover * basis_points / 10_000.0
        daily_rows.append(row)

        if rebalance:
            scores = dict(zip(group["instrument"], group["score"], strict=True))
            for name, weight in sorted(target.items()):
                holding_rows.append(
                    {
                        "datetime": timestamp,
                        "instrument": name,
                        "target_weight": weight,
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

    return _ledger(predictions, cost_bps, selector)


def build_buffered_ranked_portfolio(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    retention_rank: int,
    cost_bps: Iterable[int],
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

    return _ledger(predictions, cost_bps, selector)


def build_equal_weight_benchmark(
    predictions: pd.DataFrame,
    *,
    cost_bps: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def selector(group: pd.DataFrame, previous: dict[str, float]) -> dict[str, float]:
        del previous
        names = sorted(str(name) for name in group["instrument"].unique())
        weight = 1.0 / len(names)
        return {name: weight for name in names}

    return _ledger(predictions, cost_bps, selector)
