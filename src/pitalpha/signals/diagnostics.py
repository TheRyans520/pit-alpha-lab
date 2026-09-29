"""Signal monotonicity, decay and coverage diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def quantile_return_diagnostics(predictions: pd.DataFrame, quantiles: int) -> tuple[pd.DataFrame, dict[str, float]]:
    daily_rows: list[dict[str, object]] = []
    for timestamp, group in predictions.groupby("datetime", sort=True):
        usable = group[["score", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < quantiles:
            continue
        ranks = usable["score"].rank(method="first")
        buckets = pd.qcut(ranks, q=quantiles, labels=False) + 1
        for bucket, values in usable.assign(quantile=buckets).groupby("quantile", sort=True):
            daily_rows.append(
                {
                    "datetime": pd.Timestamp(timestamp),
                    "quantile": int(bucket),
                    "observations": int(len(values)),
                    "mean_forward_return": float(values["label"].mean()),
                }
            )
    daily = pd.DataFrame(daily_rows)
    summary = (
        daily.groupby("quantile", sort=True)
        .agg(days=("datetime", "nunique"), mean_daily_forward_return=("mean_forward_return", "mean"))
        .reset_index()
    )
    monotonicity = float(summary["quantile"].corr(summary["mean_daily_forward_return"], method="spearman"))
    spread = float(
        summary.loc[summary["quantile"] == quantiles, "mean_daily_forward_return"].iloc[0]
        - summary.loc[summary["quantile"] == 1, "mean_daily_forward_return"].iloc[0]
    )
    return summary, {"quantile_monotonicity_spearman": monotonicity, "top_minus_bottom_mean_daily_return": spread}


def signal_decay_diagnostics(predictions: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        column = f"forward_return_{int(horizon)}d"
        if column not in predictions:
            rows.append({"horizon_days": int(horizon), "days": 0, "mean_rank_ic": np.nan, "rank_ic_std": np.nan})
            continue
        daily_values = []
        for _, group in predictions.groupby("datetime", sort=True):
            usable = group[["score", column]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(usable) >= 3 and usable["score"].nunique() > 1 and usable[column].nunique() > 1:
                daily_values.append(float(usable["score"].corr(usable[column], method="spearman")))
        values = pd.Series(daily_values, dtype=float)
        rows.append(
            {
                "horizon_days": int(horizon),
                "days": int(len(values)),
                "mean_rank_ic": float(values.mean()) if len(values) else np.nan,
                "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def coverage_diagnostics(predictions: pd.DataFrame) -> dict[str, float | int]:
    daily = predictions.groupby("datetime", sort=True).agg(
        rows=("instrument", "size"),
        finite_scores=("score", lambda values: int(np.isfinite(values).sum())),
        finite_labels=("label", lambda values: int(np.isfinite(values).sum())),
        finite_next_returns=("realized_return_1d", lambda values: int(np.isfinite(values).sum())),
    )
    return {
        "rows": int(len(predictions)),
        "dates": int(len(daily)),
        "score_coverage": float(daily["finite_scores"].sum() / daily["rows"].sum()),
        "label_coverage": float(daily["finite_labels"].sum() / daily["rows"].sum()),
        "next_return_coverage": float(daily["finite_next_returns"].sum() / daily["rows"].sum()),
        "minimum_daily_score_coverage": float((daily["finite_scores"] / daily["rows"]).min()),
    }
