"""Cross-sectional prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_information_coefficients(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, group in predictions.groupby("datetime", sort=True):
        usable = group[["score", "label"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable) < 3 or usable["score"].nunique() < 2 or usable["label"].nunique() < 2:
            ic = np.nan
            rank_ic = np.nan
        else:
            ic = usable["score"].corr(usable["label"], method="pearson")
            rank_ic = usable["score"].corr(usable["label"], method="spearman")
        rows.append({"datetime": pd.Timestamp(timestamp), "observations": len(usable), "ic": ic, "rank_ic": rank_ic})
    return pd.DataFrame(rows)


def _window_summary(frame: pd.DataFrame, window: str) -> dict[str, object]:
    ic = frame["ic"].dropna()
    rank_ic = frame["rank_ic"].dropna()
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else np.nan
    return {
        "window": window,
        "days": int(len(frame)),
        "ic": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "ic_std": ic_std,
        "rank_ic_std": rank_std,
        "icir": float(ic.mean() / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        "rank_icir": float(rank_ic.mean() / rank_std) if np.isfinite(rank_std) and rank_std > 0 else np.nan,
    }


def summarize_information_coefficients(daily: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for year, frame in daily.groupby(daily["datetime"].dt.year, sort=True):
        summaries.append(_window_summary(frame, str(year)))
    summaries.append(_window_summary(daily, "ALL"))
    return pd.DataFrame(summaries)
