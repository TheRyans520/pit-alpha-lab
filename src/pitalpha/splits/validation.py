"""Purged inner validation for labels observed after their decision date."""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_validation_split(
    train: pd.DataFrame, *, validation_dates: int, embargo_trading_days: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    # Count sessions before dropping missing labels; availability must not move
    # the calendar boundary or shrink the embargo.
    finite = np.isfinite(train["label"].to_numpy(dtype=float))
    last_labeled_date = train.loc[finite, "datetime"].max()
    calendar = pd.DatetimeIndex(pd.to_datetime(train["datetime"]).drop_duplicates().sort_values())
    calendar = calendar[calendar <= last_labeled_date]
    if validation_dates <= 0 or embargo_trading_days < 6:
        raise ValueError("validation requires positive dates and at least six embargo sessions")
    split = len(calendar) - validation_dates
    if split <= embargo_trading_days:
        raise ValueError("training window lacks dates for purged chronological validation")
    start = calendar[split]
    end = calendar[split - embargo_trading_days - 1]
    subtrain = train.loc[finite & (train["datetime"] <= end)].copy().reset_index(drop=True)
    validation = train.loc[finite & (train["datetime"] >= start)].copy().reset_index(drop=True)
    if subtrain.empty or validation.empty:
        raise ValueError("purged training or validation has no finite labels")
    return subtrain, validation, {
        "validation_start": start.date().isoformat(),
        "subtrain_end": end.date().isoformat(),
        "validation_embargo_trading_days": embargo_trading_days,
        "purged_rows": int(((train["datetime"] > end) & (train["datetime"] < start)).sum()),
    }
