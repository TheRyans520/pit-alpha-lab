"""Annual expanding-window folds with trading-date embargoes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_trading_days: int

    def training_mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.train_start, self.train_end, inclusive="both")

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.test_start, self.test_end, inclusive="both")


def annual_walk_forward_folds(dates: pd.Series | pd.DatetimeIndex, config: Mapping[str, Any]) -> list[WalkForwardFold]:
    """Create one fold per test calendar year using a trading-date embargo."""

    calendar = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).drop_duplicates().sort_values()))
    split_config = config["splits"]
    train_start = pd.Timestamp(split_config["initial_train"][0])
    requested_test_start = pd.Timestamp(split_config["test"][0])
    requested_test_end = pd.Timestamp(split_config["test"][1])
    embargo = int(split_config["embargo_trading_days"])
    eligible_test = calendar[(calendar >= requested_test_start) & (calendar <= requested_test_end)]
    if eligible_test.empty:
        raise ValueError("configured test window contains no trading dates")

    folds: list[WalkForwardFold] = []
    for year in sorted(set(eligible_test.year)):
        test_dates = eligible_test[eligible_test.year == year]
        first_test_position = calendar.get_loc(test_dates[0])
        train_end_position = first_test_position - embargo - 1
        if train_end_position < 0:
            raise ValueError(f"not enough history for fold {year} and embargo {embargo}")
        train_end = calendar[train_end_position]
        if train_end < train_start:
            raise ValueError(f"empty training window for fold {year}")
        folds.append(
            WalkForwardFold(
                name=str(year),
                train_start=train_start,
                train_end=train_end,
                test_start=test_dates[0],
                test_end=test_dates[-1],
                embargo_trading_days=embargo,
            )
        )
    return folds
