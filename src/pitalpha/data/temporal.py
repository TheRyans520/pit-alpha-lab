"""Causal, calendar-aware temporal tensor construction.

The store indexes point-in-time feature rows only. It never reads labels or forward
returns, and each sequence ends at its decision date. Missing sessions and missing
feature cells remain distinct masks so a model cannot silently interpret either as
an observed zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd


_FORBIDDEN_FEATURES = {"label", "realized_return_1d"}
_FORBIDDEN_PREFIXES = ("forward_return_",)


def _validated_feature_columns(feature_columns: Iterable[str]) -> tuple[str, ...]:
    columns = tuple(str(column) for column in feature_columns)
    if not columns:
        raise ValueError("temporal feature_columns must not be empty")
    if len(columns) != len(set(columns)):
        raise ValueError("temporal feature_columns must be unique")
    forbidden = []
    for column in columns:
        normalized = column.casefold()
        if normalized in _FORBIDDEN_FEATURES or any(
            normalized.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES
        ):
            forbidden.append(column)
    if forbidden:
        raise ValueError(f"future or outcome columns cannot be temporal features: {forbidden}")
    return columns


def _naive_datetime_index(values: Iterable[object], *, name: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if dates.hasnans:
        raise ValueError(f"{name} contains missing datetimes")
    if dates.tz is not None:
        raise ValueError(f"{name} must use timezone-naive session timestamps")
    return dates


@dataclass(frozen=True)
class TemporalSequenceBatch:
    """A prediction-aligned batch of causal temporal sequences."""

    values: np.ndarray
    observed_mask: np.ndarray
    row_mask: np.ndarray
    session_dates: np.ndarray
    decision_dates: np.ndarray
    instruments: tuple[str, ...]
    source_index: tuple[Hashable, ...]
    feature_columns: tuple[str, ...]
    lookback_sessions: int

    def __post_init__(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        sample_count = len(self.instruments)
        feature_count = len(self.feature_columns)
        expected_values = (sample_count, self.lookback_sessions, feature_count)
        expected_rows = (sample_count, self.lookback_sessions)
        if self.values.shape != expected_values:
            raise ValueError(f"values must have shape {expected_values}")
        if self.observed_mask.shape != expected_values:
            raise ValueError(f"observed_mask must have shape {expected_values}")
        if self.row_mask.shape != expected_rows:
            raise ValueError(f"row_mask must have shape {expected_rows}")
        if self.session_dates.shape != expected_rows:
            raise ValueError(f"session_dates must have shape {expected_rows}")
        if self.decision_dates.shape != (sample_count,):
            raise ValueError(f"decision_dates must have shape ({sample_count},)")
        if len(self.source_index) != sample_count:
            raise ValueError("source_index must align with samples")
        if not np.isfinite(self.values).all():
            raise ValueError("temporal values must be finite after explicit filling")
        if self.observed_mask.dtype != np.bool_ or self.row_mask.dtype != np.bool_:
            raise ValueError("temporal masks must be boolean")
        if np.any(self.observed_mask & ~self.row_mask[:, :, None]):
            raise ValueError("a feature cannot be observed when its source row is missing")
        if np.any(self.row_mask & np.isnat(self.session_dates)):
            raise ValueError("a source row cannot exist in a padded calendar position")
        if sample_count:
            if not np.array_equal(self.session_dates[:, -1], self.decision_dates):
                raise ValueError("every temporal sequence must end at its decision date")
            future = self.session_dates > self.decision_dates[:, None]
            if np.any(future & ~np.isnat(self.session_dates)):
                raise ValueError("temporal sequences cannot contain future sessions")

    @property
    def sample_count(self) -> int:
        return len(self.instruments)

    @property
    def feature_count(self) -> int:
        return len(self.feature_columns)

    def to_metadata(self) -> dict[str, object]:
        feature_cells = int(self.observed_mask.size)
        observed_cells = int(self.observed_mask.sum())
        return {
            "samples": self.sample_count,
            "lookback_sessions": self.lookback_sessions,
            "features": list(self.feature_columns),
            "feature_count": self.feature_count,
            "observed_feature_fraction": (
                observed_cells / feature_cells if feature_cells else 0.0
            ),
            "complete_history_samples": int(self.row_mask.all(axis=1).sum()),
        }


@dataclass(frozen=True)
class _InstrumentHistory:
    calendar_positions: np.ndarray
    values: np.ndarray


class CausalSequenceStore:
    """Read-only index that materializes only requested temporal batches."""

    def __init__(
        self,
        *,
        calendar: pd.DatetimeIndex,
        feature_columns: tuple[str, ...],
        histories: dict[str, _InstrumentHistory],
    ) -> None:
        self._calendar = calendar
        self._feature_columns = feature_columns
        self._histories = histories

    @classmethod
    def from_panel(
        cls,
        panel: pd.DataFrame,
        feature_columns: Iterable[str],
        *,
        calendar: Iterable[object] | None = None,
    ) -> "CausalSequenceStore":
        """Index a long panel without materializing a full stock-date-feature cube."""

        columns = _validated_feature_columns(feature_columns)
        required = {"datetime", "instrument", *columns}
        missing = sorted(required - set(panel.columns))
        if missing:
            raise ValueError(f"temporal panel is missing columns: {missing}")
        if panel.empty:
            raise ValueError("temporal panel must contain at least one row")

        indexed = panel.loc[:, ["datetime", "instrument", *columns]].copy()
        indexed["datetime"] = _naive_datetime_index(indexed["datetime"], name="panel datetime")
        indexed["instrument"] = indexed["instrument"].astype(str)
        if (indexed["instrument"].str.len() == 0).any():
            raise ValueError("temporal panel contains an empty instrument")
        if indexed.duplicated(["datetime", "instrument"]).any():
            raise ValueError("temporal panel keys (datetime, instrument) must be unique")

        if calendar is None:
            resolved_calendar = pd.DatetimeIndex(indexed["datetime"].unique()).sort_values()
        else:
            resolved_calendar = _naive_datetime_index(calendar, name="calendar")
            if not resolved_calendar.is_monotonic_increasing or not resolved_calendar.is_unique:
                raise ValueError("calendar must be strictly increasing and unique")
        if resolved_calendar.empty:
            raise ValueError("calendar must contain at least one session")

        histories: dict[str, _InstrumentHistory] = {}
        for instrument, group in indexed.groupby("instrument", sort=False):
            ordered = group.sort_values("datetime", kind="stable")
            positions = resolved_calendar.get_indexer(ordered["datetime"])
            if np.any(positions < 0):
                raise ValueError("calendar does not contain every panel datetime")
            values = ordered.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
            positions = positions.astype(np.int64, copy=False)
            positions.setflags(write=False)
            values.setflags(write=False)
            histories[str(instrument)] = _InstrumentHistory(positions, values)
        return cls(
            calendar=resolved_calendar,
            feature_columns=columns,
            histories=histories,
        )

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._calendar.copy()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    def build_batch(
        self,
        decisions: pd.DataFrame,
        *,
        lookback_sessions: int,
        fill_value: float = 0.0,
        require_decision_row: bool = True,
    ) -> TemporalSequenceBatch:
        """Build sequences ending exactly at each requested decision session."""

        if lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not math.isfinite(fill_value):
            raise ValueError("fill_value must be finite")
        missing = sorted({"datetime", "instrument"} - set(decisions.columns))
        if missing:
            raise ValueError(f"decision rows are missing columns: {missing}")
        decision_dates_index = _naive_datetime_index(
            decisions["datetime"], name="decision datetime"
        )
        decision_positions = self._calendar.get_indexer(decision_dates_index)
        if np.any(decision_positions < 0):
            unknown = decision_dates_index[decision_positions < 0][0]
            raise ValueError(f"decision date is not in the session calendar: {unknown}")

        instruments = tuple(decisions["instrument"].astype(str))
        sample_count = len(decisions)
        feature_count = len(self._feature_columns)
        values = np.full(
            (sample_count, lookback_sessions, feature_count),
            np.float32(fill_value),
            dtype=np.float32,
        )
        observed_mask = np.zeros(values.shape, dtype=np.bool_)
        row_mask = np.zeros((sample_count, lookback_sessions), dtype=np.bool_)
        session_dates = np.full(
            (sample_count, lookback_sessions),
            np.datetime64("NaT", "ns"),
            dtype="datetime64[ns]",
        )

        calendar_values = self._calendar.to_numpy(dtype="datetime64[ns]", copy=False)
        for sample, (instrument, decision_position) in enumerate(
            zip(instruments, decision_positions)
        ):
            history = self._histories.get(instrument)
            if history is None:
                raise ValueError(f"unknown decision instrument: {instrument!r}")
            window_start = max(0, int(decision_position) - lookback_sessions + 1)
            window_positions = np.arange(
                window_start, int(decision_position) + 1, dtype=np.int64
            )
            destination_start = lookback_sessions - len(window_positions)
            session_dates[sample, destination_start:] = calendar_values[window_positions]

            candidate_indices = np.searchsorted(history.calendar_positions, window_positions)
            inside = candidate_indices < len(history.calendar_positions)
            matched = np.zeros(len(window_positions), dtype=np.bool_)
            matched[inside] = (
                history.calendar_positions[candidate_indices[inside]]
                == window_positions[inside]
            )
            if matched.any():
                source = history.values[candidate_indices[matched]]
                finite = np.isfinite(source)
                destination = destination_start + np.flatnonzero(matched)
                row_mask[sample, destination] = True
                observed_mask[sample, destination] = finite
                for local_row, destination_row in enumerate(destination):
                    values[sample, destination_row, finite[local_row]] = source[
                        local_row, finite[local_row]
                    ]
            if require_decision_row and not row_mask[sample, -1]:
                decision_date = decision_dates_index[sample]
                raise ValueError(
                    f"no panel row for decision key ({decision_date}, {instrument!r})"
                )

        decision_dates = decision_dates_index.to_numpy(dtype="datetime64[ns]", copy=True)
        source_index = tuple(decisions.index)
        for array in (values, observed_mask, row_mask, session_dates, decision_dates):
            array.setflags(write=False)
        return TemporalSequenceBatch(
            values=values,
            observed_mask=observed_mask,
            row_mask=row_mask,
            session_dates=session_dates,
            decision_dates=decision_dates,
            instruments=instruments,
            source_index=source_index,
            feature_columns=self._feature_columns,
            lookback_sessions=lookback_sessions,
        )
