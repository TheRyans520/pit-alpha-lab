"""Deterministic synthetic equity panel for public tests and demos."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "latent_quality",
    "momentum_5",
    "reversal_1",
    "volatility_20",
    "volume_surprise",
)


def _forward_compound(returns: np.ndarray, horizon: int) -> np.ndarray:
    rows, columns = returns.shape
    result = np.full((rows, columns), np.nan, dtype=np.float64)
    for index in range(rows - horizon):
        result[index] = np.prod(1.0 + returns[index + 1 : index + horizon + 1], axis=0) - 1.0
    return result


def _membership_intervals(dates: pd.DatetimeIndex, instruments: list[str]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Keep thirty active names while six deterministic replacements occur."""

    intervals: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for index in range(6, 30):
        intervals[instruments[index]] = (dates[0], dates[-1])
    cut_positions = np.linspace(int(len(dates) * 0.18), int(len(dates) * 0.82), 6, dtype=int)
    for index, cut_position in enumerate(cut_positions):
        intervals[instruments[index]] = (dates[0], dates[cut_position - 1])
        intervals[instruments[30 + index]] = (dates[cut_position], dates[-1])
    return intervals


def generate_synthetic_panel(config: Mapping[str, Any]) -> pd.DataFrame:
    """Generate an auditable panel with features known at ``t`` and returns after ``t``.

    The generator intentionally includes changing interval membership and missing feature cells.
    It is a software fixture, not a calibrated market simulator.
    """

    seed = int(config["experiment"]["seed"])
    data_config = config["data"]
    horizon = int(config["label"]["horizon_trading_days"])
    dates = pd.bdate_range(data_config["start"], data_config["end"], inclusive="both")
    if len(dates) < 252:
        raise ValueError("synthetic demo requires at least 252 business dates")

    rng = np.random.default_rng(seed)
    instruments = [f"SYN{i:03d}" for i in range(36)]
    n_dates = len(dates)
    n_instruments = len(instruments)

    latent = np.zeros((n_dates, n_instruments), dtype=np.float64)
    latent[0] = rng.normal(0.0, 1.0, n_instruments)
    for index in range(1, n_dates):
        latent[index] = 0.94 * latent[index - 1] + rng.normal(0.0, 0.34, n_instruments)

    market = rng.normal(0.00015, 0.0075, n_dates)
    betas = rng.uniform(0.75, 1.25, n_instruments)
    returns = np.zeros((n_dates, n_instruments), dtype=np.float64)
    returns[0] = betas * market[0] + rng.normal(0.0, 0.011, n_instruments)
    for index in range(1, n_dates):
        predictable = 0.00065 * latent[index - 1] - 0.06 * returns[index - 1]
        returns[index] = predictable + betas * market[index] + rng.normal(0.0, 0.011, n_instruments)
    returns = np.clip(returns, -0.18, 0.18)

    log_volume = 12.0 + 0.22 * np.abs(latent) + rng.normal(0.0, 0.28, latent.shape)
    returns_frame = pd.DataFrame(returns, index=dates, columns=instruments)
    volume_frame = pd.DataFrame(log_volume, index=dates, columns=instruments)
    momentum_5 = returns_frame.rolling(5, min_periods=5).mean().to_numpy()
    volatility_20 = returns_frame.rolling(20, min_periods=10).std(ddof=0).to_numpy()
    volume_surprise = (volume_frame - volume_frame.rolling(20, min_periods=10).mean()).to_numpy()
    forward_horizons = sorted({1, horizon, 10, 20})
    forward_returns = {value: _forward_compound(returns, value) for value in forward_horizons}
    label = forward_returns[horizon]
    realized_return_1d = forward_returns[1]

    feature_arrays = [latent, momentum_5, -returns, volatility_20, volume_surprise]
    missing_mask = rng.random((n_dates, n_instruments, len(FEATURE_COLUMNS))) < 0.006
    intervals = _membership_intervals(dates, instruments)
    rows: list[pd.DataFrame] = []
    for instrument_index, instrument in enumerate(instruments):
        start, end = intervals[instrument]
        active = (dates >= start) & (dates <= end)
        frame = pd.DataFrame(
            {
                "datetime": dates[active],
                "instrument": instrument,
                "membership_start": start,
                "membership_end": end,
                "label": label[active, instrument_index],
                "realized_return_1d": realized_return_1d[active, instrument_index],
            }
        )
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            values = feature_arrays[feature_index][active, instrument_index].copy()
            values[missing_mask[active, instrument_index, feature_index]] = np.nan
            frame[feature_name] = values
        for forward_horizon, values in forward_returns.items():
            frame[f"forward_return_{forward_horizon}d"] = values[active, instrument_index]
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["datetime", "instrument"], kind="stable").reset_index(drop=True)
    daily_counts = panel.groupby("datetime", sort=False)["instrument"].nunique()
    if not daily_counts.eq(30).all():
        raise RuntimeError("synthetic point-in-time universe must contain exactly 30 names per date")
    return panel
