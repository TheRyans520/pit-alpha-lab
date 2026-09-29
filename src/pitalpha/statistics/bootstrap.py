"""Deterministic circular moving-block bootstrap."""

from __future__ import annotations

import math

import numpy as np


def moving_block_mean_interval(
    values,
    *,
    block_length: int,
    replications: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("bootstrap input has no finite values")
    if block_length <= 0 or replications <= 0:
        raise ValueError("block_length and replications must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    block_length = min(block_length, clean.size)
    block_count = math.ceil(clean.size / block_length)
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replications, dtype=np.float64)
    for replication in range(replications):
        starts = rng.integers(0, clean.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % clean.size).reshape(-1)[: clean.size]
        estimates[replication] = clean[indices].mean()
    alpha = 1.0 - confidence
    return {
        "observations": int(clean.size),
        "point_estimate": float(clean.mean()),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "confidence": confidence,
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
    }
