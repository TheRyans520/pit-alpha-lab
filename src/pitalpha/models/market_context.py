"""Point-in-time cross-sectional market-context neural model.

The model conditions each stock's representation on mean and dispersion statistics computed
from the eligible universe on the same decision date. Context never crosses date boundaries and
contains no labels. A differentiable within-date correlation term can complement robust
regression without using test outcomes.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from pitalpha.splits.validation import purged_validation_split

from pitalpha.models.mlp import _fit_preprocessor, _seed_everything, _transform


class _MarketContextNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        context_hidden_dim: int,
        dropout: float,
        use_market_context: bool,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        self.use_market_context = use_market_context
        first_width = hidden_dims[0]
        self.stock_projection = nn.Linear(input_dim, first_width)
        if use_market_context:
            context_dim = input_dim * 2
            self.context_norm = nn.LayerNorm(context_dim)
            self.feature_gate = nn.Sequential(
                nn.Linear(context_dim, context_hidden_dim),
                nn.GELU(),
                nn.Linear(context_hidden_dim, input_dim),
            )
            self.context_projection = nn.Linear(context_dim, first_width, bias=False)
        else:
            self.context_norm = None
            self.feature_gate = None
            self.context_projection = None
        layers: list[nn.Module] = [nn.LayerNorm(first_width), nn.GELU(), nn.Dropout(dropout)]
        previous = first_width
        for width in hidden_dims[1:]:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, market_context: torch.Tensor) -> torch.Tensor:
        if self.use_market_context:
            assert self.context_norm is not None
            assert self.feature_gate is not None
            assert self.context_projection is not None
            normalized_context = self.context_norm(market_context)
            gate = 2.0 * torch.sigmoid(self.feature_gate(normalized_context))
            hidden = self.stock_projection(features * gate) + self.context_projection(normalized_context)
        else:
            hidden = self.stock_projection(features)
        return self.head(hidden).squeeze(-1)


@dataclass(frozen=True)
class _ContextData:
    table: np.ndarray
    date_codes: np.ndarray
    groups: tuple[np.ndarray, ...]


def _build_market_context(values: np.ndarray, dates: pd.Series) -> _ContextData:
    """Build same-date mean/std context without materializing it once per row."""

    if len(values) != len(dates):
        raise ValueError("feature rows and dates must have equal length")
    codes, _ = pd.factorize(pd.to_datetime(dates), sort=False)
    codes = codes.astype(np.int64, copy=False)
    if len(codes) == 0:
        raise ValueError("market context requires at least one row")
    if np.any(codes[1:] < codes[:-1]):
        raise ValueError("market context rows must be sorted by datetime")
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    ends = np.r_[starts[1:], len(codes)]
    feature_count = values.shape[1]
    table = np.empty((len(starts), feature_count * 2), dtype=np.float32)
    # Work one cross-section at a time. This avoids a full-panel float64 square/copy,
    # which is material for the 1.3M x 157 CSI300 matrix.
    for group_index, (start, end) in enumerate(zip(starts, ends)):
        cross_section = values[start:end]
        table[group_index, :feature_count] = cross_section.mean(axis=0, dtype=np.float64)
        table[group_index, feature_count:] = cross_section.std(axis=0, dtype=np.float64)
    groups = tuple(np.arange(start, end, dtype=np.int64) for start, end in zip(starts, ends))
    return _ContextData(table=table, date_codes=codes, groups=groups)


def _within_date_correlation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_codes: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    counts = torch.bincount(group_codes, minlength=group_count).to(predictions.dtype).clamp_min(1.0)
    prediction_sums = torch.zeros(group_count, dtype=predictions.dtype).scatter_add_(
        0, group_codes, predictions
    )
    target_sums = torch.zeros(group_count, dtype=targets.dtype).scatter_add_(0, group_codes, targets)
    prediction_centered = predictions - prediction_sums[group_codes] / counts[group_codes]
    target_centered = targets - target_sums[group_codes] / counts[group_codes]
    covariance = torch.zeros(group_count, dtype=predictions.dtype).scatter_add_(
        0, group_codes, prediction_centered * target_centered
    )
    prediction_variance = torch.zeros(group_count, dtype=predictions.dtype).scatter_add_(
        0, group_codes, prediction_centered.square()
    )
    target_variance = torch.zeros(group_count, dtype=targets.dtype).scatter_add_(
        0, group_codes, target_centered.square()
    )
    denominator = torch.sqrt(prediction_variance * target_variance)
    valid = denominator > 1e-12
    if not bool(valid.any()):
        return predictions.sum() * 0.0
    correlation = covariance[valid] / denominator[valid].clamp_min(1e-12)
    return 1.0 - correlation.mean()


def _date_batches(context: _ContextData, dates_per_batch: int, order: np.ndarray):
    for start in range(0, len(order), dates_per_batch):
        selected = order[start : start + dates_per_batch]
        parts = [context.groups[int(group)] for group in selected]
        indices = np.concatenate(parts)
        local_codes = np.repeat(
            np.arange(len(parts), dtype=np.int64),
            np.fromiter((len(part) for part in parts), dtype=np.int64),
        )
        yield indices, local_codes, len(parts)


def _batch_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_codes: torch.Tensor,
    group_count: int,
    ranking_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = nn.functional.smooth_l1_loss(predictions, targets)
    ranking = _within_date_correlation_loss(predictions, targets, group_codes, group_count)
    return regression + ranking_weight * ranking, regression, ranking


def _train_epoch(
    model: _MarketContextNetwork,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets: np.ndarray,
    context: _ContextData,
    *,
    dates_per_batch: int,
    ranking_weight: float,
    generator: torch.Generator,
) -> None:
    model.train()
    order = torch.randperm(len(context.groups), generator=generator).numpy()
    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(targets)
    context_tensor = torch.from_numpy(context.table)
    date_code_tensor = torch.from_numpy(context.date_codes)
    for indices, local_codes, group_count in _date_batches(context, dates_per_batch, order):
        index_tensor = torch.from_numpy(indices)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            feature_tensor[index_tensor],
            context_tensor[date_code_tensor[index_tensor]],
        )
        loss, _, _ = _batch_loss(
            predictions,
            target_tensor[index_tensor],
            torch.from_numpy(local_codes),
            group_count,
            ranking_weight,
        )
        loss.backward()
        optimizer.step()


def _evaluate(
    model: _MarketContextNetwork,
    features: np.ndarray,
    targets: np.ndarray,
    context: _ContextData,
    *,
    dates_per_batch: int,
    ranking_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(targets)
    context_tensor = torch.from_numpy(context.table)
    date_code_tensor = torch.from_numpy(context.date_codes)
    total_regression = 0.0
    total_rows = 0
    total_ranking = 0.0
    total_dates = 0
    order = np.arange(len(context.groups), dtype=np.int64)
    with torch.no_grad():
        for indices, local_codes, group_count in _date_batches(context, dates_per_batch, order):
            index_tensor = torch.from_numpy(indices)
            predictions = model(
                feature_tensor[index_tensor],
                context_tensor[date_code_tensor[index_tensor]],
            )
            _, regression, ranking = _batch_loss(
                predictions,
                target_tensor[index_tensor],
                torch.from_numpy(local_codes),
                group_count,
                ranking_weight,
            )
            total_regression += float(regression) * len(indices)
            total_rows += len(indices)
            total_ranking += float(ranking) * group_count
            total_dates += group_count
    regression_value = total_regression / total_rows
    ranking_value = total_ranking / total_dates
    return regression_value + ranking_weight * ranking_value, regression_value, ranking_value


def _predict(
    model: _MarketContextNetwork,
    features: np.ndarray,
    context: _ContextData,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    feature_tensor = torch.from_numpy(features)
    context_tensor = torch.from_numpy(context.table)
    date_code_tensor = torch.from_numpy(context.date_codes)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            parts.append(
                model(feature_tensor[start:stop], context_tensor[date_code_tensor[start:stop]])
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(parts)


def _new_model(input_dim: int, params: Mapping[str, Any]) -> _MarketContextNetwork:
    return _MarketContextNetwork(
        input_dim=input_dim,
        hidden_dims=[int(value) for value in params["hidden_dims"]],
        context_hidden_dim=int(params["context_hidden_dim"]),
        dropout=float(params["dropout"]),
        use_market_context=bool(params.get("use_market_context", True)),
    )


def _normalized_labels(values: np.ndarray) -> tuple[np.ndarray, np.float32, np.float32]:
    mean = np.float32(values.mean(dtype=np.float64))
    scale = np.float32(values.std(dtype=np.float64))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.float32(1.0)
    return (values - mean) / scale, mean, scale


def fit_predict_market_context(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit with chronological epoch selection and refit on the full outer train window."""

    columns = list(feature_columns)
    seed = int(params["seed"])
    threads = int(params.get("torch_num_threads", 4))
    dates_per_batch = int(params.get("dates_per_batch", 16))
    ranking_weight = float(params.get("ranking_weight", 0.0))
    prediction_batch_size = int(params.get("prediction_batch_size", 16384))
    if dates_per_batch <= 0:
        raise ValueError("dates_per_batch must be positive")
    if ranking_weight < 0:
        raise ValueError("ranking_weight must be non-negative")
    _seed_everything(seed, threads)

    finite_label = np.isfinite(train["label"].to_numpy(dtype=float))
    training = train.loc[finite_label].sort_values(["datetime"], kind="stable").reset_index(drop=True)
    testing = test.sort_values(["datetime"], kind="stable")
    if not testing.index.equals(test.index):
        raise ValueError("market-context test rows must already be sorted by datetime")
    subtrain, validation, validation_metadata = purged_validation_split(
        train,
        validation_dates=int(params["validation_dates"]),
        embargo_trading_days=int(params.get("validation_embargo_trading_days", 6)),
    )

    subtrain_rows = int(len(subtrain))
    validation_rows = int(len(validation))

    x_subtrain, medians, means, scales, _ = _fit_preprocessor(
        subtrain[columns].to_numpy(dtype=np.float32, copy=False)
    )
    x_validation = _transform(
        validation[columns].to_numpy(dtype=np.float32, copy=False), medians, means, scales
    )
    y_subtrain, label_mean, label_scale = _normalized_labels(
        subtrain["label"].to_numpy(dtype=np.float32)
    )
    y_validation = (
        validation["label"].to_numpy(dtype=np.float32) - label_mean
    ) / label_scale
    subtrain_context = _build_market_context(x_subtrain, subtrain["datetime"])
    validation_context = _build_market_context(x_validation, validation["datetime"])

    model = _new_model(len(columns), params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(params["max_epochs"])):
        _train_epoch(
            model,
            optimizer,
            x_subtrain,
            y_subtrain,
            subtrain_context,
            dates_per_batch=dates_per_batch,
            ranking_weight=ranking_weight,
            generator=generator,
        )
        validation_loss, validation_regression, validation_ranking = _evaluate(
            model,
            x_validation,
            y_validation,
            validation_context,
            dates_per_batch=dates_per_batch,
            ranking_weight=ranking_weight,
        )
        history.append(
            {
                "combined": validation_loss,
                "regression": validation_regression,
                "ranking": validation_ranking,
            }
        )
        if validation_loss < best_loss - float(params.get("minimum_delta", 0.0)):
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(params["patience"]):
                break

    x_full, full_medians, full_means, full_scales, full_all_missing = _fit_preprocessor(
        training[columns].to_numpy(dtype=np.float32, copy=False)
    )
    y_full, full_label_mean, full_label_scale = _normalized_labels(
        training["label"].to_numpy(dtype=np.float32)
    )
    x_test = _transform(
        testing[columns].to_numpy(dtype=np.float32, copy=False),
        full_medians,
        full_means,
        full_scales,
    )
    full_context = _build_market_context(x_full, training["datetime"])
    test_context = _build_market_context(x_test, testing["datetime"])
    del model, optimizer, x_subtrain, x_validation, y_subtrain, y_validation
    del subtrain_context, validation_context, subtrain, validation
    gc.collect()

    _seed_everything(seed, threads)
    final_model = _new_model(len(columns), params)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    final_generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(best_epoch + 1):
        _train_epoch(
            final_model,
            final_optimizer,
            x_full,
            y_full,
            full_context,
            dates_per_batch=dates_per_batch,
            ranking_weight=ranking_weight,
            generator=final_generator,
        )
    predictions = (
        _predict(final_model, x_test, test_context, prediction_batch_size) * full_label_scale
        + full_label_mean
    )
    metadata = {
        "training_rows": int(len(training)),
        "subtrain_rows": subtrain_rows,
        "validation_rows": validation_rows,
        **validation_metadata,
        "test_rows": int(len(testing)),
        "features": columns,
        "best_epoch": int(best_epoch + 1),
        "epochs_evaluated": len(history),
        "best_validation_loss": best_loss,
        "validation_history": history,
        "all_missing_training_features": [
            column for column, missing in zip(columns, full_all_missing) if missing
        ],
        "library": "torch",
        "library_version": torch.__version__,
        "device": "cpu",
        "seed": seed,
        "deterministic_algorithms": True,
        "architecture": "same-date-market-context-gated-mlp",
        "use_market_context": bool(params.get("use_market_context", True)),
        "context_statistics": ["eligible_cross_section_mean", "eligible_cross_section_std"],
        "ranking_weight": ranking_weight,
        "dates_per_batch": dates_per_batch,
        "parameter_count": int(sum(parameter.numel() for parameter in final_model.parameters())),
    }
    return predictions.astype(np.float64), metadata
