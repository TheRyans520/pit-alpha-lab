"""Deterministic CPU PyTorch MLP baseline with chronological validation."""

from __future__ import annotations

import gc
import random
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from pitalpha.splits.validation import purged_validation_split


class _ReturnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _fit_preprocessor(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prepared = values.astype(np.float32, copy=True)
    prepared[~np.isfinite(prepared)] = np.nan
    medians = np.nanmedian(prepared, axis=0).astype(np.float32)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    missing = ~np.isfinite(prepared)
    if missing.any():
        prepared[missing] = np.take(medians, np.nonzero(missing)[1])
    means = prepared.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = prepared.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[(~np.isfinite(scales)) | (scales < 1e-12)] = 1.0
    return prepared, medians, means, scales, all_missing


def _transform(values: np.ndarray, medians: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    prepared = values.astype(np.float32, copy=True)
    prepared[~np.isfinite(prepared)] = np.nan
    missing = ~np.isfinite(prepared)
    if missing.any():
        prepared[missing] = np.take(medians, np.nonzero(missing)[1])
    return (prepared - means) / scales


def _train_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.SmoothL1Loss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(x_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_tensor[indices]), y_tensor[indices])
            loss.backward()
            optimizer.step()


def _loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    model.eval()
    loss_function = nn.SmoothL1Loss(reduction="sum")
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            total += float(loss_function(model(x_tensor[start : start + batch_size]), y_tensor[start : start + batch_size]))
    return total / len(x_tensor)


def _predict(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x)
    parts = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            parts.append(model(tensor[start : start + batch_size]).numpy())
    return np.concatenate(parts).astype(np.float32)


def fit_predict_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select epoch count chronologically, then refit on the complete outer train window."""

    columns = list(feature_columns)
    seed = int(params["seed"])
    threads = int(params.get("torch_num_threads", 4))
    _seed_everything(seed, threads)
    finite_label = np.isfinite(train["label"].to_numpy(dtype=float))
    training = train.loc[finite_label].sort_values(["datetime"], kind="stable")
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
    y_subtrain_raw = subtrain["label"].to_numpy(dtype=np.float32)
    label_mean = np.float32(y_subtrain_raw.mean(dtype=np.float64))
    label_scale = np.float32(y_subtrain_raw.std(dtype=np.float64))
    if not np.isfinite(label_scale) or label_scale < 1e-12:
        label_scale = np.float32(1.0)
    y_subtrain = (y_subtrain_raw - label_mean) / label_scale
    y_validation = (validation["label"].to_numpy(dtype=np.float32) - label_mean) / label_scale

    hidden_dims = [int(value) for value in params["hidden_dims"]]
    dropout = float(params["dropout"])
    batch_size = int(params["batch_size"])
    model = _ReturnMLP(len(columns), hidden_dims, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"])
    )
    loss_function = nn.SmoothL1Loss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_tensor = torch.from_numpy(x_subtrain)
    y_tensor = torch.from_numpy(y_subtrain)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[float] = []
    for epoch in range(int(params["max_epochs"])):
        model.train()
        permutation = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(x_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_tensor[indices]), y_tensor[indices])
            loss.backward()
            optimizer.step()
        validation_loss = _loss(model, x_validation, y_validation, batch_size)
        history.append(validation_loss)
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
    y_full_raw = training["label"].to_numpy(dtype=np.float32)
    full_label_mean = np.float32(y_full_raw.mean(dtype=np.float64))
    full_label_scale = np.float32(y_full_raw.std(dtype=np.float64))
    if not np.isfinite(full_label_scale) or full_label_scale < 1e-12:
        full_label_scale = np.float32(1.0)
    y_full = (y_full_raw - full_label_mean) / full_label_scale
    x_test = _transform(test[columns].to_numpy(dtype=np.float32, copy=False), full_medians, full_means, full_scales)
    del x_subtrain, x_validation, y_subtrain, y_validation, x_tensor, y_tensor
    del optimizer, model, subtrain, validation
    gc.collect()
    _seed_everything(seed, threads)
    final_model = _ReturnMLP(len(columns), hidden_dims, dropout)
    _train_epochs(
        final_model,
        x_full,
        y_full,
        epochs=best_epoch + 1,
        batch_size=batch_size,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        seed=seed,
    )
    predictions = _predict(final_model, x_test, batch_size) * full_label_scale + full_label_mean
    metadata = {
        "training_rows": int(len(training)),
        "subtrain_rows": subtrain_rows,
        "validation_rows": validation_rows,
        **validation_metadata,
        "test_rows": int(len(test)),
        "features": columns,
        "best_epoch": int(best_epoch + 1),
        "epochs_evaluated": len(history),
        "best_validation_loss": best_loss,
        "validation_loss_history": history,
        "all_missing_training_features": [
            column for column, missing in zip(columns, full_all_missing) if missing
        ],
        "library": "torch",
        "library_version": torch.__version__,
        "device": "cpu",
        "seed": seed,
        "deterministic_algorithms": True,
        "parameter_count": int(sum(parameter.numel() for parameter in final_model.parameters())),
    }
    return predictions.astype(np.float64), metadata
