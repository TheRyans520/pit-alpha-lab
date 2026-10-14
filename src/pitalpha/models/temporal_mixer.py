"""Compact causal temporal mixer reference baseline.

This first-stage model intentionally excludes cross-stock context, differentiable
allocation and soft portfolio losses. Those components must earn inclusion through
separate ablations after the causal temporal baseline is stable.
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

from pitalpha.data import CausalSequenceStore, TemporalSequenceBatch
from pitalpha.models.mlp import _seed_everything


class _MixerBlock(nn.Module):
    def __init__(
        self,
        patch_count: int,
        hidden_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(patch_count, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, patch_count),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        token_state = self.token_norm(inputs).transpose(1, 2)
        inputs = inputs + self.token_mixer(token_state).transpose(1, 2)
        return inputs + self.channel_mixer(self.channel_norm(inputs))


class _TemporalMixer(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        lookback_sessions: int,
        patch_size: int,
        hidden_dim: int,
        mixer_blocks: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float,
        use_last_session_residual: bool,
    ) -> None:
        super().__init__()
        if lookback_sessions % patch_size != 0:
            raise ValueError("lookback_sessions must be divisible by patch_size")
        self.patch_size = patch_size
        self.patch_count = lookback_sessions // patch_size
        self.use_last_session_residual = use_last_session_residual
        self.session_projection = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.patch_projection = nn.Sequential(
            nn.Linear(patch_size * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                _MixerBlock(
                    self.patch_count,
                    hidden_dim,
                    token_mlp_dim,
                    channel_mlp_dim,
                    dropout,
                )
                for _ in range(mixer_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        session_hidden = self.session_projection(inputs)
        last_session = session_hidden[:, -1]
        hidden = session_hidden.reshape(
            len(session_hidden),
            self.patch_count,
            self.patch_size * session_hidden.shape[-1],
        )
        hidden = self.patch_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.output_norm(hidden).mean(dim=1)
        if self.use_last_session_residual:
            pooled = pooled + last_session
        return self.head(pooled).squeeze(-1)


@dataclass(frozen=True)
class _FeatureStatistics:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    all_missing: np.ndarray


def _fit_feature_statistics(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> _FeatureStatistics:
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    means = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[(~np.isfinite(scales)) | (scales < 1e-12)] = 1.0
    for array in (medians, means, scales, all_missing):
        array.setflags(write=False)
    return _FeatureStatistics(medians, means, scales, all_missing)


def _transform_temporal_batch(
    batch: TemporalSequenceBatch,
    statistics: _FeatureStatistics,
) -> np.ndarray:
    medians = statistics.medians.reshape(1, 1, -1)
    prepared = np.where(batch.observed_mask, batch.values, medians)
    prepared = (prepared - statistics.means.reshape(1, 1, -1)) / statistics.scales.reshape(
        1, 1, -1
    )
    return np.concatenate(
        [
            prepared.astype(np.float32, copy=False),
            batch.observed_mask.astype(np.float32),
            batch.row_mask[:, :, None].astype(np.float32),
        ],
        axis=2,
    )


def _normalized_labels(values: np.ndarray) -> tuple[np.ndarray, np.float32, np.float32]:
    mean = np.float32(values.mean(dtype=np.float64))
    scale = np.float32(values.std(dtype=np.float64))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.float32(1.0)
    return (values - mean) / scale, mean, scale


def _new_model(feature_count: int, params: Mapping[str, Any]) -> _TemporalMixer:
    return _TemporalMixer(
        input_channels=feature_count * 2 + 1,
        lookback_sessions=int(params["lookback_sessions"]),
        patch_size=int(params["patch_size"]),
        hidden_dim=int(params["hidden_dim"]),
        mixer_blocks=int(params["mixer_blocks"]),
        token_mlp_dim=int(params["token_mlp_dim"]),
        channel_mlp_dim=int(params["channel_mlp_dim"]),
        dropout=float(params["dropout"]),
        use_last_session_residual=bool(params.get("use_last_session_residual", False)),
    )


def _validate_params(params: Mapping[str, Any]) -> None:
    positive_integer_fields = (
        "lookback_sessions",
        "patch_size",
        "hidden_dim",
        "mixer_blocks",
        "token_mlp_dim",
        "channel_mlp_dim",
        "batch_size",
        "prediction_batch_size",
        "max_epochs",
        "patience",
        "validation_dates",
    )
    for field in positive_integer_fields:
        if not isinstance(params.get(field), int) or int(params[field]) <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if int(params["lookback_sessions"]) % int(params["patch_size"]) != 0:
        raise ValueError("lookback_sessions must be divisible by patch_size")
    dropout = float(params["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if float(params["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(params["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be non-negative")


def _input_tensor(
    store: CausalSequenceStore,
    decisions: pd.DataFrame,
    statistics: _FeatureStatistics,
    lookback_sessions: int,
) -> torch.Tensor:
    batch = store.build_batch(
        decisions.loc[:, ["datetime", "instrument"]],
        lookback_sessions=lookback_sessions,
    )
    return torch.from_numpy(_transform_temporal_batch(batch, statistics))


def _train_epoch(
    model: _TemporalMixer,
    optimizer: torch.optim.Optimizer,
    store: CausalSequenceStore,
    decisions: pd.DataFrame,
    targets: np.ndarray,
    statistics: _FeatureStatistics,
    *,
    lookback_sessions: int,
    batch_size: int,
    generator: torch.Generator,
) -> None:
    model.train()
    order = torch.randperm(len(decisions), generator=generator).numpy()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        features = _input_tensor(
            store,
            decisions.iloc[indices],
            statistics,
            lookback_sessions,
        )
        target_tensor = torch.from_numpy(targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.smooth_l1_loss(model(features), target_tensor)
        loss.backward()
        optimizer.step()


def _loss(
    model: _TemporalMixer,
    store: CausalSequenceStore,
    decisions: pd.DataFrame,
    targets: np.ndarray,
    statistics: _FeatureStatistics,
    *,
    lookback_sessions: int,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            stop = min(start + batch_size, len(decisions))
            features = _input_tensor(
                store,
                decisions.iloc[start:stop],
                statistics,
                lookback_sessions,
            )
            target_tensor = torch.from_numpy(targets[start:stop])
            total += float(
                nn.functional.smooth_l1_loss(
                    model(features), target_tensor, reduction="sum"
                )
            )
    return total / len(decisions)


def _predict(
    model: _TemporalMixer,
    store: CausalSequenceStore,
    decisions: pd.DataFrame,
    statistics: _FeatureStatistics,
    *,
    lookback_sessions: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            features = _input_tensor(
                store,
                decisions.iloc[start : start + batch_size],
                statistics,
                lookback_sessions,
            )
            predictions.append(model(features).numpy().astype(np.float32))
    return np.concatenate(predictions)


def fit_predict_temporal_mixer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a compact causal temporal mixer and predict in original test-row order."""

    _validate_params(params)
    columns = list(feature_columns)
    required = {"datetime", "instrument", "label", *columns}
    for name, frame in (("train", train), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"temporal mixer {name} frame is missing columns: {missing}")

    seed = int(params["seed"])
    threads = int(params.get("torch_num_threads", 4))
    lookback_sessions = int(params["lookback_sessions"])
    batch_size = int(params["batch_size"])
    prediction_batch_size = int(params["prediction_batch_size"])
    _seed_everything(seed, threads)

    finite_label = np.isfinite(train["label"].to_numpy(dtype=float))
    training = train.loc[finite_label].sort_values(
        ["datetime", "instrument"], kind="stable"
    ).reset_index(drop=True)
    if training.empty:
        raise ValueError("temporal mixer training window has no finite labels")
    subtrain, validation, validation_metadata = purged_validation_split(
        train,
        validation_dates=int(params["validation_dates"]),
        embargo_trading_days=int(params.get("validation_embargo_trading_days", 6)),
    )


    history = pd.concat(
        [
            train.loc[:, ["datetime", "instrument", *columns]],
            test.loc[:, ["datetime", "instrument", *columns]],
        ],
        ignore_index=True,
    )
    store = CausalSequenceStore.from_panel(history, columns)
    subtrain_statistics = _fit_feature_statistics(subtrain, columns)
    y_subtrain, label_mean, label_scale = _normalized_labels(
        subtrain["label"].to_numpy(dtype=np.float32)
    )
    y_validation = (
        validation["label"].to_numpy(dtype=np.float32) - label_mean
    ) / label_scale

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
    validation_history: list[float] = []
    for epoch in range(int(params["max_epochs"])):
        _train_epoch(
            model,
            optimizer,
            store,
            subtrain,
            y_subtrain,
            subtrain_statistics,
            lookback_sessions=lookback_sessions,
            batch_size=batch_size,
            generator=generator,
        )
        validation_loss = _loss(
            model,
            store,
            validation,
            y_validation,
            subtrain_statistics,
            lookback_sessions=lookback_sessions,
            batch_size=prediction_batch_size,
        )
        validation_history.append(validation_loss)
        if validation_loss < best_loss - float(params.get("minimum_delta", 0.0)):
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(params["patience"]):
                break

    full_statistics = _fit_feature_statistics(training, columns)
    y_full, full_label_mean, full_label_scale = _normalized_labels(
        training["label"].to_numpy(dtype=np.float32)
    )
    del model, optimizer, subtrain_statistics, y_subtrain, y_validation
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
            store,
            training,
            y_full,
            full_statistics,
            lookback_sessions=lookback_sessions,
            batch_size=batch_size,
            generator=final_generator,
        )
    normalized_predictions = _predict(
        final_model,
        store,
        test.reset_index(drop=True),
        full_statistics,
        lookback_sessions=lookback_sessions,
        batch_size=prediction_batch_size,
    )
    predictions = normalized_predictions * full_label_scale + full_label_mean
    metadata = {
        "training_rows": int(len(training)),
        "history_context_rows": int((~finite_label).sum()),
        "subtrain_rows": int(len(subtrain)),
        "validation_rows": int(len(validation)),
        **validation_metadata,
        "test_rows": int(len(test)),
        "features": columns,
        "lookback_sessions": lookback_sessions,
        "patch_size": int(params["patch_size"]),
        "patch_count": lookback_sessions // int(params["patch_size"]),
        "best_epoch": int(best_epoch + 1),
        "epochs_evaluated": len(validation_history),
        "best_validation_loss": best_loss,
        "validation_loss_history": validation_history,
        "all_missing_training_features": [
            column
            for column, missing in zip(columns, full_statistics.all_missing)
            if missing
        ],
        "library": "torch",
        "library_version": torch.__version__,
        "device": "cpu",
        "seed": seed,
        "deterministic_algorithms": True,
        "architecture": "causal-temporal-mixer-v0",
        "use_last_session_residual": bool(
            params.get("use_last_session_residual", False)
        ),
        "parameter_count": int(
            sum(parameter.numel() for parameter in final_model.parameters())
        ),
        "tensor_contract": "calendar-aligned-values-observed-mask-row-mask",
    }
    return predictions.astype(np.float64), metadata
