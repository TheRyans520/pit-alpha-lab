"""Versioned experiment configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the public contract."""


REQUIRED_SECTIONS = (
    "experiment",
    "data",
    "data_quality",
    "features",
    "label",
    "splits",
    "model",
    "portfolio",
    "evaluation",
    "artifacts",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _iso_date(value: Any, path: str) -> date:
    raw = _non_empty_string(value, path)
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(f"{path} must use YYYY-MM-DD") from exc


def _window(value: Any, path: str) -> tuple[date, date]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{path} must be a two-item date list")
    start = _iso_date(value[0], f"{path}[0]")
    end = _iso_date(value[1], f"{path}[1]")
    if start > end:
        raise ConfigError(f"{path} starts after it ends")
    return start, end


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate invariants needed before an experiment can be identified."""

    root = _mapping(config, "config")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version must equal 1")
    for section in REQUIRED_SECTIONS:
        _mapping(root.get(section), section)

    experiment = _mapping(root["experiment"], "experiment")
    _non_empty_string(experiment.get("name"), "experiment.name")
    if not isinstance(experiment.get("seed"), int):
        raise ConfigError("experiment.seed must be an integer")

    data = _mapping(root["data"], "data")
    source = _non_empty_string(data.get("source"), "data.source")
    if source not in {"synthetic", "qlib"}:
        raise ConfigError("data.source must be synthetic or qlib")
    _non_empty_string(data.get("snapshot_id"), "data.snapshot_id")
    _non_empty_string(data.get("universe"), "data.universe")
    data_start = _iso_date(data.get("start"), "data.start")
    data_end = _iso_date(data.get("end"), "data.end")
    if data_start > data_end:
        raise ConfigError("data.start must not be after data.end")

    features = _mapping(root["features"], "features")
    _non_empty_string(features.get("set"), "features.set")

    label = _mapping(root["label"], "label")
    _non_empty_string(label.get("expression"), "label.expression")
    horizon = label.get("horizon_trading_days")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ConfigError("label.horizon_trading_days must be a positive integer")
    winsorization = label.get("cross_sectional_winsorization")
    if (
        not isinstance(winsorization, list)
        or len(winsorization) != 2
        or any(not isinstance(value, (int, float)) for value in winsorization)
        or not 0 <= winsorization[0] < winsorization[1] <= 1
    ):
        raise ConfigError("label.cross_sectional_winsorization must be two increasing values in [0, 1]")

    if source == "qlib":
        if data.get("adapter") != "audited_parquet":
            raise ConfigError("qlib data.adapter must be audited_parquet")
        for key in ("audit_root_env", "release_tag", "snapshot_manifest"):
            _non_empty_string(data.get(key), f"data.{key}")

    quality = _mapping(root["data_quality"], "data_quality")
    for key in ("minimum_rows", "minimum_daily_members", "maximum_daily_members"):
        value = quality.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"data_quality.{key} must be a positive integer")
    if quality["minimum_daily_members"] > quality["maximum_daily_members"]:
        raise ConfigError("data_quality daily member bounds are reversed")
    for key in (
        "minimum_feature_coverage",
        "minimum_label_coverage",
        "minimum_realized_return_coverage",
        "minimum_ohlc_coverage",
    ):
        value = quality.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ConfigError(f"data_quality.{key} must be in [0, 1]")
    maximum_return = quality.get("maximum_absolute_realized_return")
    if not isinstance(maximum_return, (int, float)) or maximum_return <= 0:
        raise ConfigError("data_quality.maximum_absolute_realized_return must be positive")

    splits = _mapping(root["splits"], "splits")
    train_start, train_end = _window(splits.get("initial_train"), "splits.initial_train")
    valid_start, valid_end = _window(splits.get("validation"), "splits.validation")
    test_start, test_end = _window(splits.get("test"), "splits.test")
    if not (train_end < valid_start <= valid_end < test_start):
        raise ConfigError("train, validation and test windows must be ordered and disjoint")
    if train_start < data_start or test_end > data_end:
        raise ConfigError("split windows must remain inside the configured data range")
    embargo = splits.get("embargo_trading_days")
    if not isinstance(embargo, int) or embargo < horizon:
        raise ConfigError("splits.embargo_trading_days must cover the label horizon")

    model = _mapping(root["model"], "model")
    _non_empty_string(model.get("name"), "model.name")
    _mapping(model.get("params"), "model.params")

    portfolio = _mapping(root["portfolio"], "portfolio")
    top_k = portfolio.get("top_k")
    retention_rank = portfolio.get("retention_rank")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigError("portfolio.top_k must be a positive integer")
    if not isinstance(retention_rank, int) or retention_rank < top_k:
        raise ConfigError("portfolio.retention_rank must be an integer at least as large as top_k")
    costs = portfolio.get("one_way_cost_bps")
    if not isinstance(costs, list) or not costs or any(not isinstance(v, int) or v < 0 for v in costs):
        raise ConfigError("portfolio.one_way_cost_bps must be a non-empty list of non-negative integers")
    primary_cost = portfolio.get("primary_cost_bps")
    if primary_cost not in costs:
        raise ConfigError("portfolio.primary_cost_bps must appear in portfolio.one_way_cost_bps")

    evaluation = _mapping(root["evaluation"], "evaluation")
    quantiles = evaluation.get("quantiles")
    if not isinstance(quantiles, int) or quantiles < 2:
        raise ConfigError("evaluation.quantiles must be an integer of at least two")
    decay_horizons = evaluation.get("signal_decay_horizons")
    if (
        not isinstance(decay_horizons, list)
        or not decay_horizons
        or any(not isinstance(value, int) or value <= 0 for value in decay_horizons)
    ):
        raise ConfigError("evaluation.signal_decay_horizons must contain positive integers")
    bootstrap = _mapping(evaluation.get("bootstrap"), "evaluation.bootstrap")
    if bootstrap.get("method") != "moving_block":
        raise ConfigError("evaluation.bootstrap.method must be moving_block")
    block_length = bootstrap.get("block_length_dates")
    replications = bootstrap.get("replications")
    if not isinstance(block_length, int) or block_length <= 0:
        raise ConfigError("evaluation.bootstrap.block_length_dates must be positive")
    if not isinstance(replications, int) or replications <= 0:
        raise ConfigError("evaluation.bootstrap.replications must be positive")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _non_empty_string(artifacts.get("root"), "artifacts.root")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML config, validate it and return a plain dictionary."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(payload, "config")
    config = dict(root)
    validate_config(config)
    return config


def canonical_config_json(config: Mapping[str, Any]) -> str:
    """Return the canonical serialization used for content identity."""

    validate_config(config)
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a validated configuration."""

    encoded = canonical_config_json(config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
