"""Lazy model registry and explicit third-party plugin boundary."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Literal

import numpy as np
import pandas as pd


FitPredictFunction = Callable[
    [pd.DataFrame, pd.DataFrame, Iterable[str], Mapping[str, Any]],
    tuple[np.ndarray, Mapping[str, Any]],
]

ENTRY_POINT_GROUP = "pitalpha.models"


@dataclass(frozen=True)
class ModelDescriptor:
    """Machine-readable capabilities; claims remain subject to conformance tests."""

    name: str
    provider: Literal["built_in", "external"]
    input_kind: Literal["tabular", "cross_sectional", "temporal"]
    description: str
    uses_market_context: bool = False
    produces_uncertainty: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelAdapter:
    descriptor: ModelDescriptor
    fit_predict: FitPredictFunction

    def __post_init__(self) -> None:
        if not self.descriptor.name.strip():
            raise ValueError("model descriptor name must be non-empty")
        if not callable(self.fit_predict):
            raise TypeError("model fit_predict must be callable")


@dataclass(frozen=True)
class ModelContractReport:
    model: str
    test_rows: int
    finite_scores: bool
    deterministic: bool
    test_label_independent: bool
    inputs_unchanged: bool
    maximum_repeat_difference: float
    maximum_label_difference: float

    @property
    def passed(self) -> bool:
        return (
            self.finite_scores
            and self.deterministic
            and self.test_label_independent
            and self.inputs_unchanged
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


_BUILTINS: dict[str, tuple[ModelDescriptor, str, str]] = {
    "ridge": (
        ModelDescriptor(
            name="ridge",
            provider="built_in",
            input_kind="tabular",
            description="Leakage-safe standardized Ridge anchor.",
        ),
        "pitalpha.models.ridge",
        "fit_predict_ridge",
    ),
    "lightgbm": (
        ModelDescriptor(
            name="lightgbm",
            provider="built_in",
            input_kind="tabular",
            description="Frozen tree-ensemble comparison baseline.",
        ),
        "pitalpha.models.lightgbm",
        "fit_predict_lightgbm",
    ),
    "mlp": (
        ModelDescriptor(
            name="mlp",
            provider="built_in",
            input_kind="tabular",
            description="Deterministic PyTorch flat-feature neural control.",
        ),
        "pitalpha.models.mlp",
        "fit_predict_mlp",
    ),
    "market_context": (
        ModelDescriptor(
            name="market_context",
            provider="built_in",
            input_kind="cross_sectional",
            description="Same-date context gate with a within-date ranking objective.",
            uses_market_context=True,
        ),
        "pitalpha.models.market_context",
        "fit_predict_market_context",
    ),
    "temporal_mixer": (
        ModelDescriptor(
            name="temporal_mixer",
            provider="built_in",
            input_kind="temporal",
            description="Compact causal 60-session patch and temporal mixing baseline.",
        ),
        "pitalpha.models.temporal_mixer",
        "fit_predict_temporal_mixer",
    ),
}


def list_model_descriptors(*, include_external_names: bool = False) -> tuple[dict[str, object], ...]:
    """List built-ins without importing optional model libraries."""

    rows = [descriptor.to_dict() for descriptor, _, _ in _BUILTINS.values()]
    if include_external_names:
        for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
            if entry_point.name not in _BUILTINS:
                rows.append(
                    {
                        "name": entry_point.name,
                        "provider": "external",
                        "input_kind": "unknown_until_loaded",
                        "description": f"Installed entry point: {entry_point.value}",
                        "uses_market_context": None,
                        "produces_uncertainty": None,
                    }
                )
    return tuple(sorted(rows, key=lambda row: (str(row["provider"]), str(row["name"]))))


def resolve_model(name: str, *, allow_external: bool = False) -> ModelAdapter:
    """Resolve one adapter; third-party code is never loaded implicitly."""

    if name in _BUILTINS:
        descriptor, module_name, attribute = _BUILTINS[name]
        module = importlib.import_module(module_name)
        return ModelAdapter(descriptor=descriptor, fit_predict=getattr(module, attribute))

    candidates = [
        entry_point
        for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP)
        if entry_point.name == name
    ]
    if not candidates:
        raise RuntimeError(f"unsupported model: {name}")
    if not allow_external:
        raise RuntimeError(
            f"model {name!r} is an external plugin; rerun with --allow-external-model "
            "only after reviewing its installed code"
        )
    if len(candidates) != 1:
        raise RuntimeError(f"model entry point {name!r} is ambiguous")
    factory = candidates[0].load()
    adapter = factory()
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"external model {name!r} must return ModelAdapter from its entry-point factory")
    if adapter.descriptor.name != name or adapter.descriptor.provider != "external":
        raise ValueError("external model descriptor must match its entry-point name and provider")
    return adapter


def validate_model_response(
    scores: np.ndarray,
    model_metadata: Mapping[str, Any],
    *,
    expected_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enforce the artifact contract shared by built-in and user models."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(f"model scores must have shape ({expected_rows},)")
    if not np.isfinite(values).all():
        raise ValueError("model scores must all be finite")
    metadata_copy = dict(model_metadata)
    try:
        json.dumps(metadata_copy, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata must be finite JSON-serializable data") from exc
    return values, metadata_copy


def run_model_adapter(
    adapter: ModelAdapter,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    scores, model_metadata = adapter.fit_predict(train, test, feature_columns, params)
    values, validated_metadata = validate_model_response(
        scores,
        model_metadata,
        expected_rows=len(test),
    )
    validated_metadata["model_contract"] = adapter.descriptor.to_dict()
    return values, validated_metadata


def audit_test_label_independence(
    adapter: ModelAdapter,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
    *,
    absolute_tolerance: float = 0.0,
) -> ModelContractReport:
    """Detect adapters that read test labels or are not deterministic under a fixed config."""

    if "label" not in test:
        raise ValueError("test frame must contain label for the independence audit")
    train_probe = train.copy(deep=True)
    test_probe = test.copy(deep=True)
    train_expected = train_probe.copy(deep=True)
    test_expected = test_probe.copy(deep=True)
    baseline, _ = run_model_adapter(adapter, train_probe, test_probe, feature_columns, params)
    inputs_unchanged = train_probe.equals(train_expected) and test_probe.equals(test_expected)
    repeated, _ = run_model_adapter(
        adapter,
        train.copy(deep=True),
        test.copy(deep=True),
        feature_columns,
        params,
    )
    perturbed = test.copy()
    labels = perturbed["label"].to_numpy(dtype=np.float64, copy=True)
    perturbed["label"] = np.flip(labels) + 123.456
    label_perturbed, _ = run_model_adapter(
        adapter,
        train.copy(deep=True),
        perturbed,
        feature_columns,
        params,
    )
    repeat_difference = np.abs(baseline - repeated)
    label_difference = np.abs(baseline - label_perturbed)
    maximum_repeat_difference = float(repeat_difference.max(initial=0.0))
    maximum_label_difference = float(label_difference.max(initial=0.0))
    deterministic = bool(np.allclose(baseline, repeated, rtol=0.0, atol=absolute_tolerance))
    independent = bool(np.allclose(baseline, label_perturbed, rtol=0.0, atol=absolute_tolerance))
    return ModelContractReport(
        model=adapter.descriptor.name,
        test_rows=len(test),
        finite_scores=bool(
            np.isfinite(baseline).all()
            and np.isfinite(repeated).all()
            and np.isfinite(label_perturbed).all()
        ),
        deterministic=deterministic,
        test_label_independent=independent,
        inputs_unchanged=inputs_unchanged,
        maximum_repeat_difference=maximum_repeat_difference,
        maximum_label_difference=maximum_label_difference,
    )
