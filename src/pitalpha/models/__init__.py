"""Common model adapters."""

from pitalpha.models.lightgbm import fit_predict_lightgbm
from pitalpha.models.registry import (
    ENTRY_POINT_GROUP,
    ModelAdapter,
    ModelContractReport,
    ModelDescriptor,
    audit_test_label_independence,
    list_model_descriptors,
    resolve_model,
    run_model_adapter,
    validate_model_response,
)
from pitalpha.models.ridge import fit_predict_ridge

__all__ = [
    "ENTRY_POINT_GROUP",
    "ModelAdapter",
    "ModelContractReport",
    "ModelDescriptor",
    "audit_test_label_independence",
    "fit_predict_lightgbm",
    "fit_predict_ridge",
    "list_model_descriptors",
    "resolve_model",
    "run_model_adapter",
    "validate_model_response",
]
