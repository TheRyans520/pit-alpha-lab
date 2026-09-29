"""Common model adapters."""

from pitalpha.models.lightgbm import fit_predict_lightgbm
from pitalpha.models.ridge import fit_predict_ridge

__all__ = ["fit_predict_lightgbm", "fit_predict_ridge"]
