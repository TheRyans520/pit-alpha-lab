"""Leakage-safe Ridge baseline."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def fit_predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Iterable[str],
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit all preprocessing on training rows and predict the untouched test rows."""

    columns = list(feature_columns)
    finite_label = np.isfinite(train["label"].to_numpy(dtype=float))
    training = train.loc[finite_label]
    if training.empty:
        raise ValueError("Ridge training window has no finite labels")
    x_train = training[columns].to_numpy(dtype=np.float32, copy=True)
    x_test = test[columns].to_numpy(dtype=np.float32, copy=True)
    x_train[~np.isfinite(x_train)] = np.nan
    x_test[~np.isfinite(x_test)] = np.nan
    medians = np.nanmedian(x_train, axis=0).astype(np.float32)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    missing_train = ~np.isfinite(x_train)
    missing_test = ~np.isfinite(x_test)
    if missing_train.any():
        x_train[missing_train] = np.take(medians, np.nonzero(missing_train)[1])
    if missing_test.any():
        x_test[missing_test] = np.take(medians, np.nonzero(missing_test)[1])
    means = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[(~np.isfinite(scales)) | (scales < 1e-12)] = 1.0
    x_train = (x_train - means) / scales
    x_test = (x_test - means) / scales
    model = Ridge(**dict(params))
    model.fit(x_train, training["label"].to_numpy(dtype=np.float32))
    predictions = model.predict(x_test)
    metadata = {
        "training_rows": int(len(training)),
        "test_rows": int(len(test)),
        "features": columns,
        "all_missing_training_features": [column for column, missing in zip(columns, all_missing) if missing],
    }
    return predictions.astype(np.float64), metadata
