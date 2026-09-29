"""Atomic artifact writers and content hashing."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw_path)


def write_frame_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        if destination.suffix.lower() == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif destination.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False, lineterminator="\n")
        else:
            raise ValueError(f"unsupported dataframe artifact extension: {destination.suffix}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_text_atomic(text: str, path: str | Path) -> Path:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
