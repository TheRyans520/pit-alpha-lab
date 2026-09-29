"""Read-only adapter for checksum-audited Qlib parquet exports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


class SnapshotAuditError(RuntimeError):
    """Raised when local data do not match the frozen public manifest."""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("year=*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise SnapshotAuditError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotAuditError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotAuditError(f"{description} must contain a JSON object: {path}")
    return value


def _verify_partitions(directory: Path, expected: Mapping[str, str], expected_digest: str) -> None:
    actual_names = sorted(path.name for path in directory.glob("year=*.parquet"))
    if actual_names != sorted(expected):
        raise SnapshotAuditError(f"partition set differs from metadata in {directory}")
    aggregate = hashlib.sha256()
    for name in actual_names:
        actual = _sha256(directory / name)
        if actual != expected[name]:
            raise SnapshotAuditError(f"checksum mismatch for {directory / name}")
        aggregate.update(name.encode("utf-8"))
        aggregate.update(bytes.fromhex(actual))
    if aggregate.hexdigest() != expected_digest:
        raise SnapshotAuditError(f"aggregate partition digest mismatch in {directory}")


def load_audited_qlib_panel(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Load a frozen feature export after verifying its complete checksum chain.

    The adapter never initializes Qlib or mutates the provider. The local audit root
    must contain ``<tag>_features``, ``<tag>_raw`` and extraction metadata generated
    from the separately obtained provider archive.
    """

    data = config["data"]
    env_name = str(data["audit_root_env"])
    env = os.environ if environment is None else environment
    raw_root = env.get(env_name)
    if not raw_root:
        raise SnapshotAuditError(
            f"environment variable {env_name} must point to the Phase 2A.7 audits directory"
        )
    audit_root = Path(raw_root).expanduser().resolve()
    tag = str(data["release_tag"])
    metadata = _load_json(audit_root / f"{tag}_extraction_metadata.json", "extraction metadata")
    manifest_path = repository_root / str(data["snapshot_manifest"])
    manifest = _load_json(manifest_path, "snapshot manifest")
    if metadata.get("tag") != tag or manifest.get("release_tag") != tag:
        raise SnapshotAuditError("release tag differs across config, manifest and extraction metadata")
    if manifest.get("snapshot_id") != data["snapshot_id"]:
        raise SnapshotAuditError("snapshot ID differs between config and manifest")
    comparisons = {
        "archive_sha256": "archive_sha256",
        "raw_partition_digest": "raw_partition_digest",
        "features_partition_digest": "feature_partition_digest",
    }
    for metadata_key, manifest_key in comparisons.items():
        if metadata.get(metadata_key) != manifest.get(manifest_key):
            raise SnapshotAuditError(f"{metadata_key} differs between manifest and extraction metadata")

    feature_dir = audit_root / f"{tag}_features"
    raw_dir = audit_root / f"{tag}_raw"
    if not feature_dir.is_dir() or not raw_dir.is_dir():
        raise SnapshotAuditError(f"feature/raw partition directories are missing under {audit_root}")
    _verify_partitions(feature_dir, metadata["feature_partitions"], metadata["features_partition_digest"])
    _verify_partitions(raw_dir, metadata["raw_partitions"], metadata["raw_partition_digest"])

    start_year = pd.Timestamp(data["start"]).year
    end_year = pd.Timestamp(data["end"]).year
    names = [f"year={year}.parquet" for year in range(start_year, end_year + 1)]
    missing = [name for name in names if name not in metadata["feature_partitions"]]
    if missing:
        raise SnapshotAuditError(f"configured window needs unavailable partitions: {missing}")
    feature_columns = tuple(str(name) for name in metadata["feature_names"])
    features = pd.concat(
        [pd.read_parquet(feature_dir / name, columns=["instrument", "datetime", *feature_columns]) for name in names],
        ignore_index=True,
    )
    raw = pd.concat(
        [
            pd.read_parquet(
                raw_dir / name,
                columns=[
                    "instrument",
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "factor",
                    "adjclose",
                    "vwap",
                    "label_5d",
                    "return_1d",
                ],
            )
            for name in names
        ],
        ignore_index=True,
    )
    keys = ["instrument", "datetime"]
    if features.duplicated(keys).any() or raw.duplicated(keys).any():
        raise SnapshotAuditError("snapshot contains duplicate instrument-date keys")
    panel = features.merge(raw, on=keys, how="inner", validate="one_to_one")
    if len(panel) != len(features) or len(panel) != len(raw):
        raise SnapshotAuditError("feature and raw partitions do not have identical keys")
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    window = panel["datetime"].between(pd.Timestamp(data["start"]), pd.Timestamp(data["end"]))
    panel = panel.loc[window].copy()
    panel["label_5d"] = panel["label_5d"].replace([np.inf, -np.inf], np.nan)
    lower, upper = (float(value) for value in config["label"]["cross_sectional_winsorization"])
    grouped = panel.groupby("datetime", sort=False)["label_5d"]
    low = grouped.transform(lambda values: values.quantile(lower))
    high = grouped.transform(lambda values: values.quantile(upper))
    panel["label"] = panel["label_5d"].clip(lower=low, upper=high)
    panel["realized_return_1d"] = panel["return_1d"].replace([np.inf, -np.inf], np.nan)
    panel["forward_return_1d"] = panel["realized_return_1d"]
    panel["forward_return_5d"] = panel["label_5d"]
    panel = panel.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
            "volume": "raw_volume",
            "factor": "raw_factor",
            "adjclose": "raw_adjclose",
            "vwap": "raw_vwap",
        }
    ).drop(columns=["label_5d", "return_1d"])
    panel = panel.sort_values(["datetime", "instrument"], kind="stable").reset_index(drop=True)
    provenance = {
        "adapter": "audited_qlib_parquet",
        "snapshot_id": manifest["snapshot_id"],
        "release_tag": tag,
        "rows": int(len(panel)),
        "feature_count": len(feature_columns),
        "raw_partition_digest": metadata["raw_partition_digest"],
        "feature_partition_digest": metadata["features_partition_digest"],
        "verified_partition_files": len(metadata["raw_partitions"]) + len(metadata["feature_partitions"]),
    }
    return panel, feature_columns, provenance
