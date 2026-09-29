"""Run-manifest construction and atomic persistence."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pitalpha import __version__
from pitalpha.config import config_digest, validate_config


def build_run_manifest(
    config: Mapping[str, Any],
    config_path: str | Path,
    *,
    created_at: datetime | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Build metadata without running or mutating an experiment."""

    validate_config(config)
    instant = created_at or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    digest = config_digest(config)
    experiment_name = str(config["experiment"]["name"])
    timestamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema_version": 1,
        "status": "planned",
        "content_id": digest,
        "run_id": f"{experiment_name}-{digest[:12]}-{timestamp}",
        "created_at_utc": instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project": {"name": "pit-alpha-lab", "version": __version__},
        "config": {"path": str(Path(config_path)), "sha256": digest},
        "data": {
            "source": config["data"]["source"],
            "snapshot_id": config["data"]["snapshot_id"],
            "universe": config["data"]["universe"],
            "start": config["data"]["start"],
            "end": config["data"]["end"],
            "checksum_manifest": config["data"].get("checksum_manifest"),
        },
        "code_revision": code_revision,
        "execution": {
            "python": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
