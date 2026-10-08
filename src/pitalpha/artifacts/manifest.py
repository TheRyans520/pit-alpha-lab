"""Run-manifest construction and atomic persistence."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pitalpha import __version__
from pitalpha.config import config_digest, validate_config


def source_identity(root: Path) -> dict[str, Any]:
    """Record local revision and actual source bytes, including uncommitted code."""
    revision = None
    dirty = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    files = sorted(set(root.glob("src/**/*.py")) | set(root.glob("configs/*.yaml")) |
                   set(root.glob("scripts/*.py")) | set(root.glob("manifests/*.json")) |
                   set(root.glob("requirements*.txt")) | set(root.glob("pyproject.toml")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"revision": revision, "dirty": dirty, "source_sha256": digest.hexdigest(),
            "source_files": len(files)}


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
    source = source_identity(Path(__file__).resolve().parents[3])
    return {
        "schema_version": 1,
        "status": "planned",
        "content_id": digest,
        "experiment_id": hashlib.sha256((digest + source["source_sha256"]).encode()).hexdigest(),
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
        "code_revision": code_revision or source["revision"],
        "source": source,
        "execution": {
            "python": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
                         if dist.metadata["Name"]},
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
