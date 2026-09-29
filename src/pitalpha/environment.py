"""Environment diagnostics that never mutate Python installations."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def environment_report(repo_root: Path | None = None) -> dict[str, Any]:
    """Describe the interpreter and whether it is an expected isolated environment."""

    root = (repo_root or repository_root()).resolve()
    prefix = Path(sys.prefix).resolve()
    expected_prefixes = {(root / ".venv").resolve(), (root.parent / ".venv").resolve()}
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(prefix),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "isolated_environment": prefix != Path(sys.base_prefix).resolve(),
        "expected_project_environment": prefix in expected_prefixes,
        "repository_root": str(root),
        "platform": platform.platform(),
    }
