#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_path=${PITALPHA_PYTHON:-"$repo_root/.venv/bin/python"}

if [ ! -x "$python_path" ]; then
    echo "No project virtual-environment Python found at $python_path." >&2
    echo "Create .venv or set PITALPHA_PYTHON to an isolated interpreter." >&2
    exit 1
fi

if ! "$python_path" -c 'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)'; then
    echo "Refusing to verify with a system/global Python environment." >&2
    exit 1
fi

cd "$repo_root"
PYTHONPATH="$repo_root/src" "$python_path" -B -m unittest discover -s tests -v
PYTHONPATH="$repo_root/src" "$python_path" -B -m pitalpha doctor
PYTHONPATH="$repo_root/src" "$python_path" -B -m pitalpha config validate configs/demo_synthetic.yaml

if [ -d "$repo_root/apps/web/node_modules" ] && command -v npm >/dev/null 2>&1; then
    (cd "$repo_root/apps/web" && npm run build)
else
    echo "Skipping web build: local node_modules or npm is unavailable."
fi

echo "PIT Alpha Lab verification passed."
