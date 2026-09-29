# Contributing

## Environment rule

Never install dependencies into system or global Python. In the current workspace, use:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

The verifier refuses a system/global interpreter. If a trusted isolated environment already exists
outside the repository, pass its interpreter with `-Python`; otherwise use the project-local
`.venv`. Do not reuse unrelated Conda or system environments.

## Change contract

- Add or update tests with behavior changes.
- Keep data, split, label, execution and cost assumptions in versioned YAML.
- Never refresh a dataset silently; create a new manifest and checksum.
- Never commit raw licensed market data or generated large artifacts.
- Keep notebooks exploratory. Reusable logic belongs under `src/pitalpha`.
- Record material architectural changes as an ADR under `docs/decisions`.

## Q0 checks

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

The CI additionally runs `compileall` inside its own clean project `.venv`. Some restricted
desktop sandboxes block creation of `__pycache__`; `-B` keeps local verification read-only.
