# Public release checklist

## Repository identity

- [x] Choose the GitHub owner and repository slug: `TheRyans520/pit-alpha-lab`.
- [x] Use the GitHub `noreply` address for public commits.
- [ ] Add the remote only after the destination repository has been confirmed.
- [ ] Add repository description and topics such as `quantitative-finance`, `backtesting`,
  `point-in-time`, `machine-learning`, `fastapi` and `react`.

## Content and safety

- [ ] Review `git status --short` before the first commit.
- [ ] Confirm that `artifacts/`, `.venv/`, `node_modules/`, raw data and `.env` are ignored.
- [ ] Scan staged files for credentials, local absolute data paths and oversized binaries.
- [ ] Confirm that the tracked dashboard screenshot contains no private information.
- [ ] Confirm all third-party data remain outside Git and are described by manifests only.

## Verification

- [ ] Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1` on Windows.
- [ ] Confirm the synthetic demo creates a complete hashed run.
- [ ] Run the live and offline browser smoke test.
- [ ] Push only after the local checks pass.
- [ ] Confirm all GitHub Actions jobs pass on the initial branch.

## Presentation

- [ ] Pin the repository on the GitHub profile.
- [ ] Keep the dashboard screenshot near the top of the README.
- [ ] Add a short demo recording only if it remains lightweight and reproducible.
- [ ] Use the technical brief and interview guide to keep public claims consistent.
- [ ] Tag `v0.1.0` only after the remote CI result is green.
