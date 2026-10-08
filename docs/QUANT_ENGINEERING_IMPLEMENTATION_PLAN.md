# Quant engineering implementation plan — 2026-10-08

## Scope and handoff

Base main: `239dab5b823fb3087b8f976995aa994120d10ad5`.
Existing draft PR #1: `audit/ex-ante-portfolio-selection-20261008` at
`f743e847740ad41b20a4b3fbc8250371ab092ded`.
Working branch: `codex/quant-resume-readiness`, descended from PR #1.
The owner confirmed that audited CSI300 partitions are unavailable on this computer.
Do not merge main without owner review. Do not overwrite frozen case-study tables.

## Delivered in this branch

1. Retain PR #1's outcome-independent selection and explicit failure for unvalued holdings.
   Add full-pipeline regression coverage, carried-position coverage and calendar-boundary tests.
2. Crop only whole terminal dates: one future session for the synthetic return generator;
   two for the declared real next-open-to-next-open contract. Persist this policy in each run.
3. Reject unexplained disappearing real holdings. Preserve a separately named, explicit
   synthetic membership-exit assumption with exit rows, turnover and cash diagnostics.
4. Export daily positions and exits, prior/target weights and per-security P&L/trade contributions.
   Reject missing metric inputs and insolvent portfolios instead of silently fabricating a series.
5. Purge six sessions between neural subtraining and validation, counting the original calendar
   before missing-label removal. Apply to MLP, market-context and temporal-mixer adapters.
   Keep final refitting on all eligible outer-fold training labels.
6. Record local revision, dirty state, source/config/snapshot-file digest and installed versions.
   Preserve `content_id` as the existing config hash; add `experiment_id` for config plus source identity.
7. Add independent artifact/ledger/metric reconciliation and a deterministic subprocess benchmark.
8. Mark historical performance provisional in README, technical brief, case study and dashboard.
   Keep negative results visible. Add an explicit neural CI job and update the vulnerable
   transitive source-map-js dependency from 1.2.1 to 1.2.2.

## Acceptance matrix

| Gate | Result | Evidence |
|---|---|---|
| Missing future return cannot promote the runner-up | Passed locally | `tests/test_pipeline.py`, `tests/test_backtest.py` |
| Held disappearance requires real evidence | Passed locally | `tests/test_backtest.py` |
| Six-session inner embargo | Passed locally | `tests/test_validation.py`, all three neural model tests |
| Complete local Python contract including API and PyTorch | 75 tests passed | `bash scripts/verify.sh` |
| CLI doctor/config validation | Passed; isolated Python 3.10.20 | Same command |
| TypeScript and web production build | Passed | Same command with Node 24 on PATH |
| Independent synthetic accounting | Passed | `benchmarks/synthetic_reconciliation.json` |
| Repeated synthetic artifacts | Passed | `benchmarks/synthetic_repeatability.json` |
| Real CSI300 reproduction | Unavailable | Owner confirmed no audit partitions |
| Historical performance delta | Unknown | Requires real outcomes and independently verified exit rules |
| Live/executable constraints | Not claimed | No verified market masks connected to ledger |

No browser interaction smoke test was run in this handoff. A production build is not a UI smoke test.
New remote CI must be inspected on the published head; local success is not remote CI success.

## Reproduce locally

Use Python 3.10 in `.venv`; install `requirements-api.txt`, `requirements-dl.txt`, and the editable
package. Use Node 24 with `npm ci` inside `apps/web`. Then:

```bash
bash scripts/verify.sh
.venv/bin/python -m pitalpha run configs/demo_synthetic.yaml
.venv/bin/python scripts/reconcile_run.py artifacts/demo-synthetic-ridge/<run-id>
.venv/bin/python scripts/benchmark_ledger.py --days 1260 --instruments 300 --repeats 3 --output /tmp/ledger-benchmark.json
```

macOS handoff environment: Python 3.10.20 and Node 24.21.0 were downloaded into the parent workspace,
not installed globally. LightGBM needs OpenMP. This machine's isolated Python runtime uses a local
symlink to the installed PyTorch wheel's `torch/lib/libomp.dylib`; both libraries imported and all
model tests passed. Other installations should supply a compatible platform OpenMP runtime.
Exact transitive Python versions are captured in each generated run manifest; requirements pin
direct packages, not a complete transitive lock. Benchmark RSS is a Unix/macOS process measure.

## Remaining P1 work: real data and economics

1. Obtain the same licensed/permitted audit export privately. Set `PITALPHA_QLIB_AUDIT_ROOT`.
   Verify partition hashes, raw/feature key equality, membership intervals and exchange calendar.
2. Independently reconstruct open(t+1)/open(t+2) returns and open(t+1)/open(t+6) labels from
   appropriately adjusted raw observations. Exporter code and original membership construction
   are not included here; matching a checksum does not prove those semantics.
3. Classify every missing outcome or disappearing security using PIT security-master evidence.
   Supply suspension marks, corporate-action economics and documented delisting settlement.
   No zero-return or last-price substitution is allowed merely because data are unavailable.
4. Build a separate all-held-security valuation table plus timestamped buy/sell/ADV masks.
   Only then connect `enforce_execution_constraints` to sequential cash and order accounting.
   Validate blocked sells, cash shortfalls, drifted caps, fees, final liquidation and settlement.
5. Rerun the preserved historical protocol and report changed selections, positions, costs and
   every model's performance delta. Keep old and new run IDs. A failed strict valuation gate is
   a result to investigate, not a reason to relax the gate until the backtest runs.
6. For new research, lock a genuinely unobserved period/market before accessing labels.
   Compare unchanged baselines with paired uncertainty; do not tune against observed 2021–2025.

## Hiring priorities

QD: reproducibility, failure semantics, independent reconciliation and measured throughput are
ready to demonstrate. C++/concurrency/networking require separate work for roles that ask for them.
QR: data economics and a fresh confirmatory holdout remain the main gaps.
Technical QT: this daily research tool supports risk/cost reasoning, but does not demonstrate
order-book analysis, queue behavior, exchange execution or live risk management.
