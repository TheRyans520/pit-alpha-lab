# PIT Alpha Lab — verified resume facts (2026-10-08)

## Identity and positioning

- Repository: https://github.com/TheRyans520/pit-alpha-lab
- Base main: `239dab5b823fb3087b8f976995aa994120d10ad5`.
- PR #1 reviewed and inherited: `f743e847740ad41b20a4b3fbc8250371ab092ded`.
- Working branch: `codex/quant-resume-readiness`. Read the final commit from this branch's Git log.
- Verified source digest: `708c8d852f275b6fbaac8d9f8cef71c27d4d8f4353f4d0d30b0be36d35252121`.
- Best current positioning: **Python quantitative research engineering / QD**, with QR methodology
  evidence. Technical QT relevance is cost, turnover and risk reasoning, not exchange execution.
- Historical real-market returns: **not independently reproduced after the fixes**. The owner
  confirmed the required audit partitions are not on this computer.

## Evidence hierarchy

| Claim | Verified here | Evidence |
|---|---|---|
| Public synthetic pipeline | Yes | 62,640 rows, 2,088 business dates, 36 instruments, 30 active per date |
| Walk-forward execution | Yes on synthetic | Three annual test folds, 2023–2025 |
| Ledger reconciliation | Yes | 23,490 predictions, 40,006 position/exit rows, 2,346 strategy/date rows across three strategies |
| Artifact integrity | Yes | 14 checked hashes; 13 deterministic artifacts identical on repeated execution |
| Local tests | Yes | 75 tests passed, including API and installed PyTorch models; no optional skips |
| Frontend | Build verified | TypeScript + Vite production build; browser smoke not run |
| Real CSI300 scale | Manifest claim only | 1,312,314 rows / 852 instruments / 4,376 dates; not recounted here |
| Ridge 8.49%, Sharpe 0.445 | Historical provisional only | Preserved case-study CSV; do not use as a verified impact metric |
| Market execution constraints | Separate component only | Not integrated into the flagship daily ledger |

Sources: [`synthetic_reconciliation.json`](benchmarks/synthetic_reconciliation.json),
[`synthetic_repeatability.json`](benchmarks/synthetic_repeatability.json),
[`ledger_1260x300.json`](benchmarks/ledger_1260x300.json),
[`implementation plan`](QUANT_ENGINEERING_IMPLEMENTATION_PLAN.md).

## Three English resume bullets

1. **Developed a Python quantitative equity research pipeline with five model adapters, purged
   walk-forward evaluation, transaction-cost analysis and a read-only FastAPI/React interface;
   validated 75 automated tests covering data contracts, leakage, portfolio accounting and APIs.**
   Evidence: `src/pitalpha/models/registry.py`, `pipeline.py`, `splits/walk_forward.py`,
   `splits/validation.py`, `api.py`, `apps/web/`, and `bash scripts/verify.sh`.
   Five adapters exist; their contract/neural tests run here. This does not claim five fresh
   full-market model experiments.
2. **Hardened backtests against future-return selection bias and missing valuations, adding
   daily position/exit ledgers and independent reconciliation of 23,490 synthetic
   predictions and 40,006 position records; reproduced 13 deterministic artifacts exactly.**
   Evidence: `src/pitalpha/pipeline.py:50`, `backtest/daily.py:31`,
   `scripts/reconcile_run.py`, `tests/test_pipeline.py`, and the two synthetic evidence JSON files.
   The P0 fix originated in PR #1 and is preserved, extended and tested here.
3. **Benchmarked a deterministic daily portfolio engine on 378,000 synthetic stock-date
   observations, achieving 0.385 s median runtime across three fresh processes
   with identical output hashes and 184.9 MiB maximum process peak RSS.**
   Evidence: `scripts/benchmark_ledger.py` and `docs/benchmarks/ledger_1260x300.json`.
   Environment: macOS ARM64, Python 3.10.20, NumPy 2.2.6,
   Pandas 2.3.3, 6 logical CPUs, BLAS/OpenMP thread settings = 1.
   Timing excludes fixture creation/imports; RSS includes the whole process. This is a throughput
   baseline, not a measured speedup, execution latency or a cross-machine guarantee.

These bullets describe repository capabilities. Use them only after personally reviewing and
being able to explain the implementation; AI-assisted implementation alone does not prove the
owner's interview fluency. The separate resume assistant should select/shorten them for each JD.

## Interview explanations to prepare

- Show why removing a high-scoring stock because its future return is missing changes selection.
- Derive t-close → next-open → following-open P&L; explain the two-session terminal boundary.
- Explain the six-session embargo, including the newly purged neural validation split.
- Derive one-way turnover: initial 100% purchase = 1; complete sell-and-replace = 2.
- Explain gross-weight drift and why the cost overlay is not fee-funded share accounting.
- Explain which missing outcomes now fail and why a synthetic exit assumption cannot validate
  real suspension, limit, delisting or corporate-action economics.
- Demonstrate that a checksum-consistent manipulated return fails independent reconciliation.
- Explain why 2021–2025 is observed, why negative neural findings remain visible, and what a new
  holdout would establish beyond software correctness.

## Remaining hiring gaps

| Track | Supported now | Still missing |
|---|---|---|
| Python QD | Packaging, model API, tests, reproducible artifacts, API/UI, measured ledger throughput | Production operations, concurrency/networking, cost-funded execution ledger; C++ if required by JD |
| QR | Chronological validation, leakage regressions, baseline comparisons, IC/uncertainty/cost methods | Revalidated real-data economics, untouched holdout, paired model inference and independent market data |
| Technical QT | Position/risk/turnover/cost reasoning | Microstructure, order book/queue data, executable market rules and live risk controls |

Do not claim a production trading system, live profitability, institutional security-master
coverage, calibrated capacity/impact, or HFT/low-latency infrastructure.
