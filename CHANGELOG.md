# Changelog

All notable project changes will be recorded here.

## Unreleased

- Added a literature-first architecture gate covering finance-specific and recent general
  time-series models before implementing the built-in temporal reference model.
- Added a lazy built-in/external model registry, explicit external-code opt-in, response validation,
  label-independence and input-mutation audits, CLI discovery and a read-only model API.
- Added a deterministic hard-constraint kernel for tradeability, liquidity participation,
  concentration, turnover and cash conservation.
- Added a preregistration candidate with evidence and claim registries for a fresh-holdout,
  constraint-aware temporal experiment.
- Added a read-only, calendar-aware temporal tensor store with explicit row/feature masks,
  outcome-column rejection and future-perturbation causality tests.
- Added a compact built-in causal temporal mixer, a synthetic config and embargo-history handling
  through the common model adapter contract.
- Retained the last-session residual as the stronger synthetic temporal baseline while keeping
  Ridge champion; published both temporal variants and the negative promotion decision.
- Reworked the public README around audited results, reproducibility and the interactive dashboard.
- Added a reviewed dashboard image, technical brief, interview guide and public-release checklist.
- Added cross-platform verification scripts that reject global Python environments.
- Removed machine-specific workspace paths from contributor and web documentation.

## 0.1.0.dev0 - 2026-09-28

- Established the Q0 package and repository scaffold.
- Added configuration validation and experiment-manifest primitives.
- Added isolated-environment diagnostics, unit tests and cross-platform CI.
- Recorded project, data and reproducibility architecture decisions.
- Added the deterministic point-in-time synthetic market fixture.
- Added annual embargoed walk-forward Ridge fitting and prediction diagnostics.
- Added weekly ranked and equal-weight portfolio ledgers with cost accounting.
- Added one-command, content-hashed artifacts and a generated Markdown report.
- Added quantile returns, signal decay, coverage and moving-block bootstrap uncertainty.
- Added label-independence and train-only preprocessing leakage tests.
- Added a rank-retention buffer with side-by-side Top-K and equal-weight ledgers.
- Added volatility, downside, tail-loss and concentration diagnostics.
- Added terminal-return break-even transaction-cost sensitivity.
- Added a read-only, per-partition checksum-verified adapter for frozen Qlib parquet exports.
- Reproduced 363,409 Phase 2A.7 Ridge predictions exactly through the public package pipeline.
- Added the first tracked CSI300 Ridge case-study result tables.
- Made portfolio aggregation byte-stable across Python hash seeds with ordered compensated sums.
- Added the frozen LightGBM baseline and exact Phase 2A.7 prediction reproduction.
- Recorded the pre-registered Ridge-over-LightGBM net-performance decision.
- Added configurable semantic data-quality gates and report artifacts.
- Added explicit L0-L5 data evidence levels and source-expansion policy.
- Added an optional deterministic PyTorch MLP with chronological validation and full-window refit.
- Added the paper-grounded market-context model and ablation plan.
- Ran the real CSI300 MLP baseline and retained its statistically weak, negative net result.
- Added a path-safe, read-only FastAPI over completed checksum-audited runs.
- Added a responsive React/TypeScript research dashboard with live cost, model and turnover controls.
- Added deterministic desktop and narrow-screen visual QA for the dashboard and split chart loading.
- Added a point-in-time same-date market-context gated neural model and within-date ranking loss.
- Added auditable CLI overrides for multi-seed runs and context/ranking ablations.
- Completed five CSI300 market-context seeds and rejected the model on unstable negative net returns.
- Added checksum-gated multi-seed and ablation CSV/JSON/Markdown summaries.
