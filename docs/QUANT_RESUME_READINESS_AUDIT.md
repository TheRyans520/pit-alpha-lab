# Quant resume readiness audit — 2026-10-08

Scope: static review of public `main` at `239dab5`. Full real-market dataset is not redistributed; no independent real-market rerun has been performed for this review.

## Confirmed strengths
- Research package with model registry, synthetic reproducibility path, CSI300 Qlib audited snapshot loader, walk-forward splits, signal metrics and cost-aware daily ledger.
- First-party tests for leakage, splits, quality, portfolio, CLI and API. GitHub Actions CI on the reviewed main commit reported success.
- Explicit frozen historical case study, public provenance manifest, SHA256 verification and honest negative neural-model results.

## P0 — selection conditioned on future return availability
Evidence: `src/pitalpha/pipeline.py` previously built `portfolio_evaluation` by filtering `realized_return_1d.notna()` **before** computing Top-K holdings. Whether next-session P&L is available cannot be treated as a decision-time eligibility signal. It can change ranks and cause survivorship / execution-timing bias.

Remediation in this PR:
- Preserve all ex-ante candidate securities on evaluated dates and cut only the terminal **calendar date**, which has no next-session observations by design.
- Fail explicitly when a selected holding lacks a next-period return, rather than silently attributing a zero return. The error indicates that real delisting/suspension/corporate-action semantics need a verified outcome rule.
- Add targeted regression test for missing selected returns.

Acceptance:
- CI and unit tests pass.
- A synthetic panel with a higher-scored instrument whose future return is missing must fail loudly, not silently select the runner-up.
- Reproduce all CSI300 metrics against underlying frozen market data after the selection policy is fixed; do not reuse old headline return figures as revalidated numbers.

## P1 — execution constraints are not active in flagship ledger
Evidence: `src/pitalpha/portfolio/constraints.py` defines tradeability masks, participation caps, and no-leverage constraints; `src/pitalpha/pipeline.py` currently invokes `build_ranked_portfolio` and `build_buffered_ranked_portfolio` without invoking `enforce_execution_constraints`.

Action: connect **point-in-time verified** tradeability data, price-limit and suspension rules to the portfolio ledger, accounting for blocked sells, cash and realized return. Otherwise describe current CSI300 outcome only as a simplified long-only cost-sensitive backtest, not executable strategy returns.

## P1 — unavailable actual data prevents independent validation
Snapshot parquet partitions are intentionally excluded from Git. Published CI exercises the software/synthetic contracts but does not rerun frozen L1 CSI300. A verified local rerun and comparison with stored CSVs is mandatory before quoting returns as independently audited.

## P2 — quant-developer differentiation
- Add benchmarked profiling and deterministic throughput/latency/memory figures.
- Keep false promises about live trading, exchange microstructure, market impact and production trading out of resume.

## Resume decisions
QD/QR project is credible as **research software implementation** already. Historical performance metrics are **provisional** pending P0 remediation and rerun. Do not describe the separate constraints component as enforced by the flagship pipeline before integration.

Current PR is a defensive hardening patch, not a validated update to the case-study numbers.
