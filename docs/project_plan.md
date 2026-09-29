# Project plan

## Objective

Build a public, interview-ready quantitative research platform that demonstrates the complete
path from a point-in-time dataset to a cost-aware and uncertainty-aware portfolio conclusion.
The primary role fit is quantitative research and quantitative development; trading relevance
comes from decision timing, costs, risk and execution assumptions.

## Q0 - Engineering scaffold

Deliverables:

- Clean standalone repository with no research archives or vendored upstream source.
- Installable `src` package and minimal CLI.
- Versioned experiment configuration and manifest contracts.
- Cross-platform smoke CI and standard-library unit tests.
- Architecture, data policy and scope ADRs.

Exit criteria:

- All Q0 tests pass under the existing isolated project environment.
- `doctor`, `config validate` and `manifest create` work from PowerShell.
- Git ignores all known large-data and generated-artifact formats.

## Q1 - Golden end-to-end pipeline

Target duration: one week.

- Implement a deterministic synthetic market fixture.
- Port the verified point-in-time membership and annual walk-forward logic.
- Add Ridge training, predictions, Top-30 portfolio and 10 bps cost ledger.
- Produce one manifest-linked Markdown report from a single command.
- Treat the existing Phase 2A.7 numbers as a regression oracle, not copied implementation.

Exit criterion: a fresh CPU-only clone reproduces the synthetic demo in under five minutes.

## Q2 - Research diagnostics

Target duration: one week.

- IC, RankIC, ICIR and RankICIR by year and overall.
- Quantile monotonicity, signal decay and coverage.
- Walk-forward and embargo audit reports.
- Seed dispersion and moving-block confidence intervals.
- Tests designed to fail on future leakage, cross-sectional fit leakage and bad membership.

## Q3 - Portfolio, risk and costs

Target duration: one to two weeks.

- Retained Top-K baseline and constrained long-only optimizer.
- Weight, turnover and exposure constraints.
- Linear costs at 0/5/10/20 bps and breakeven cost.
- Transparent ADV/volatility/participation-rate capacity sensitivity.
- Gross-to-net attribution, drawdown contribution and concentration diagnostics.

## Q4 - Model layer

Target duration: one week.

- Preserve Ridge as the deterministic anchor.
- Add fixed LightGBM through the common model interface.
- Add one seeded PyTorch MLP only after the comparison protocol is stable.
- Report prediction, portfolio and computational metrics under identical splits.

StockMixer remains optional and cannot block the first public release.

## Q5 - Production-minded engineering

Target duration: one week.

- Content-addressed caches and safe interrupted-run recovery.
- Parallel fold execution with deterministic aggregation.
- Static checks, typing, structured logs and core coverage target of at least 80%.
- Runtime and peak-memory benchmarks.
- Windows and Linux CI paths.

## Q6 - Portfolio presentation

Target duration: one week.

- Results-first README with architecture, one command, key tables and limitations.
- Five-minute demo and two-page technical brief.
- Ninety-second English explanation and role-specific resume bullets.
- Interview appendix covering design choices, failed hypotheses and production gaps.

## Optional QT extension

After the flagship release, create a separate event-replay module using appropriately licensed
public tick/order-book data or a synthetic exchange. Compare TWAP, VWAP and participation-rate
policies under fees, latency and inventory constraints. Do not present daily bars as an HFT
simulation and do not let this extension delay Q1-Q6.

## Q7 - Dual product: evaluation engine and reference model

Track A makes the framework useful for any compatible model:

- a versioned built-in/external model adapter contract;
- explicit opt-in before third-party Python code can execute;
- common point-in-time splits, constraints, portfolio ledger, costs and artifacts;
- model-independent hard tradeability, liquidity, concentration, turnover and cash rules;
- API and dashboard diagnostics for requested versus executable positions.

Track B develops one built-in reference model through a literature-first gate:

- compare recent finance and time-series models using primary papers and official code;
- freeze a causal 60-session tensor and missingness contract;
- implement a compact temporal mixer before trying larger Transformers;
- add context, multiscale, uncertainty and soft constraint terms as isolated ablations;
- use a genuinely fresh holdout and retain rejected or negative variants.

Shared exit criterion: built-in and user-supplied models must traverse the same engine-owned
constraints and reporting path. The reference model is a demonstrated option, not privileged
evidence for the framework.

## Public-release definition of done

- No licensed raw data, archives, credentials, virtual environment or vendor source in Git.
- CPU synthetic demo is deterministic and CI-tested.
- Full experiment assumptions are versioned and machine-readable.
- Every displayed result traces to a manifest and generated artifact.
- Costs, turnover, drawdown, annual stability and uncertainty accompany prediction metrics.
- Negative and null results remain visible.
- README distinguishes historical research from live trading capability.
