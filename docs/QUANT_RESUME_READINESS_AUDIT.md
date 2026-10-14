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

---

# Cross-device follow-up audit — 2026-10-08

The preceding report is preserved as the original PR #1 review. This section supersedes its
acceptance status where stated; it does not upgrade the historical results.

## PR #1 reconciliation

Reviewed the actual diff against `239dab5`, not just its description. PR #1 remains draft/open,
unmerged at `f743e847740ad41b20a4b3fbc8250371ab092ded`. Its CI run
[37738742795](https://github.com/TheRyans520/pit-alpha-lab/actions/runs/37738742795)
reported success when checked. This working branch descends from that exact head, retaining its
P0 fix, test and audit history. It extends the patch rather than implementing an alternative.

## Confirmed defects and fixes

| Priority | Confirmed behavior at the reviewed base | Current evidence and resolution |
|---|---|---|
| P0 | `pipeline.py` prefiltered candidates using `realized_return_1d.notna()` | Retained PR fix; `pipeline.py:50` and `:247` preserve missing-outcome candidates. Full-pipeline adversarial test makes the top scorer's future return missing and requires failure. |
| P0 | Base ledger skipped missing held returns in P&L; PR #1 stopped this but did not test carried holdings | `backtest/daily.py:76` fails for selected or carried missing outcomes. Tests cover both ranked and buffered selectors. |
| P1 | Base ledger silently removed disappeared held names and treated the change as a sale without exit evidence | `backtest/daily.py:61` fails by default. Synthetic-only explicit last-mark exit assumption is recorded in positions and accounting policy. |
| P1 | PR #1 excluded only one end date for both sources | `pipeline.py:58` uses the source contract's calendar lag. Two sessions are needed for the declared real open(t+1)→open(t+2) interval. This is conservative until raw exporter semantics are independently checked. |
| P1 | MLP, market-context and temporal mixer had adjacent inner train/validation dates despite forward labels | Shared `splits/validation.py:9`; integrated at `models/mlp.py:129`, `market_context.py:302`, `temporal_mixer.py:339`. Six-session purge, missing-label calendar test and actual neural execution tests pass. |
| P1 | Config allowed five embargo sessions despite the declared t+6 label endpoint | `config.py` requires horizon + execution lag; regression test rejects five sessions. |
| P1 | `return_metrics` silently replaced NaN with zero | `metrics/returns.py:11` rejects missing, infinite and below-minus-100% values. |
| P1 | Local manifests had null code revision unless `GITHUB_SHA` was provided | `artifacts/manifest.py:21` captures Git revision/dirty state, actual source digest and environment; `experiment_id` combines source/config identity. |

## Missing-outcome and liquidation policy

| Situation | Implemented rule | Evidence still required for real economics |
|---|---|---|
| Non-held candidate has missing future return | Retain it in the decision universe; ranked portfolio may proceed if it is not held | Do not infer availability from the future |
| Selected or carried name has missing/invalid future return | Stop with date and instrument; no replacement or zero mark | Verified price/settlement outcome |
| Held name disappears, including constituent exit | Stop for real data | Tradable exit observation or continuing all-security valuation feed |
| Synthetic membership exit | Explicit `synthetic_liquidate_at_last_mark`, proceeds become cash, sale costs apply | Fixture assumption only; not usable evidence of real tradability |
| Suspension | Stop if valuation missing | PIT suspension state and a defensible mark; no automatic zero |
| Delisting | Stop if outcome/position disappears | Delisting return, cash settlement, timing and security-master event |
| Corporate action | Stop when necessary return missing/invalid | Split/dividend/merger adjustment and cash/share entitlements |
| Corrupt or absent quote | Stop if held; retain non-held candidate | Source correction or documented valuation rule |
| Terminal data boundary | Remove whole decision dates using calendar lag, regardless of individual returns | Actual exchange calendar and source timing |
| End of evaluation | Mark-to-market terminal wealth; no automatic terminal liquidation charge | Separate liquidation protocol if needed |

Daily position artifacts now contain retained positions and zero-target exit rows. A zero asset
return on an exit row is a zero-exposure bookkeeping convention, never a mark for a held asset.
The standalone reconciler rederives P&L from predictions and position weights and rejects tampered
cost returns even if an attacker refreshes the file checksum.

## Unverified risks and deliberate limitations

- Historical interval membership, adjustment factors, delisting completeness and forward-return
  construction originate in an external export. `data/audited_qlib.py:47`, `:92`, `:145` verify
  partition identity and key alignment; `:151` consumes supplied labels/returns. The adapter does
  not independently construct an exchange calendar or historical membership/security master.
- Separate execution constraints exist but are not called by the flagship pipeline. No reliable
  suspension/limit/ADV masks are available here. README wording now matches that boundary.
- Costs are a **linear overlay**: net = gross − sum(abs(weight changes)) × one-way bps. Both sale
  and purchase legs and initial entry are charged. Weights drift on gross wealth; this is not a
  fee-funded share/cash account with cost-dependent positions. Residual cash earns zero.
- Bootstrap intervals concern each model's mean IC and do not establish a paired model-performance
  difference, correct for multiple testing, or cure selection against an observed holdout.
- Config validation windows are not a universal hyperparameter-search implementation: annual
  outer folds expand over earlier years; neural selection uses a trailing purged inner window.
- Frozen real-data summaries, including Ridge 8.49% / Sharpe 0.445, are preserved and labelled
  provisional. No direction or size of correction is known without real data.
- New ledger exports use a daily-position v2 schema; legacy runs remain readable but the independent
  reconciler explicitly refuses to certify them as v2. Hash consistency is not economic validation.
- Failed experiments may leave a planned manifest; API listings expose completed runs only.

## Optional enhancements

Verified market-mask integration; a cost-funded order/share ledger; paired block-bootstrap model
comparisons; fully locked transitive Python dependencies; independent exchange-calendar validation;
and C++/concurrency work for specific QD roles. These are not delivered trading capabilities.

## Actual validation and resume decision

See `QUANT_ENGINEERING_IMPLEMENTATION_PLAN.md` for commands and the acceptance matrix, and
`QUANT_RESUME_FACTS.md` for claims mapped to evidence. Local Python/API/neural tests, the synthetic
pipeline, independent reconciliation, repeated-artifact checks, frontend build and a measured
benchmark are complete. Real-data reproduction is unavailable, as confirmed by the owner.
The defensible current positioning is quantitative research engineering, strongest for Python QD
and supportive of QR. There is no validated live execution or HFT claim.

## Follow-up: recovered artifacts and cash-funded execution

`docs/LEGACY_REPLAY_FINDINGS.md` supersedes the prior statement that saved real-data run artifacts
were unavailable. The owner supplied them privately: 42 hashes and 216 summary rows from the three
headline runs verify. Raw feature/price partitions remain unavailable. Strict saved-score replay
stops at an unavailable held return on 2021-01-27; no corrected performance is asserted.

The separate synthetic execution path now integrates `enforce_execution_constraints` with a
fractional-share/cash account and actual fee deductions (`backtest/execution.py`). An independent
checker (`backtest/execution_audit.py`) verifies the inventory/cash identities from inputs and fills.
This does not retrofit real tradeability evidence into the historical CSI300 case.

The review also found a concrete projection defect: clipping desired weights before solving for the
simplex threshold changed the advertised Euclidean solution. Desired weights (1.0, 0.3), a 0.4
single-name cap and a 0.5 gross cap must project to (0.4, 0.1), not (0.3, 0.2). This is fixed and
regression-tested. Nonfinite portfolio notionals and non-boolean side masks are now rejected.

All 91 local tests pass, including the cash-scenario and legacy-audit checks. New public files contain
only synthetic fixtures and small derived diagnostics; private archived predictions are not uploaded.
