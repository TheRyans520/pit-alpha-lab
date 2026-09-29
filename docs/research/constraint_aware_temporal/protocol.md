# Constraint-Aware Temporal Alpha protocol

> Status: **design-freeze candidate, not yet confirmatory**
>
> Created: 2026-09-29
>
> Evidence status: primary sources were opened for discovery by the research agent; every external
> claim remains pending accountable human verification in `source_manifest.json`.

## Research question

Does a causal 60-day temporal model with explicit market-feasibility constraints improve stable,
after-cost portfolio performance over unchanged Ridge and LightGBM baselines on a genuinely fresh
2026 holdout?

The project uses **market-structure constraints** rather than claiming literal physical laws. The
core product promise is narrow and falsifiable:

> A model may propose alpha, but it cannot bypass information timing, capital conservation,
> tradeability, liquidity or portfolio-risk limits.

Constraints are an inductive bias and an execution guarantee, not proof of out-of-sample
generalization. Generalization is earned only through a frozen new-sample result and ablations.

## Why this is a meaningful extension

The existing platform already enforces point-in-time membership, train-only preprocessing, purged
walk-forward splits and next-session execution. It does not yet model whether every requested trade
can actually occur. The current ledger assumes target weights are reachable, so suspension,
price-limit, settlement and liquidity effects can still create an implementation gap.

Exchange rules can make a requested trade unavailable even when the signal is known. The current
Shanghai Stock Exchange rules state that purchased securities cannot be sold before settlement,
except for instruments with same-day turnaround, and specify price-limit and suspension mechanisms.
The Shenzhen exchange maintains its own current rule book. [claim:C001] [evidence:E001,E002]

Differentiable disciplined convex programs can be embedded as layers whose solutions support
backpropagation, providing a route from predicted alpha to constrained allocation without treating
portfolio construction as an unrelated afterthought. [claim:C002] [evidence:E003]

## Constraint hierarchy

Higher rows have priority over lower rows. A lower-priority objective may never relax a higher-level
constraint to improve a metric.

| Priority | Constraint | Enforcement | Failure behavior |
|---:|---|---|---|
| 1 | Causal information | 60-day window ends at decision time; PIT universe; train-only state | Stop the run |
| 2 | Tradeability | Separate `can_buy`/`can_sell` masks for suspension, limits and settlement | Block the side and retain cash/position |
| 3 | Capital conservation | Long-only, gross exposure ≤ 1, sells before buys | Scale buys; never create leverage |
| 4 | Liquidity | Trade value ≤ configured fraction of trailing ADV | Clip order; missing ADV fails closed |
| 5 | Concentration | Per-name cap and later sector/factor exposure bounds | Project into feasible set |
| 6 | Turnover | Per-rebalance cap plus explicit one-way costs | Scale executable deltas |
| 7 | Learned preference | Alpha, uncertainty and risk-adjusted objective | Optimize only inside feasible set |

The first implementation covers priorities 2–6 at the weight-transition level. Exact board-lot,
corporate-action and board-specific price-limit logic remains gated on audited fields; it must not be
approximated silently.

## Model design

Architecture selection is governed by the
[literature-first model design review](model_design_review.md). Candidate models are screened by
their original task and protocol, reproducibility and fit to the point-in-time contract before any
component enters code.

```text
PIT raw bars + masks
        |
        v
60-day causal relative features -----> missingness / tradeability channels
        |
        v
small temporal encoder (TCN or temporal mixer)
        |
        +----> same-date market context and regime token
        |
        v
return score + uncertainty head
        |
        v
bounded long-only allocation proposal
        |
        v
hard execution projector: tradeability -> liquidity -> turnover -> cash
        |
        v
audited holdings, costs, violations and net returns
```

The first temporal baseline will be a compact temporal mixer, not a large Transformer.
StockMixer explicitly mixes indicator, temporal and stock dimensions with a lightweight MLP design.
[claim:C006] [evidence:E007] PatchTST motivates patching and shared channel-independent temporal
weights, but its published forecasting benchmarks do not by themselves establish equity-alpha
performance. [claim:C004] [evidence:E005]

Market conditioning remains an ablation rather than an assumed benefit. MASTER combines
market-guided feature selection with intra-stock and inter-stock aggregation, which motivates—but
does not validate—the planned context token for this dataset. [claim:C003] [evidence:E004]

### Inputs known by decision time

- 60 sessions of close-to-close return, open gap, high-low range and close location;
- log volume/value changes and trailing realized volatility;
- missingness and stale-price masks;
- PIT membership and instrument metadata;
- same-date market return, breadth, cross-sectional dispersion and volume state;
- later, PIT industry and float-cap fields when their availability timestamps are audited.

Absolute price levels are excluded from the core encoder. Relative/log features and training-only
normalization reduce unit and split-adjustment sensitivity. Missing observations remain explicit
channels rather than being silently interpreted as zero.

### Outputs and objective

- primary head: five-session cross-sectional score;
- auxiliary heads: one- and twenty-session score, used only when all split embargoes cover them;
- uncertainty head: conditional scale or quantiles;
- objective: robust regression + within-date ranking + cost/turnover-aware portfolio term;
- hard constraints: always applied outside the loss, even when a differentiable allocation layer is
  later introduced.

Direct portfolio-learning research shows one possible design that optimizes portfolio performance
and reports transaction-cost and volatility-scaling sensitivity rather than relying on prediction
loss alone. [claim:C005] [evidence:E006] That evidence is architecture motivation, not a transferable
performance claim for CSI300 stocks.

## Fresh-sample protocol

The 2021–2025 CSI300 result is permanently historical/exploratory for future model selection. It is
not reused as confirmatory evidence.

| Use | Dates | Permitted use |
|---|---|---|
| Development | 2008-01-01 to 2023-12-31 | Fit model parameters and debug code |
| Validation | 2024-01-01 to 2025-12-31 | Select architecture and fixed hyperparameters; already observed |
| Confirmatory test | 2026-01-05 to 2026-08-31 | One locked evaluation after snapshot and code freeze |

The exact test dates are fixed before acquiring the new test labels. A future snapshot must include
enough subsequent sessions to construct the longest declared label. If the required data are
unavailable or fail quality checks, the experiment is blocked; the interval is not shortened based
on observed performance.

### Data gate

Promotion requires at least L2 evidence:

1. immutable source and extraction checksums;
2. a second-source discrepancy report for prices and corporate actions;
3. historical universe membership by effective date;
4. suspension, price-limit, ST/board status and delisting coverage;
5. value traded or ADV fields with units and adjustment policy;
6. explicit missing/stale observations rather than forward-filled tradability.

Hong Kong is a valuable later transfer market for the portfolio, but it will not become the primary
confirmatory set until constituent history, corporate actions and tradeability can meet the same
evidence gate.

## Baselines and ablations

All variants share the same snapshot, folds, labels, costs and constraint engine.

1. Frozen Ridge baseline.
2. Frozen LightGBM baseline.
3. Compact temporal encoder without market context.
4. Temporal encoder with context token.
5. Variant 4 with soft cost/risk loss.
6. Variant 5 with the hard execution projector.

Additional ablations remove one component at a time: temporal patching, context, ranking loss,
uncertainty head and constraint-aware training. The unconstrained portfolio may be reported only as
a diagnostic upper bound; it is not eligible for promotion.

## Promotion gates

The temporal model is promoted only if all conditions pass on the locked 2026 test:

- zero causal, capital, tradeability and configured exposure violations;
- five predeclared seeds, with no seed selected after test inspection;
- median 10 bps net Sharpe above the unchanged Ridge baseline;
- at least four of five seeds with positive net excess return versus Ridge;
- moving-block 95% interval for mean daily excess net return above zero;
- median 20 bps net return above zero;
- worst-seed maximum drawdown no more than five percentage points worse than Ridge;
- constraint ablation reported, including blocked orders, cash drag and liquidity clipping.

Failure is retained as evidence. A higher IC, a single profitable seed or an attractive gross curve
cannot override these gates.

## Current implementation boundary

`pitalpha.portfolio.constraints` now provides a deterministic, dependency-free hard-constraint
kernel:

- bounded long-only projection;
- independent buy/sell masks;
- participation-rate clipping against ADV and portfolio notional;
- turnover scaling;
- sell-before-buy cash conservation;
- fail-closed behavior for missing tradeability or liquidity state;
- diagnostics for every blocked or scaled decision.

It is intentionally not wired into the real case-study ledger yet. Integration is blocked until the
new data contract supplies audited point-in-time execution fields; inventing those fields would make
the backtest look more realistic while reducing its evidential value.

## Evidence and accountability

This protocol follows an explicit source/claim registry workflow based on the Scientific Agent
Skills procedural library. [claim:C008] [evidence:E009] The library reports procedures rather than
task-level proof of correctness, so the human project owner remains responsible for verifying every
source and approving the final freeze.

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A
Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065.
<https://doi.org/10.48550/arXiv.2609.00065>
