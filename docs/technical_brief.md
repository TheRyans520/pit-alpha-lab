# PIT Alpha Lab technical brief

> **2026-10-08 audit status:** Historical CSI300 performance below is provisional.
> Portfolio selection previously depended on future-return availability, and neural
> inner validation lacked a label embargo. Code has been hardened, but the frozen
> market partitions are unavailable on the handoff computer, so corrected real-data
> returns have **not** been reproduced. Use the project as research-engineering
> evidence; do not quote these historical returns as revalidated resume results.
> See [the audit](QUANT_RESUME_READINESS_AUDIT.md).


## The problem

Quantitative research often fails between a promising prediction metric and a realizable portfolio.
Historical constituent leakage, inconsistent preprocessing, optimistic execution timing, ignored
turnover and repeated test-set tuning can each manufacture apparent alpha. PIT Alpha Lab makes those
assumptions explicit, executable and auditable.

The flagship study asks a narrow question: under one frozen point-in-time protocol, do Ridge,
LightGBM or neural models produce a stronger net long-only CSI300 portfolio?

## Research contract

- **Data:** community Qlib China daily data, frozen as
  `qlib-cn-2026-09-09-common-2025`; raw data are not redistributed.
- **Universe:** historical CSI300 interval membership evaluated on every decision date.
- **Features:** 157 Alpha158 OHLCV-derived features.
- **Label:** open `t+1` to open `t+6`, cross-sectionally winsorized at 1% and 99%.
- **Splits:** annual expanding-window refits for 2021–2025 with a six-trading-day embargo.
- **Execution:** score after close `t`; trade no earlier than open `t+1`.
- **Portfolio:** weekly equal-weight Top 30, plus a rank-40 retention buffer sensitivity.
- **Costs:** 0, 5, 10 and 20 bps one-way, including initial entry.
- **Uncertainty:** 1,000-replication moving-block bootstrap with a ten-date block.

Before fitting, the shared panel must pass 14 semantic checks covering observation keys, member and
date coverage, feature/label coverage, price and volume validity, OHLC ordering and content hashes.
The checked real panel contains 1,312,314 rows and 298–300 eligible members per date.

## Architecture and reproducibility

The package separates data identity, feature transforms, splits, model adapters, signal diagnostics,
portfolio decisions, backtest accounting, metrics and artifacts. Models cannot own portfolio logic,
and test-period data cannot fit preprocessing state. A run records its resolved configuration,
snapshot identity, feature schema, environment metadata and artifact SHA-256 hashes.

The same pipeline supports a deterministic synthetic fixture and the frozen CSI300 adapter. The
synthetic path gives a public, data-license-safe regression target; the real adapter refuses to load
partitions that do not match extraction and snapshot metadata.

Completed artifacts are exposed by a path-safe, read-only FastAPI service. The React dashboard reads
that API and falls back to a small, explicitly labelled audited snapshot when the service is absent.
It never accepts arbitrary filesystem paths or mutates experiment outputs.

## Findings

At the pre-specified 10 bps cost, Ridge produced 8.49% annualized return, 0.445 Sharpe and -25.61%
maximum drawdown. LightGBM produced 5.36%, 0.323 and -43.73%, respectively. Their Rank IC values were
similar—0.02535 and 0.02485—but Ridge converted the signal into a more stable portfolio and remains
the preferred baseline.

The deterministic MLP produced -11.33% annualized return. A 48,402-parameter market-context network
combined same-date universe statistics, a feature gate and differentiable within-date ranking loss.
Across five seeds its mean Rank IC was positive at 0.0146, yet all five unbuffered portfolios lost
money after 10 bps, averaging -5.18% annualized. Ablations showed that ranking loss improved Rank IC,
but neither it nor the context gate produced a net investable improvement.

This negative evidence is a core result: predictive association, portfolio value and robustness are
different claims. Complexity did not win under the frozen protocol.

## Engineering evidence

- Python 3.10 and every dependency are pinned and run inside an isolated virtual environment.
- Unit tests cover data checks, leakage, split timing, models, ledgers, artifacts, CLI and API.
- CI runs the Python contract on Windows and Linux, then independently builds the API and web app.
- The browser smoke test exercises live model/cost/buffer controls, mobile overflow and offline mode.
- Real multi-seed aggregation revalidates 70 artifact hashes before writing compact summaries.

## Boundaries and next research gate

The data evidence level is L1: reproducible public daily bars with historical interval membership.
It is not an institutional security master. Daily OHLCV cannot support queue, latency, market-impact
or high-frequency claims, and capacity is not calibrated without appropriate liquidity fields.

The 2021–2025 test period has been observed and is now frozen as historical evidence. A credible next
experiment will acquire a fresh period or new market, pre-register the protocol, construct true
60-day sequences and compare a small TCN or temporal Transformer against the unchanged Ridge and
LightGBM baselines. Promotion will require multi-seed net performance, not a single attractive run.
