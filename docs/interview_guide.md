# Interview guide

## 90-second project explanation

> PIT Alpha Lab is a point-in-time, cost-aware equity research platform rather than a notebook-only
> backtest. I built a frozen data and experiment contract around historical CSI300 membership,
> purged annual walk-forward splits and next-session execution. Ridge, LightGBM and neural models all
> use the same features, folds and portfolio rules, so the comparison is controlled. Model scores
> flow into weekly Top-30 holdings, turnover, transaction costs, risk and bootstrap uncertainty, and
> every result is linked to checksum-verified artifacts. The interesting result is that complexity
> did not win: Ridge returned 8.49% annualized after 10 bps with a 0.445 Sharpe, while the MLP and a
> five-seed market-context network lost money. I kept those negative results because positive Rank IC
> did not translate into a viable strategy. I also exposed the evidence through a read-only FastAPI
> service and a responsive React dashboard. The next experiment is deliberately gated on a fresh
> holdout and true temporal sequences, because the current test years have already been observed.

## Five-minute demo route

1. Start at the dashboard hero and state the frozen universe, 2021–2025 out-of-sample window and
   10 bps cost assumption.
2. Switch Ridge, LightGBM and MLP. Point out that every model shares the same split and portfolio
   contract; the UI is reading audited result artifacts rather than recalculating metrics in-browser.
3. Change the cost scenario and turnover buffer. Explain why an IC result is incomplete without its
   holdings, trading frequency and cost sensitivity.
4. Open the signal evidence and data-integrity panels. Mention the moving-block confidence interval,
   14 semantic checks and content hashes.
5. Finish on the neural stability panel. Explain why all five negative net runs prevent promotion and
   why the next test must move to fresh data instead of tuning the observed period again.

## Resume bullets

- Built an end-to-end point-in-time equity research platform covering leakage-safe walk-forward
  training, signal diagnostics, portfolio construction, transaction costs, risk and artifact lineage.
- Audited 1.31M CSI300 observations across 157 features with 14 semantic data-quality gates and
  checksum-verified raw/feature partitions.
- Benchmarked Ridge, LightGBM and PyTorch models under identical 2021–2025 splits; retained Ridge at
  8.49% net annualized return and 0.445 Sharpe after 10 bps while documenting failed neural hypotheses.
- Developed a read-only FastAPI service and responsive React/TypeScript dashboard with interactive
  model, cost and turnover controls, audited fallback data and desktop/mobile browser tests.
- Established deterministic Windows/Linux CI, isolated pinned environments and manifest-linked
  outputs for reproducible research.

Use only bullets that accurately match the role and be ready to explain every number and assumption.

## Likely interview questions

### Why did Ridge beat the more complex models?

The sample is noisy and the features are engineered trailing summaries rather than raw sequences.
Regularization can be more valuable than nonlinear capacity in that setting. The important point is
that the protocol—not model prestige—decided the winner.

### How did you prevent look-ahead bias?

Membership is evaluated by date, transforms fit only on training rows, labels start at the next open,
and fold boundaries apply a six-trading-day embargo. Tests deliberately inject future information and
invalid timing so those failures are observable.

### Why is positive Rank IC not enough?

Rank IC measures cross-sectional ordering, not concentration, turnover, path risk or implementation
cost. The market-context model had positive average Rank IC but lost money after the exact portfolio
and cost rules were applied.

### Would you deploy this strategy?

No. It is a research benchmark using L1 public daily data. Deployment would require stronger
corporate-action and delisting validation, explicit suspension/limit handling, liquidity and impact
calibration, paper trading, monitoring and a fresh untouched evaluation period.

### What would you build next?

I would pre-register a new-market or fresh-period experiment, add raw 60-day OHLCV sequences and
compare a compact TCN or temporal Transformer with the frozen Ridge baseline. Promotion would require
consistent multi-seed improvement after costs and stable risk, not only a higher prediction metric.
