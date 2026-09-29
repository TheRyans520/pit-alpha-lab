# Deep-learning research plan

## Principle

Deep learning is a hypothesis about representation, not an assumed improvement. Every neural
model must use the same point-in-time data, outer walk-forward folds, costs and portfolio ledger
as Ridge and LightGBM. A neural model is promoted only when its multi-seed net result is both
statistically and economically defensible.

## Stage 1: deterministic MLP control

The first neural baseline is intentionally small:

- training-only median imputation and standardization;
- chronological inner validation rather than random row splitting;
- early stopping selects only the epoch count;
- a fresh model is then refitted on the complete outer training window;
- LayerNorm, GELU, dropout, AdamW and Smooth L1 loss;
- CPU deterministic algorithms and an explicit seed;
- test labels never enter preprocessing, selection or fitting.

This baseline separates a generic nonlinear neural effect from any benefit of a more elaborate
temporal or relational architecture.

## Stage 2A: PIT same-date Market-Context model (completed)

The first context model now implements same-date cross-sectional mean/dispersion tokens,
context-conditioned feature gates, date-batched robust regression and a differentiable within-date
correlation loss. It is intentionally not described as temporal: Alpha158 contains trailing-window
summaries, but this implementation does not yet encode a raw 60-day path.

Five CSI300 seeds and two seed-17 ablations are complete. The model is not promoted: all five
unbuffered 10 bps portfolios lose money, and only one buffered seed is marginally positive. The
ranking objective improves Rank IC in the seed-17 ablation, while the current context gate does not.
See `case_studies/csi300_alpha/market_context_multiseed.md` and
`case_studies/csi300_alpha/market_context_ablation.md`.

## Stage 2B: PIT temporal and relational model (fresh-holdout gated)

The next project-specific architecture may combine:

1. A 60-trading-day per-stock temporal encoder over raw returns, ranges and volume changes.
2. A market context token formed only from the eligible cross-section at decision time.
3. Context-conditioned feature gates inspired by MASTER's market-guided selection.
4. Optional relation attention using point-in-time industry links and trailing-return similarity,
   inspired by HIST without using future graph information.
5. Multi-horizon 1/5/20-day heads with a shared encoder.
6. A robust regression loss plus a within-date ranking loss.
7. Quantile heads for uncertainty rather than a single unqualified point score.

Primary references:

- MASTER official implementation: <https://github.com/SJTU-DMTai/MASTER>
- HIST paper: <https://arxiv.org/abs/2110.13716>
- StockMixer official implementation: <https://github.com/SJTU-DMTai/StockMixer>
- Qlib model benchmark protocol: <https://github.com/microsoft/qlib/tree/main/examples/benchmarks>

## Ablation status

- Completed: MLP versus same-date context gate.
- Completed: regression-only versus regression-plus-ranking loss.
- Completed: fixed-seed debugging followed by five model seeds.
- Pending fresh holdout: MLP versus true temporal encoder.
- Pending fresh holdout: temporal encoder with and without market context.
- Pending suitable point-in-time graph data: relation attention.
- Pending fresh holdout: single-horizon versus multi-horizon supervision.

The comparison reports parameter count, IC/RankIC dispersion,
turnover, net Sharpe, drawdown and break-even cost. A higher gross IC alone is insufficient.
