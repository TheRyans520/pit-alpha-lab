# Market-context ablation

This seed-17 comparison is explanatory, not a new confirmatory test. Every row uses the same frozen
CSI300 data, walk-forward folds, Top-30 portfolio and 10 bps one-way cost.

| Variant | Rank IC | Annualized return | Sharpe | Buffered return |
|---|---:|---:|---:|---:|
| Ridge | 0.0253 | 8.49% | 0.45 | 7.72% |
| MLP | 0.0075 | -11.33% | -0.36 | -9.31% |
| Context, no ranking loss | -0.0079 | -2.01% | 0.06 | -0.84% |
| Ranking loss, no context | 0.0240 | -7.64% | -0.20 | -6.78% |
| Context + ranking loss | 0.0087 | -7.01% | -0.14 | -5.31% |

The within-date ranking objective raises Rank IC relative to the plain MLP, but this does not become
positive net performance. Adding the current same-date context to that objective lowers Rank IC in
seed 17. Ridge remains ahead of the best neural variant by
10.49% annualized.
The next credible neural step requires true temporal sequences and a fresh holdout, not more tuning
against 2021-2025.
