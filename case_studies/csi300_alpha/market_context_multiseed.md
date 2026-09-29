# Market-context model: five-seed result

The model is **not promoted**. All five runs use the frozen CSI300 snapshot, identical walk-forward
folds and identical portfolio/cost rules; only neural initialization and training order change.

| Metric | Mean | Seed range |
|---|---:|---:|
| Rank IC | 0.0146 | 0.0055 to 0.0263 |
| Annualized return, 10 bps | -5.18% | -7.01% to -2.10% |
| Sharpe, 10 bps | -0.08 | -0.15 to 0.06 |
| Buffered annualized return, 10 bps | -3.85% | -6.93% to 0.31% |

Rank IC is positive in all seeds, but unbuffered net annualized return is negative in all seeds.
Only 20% of buffered seeds are positive,
and the positive case is economically small. This is evidence that the current same-date context and
ranking objective do not overcome turnover and temporal-regime limitations. The result remains
exploratory because the 2021-2025 test period had already been observed before this architecture.

Every included run passed data quality and artifact checksum verification. See
`results/market_context_multiseed_10bps.csv` and `results/market_context_multiseed_summary.json`.
