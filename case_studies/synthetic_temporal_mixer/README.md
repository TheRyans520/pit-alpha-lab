# Synthetic Temporal Mixer development result

> Historical development outputs below predate the inner-validation embargo fix.
> Their files are preserved; they are not measurements of the current model code.


This is a software and causality fixture, not evidence that the model can earn money in a real
market. It exercises the new 60-session tensor, temporal adapter, annual walk-forward refits,
portfolio construction, costs and artifact lineage on 62,640 deterministic synthetic panel rows.

## Decision

The last-session residual is retained as the stronger temporal development baseline, but neither
temporal variant is promoted over Ridge.

| Model | IC | Rank IC | Buffered 10bps annual return | Sharpe | Max drawdown | Mean turnover | Break-even cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | **0.1029** | **0.0953** | **7.74%** | **0.645** | **-10.21%** | 0.0882 | **43.52 bps** |
| Temporal Mixer v0 | 0.0587 | 0.0582 | 3.70% | 0.347 | -14.99% | **0.0861** | 26.71 bps |
| Temporal Mixer + last-session residual | 0.0769 | 0.0764 | 4.81% | 0.428 | -13.48% | 0.1176 | 25.83 bps |

The residual repairs part of the information loss caused by averaging patch representations: it
raises IC, Rank IC, net return and Sharpe relative to v0. It also raises turnover, and its lower
break-even cost shows that the stronger prediction did not translate into better cost robustness.
This is precisely why promotion is based on the portfolio ledger rather than prediction metrics
alone.

## Reproduction identity

| Variant | Config | Run ID |
|---|---|---|
| Ridge | `configs/demo_synthetic.yaml` | `demo-synthetic-ridge-349c20d9028e-20260929T163033Z` |
| Temporal v0 | `configs/demo_synthetic_temporal_mixer.yaml` | `demo-synthetic-temporal-mixer-f605ac7f0006-20260929T162531Z` |
| Temporal residual | `configs/demo_synthetic_temporal_mixer_residual.yaml` | `demo-synthetic-temporal-mixer-residual-603f828cde31-20260929T163406Z` |

Exact unrounded rows are in `comparison.csv`; machine-readable decisions and config hashes are in
`result.json`. Large generated predictions and ledgers remain ignored by Git and can be reproduced
from the deterministic fixture.

The next permitted step is not more tuning on this synthetic result. It is to audit the raw
60-session real-data fields, then evaluate the frozen temporal baseline only in the development
window declared by the constraint-aware protocol.
