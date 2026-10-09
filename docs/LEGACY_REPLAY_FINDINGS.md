# Archived CSI300 prediction diagnosis — 2026-10-08

The owner supplied an old project backup. Its 140 tracked files match original main `239dab5`;
29 completed experiments and 385 manifest-listed artifacts matched their recorded hashes.
The archive contains predictions and backtests, but not the source raw/feature audit partitions.

The reusable public command is:

```bash
.venv/bin/python -m pitalpha audit-run /path/to/saved/run --output /tmp/run-audit.json
```

It reads artifacts only. No archived scripts or third-party plugins are executed; no model is refitted.
Exit code zero means the diagnostic completed, not that the strategy passed an economic gate.
Inspect `strict_replay.status` and `corrected_market_performance` in the returned JSON.

## Confirmed evidence

The three headline runs each contain 363,409 saved predictions: **1,090,227 model-prediction rows**
in total, not that many unique stock-date observations. Their 42 manifest-listed artifacts pass
hash checks. Recomputing the return summaries checks **216 strategy/window/cost rows**.

| Model | Weekly unbuffered Top-30 dates affected by future-return prefilter | First strict saved-score replay failure |
|---|---:|---|
| Ridge | 1 | 2021-01-27, SH600340 held return unavailable |
| LightGBM | 3 | 2021-01-27, SH600340 held return unavailable |
| MLP | 6 | 2021-01-27, SH600340 held return unavailable |

Each model has 405 missing future returns after the conservative two-session calendar crop.
Affected-date counts compare weekly ranks only; they do not measure all carried-position,
buffered-portfolio or benchmark effects. Do not infer a performance delta from these counts.

The old saved returns recompute to the published Ridge 8.49%, LightGBM 5.36% and MLP -11.33%
annualized figures. **This establishes arithmetic consistency of old outputs, not corrected
market performance.** A null source code revision in the old manifests remains a lineage limit.

Compact evidence, including source artifact hashes and exact diagnostic messages, is in
[`legacy_replay_audit.json`](../case_studies/csi300_alpha/results/legacy_replay_audit.json).
No private predictions, source bars or archive are redistributed.

## What is still unknown

The cause and correct economic treatment of each missing return require source price/valuation,
security-event and tradeability evidence. A stock code and a missing value do not establish
suspension, delisting or a corporate action. The replay uses old scores and saved dates; it is
neither a model refit nor an independent reconstruction of the exchange calendar or source returns.

New performance claims remain blocked until those inputs are available. Meanwhile, this is a
concrete QR/QD audit result: the project detects invalid evaluation assumptions and retains the
failure instead of replacing a holding or reporting an invented corrected return.
