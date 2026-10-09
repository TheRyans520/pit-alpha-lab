# CSI300 point-in-time model study

> **2026-10-08 audit status:** Historical CSI300 performance below is provisional.
> Portfolio selection previously depended on future-return availability, and neural
> inner validation lacked a label embargo. Code has been hardened, but the frozen
> market partitions are unavailable on the handoff computer, so corrected real-data
> returns have **not** been reproduced. Use the project as research-engineering
> evidence; do not quote these historical returns as revalidated resume results.
> See [the audit](../../docs/QUANT_RESUME_READINESS_AUDIT.md).


This case study is the first real-data model comparison from the public package pipeline. It uses the frozen
`qlib-cn-2026-09-09-common-2025` snapshot, historical interval-at-date CSI300 membership,
Alpha158-OHLCV-157 features and annual expanding-window refits. Ridge, LightGBM, MLP and the
market-context experiments share every data, label, split, portfolio, cost and evaluation assumption.

## Frozen protocol

- Data window: 2008-01-01 through 2025-12-31.
- Test window: 2021-2025, with one refit per year and a six-trading-day label embargo.
- Label: open `t+1` to open `t+6`, winsorized cross-sectionally at 1% and 99% per date.
- Execution: score after close `t`, apply to the next open-to-open return.
- Portfolio: weekly equal-weight Top 30; buffered variant retains names through rank 40.
- Costs: 0, 5, 10 and 20 bps one-way; initial entry is charged.
- Uncertainty: 1,000-replication moving-block bootstrap, block length 10 dates.

## Prediction result

Ridge's all-period IC was 0.02785 and RankIC was 0.02535. Their 95% moving-block intervals were
[0.01483, 0.04032] and [0.01221, 0.03817]. LightGBM's IC was 0.02409 and RankIC was 0.02485,
with intervals [0.01232, 0.03596] and [0.01378, 0.03705]. Both models produced monotonic five-day
return quantiles; their uncertainty intervals overlap substantially.

Before either model was fitted, the shared data passed all 14 semantic quality checks: zero
duplicate keys, 298-300 daily members, 96.08% minimum feature coverage, 97.05% label coverage,
97.36% next-return coverage and zero OHLC ordering violations.

## Portfolio result and decision

At 10 bps one-way cost, the plain Top-30 portfolio produced 8.49% annualized return, 0.445 Sharpe
and -25.61% maximum drawdown. The eligible-universe equal-weight benchmark produced 1.63%
annualized return, 0.179 Sharpe and -34.99% maximum drawdown. The rank-40 buffer reduced mean daily
turnover from 0.3063 to 0.2816 but also reduced annualized return to 7.72%; in this sample the
turnover reduction did not improve the primary net result.

LightGBM's plain Top-30 portfolio produced 5.36% annualized return, 0.323 Sharpe and -43.73%
maximum drawdown at the same 10 bps cost. Its buffer also reduced turnover but weakened the net
result. Under the pre-registered rule—higher 10 bps after-cost Sharpe, with annualized return only
as a tie-breaker—**Ridge is the preferred baseline**. No post-hoc tuning was used to manufacture
that ranking.

The first deterministic PyTorch MLP is retained as a negative result. Its IC was 0.0070 with a
95% moving-block interval crossing zero, while the 10 bps Top-30 portfolio returned -11.33%
annualized with a -0.363 Sharpe. All five folds selected one epoch by chronological validation.
This rejects the idea that adding a generic neural network to flat Alpha158 rows is automatically
an improvement; the next neural hypothesis must add temporal or market-context structure.

## Market-context extension

A 48,402-parameter gated network then conditioned each stock on same-date eligible-universe feature
means and dispersions and added a differentiable within-date ranking objective. Its context is
point-in-time: no statistic crosses a decision-date boundary and no label enters the context token.

Five model seeds (7, 17, 29, 41 and 53) were run under the frozen protocol. Mean Rank IC was 0.0146
(range 0.0055-0.0263), but all five unbuffered portfolios lost money after 10 bps: mean annualized
return was -5.18%. Buffering improved the mean to -3.85%, with only one of five seeds marginally
positive. The model is therefore **not promoted**.

Seed-17 ablations show that the ranking loss raises Rank IC relative to the plain MLP, but not net
performance. Removing market context while retaining the ranking loss produced Rank IC 0.0240 but
still lost 7.64% annualized after costs. Adding the current context gate lowered Rank IC to 0.0087.
Ridge remained 10.49 percentage points ahead of the best neural variant in annualized return.

These neural comparisons are exploratory because the 2021-2025 test period had already been observed.
The next credible neural experiment requires true temporal sequences and a fresh holdout or new market,
not further tuning against the same test years. See `market_context_multiseed.md`,
`market_context_ablation.md` and their compact CSV/JSON files under `results/`.

This is a historical result on a public daily dataset, not evidence of live tradability. The
cost break-even statistic is not a calibrated market-impact or capacity model.

## Reproduction and independent consistency check

- Configs: `configs/csi300_ridge.yaml`, `configs/csi300_lightgbm.yaml`,
  `configs/csi300_mlp.yaml` and `configs/csi300_market_context_seed17.yaml`.
- Multi-seed reproduction: rerun the market-context config with `--model-seed 7`, `29`, `41` and
  `53`; every override is embedded in `config.resolved.json` and the run manifest.
- Ablation reproduction: use `--ranking-weight 0` and `--disable-market-context`.
- Snapshot manifest: `manifests/csi300_2026-09-09.json`
- Ridge run: `csi300-pit-ridge-5f53b46fa6c9-20260928T085135Z`, config SHA-256
  `5f53b46fa6c9a0a32ec39e6be7c0ca05da9493c0b6e040521b044e76279d330a`.
- LightGBM run: `csi300-pit-lightgbm-925a4b7a8035-20260928T085342Z`, config SHA-256
  `925a4b7a80359b6c8084214233764f391996872c4d05ec0ab46e0bb16fbcd9a0`.
- MLP run: `csi300-pit-mlp-seed17-d348ccba82b9-20260928T090525Z`, config SHA-256
  `d348ccba82b9b4adb9212d5e6184c3fc5f24b979cfd2c479c3c05dd21bc50b7e`.
- Market-context seed-17 run: `csi300-pit-market-context-seed17-51b7f2c4fba5-20260928T094758Z`,
  config SHA-256 `51b7f2c4fba5c80b640868f7e158f331b4d10d9e4047b619c9b82a63d10deae5`.
- All 14 generated artifacts in every included run passed their recorded SHA-256 checks; the
  five-seed aggregator rechecks 70 files before writing its summary.
- All 363,409 Ridge/LightGBM predictions per model matched the independently generated Phase 2A.7 outputs:
  key coverage 100%, maximum score difference 0 and maximum label difference 0.

The source market data are intentionally absent from Git. The compact tables in `results/` are
derived outputs tied to the run above.

## Recovered archive follow-up

The original prediction and ledger artifacts were supplied privately after the initial handoff.
Their hashes and saved-return arithmetic have been checked. The new
[archived-run diagnostic](../../docs/LEGACY_REPLAY_FINDINGS.md) confirms future-outcome selection
changes and unresolved held returns. Raw source audit partitions are still unavailable;
this follow-up does not establish corrected historical performance.
