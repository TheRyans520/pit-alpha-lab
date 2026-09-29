# Architecture

## System boundary

PIT Alpha Lab owns experiment orchestration, validation, model adapters, signal evaluation,
portfolio simulation and artifact lineage. It does not own a market-data license, a live order
gateway or a broker connection.

```text
External provider / synthetic fixture
              |
              v
  Snapshot + checksum manifest
              |
              v
  Point-in-time panel builder ----> leakage and coverage audits
              |
              v
  Feature / label / split contract
              |
              v
  Model adapter -> prediction table -> signal diagnostics
                                      |
                                      v
                          portfolio and cost engine
                                      |
                                      v
                       metrics, attribution, report

Every stage writes immutable artifacts linked by one run manifest.
```

## Package boundaries

| Package | Owns | Must not own |
|---|---|---|
| `data` | Snapshot identity, point-in-time observations, coverage audits | Model preprocessing |
| `features` | Feature definitions and train-fitted transforms | Test-period fitting |
| `splits` | Walk-forward windows, embargo and date eligibility | Portfolio logic |
| `models` | Common fit/predict adapters | Holdings and costs |
| `signals` | IC, rank diagnostics, decay and selection scores | Execution accounting |
| `portfolio` | Target weights and portfolio constraints | Future returns |
| `backtest` | Timing, holdings, turnover, costs and P&L ledger | Model fitting |
| `metrics` | Prediction, return, risk and uncertainty summaries | Mutable experiment state |
| `artifacts` | Manifests, hashes and atomic persistence | Business logic |

## Canonical tables

The internal data contract will use narrow, explicit tables:

- Observation key: `(datetime, instrument)`.
- Feature table: key plus named numeric feature columns.
- Label table: key plus label and availability timestamps.
- Prediction table: key, model, fold, seed and score.
- Target table: decision time, instrument, target weight and selection reason.
- Ledger table: execution time, prior/target/realized weight, return, turnover, costs and P&L.

All persisted tables require schema version, snapshot ID, config digest and content checksum.

## Timing contract

The first case study observes information through close `t`, forms a score at close `t`, and
executes no earlier than open `t+1`. The five-day label is open `t+1` to open `t+6` and split
boundaries use an explicit embargo. These assumptions live in config and tests, never only in
prose.

## Reproducibility model

A run is identified by:

```text
config digest + data snapshot checksum + feature schema + code revision
```

Time, machine and Python metadata describe an execution but do not redefine the experiment.
Generated artifacts are written atomically and are never silently replaced by refreshed data.

The Qlib production adapter consumes audit Parquet exports through an explicit local environment
variable. Before reading a row it checks the release tag, snapshot ID, archive identity, every
partition checksum and the aggregate raw/feature digests. Absolute paths stored by an earlier
machine are intentionally ignored.

## Deliberate limitations

- Daily OHLCV cannot support claims about queue position, latency or high-frequency execution.
- Public community data do not provide institutional-grade delisting guarantees.
- Impact and capacity models are sensitivity analyses until calibrated with suitable data.
- A strong backtest is evidence about a frozen historical protocol, not a live profit claim.
