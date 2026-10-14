# Local engineering measurements

These are synthetic software measurements on the handoff machine, not market returns.

- `ledger_1260x300.json`: seed-17 fixture, 1,260 dates × 300 instruments = 378,000 input rows;
  weekly Top-30, four cost overlays, three fresh subprocesses and identical output hashes.
  Elapsed time measures the ledger including daily position construction. It excludes imports,
  fixture generation and output serialization. Peak RSS includes the whole child process,
  imports, fixture and ledger, and is not incremental allocation or Python-only memory.
- `synthetic_reconciliation.json`: independently recomputed held outcomes, prior-weight drift,
  turnover, cost overlays, CAGR, Sharpe, drawdown and daily IC/Rank IC; all 14 artifact hashes checked.
- `synthetic_repeatability.json`: two runs of the same config/source, matching 13 deterministic
  artifact hashes. The report embeds a run timestamp and is intentionally excluded.

Run `scripts/benchmark_ledger.py` to reproduce. Do not infer a speedup without a comparable baseline.
No latency SLA or HFT capability is implied. Source digests include Python code, scripts, configs,
requirements and frozen manifests. Measurements made before commit record `dirty: true`; the
source digest identifies the exact measured implementation independently of the parent commit.
