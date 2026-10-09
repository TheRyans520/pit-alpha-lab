# Resume project entry — PIT Alpha Lab

**PIT Alpha Lab | Quantitative Research & Portfolio Simulation Platform**  
Python · NumPy/Pandas · scikit-learn · LightGBM · PyTorch · FastAPI · React · GitHub Actions

- Built a Python equity-research platform with five model adapters, purged walk-forward evaluation and reproducible experiment artifacts; validated 91 automated tests covering leakage, accounting and API contracts.
- Implemented fee-funded share/cash accounting with buy/sell restrictions, ADV participation and turnover limits; independently reconciled inventory, fees and NAV under synthetic execution scenarios.
- Audited 1.09 million archived model predictions across Ridge, LightGBM and MLP, identified future-outcome selection bias, and added explicit failures for unresolved held valuations.

## Role-specific selection

Use the three bullets above for a general QR/QD-oriented resume. Keep the title and claims consistent
with what you can explain. For engineering-heavy QD roles, replace the third bullet with:

- Benchmarked a deterministic portfolio engine on 378,000 synthetic stock-date records, recording a 0.385 s median runtime across three isolated processes with identical output hashes.

For research-heavy QR roles, keep the archive-audit bullet and explain the historical negative neural
results, validation embargo and observed-test-set limitation in the interview. For technical QT
roles, lead with the cash/fee/blocked-order bullet, but do not label this daily fixture an exchange
simulator or HFT system. Actual job descriptions determine whether C++, derivatives or order-book
experience needs a separate project.

## Evidence map

| Resume claim | Evidence |
|---|---|
| Five model adapters; purged evaluation | `src/pitalpha/models/registry.py`, `splits/validation.py`, `pipeline.py` |
| 91 local tests, API and neural dependencies installed | `bash scripts/verify.sh`; `tests/` |
| Fee-funded quantities, cash and order restrictions | `src/pitalpha/backtest/execution.py`, `portfolio/constraints.py` |
| Independent cash/inventory/NAV reconciliation | `src/pitalpha/backtest/execution_audit.py`, `tests/test_execution.py` |
| Deterministic synthetic scenario evidence | `case_studies/synthetic_execution/results/evidence.json` |
| 1,090,227 archived prediction rows; 216 summary rows | `case_studies/csi300_alpha/results/legacy_replay_audit.json` |
| Throughput benchmark | `docs/benchmarks/ledger_1260x300.json`; measured on the prior audited source revision, not a speedup claim |

## Five-minute interview walkthrough

1. Describe the problem: a good IC or a saved positive return does not prove a defensible portfolio.
2. Run `python -m pitalpha run configs/demo_synthetic.yaml` and show the manifest and leakage tests.
3. Run `python -m pitalpha execution-demo`; show that a blocked sell does not fund a buy and that
   purchase fees reduce affordable shares. Reconcile quantities/cash rather than only graphing NAV.
4. Show `docs/LEGACY_REPLAY_FINDINGS.md`: arithmetic matches the old published output, but strict
   replay stops on an unvalued holding. Explain why that prevents a corrected return claim.
5. Finish with the benchmark, one rejected model hypothesis and the next evidence gap: raw source
   data plus event/valuation coverage, then a new unobserved evaluation sample.

The case-study price path is artificial. Never put its positive NAV change on a resume as strategy
performance. Do not quote the old 8.49% as corrected performance. AI-assisted work is usable only to
the extent that the owner can explain and defend its design, limits and evidence.
