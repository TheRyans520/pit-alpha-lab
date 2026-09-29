# Temporal tensor contract

> Status: implemented engineering contract; real-data availability remains gated.

`pitalpha.data.CausalSequenceStore` is the boundary between a point-in-time long panel and every
future temporal model. It is model independent and materializes only requested batches, avoiding a
full stock-by-date-by-feature cube for the 1.3-million-row research panel.

## Batch schema

For `N` decision rows, `L` lookback sessions and `F` declared features:

| Field | Shape | Meaning |
|---|---:|---|
| `values` | `[N, L, F]` float32 | Finite feature values; unavailable cells use an explicit fill value |
| `observed_mask` | `[N, L, F]` bool | Whether each feature cell was actually finite and observed |
| `row_mask` | `[N, L]` bool | Whether the instrument had a source row on that market session |
| `session_dates` | `[N, L]` datetime64 | Exact calendar sessions; pre-calendar padding is `NaT` |
| `decision_dates` | `[N]` datetime64 | Decision session for each sample |
| `instruments` | `[N]` strings | Prediction-aligned security identifiers |
| `source_index` | `[N]` | Original decision-frame indices for lossless score alignment |

The row mask and feature mask are intentionally separate. A suspension, listing boundary or absent
source row is not the same event as one unavailable feature in an otherwise present observation.

## Causality invariants

- every sequence ends exactly at its decision date;
- earlier positions follow the declared market-session calendar, never row offsets;
- dates before the start of the calendar are left padded;
- missing instrument sessions remain masked instead of being forward filled;
- `label`, `realized_return_1d` and `forward_return_*` columns are rejected as features;
- unknown instruments, unknown decision dates, duplicate panel keys and missing decision rows fail
  closed;
- output arrays are read-only so a model cannot mutate the shared batch accidentally.

The store ignores all undeclared columns. This blocks known outcome fields, but it cannot prove that
an arbitrarily named vendor field was available at the decision time. The real-data adapter must
still supply field-level availability metadata and pass the protocol's data gate.

## Test obligations

`tests/test_temporal.py` verifies exact window endpoints, session-order padding, missing-row versus
missing-cell semantics, prediction alignment, rejection of known outcome fields and invariance to
arbitrary perturbations strictly after the decision date.

The next model step may depend on this contract, but may not reconstruct sequences privately from
raw frames or replace masked cells without recording the training-only preprocessing policy.
