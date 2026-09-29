# ADR 0005: Separate reproducibility from data evidence level

- Status: accepted
- Date: 2026-09-28

## Decision

Each data source receives an explicit evidence level from L0 synthetic to L5 microstructure.
Checksum verification and deterministic reproduction establish identity, not institutional
completeness. Model reports must carry both the snapshot identity and semantic quality result.

## Consequences

- Public Qlib daily data can support historical methodology and portfolio comparisons.
- It cannot support claims about live execution, point-in-time fundamentals or order-book alpha.
- Adding more model complexity does not raise the evidence level.
- A failed semantic quality gate stops fitting before predictions are produced.
