# ADR 0003: Package and CLI are authoritative

- Status: Accepted
- Date: 2026-09-28

## Decision

Reusable logic lives in `src/pitalpha` and production experiments run through versioned config
and CLI commands. Notebooks may explore or visualize artifacts but may not become the only
implementation of a published result.

## Consequences

- Tests can cover the same code used for reported results.
- Interviewers can inspect clear module boundaries instead of a monolithic notebook.
- Exploratory speed remains available without sacrificing reproducibility.
