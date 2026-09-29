# ADR 0002: Publish lineage, not market-data payloads

- Status: Accepted
- Date: 2026-09-28

## Context

The source workspace contains multi-gigabyte archives, generated Parquet files and upstream
vendor checkouts. These are unsuitable for Git and may have redistribution restrictions.

## Decision

Commit only source code, configuration, manifests, checksums, retrieval instructions, synthetic
fixtures and small derived result tables. Keep raw and generated market data ignored locally.

## Consequences

- Clones stay small and legally safer.
- Full empirical reproduction requires users to obtain provider data separately.
- CI uses deterministic synthetic fixtures rather than private or downloaded market data.
