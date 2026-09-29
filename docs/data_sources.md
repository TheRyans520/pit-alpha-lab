# Data source and evidence policy

## Current source

The first case study uses the community-maintained China-market Qlib release dated 2026-09-09.
The source archive is not redistributed. A public manifest records its release URL, archive hash,
runtime provenance and the aggregate digests of independently extracted raw and feature
partitions.

This source is suitable for a reproducible public research benchmark. It is not classified as an
institutional-grade point-in-time security master or execution feed. Reproducibility, economic
completeness and live tradability are separate claims.

## Evidence levels

| Level | Meaning | Current status |
|---|---|---|
| L0 | Synthetic fixture for software tests | Complete |
| L1 | Public daily OHLCV with frozen snapshot and historical interval membership | Complete |
| L2 | Cross-source price, corporate-action, suspension, limit and delisting validation | Planned |
| L3 | Point-in-time fundamentals keyed by publication timestamp | Planned |
| L4 | Licensed institutional reference and execution data | Out of public-project scope |
| L5 | Tick/order-book data suitable for microstructure research | Optional QT extension |

Results must state their evidence level. A model result cannot upgrade the quality level of its
input data.

## Mandatory quality gates

Every production experiment checks:

- unique `(datetime, instrument)` keys and canonical ordering;
- configured row and daily-universe bounds;
- per-feature finite coverage and all-missing columns;
- label and realizable next-return coverage;
- maximum absolute next-day return;
- OHLC completeness and ordering when raw bars are available;
- positive observed prices and adjustment factors;
- nonnegative observed volume;
- snapshot, partition and aggregate SHA-256 identities.

An error-level failure stops model fitting. Thresholds live in the experiment configuration so a
researcher cannot silently weaken them in prose.

## Expansion priorities

1. Add a second independent daily-price source and discrepancy report.
2. Add explicit suspension, price-limit, ST and delisting status.
3. Add point-in-time fundamental statements using announcement timestamps rather than fiscal
   period alone.
4. Add industry, market-cap and benchmark-weight fields for risk attribution.
5. Keep news, alternative data and order books as separately licensed modules with their own
   availability timestamps.

- Qlib documentation: <https://github.com/microsoft/qlib/blob/main/docs/component/data.rst>
- Current public-data notice: <https://github.com/microsoft/qlib/blob/main/README.md>
