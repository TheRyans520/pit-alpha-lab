# Data snapshot audit: null result retained

Before this repository was created, three public Qlib China-data releases dated 2026-06-30,
2026-07-29 and 2026-09-09 were independently downloaded, checksum-verified and compared over a
common 2008-2025 CSI300 window.

The matched 1,312,314 stock-date rows, historical membership, OHLCV/factor values, 157 frozen
features and five-day labels were identical. Independently fitted Ridge and LightGBM models
therefore produced identical predictions, selections and portfolio results across snapshots.
The proposed data-vintage paper gap was classified **not confirmed**.

This result is retained because it demonstrates an important project rule: falsified hypotheses
remain visible, and reusable validation infrastructure is separated from the original claim.
Curated machine-readable result tables will be added in Q1 without publishing the source data.
