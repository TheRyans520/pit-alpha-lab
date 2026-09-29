# ADR 0001: Build a QR/QD-first flagship with bounded QT scope

- Status: Accepted
- Date: 2026-09-28

## Context

The project should support applications to quantitative research, quantitative development and
quantitative trading roles. Existing assets are strongest in point-in-time equity research,
machine learning evaluation and cost-aware portfolio simulation. They do not contain licensed
order-book data or a live execution stack.

## Decision

Build one vertical equity-research platform optimized for QR and QD evidence. Include timing,
cost, turnover and risk decisions relevant to QT, but defer genuine microstructure replay to an
optional, separately bounded extension.

## Consequences

- The first public release can be credible without pretending to be an HFT engine.
- Engineering and research share one data and artifact contract.
- QT interview preparation remains a parallel activity rather than an inflated repo feature.
