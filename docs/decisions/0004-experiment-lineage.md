# ADR 0004: Content identity is separate from execution identity

- Status: Accepted
- Date: 2026-09-28

## Context

Timestamps alone cannot identify whether two experiments used the same data or protocol.

## Decision

Record a stable configuration digest, snapshot/checksum identity, feature schema and code
revision for every run. Record time, host and Python metadata separately as execution context.
Write manifests atomically.

## Consequences

- Equivalent experiments can be recognized across machines and dates.
- Refreshed data cannot silently masquerade as an old run.
- Later artifact caches can use content identities without relying on directory names.
