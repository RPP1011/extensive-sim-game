# Architecture Decision Records

Locked decisions that emerge from design specs in `docs/superpowers/specs/`. ADRs are immutable once `Accepted`; changing direction = a new ADR that supersedes the old one.

## Numbering

Four digits, monotonic, never reused. The next ADR number is `(highest existing + 1)`. Use:

```bash
ls docs/adr/[0-9]*.md 2>/dev/null | sort -V | tail -1
```

## Template

```markdown
# ADR-NNNN: <Decision Title>

**Status:** Accepted | Superseded by ADR-NNNN | Rejected
**Date:** YYYY-MM-DD
**Spec:** `docs/superpowers/specs/<spec>.md`

## Context

What problem prompted this decision (1 paragraph).

## Decision

The chosen approach (1 paragraph).

## Consequences

Downstream impact (positive and negative bullets).

## Alternatives considered

What was rejected and why (1 paragraph).
```

## When to write an ADR

Write an ADR when a decision:
- Locks an external interface (API, file format, schema).
- Commits to an architectural direction that future plans build on.
- Settles a contested design choice that would otherwise re-litigate.

Internal implementation choices stay in specs/plans, not ADRs.

## When NOT to write an ADR

- Tactical refactors that don't change interfaces.
- Per-task decisions inside a plan.
- Spec drafts that haven't been approved yet (those live in `docs/superpowers/specs/`).

## Index

(Hand-maintained; one line per ADR.)
