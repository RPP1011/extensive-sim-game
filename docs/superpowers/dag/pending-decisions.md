# Pending Decisions

> Append-only log of human-gated decisions surfaced by the autonomous DAG run.
> Entries land here when the run encounters `plan-writer`, `spec-needed`, or
> `human-needed` work. The run continues with other eligible tasks (or
> terminates if none remain).
>
> User resolves each entry by editing the section to add an `**APPROVED:**`
> line (for plan-writer) or by starting an interactive brainstorm session
> (for spec-needed). Then re-run `dag-bootstrap.sh` to incorporate the
> resolution.

---

<!-- entries appended below by the agent -->
