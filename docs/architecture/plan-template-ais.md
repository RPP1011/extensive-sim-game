# Architectural Impact Statement (plan preamble)

> Required preamble for every plan in `docs/superpowers/plans/`. Per P8.
>
> Copy this section into your new plan file, between the header (Goal/Architecture/Tech Stack) and the first task. Fill in every field. The constitution check is non-optional.

```markdown
## Architectural Impact Statement

- **Existing primitives searched:**
  - `<symbol or term>` at `<file:line>`
  - `<symbol or term>` at `<file:line>`
  Search method: `ast-grep` / `rg` / direct `Read`.

- **Decision:** extend `<existing primitive>` / new because `<reason>`.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `<files>`
  - Generated outputs re-emitted: `<files>`
  (Or "none" if this plan is non-engine work.)

- **Hand-written downstream code:** NONE.
  (Or, per file: `<path>: <justification>`. Each justification must explain why an emitter expansion is impossible or inappropriate.)

- **Constitution check:**
  - P1 (Compiler-First): PASS / FAIL — evidence: `<file:line>`
  - P2 (Schema-Hash on Layout): PASS / N/A — evidence: ...
  - P3 (Cross-Backend Parity): PASS / N/A
  - P4 (`EffectOp` Size Budget): PASS / N/A
  - P5 (Determinism via Keyed PCG): PASS / N/A
  - P6 (Events Are the Mutation Channel): PASS / N/A
  - P7 (Replayability Flagged): PASS / N/A
  - P8 (AIS Required): PASS — this section satisfies it
  - P9 (Tasks Close With Verified Commit): PASS / N/A
  - P10 (No Runtime Panic): PASS / N/A
  - P11 (Reduction Determinism): PASS / N/A

- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).
```

The Constitution check uses `PASS` / `FAIL` / `N/A`. `FAIL` requires a remediation step in the plan or an explicit waiver justification immediately below.

The "Re-evaluation" checkboxes are part of the plan execution: tick the post-design box only after the task list and design have stabilised, not at initial draft.
