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

- **Runtime gate:** every plan that touches engine, engine_gpu, engine_rules, or any per-tick code path MUST include at least one task that runs the changed code path and asserts an observable post-condition (tick advance, event count, byte equality). Compile-clean is not runtime-clean. List the gate test(s) here:
  - `<test name>` at `<crates/.../tests/...rs>` — `<one-line invariant>`
  (Or "N/A — pure docs / pure types" if literally no code path runs.)

- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).
```

The Constitution check uses `PASS` / `FAIL` / `N/A`. `FAIL` requires a remediation step in the plan or an explicit waiver justification immediately below.

The "Runtime gate" requirement was added 2026-04-28 after the dispatch-emit plan landed T15+T16+Stream A as "complete" with critics PASS — only to reveal at Stream C prep that `step_batch` panicked on every call (hit `unimplemented!()` inside `engine::step::step`, deleted by Plan B1' Task 11). Critics check architectural compliance, not buildability or runtime. Compile-clean is necessary but insufficient. Every plan that touches a runtime path needs a test that actually runs it.

The "Re-evaluation" checkboxes are part of the plan execution: tick the post-design box only after the task list and design have stabilised, not at initial draft.
