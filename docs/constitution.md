# Project Constitution

> Architectural principles that govern this project. Each principle is a numbered, stable handle (P-N). Other docs cite by number; do not paraphrase. Edit the prose freely; the number is the contract.
>
> This file is auto-loaded into every agent's context via the `SessionStart` hook in `.claude/settings.json`.

## P1 — Compiler-First Engine Extension

**Statement.** All engine-rule behavior originates from the DSL compiler. Hand-written rule logic in `crates/engine/src/handlers/`, `crates/engine/src/cascade/handlers/`, or `crates/engine/src/generated/` (without `// @generated` header) is forbidden.

**Rationale.** Hand-written behavior drifts from the DSL surface, breaks emitter-pattern wins (alive-bitmap, fold-body lowering, etc.), and silently violates cross-backend parity. Compile-time enforcement keeps every rule subject to the emitter's lowering.

**Enforcement.**
- Compile-time: sealed `Rule` trait (Spec B); `build.rs` sentinel checking `// @generated` headers.
- CI: ast-grep rule rejecting `impl Rule` outside `rules-generated/`.
- Agent-write-time: `PreToolUse` hook on `Edit|Write` to engine handler paths.
- Critic: project-critic checks any task tagged `engine-extension`.
- Manual: code review.

**Source.** `docs/spec/compiler.md` §1 (kept for archaeology only — readers do not follow pointers).

---

## P2 — Schema-Hash Bumps on Layout Change

**Statement.** Any change to `SimState` SoA layout, event variant set, mask-predicate semantics, or scoring-row contract requires a `crates/engine/.schema_hash` regeneration.

**Rationale.** Schema-hash mismatches block snapshot loads and trace consumption; silent layout drift produces undefined behavior at the snapshot/migration boundary.

**Enforcement.**
- CI: `tests/schema_hash.rs` baseline-comparison test.
- Agent-write-time: `PreToolUse` hook on `Edit|Write` to `crates/engine/src/state/**` requires an open task tagged `schema-change`.
- Manual: regen via the documented procedure.

**Source.** `docs/spec/runtime.md` §22.

---

## P3 — Cross-Backend Parity

**Statement.** Every engine behavior runs on both `SerialBackend` (reference) and `GpuBackend` (performance), or is annotated `@cpu_only` in DSL with explicit justification.

**Rationale.** GPU is co-equal, not auxiliary. Behavior that exists on only one backend produces a non-determinism boundary at backend selection.

**Enforcement.**
- CI: `tests/parity_*.rs` cross-backend determinism test.
- Compile-time: DSL `@cpu_only` annotation is a parser-recognized exception.
- Critic: project-critic flags new behavior without parity test.

**Source.** `docs/spec/compiler.md` §1, `docs/spec/runtime.md` §24.

---

## P4 — `EffectOp` Size Budget (≤16 bytes)

**Statement.** Each variant of `EffectOp` (and analogous IR enums) stays at or under 16 bytes after Rust enum tagging.

**Rationale.** Small variants keep `SmallVec<[EffectOp; 4]>` cheap (≤64 B + tag) and compile to GPU-friendly POD.

**Enforcement.**
- Compile-time: `static_assert::const_assert!(size_of::<EffectOp>() <= 16);` (or equivalent build-time check).
- Manual: when adding a variant, verify with `mem::size_of::<EffectOp>()`.

**Source.** `docs/spec/ability.md` §22.

---

## P5 — Determinism via Keyed PCG

**Statement.** All sim randomness flows through `per_agent_u32(world_seed, agent_id, tick, purpose)`. `thread_rng()`, system time, and any non-keyed RNG are forbidden in the deterministic path.

**Rationale.** Replay equivalence + cross-backend parity require RNG inputs to be a pure function of (state, tick, agent, purpose).

**Enforcement.**
- CI: ast-grep rule rejecting `thread_rng()` / `SystemTime::now()` in `crates/engine/src/`.
- Manual: code review.

**Source.** `docs/spec/runtime.md` §7, `docs/spec/language.md` §9.

---

## P6 — Events Are the Mutation Channel

**Statement.** All sim state mutations are events. Current state is a fold over events + entity baseline. Direct field writes outside the documented kernel API are forbidden.

**Rationale.** Replayability requires events to be the only authoritative log; anything else creates state that exists outside the snapshot/trace.

**Enforcement.**
- Manual: code review on `&mut SimState` writes.
- CI (future): visibility rule that confines field writes to `step::*` and `snapshot::*`.

**Source.** `docs/spec/runtime.md` §5.

---

## P7 — Replayability Flagged at Declaration

**Statement.** Every `event` declaration carries an explicit `replayable: bool`. Non-replayable events (chronicle prose, telemetry side channels) are tagged at the source; the runtime fold ignores them.

**Rationale.** Mixing replayable + non-replayable events into a single fold breaks deterministic replay.

**Enforcement.**
- Compile-time: parser requires the annotation; emitter generates flagged enum variants.
- CI: trace-format roundtrip test.

**Source.** `docs/spec/runtime.md` §5, `docs/spec/language.md` §2.2.

---

## P8 — Architectural Impact Statement Required

**Statement.** Every new plan in `docs/superpowers/plans/` MUST include an Architectural Impact Statement (AIS) preamble before any task list. The AIS makes the constitution check explicit.

**Rationale.** Plan-execution discipline is downstream of plan-design discipline. Forcing the AIS surfaces invariant violations at design time, not at audit time.

**Enforcement.**
- Manual: human review on plan creation; the project-DAG skill (Spec C) rejects plans without AIS.
- Template: `docs/architecture/plan-template-ais.md`.

**Source.** This spec §3.3.

---

## P9 — Tasks Close With a Verified Commit

**Statement.** Tasks marked `completed` carry a non-empty `closes_commit` UDA pointing at a real, non-reverted commit SHA on the active branch.

**Rationale.** Prevents stale "done" claims (the failure mode the 2026-04-24 docs audit found).

**Enforcement.**
- Skill: project-DAG skill (Spec C) rejects close without `closes_commit`.
- `dag-validate`: re-checks SHA existence; flags reverts.

**Source.** This spec, derived from the 2026-04-24 audit.

---

## P10 — No Runtime Panic on Deterministic Path

**Statement.** The deterministic sim hot path (`step()`, kernels, fold dispatch) does not panic. Saturating ops, `Result`, and contract assertions are the failure mode; runtime panics escape only as bugs.

**Rationale.** Panics at scale (200k agents) corrupt replay state and abort training runs.

**Enforcement.**
- CI: `tests/proptest_baseline.rs` runs `step` on randomized inputs; never panics.
- Manual: code review on `unwrap()` in hot paths.

**Source.** `docs/spec/runtime.md` §14.
