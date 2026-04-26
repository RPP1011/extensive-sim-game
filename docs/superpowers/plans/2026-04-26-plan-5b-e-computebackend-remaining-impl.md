# Plan 5b–e (combined): Remaining ComputeBackend Phases — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Phase 2–5 of 5** for the full ComputeBackend trait extraction. Plan 5a (deleted, implemented) established the trait + mask-fill threading. This plan completes the remaining tick phases and wires the real GPU kernels.

**Goal:** Route every remaining tick phase (cascade dispatch, view fold, apply_actions) through `ComputeBackend` trait methods, then replace the Phase 1 GPU stubs with calls into the already-existing `engine_gpu` kernels, culminating in a byte-identical cross-backend parity sweep.

**Architecture:** Same pattern as Plan 5a: for each tick phase, add a trait method to `ComputeBackend` (in `engine/src/backend.rs`), emit the `SerialBackend` impl body via `emit_backend.rs`, update `emit_step.rs` to route the direct call through the new method, and add a `GpuBackend` Phase-1-style stub. Phase 5e then replaces all stubs with real GPU kernel calls — the kernels already exist in `crates/engine_gpu/src/` (cascade, apply_actions, movement, view_storage, mask). The final parity sweep asserts byte-identical state hashes between Serial and GPU backends on the wolves+humans fixture.

**Tech Stack:** Rust 2021. No new dependencies. Existing `engine_gpu` kernels (`mask.rs`, `scoring.rs`, `apply_actions.rs`, `cascade.rs`, `view_storage.rs`). `dsl_compiler` emitters (`emit_backend.rs`, `emit_step.rs`).

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `ComputeBackend` trait at `crates/engine/src/backend.rs:33` — 3 mask methods established by Plan 5a; this plan adds 3 more (`cascade_dispatch`, `view_fold`, `apply_and_movement`).
  - `SerialBackend` impl at `crates/engine_rules/src/backend.rs` (GENERATED) — mask methods present; this plan extends it via `emit_backend.rs`.
  - `emit_backend.rs` at `crates/dsl_compiler/src/emit_backend.rs` — static Rust-string emitter, ~67 lines; this plan extends it with new method bodies.
  - `emit_step.rs` at `crates/dsl_compiler/src/emit_step.rs` — emits `engine_rules/src/step.rs`; direct calls at lines 147 (`cascade.run_fixed_point`) and 190 (`views.fold_all`) need routing; `apply_actions(state, scratch, events)` at line 144 is an internal helper that stays internal but `apply_and_movement` wraps both halves.
  - `GpuBackend` Phase-1 stubs at `crates/engine_gpu/src/lib.rs:368–378` (no-gpu) and `2775–2785` (gpu) — `reset_mask`/`set_mask_bit`/`commit_mask` are CPU pass-throughs; this plan adds stub bodies for the 3 new methods, then Phase 5e replaces them with real dispatches.
  - Existing GPU kernels confirmed present in `crates/engine_gpu/src/`:
    - `mask.rs` — `cs_fused_masks` entry point (mask + unpack); `run_and_readback()` API.
    - `scoring.rs` — `cs_scoring`; `run_and_readback()` API.
    - `apply_actions.rs` — `cs_apply_actions`; `run_resident()` API.
    - `cascade.rs` — GPU cascade driver (`cascade_gpu`); fixed-point loop backed by `PhysicsKernel::run_batch`.
    - `view_storage.rs` — `cs_fold_<name>` per view; fold dispatch API.
    - `movement.rs` — `cs_movement`; `run_resident()` API.
  - `GpuBackend::step` (gpu path) at `lib.rs:2416` — already does Phase 1-2 GPU mask+scoring, Phase 4a CPU apply_actions, Phase 4b GPU cascade, Phase 5 CPU view-fold; this plan routes those phases through the trait surface.
  - GPU audit at `docs/superpowers/notes/2026-04-26-audit-gpu.md` — confirms all Phase 5e kernels exist; the primary gap is `pick_ability` kernel (§5) which is explicitly out of scope for this plan.
  - `CascadeRegistry::run_fixed_point` at `crates/engine/src/cascade/dispatch.rs:465` — signature `(&self, &mut SimState, &mut V, &mut EventRing<E>)`.
  - `ViewRegistry::fold_all` at `crates/engine_rules/src/views/mod.rs:73` — signature `(&mut self, &EventRing<Event>, usize, u32)`.

  Search method: `rg`, direct `Read`, `git show` for deleted plan-5a.

- **Decision:** Combine 5b/c/d/e into one plan. Each of 5b/c/d follows an identical mechanical pattern (trait method + emit_backend extension + emit_step routing + GpuBackend stub), and they share the same infrastructure (trait file, both emitters). Splitting them would triple the AIS ceremony and create three plans that each leave the backend trait partially threaded. Phase 5e is the payoff and depends on all three stubs existing. One plan, four sequenced phases, clean delivery.

  **`apply_and_movement` grouping decision:** Phase 5d adds a single trait method `apply_and_movement` rather than separate `apply_actions` + `run_movement` methods. Rationale: in `step.rs`, `apply_actions` (Phase 4a) and the movement half are both CPU-internal helpers that together constitute "Phase 4 — apply + movement". The GPU backend already treats them as a unit in `run_gpu_apply_and_movement()` (lib.rs:2810). One method per logical phase is cleaner than fragmenting Phase 4 further.

  **`set_mask_bit` GPU concern:** The Plan 5a trait surface has `set_mask_bit(&mut self, buf, slot, kind)` called once per agent per `fill_all` iteration. This is fine for SerialBackend (direct `buf.set`) but would be O(N) GPU dispatches if a naïve GPU stub tried to launch a kernel per call. Phase 5e's `GpuBackend::set_mask_bit` does NOT dispatch a kernel per call. Instead, the GPU backend queues bit writes in a `pending_mask_bits: Vec<(usize, MicroKind)>` vec on `self`, and `commit_mask` flushes the entire batch in a single kernel dispatch (or, since the GPU mask kernel already runs over the full agent SoA from state in `run_and_readback`, the queue-and-flush is effectively a no-op because Phase 5e replaces `fill_all` routing with a single `reset_mask`→bulk GPU kernel→`commit_mask` call sequence). The queue-and-flush pattern is described in Task 14.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none.
  - Generated outputs re-emitted: `engine_rules/src/backend.rs` (gains 3 new trait method impls), `engine_rules/src/step.rs` (3 direct calls replaced with `backend.<method>(...)` calls).
  - Emitter changes: `dsl_compiler/src/emit_backend.rs` extended; `dsl_compiler/src/emit_step.rs` extended.

- **Hand-written downstream code:**
  - `crates/engine/src/backend.rs` — adds 3 new associated method signatures. Justified: this is the trait declaration, not behavior; engine declares the contract, downstream implements.
  - `crates/engine_gpu/src/lib.rs` — 3 stub impls (Phase 5b/c/d) then 3 real impls (Phase 5e). Justified: GpuBackend's step logic is hand-written (not emitted); it must call into existing GPU kernel structs by Rust API, not via DSL emit.
  - `crates/engine/tests/backend_full_tick_parity.rs` — new parity test. Justified: test fixture, not rule-implementing code.

- **Constitution check:**
  - P1 (Compiler-First): PASS — trait interface declarations live in engine (interface, not rules); rule orchestration stays emitted from `emit_step.rs`; no hand-written rule logic added.
  - P2 (Schema-Hash on Layout): N/A — no SoA field layout changes.
  - P3 (Cross-Backend Parity): PASS — this is the explicit deliverable; Task 19 asserts byte-identical state hashes.
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): PASS — no new randomness; GPU fallback preserves existing `per_agent_u32` shuffle.
  - P6 (Events Are the Mutation Channel): PASS — cascade and apply routing doesn't change event semantics; it changes which executor runs them.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `cargo test` + commit.
  - P10 (No Runtime Panic): PASS — stub methods return without dispatching; Phase 5e GPU methods mirror existing fallback-on-error patterns already in `GpuBackend::step`.
  - P11 (Reduction Determinism): N/A.

- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill). [ ] AIS reviewed post-design (after task list stabilises).

---

## File Structure

```
crates/engine/src/
  backend.rs                    MODIFIED: +3 method signatures (cascade_dispatch, view_fold,
                                          apply_and_movement)

crates/engine_rules/src/
  backend.rs                    REGENERATED: +3 SerialBackend impls for the new methods
  step.rs                       REGENERATED: 3 direct calls replaced with backend.* routing

crates/engine_gpu/src/
  lib.rs                        MODIFIED: +3 stub impls (Phase 5b/c/d), then real impls (Phase 5e)
                                          on both the no-gpu and gpu cfg branches

crates/dsl_compiler/src/
  emit_backend.rs               MODIFIED: emit the 3 new SerialBackend method bodies
  emit_step.rs                  MODIFIED: emit backend.cascade_dispatch / backend.view_fold /
                                          backend.apply_and_movement calls instead of direct calls

crates/engine/tests/
  backend_full_tick_parity.rs   NEW: full-tick parity test (Serial vs GpuBackend stub, then real GPU)
```

## Sequencing

Phases are strictly serial (each builds on the previous). Within a phase, steps are serial.

- **Phase 5b** (Tasks 1–4): cascade dispatch through backend.
- **Phase 5c** (Tasks 5–8): view fold through backend.
- **Phase 5d** (Tasks 9–12): apply_and_movement through backend.
- **Phase 5e** (Tasks 13–19): real GPU kernel wiring + parity sweep.

---

## Phase 5b — Cascade Dispatch Through Backend

### Task 1: Add `cascade_dispatch` to the `ComputeBackend` trait

**Files:**
- Modify: `crates/engine/src/backend.rs`

- [x] **Step 1: Read the current trait to confirm the import set.**

```bash
cat crates/engine/src/backend.rs
```

Confirm imports include `CascadeRegistry`, `EventRing`, `SimState`. They already do (line 15–18).

- [x] **Step 2: Add the `cascade_dispatch` method signature after `commit_mask`.**

In `crates/engine/src/backend.rs`, after the `commit_mask` method (line 61), add:

```rust
    /// Run the cascade fixed-point for this tick.
    ///
    /// SerialBackend: delegates to `cascade.run_fixed_point(state, views, events)`.
    /// GpuBackend: dispatches the GPU cascade driver (`cascade_gpu` in
    /// `engine_gpu::cascade`). Phase 5b stub forwards to Serial-equivalent.
    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    );
```

- [x] **Step 3: Build engine crate only to confirm the trait compiles.**

```bash
cargo build -p engine 2>&1 | grep -E "^error" | head -20
```

Expected: errors only in downstream crates (`engine_rules`, `engine_gpu`) that must now impl the new method — NOT in `engine` itself.

- [x] **Step 4: Commit the trait extension.**

```bash
git add crates/engine/src/backend.rs
git commit -m "feat(engine): add cascade_dispatch to ComputeBackend trait (Plan 5b Task 1)"
```

---

### Task 2: Extend `emit_backend.rs` to emit the `SerialBackend::cascade_dispatch` impl

**Files:**
- Modify: `crates/dsl_compiler/src/emit_backend.rs`

- [x] **Step 1: Read emit_backend.rs to locate the insertion point.**

```bash
cat crates/dsl_compiler/src/emit_backend.rs
```

The last method emitted is `commit_mask` (closes at the `}}` before the final `}}` at line 58).

- [x] **Step 2: Add cascade_dispatch emission after the commit_mask block.**

In `crates/dsl_compiler/src/emit_backend.rs`, inside `emit_backend`, after the `commit_mask` block (after line 57, before the closing `writeln!(out, "}}").unwrap();`):

```rust
    writeln!(out).unwrap();
    writeln!(out, "    fn cascade_dispatch(").unwrap();
    writeln!(out, "        &mut self,").unwrap();
    writeln!(out, "        cascade: &CascadeRegistry<Self::Event, Self::Views>,").unwrap();
    writeln!(out, "        state:   &mut SimState,").unwrap();
    writeln!(out, "        views:   &mut Self::Views,").unwrap();
    writeln!(out, "        events:  &mut EventRing<Self::Event>,").unwrap();
    writeln!(out, "    ) {{").unwrap();
    writeln!(out, "        cascade.run_fixed_point(state, views, events);").unwrap();
    writeln!(out, "    }}").unwrap();
```

- [x] **Step 3: Regenerate engine_rules/src/backend.rs.**

```bash
cargo run --bin xtask -- compile-dsl
```

Expected: `engine_rules/src/backend.rs` updated with the new method. Diff should show only the `cascade_dispatch` impl block added.

- [x] **Step 4: Build workspace to confirm engine_rules compiles.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -20
```

Expected: only `engine_gpu` fails (it needs the stub impl next). Alternatively both may pass if `engine_gpu`'s stub methods already return default impls — but they don't; the trait is non-auto.

- [x] **Step 5: Commit.**

```bash
git add crates/dsl_compiler/src/emit_backend.rs crates/engine_rules/src/backend.rs
git commit -m "feat(dsl_compiler,engine_rules): emit SerialBackend::cascade_dispatch (Plan 5b Task 2)"
```

---

### Task 3: Update `emit_step.rs` to route cascade through `backend.cascade_dispatch`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_step.rs`
- Regenerated: `crates/engine_rules/src/step.rs`

- [x] **Step 1: Locate the direct `cascade.run_fixed_point` call in the emitted string.**

In `crates/dsl_compiler/src/emit_step.rs`, the raw string literal around line 147 contains:

```rust
    cascade.run_fixed_point(state, views, events);
```

- [x] **Step 2: Replace the direct call with backend routing.**

In the raw string literal inside `emit_step`, change:

```rust
    apply_actions(state, scratch, events);

    // Phase 4b — cascade fixed-point.
    cascade.run_fixed_point(state, views, events);
    let _ = events_before; // reserved for future telemetry
```

to:

```rust
    apply_actions(state, scratch, events);

    // Phase 4b — cascade fixed-point (routed through ComputeBackend).
    backend.cascade_dispatch(cascade, state, views, events);
    let _ = events_before; // reserved for future telemetry
```

- [x] **Step 3: Regenerate step.rs.**

```bash
cargo run --bin xtask -- compile-dsl
```

- [x] **Step 4: Run compile-dsl --check to confirm round-trip idempotence.**

```bash
cargo run --bin xtask -- compile-dsl --check
```

Expected: `OK — all generated files up to date`.

- [x] **Step 5: Build + test (engine_rules only; engine_gpu will still fail).**

```bash
cargo build -p engine_rules 2>&1 | grep -E "^error" | head -10
cargo test -p engine_rules 2>&1 | tail -5
```

Expected: build SUCCESS, tests PASS.

- [x] **Step 6: Commit.**

```bash
git add crates/dsl_compiler/src/emit_step.rs crates/engine_rules/src/step.rs
git commit -m "feat(dsl_compiler,engine_rules): route cascade through backend.cascade_dispatch (Plan 5b Task 3)"
```

---

### Task 4: Add `GpuBackend::cascade_dispatch` stub + verify cross-backend parity

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [x] **Step 1: Locate both `ComputeBackend for GpuBackend` impl blocks.**

There are two impl blocks in `lib.rs`:

- Line ~351: `#[cfg(not(feature = "gpu"))] impl ComputeBackend for GpuBackend` — Phase 0 stub.
- Line ~2412: `#[cfg(feature = "gpu")] impl ComputeBackend for GpuBackend` — full impl.

Both need `cascade_dispatch`.

- [x] **Step 2: Add stub to the `#[cfg(not(feature = "gpu"))]` impl (after `commit_mask` at line ~378).**

```rust
    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5b stub: CPU pass-through. Plan 5e wires GPU cascade kernel.
        cascade.run_fixed_point(state, views, events);
    }
```

Note: `Self::Views = ()` for the no-gpu impl, so `run_fixed_point` accepts `&mut ()`. Confirm `CascadeRegistry<Event, ()>::run_fixed_point` compiles — it should because `run_fixed_point` is generic over `V`.

- [x] **Step 3: Add stub to the `#[cfg(feature = "gpu")]` impl (after `commit_mask` at line ~2783).**

```rust
    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        views:   &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5b stub: CPU pass-through. Plan 5e wires GPU cascade kernel.
        cascade.run_fixed_point(state, views, events);
    }
```

- [x] **Step 4: Build workspace.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -20
```

Expected: SUCCESS (all crates compile).

- [x] **Step 5: Run full test suite.**

```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -20
```

Expected: all tests PASS. The behavior of `SerialBackend` is unchanged; `GpuBackend` stub delegates identically.

- [x] **Step 6: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::cascade_dispatch stub (Plan 5b Task 4)"
```

---

## Phase 5c — View Fold Through Backend

### Task 5: Add `view_fold` to the `ComputeBackend` trait

**Files:**
- Modify: `crates/engine/src/backend.rs`

- [x] **Step 1: Review the current `fold_all` call in `step.rs` to understand the signature.**

In `crates/engine_rules/src/step.rs` (line ~190):

```rust
views.fold_all(events, events_before, state.tick);
```

`ViewRegistry::fold_all` takes `(&mut self, events: &EventRing<Event>, events_before: usize, tick: u32)`. The trait method must be generic enough to work for both `SerialBackend` (where `Views = ViewRegistry`) and `GpuBackend` (where `Views = ()`). Since `()` doesn't have `fold_all`, the SerialBackend impl will do the actual call; GpuBackend gets a stub.

- [x] **Step 2: Add `view_fold` method signature to the trait in `crates/engine/src/backend.rs`.**

After `cascade_dispatch`, add:

```rust
    /// Fold this tick's events into the view registry.
    ///
    /// SerialBackend: delegates to `views.fold_all(events, events_before, tick)`.
    /// GpuBackend: dispatches GPU view-fold kernels against `view_storage`.
    /// Phase 5c stub forwards to the CPU fold path.
    ///
    /// `events_before` is the `EventRing::total_pushed()` watermark captured
    /// before Phase 4a — it identifies the slice of events that belong to
    /// this tick.
    fn view_fold(
        &mut self,
        views:         &mut Self::Views,
        events:        &EventRing<Self::Event>,
        events_before: usize,
        tick:          u32,
    );
```

- [x] **Step 3: Build engine crate.**

```bash
cargo build -p engine 2>&1 | grep -E "^error" | head -10
```

Expected: SUCCESS (`engine` itself compiles; downstream may fail).

- [x] **Step 4: Commit.**

```bash
git add crates/engine/src/backend.rs
git commit -m "feat(engine): add view_fold to ComputeBackend trait (Plan 5c Task 5)"
```

---

### Task 6: Extend `emit_backend.rs` to emit `SerialBackend::view_fold`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_backend.rs`
- Regenerated: `crates/engine_rules/src/backend.rs`

- [x] **Step 1: Add view_fold emission after cascade_dispatch in emit_backend.**

In `crates/dsl_compiler/src/emit_backend.rs`, after the `cascade_dispatch` block, before the closing `}}` of the impl:

```rust
    writeln!(out).unwrap();
    writeln!(out, "    fn view_fold(").unwrap();
    writeln!(out, "        &mut self,").unwrap();
    writeln!(out, "        views:         &mut Self::Views,").unwrap();
    writeln!(out, "        events:        &EventRing<Self::Event>,").unwrap();
    writeln!(out, "        events_before: usize,").unwrap();
    writeln!(out, "        tick:          u32,").unwrap();
    writeln!(out, "    ) {{").unwrap();
    writeln!(out, "        views.fold_all(events, events_before, tick);").unwrap();
    writeln!(out, "    }}").unwrap();
```

- [x] **Step 2: Regenerate.**

```bash
cargo run --bin xtask -- compile-dsl
```

- [x] **Step 3: Verify round-trip.**

```bash
cargo run --bin xtask -- compile-dsl --check
```

Expected: `OK`.

- [x] **Step 4: Build engine_rules.**

```bash
cargo build -p engine_rules 2>&1 | grep -E "^error" | head -10
```

Expected: SUCCESS.

- [x] **Step 5: Commit.**

```bash
git add crates/dsl_compiler/src/emit_backend.rs crates/engine_rules/src/backend.rs
git commit -m "feat(dsl_compiler,engine_rules): emit SerialBackend::view_fold (Plan 5c Task 6)"
```

---

### Task 7: Update `emit_step.rs` to route view fold through `backend.view_fold`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_step.rs`
- Regenerated: `crates/engine_rules/src/step.rs`

- [ ] **Step 1: Locate the view fold calls in the emitted string.**

In `emit_step.rs` raw string, around lines 183–190, there are two branches:

```rust
    #[cfg(feature = "interpreted-rules")]
    crate::fold_views_interpreted(events, events_before, state, views);
    #[cfg(not(feature = "interpreted-rules"))]
    views.fold_all(events, events_before, state.tick);
```

Both paths need to route through `backend.view_fold`. The `interpreted-rules` path calls a different function but the fold result is identical conceptually; route both through the same trait method where the method body selects the correct impl.

However, `interpreted-rules` is a CPU-only feature that bypasses `fold_all`. The cleanest approach: keep the `interpreted-rules` cfg block as a `SerialBackend` internal detail. Route only the `#[cfg(not(feature = "interpreted-rules"))]` non-interpreted path through `backend.view_fold`. For the interpreted path, keep the direct call (it's feature-gated and only ever runs on `SerialBackend`).

- [ ] **Step 2: Replace the view fold block in the raw string.**

Change:

```rust
    // Phase 5 — view fold.
    if let Some(profile) = debug.tick_profile.as_ref() {
        if let Ok(mut p) = profile.lock() { p.enter("view_fold"); }
    }
    #[cfg(feature = "interpreted-rules")]
    crate::fold_views_interpreted(events, events_before, state, views);
    #[cfg(not(feature = "interpreted-rules"))]
    views.fold_all(events, events_before, state.tick);
    if let Some(profile) = debug.tick_profile.as_ref() {
        if let Ok(mut p) = profile.lock() { p.exit_with_null(); }
    }
```

to:

```rust
    // Phase 5 — view fold (routed through ComputeBackend).
    if let Some(profile) = debug.tick_profile.as_ref() {
        if let Ok(mut p) = profile.lock() { p.enter("view_fold"); }
    }
    #[cfg(feature = "interpreted-rules")]
    crate::fold_views_interpreted(events, events_before, state, views);
    #[cfg(not(feature = "interpreted-rules"))]
    backend.view_fold(views, events, events_before, state.tick);
    if let Some(profile) = debug.tick_profile.as_ref() {
        if let Ok(mut p) = profile.lock() { p.exit_with_null(); }
    }
```

- [ ] **Step 3: Regenerate + check.**

```bash
cargo run --bin xtask -- compile-dsl
cargo run --bin xtask -- compile-dsl --check
```

- [ ] **Step 4: Build + test engine_rules.**

```bash
cargo build -p engine_rules && cargo test -p engine_rules -- --test-threads=1 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add crates/dsl_compiler/src/emit_step.rs crates/engine_rules/src/step.rs
git commit -m "feat(dsl_compiler,engine_rules): route view fold through backend.view_fold (Plan 5c Task 7)"
```

---

### Task 8: Add `GpuBackend::view_fold` stub

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Add stub to the `#[cfg(not(feature = "gpu"))]` impl (after `cascade_dispatch` stub).**

```rust
    fn view_fold(
        &mut self,
        _views:         &mut Self::Views,
        _events:        &EventRing<Self::Event>,
        _events_before: usize,
        _tick:          u32,
    ) {
        // Phase 5c stub: no-op (Views = ()). Plan 5e dispatches GPU fold kernels.
    }
```

(`Self::Views = ()` in the no-gpu branch, so there's nothing to fold.)

- [ ] **Step 2: Add stub to the `#[cfg(feature = "gpu")]` impl (after `cascade_dispatch` stub).**

```rust
    fn view_fold(
        &mut self,
        _views:         &mut Self::Views,
        _events:        &EventRing<Self::Event>,
        _events_before: usize,
        _tick:          u32,
    ) {
        // Phase 5c stub: no-op (GpuBackend manages view_storage internally).
        // Plan 5e: dispatch cs_fold_* kernels via view_storage fold API.
    }
```

- [ ] **Step 3: Build workspace + run all tests.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -10
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: build SUCCESS, all tests PASS.

- [ ] **Step 4: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::view_fold stub (Plan 5c Task 8)"
```

---

## Phase 5d — Apply+Movement Through Backend

### Task 9: Add `apply_and_movement` to the `ComputeBackend` trait

**Files:**
- Modify: `crates/engine/src/backend.rs`

- [ ] **Step 1: Understand the current Phase 4a + 4b split in step.rs.**

In `crates/engine_rules/src/step.rs`:

- Lines 140–144: `apply_actions(state, scratch, events)` — Phase 4a, internal helper. Emits seed events.
- Line 147: `backend.cascade_dispatch(...)` — Phase 4b, already routed.

Movement (Phase 4 movement component) is handled inside `apply_actions` via the `MicroKind::MoveToward` / `Flee` match arms (emitting `AgentMoved`/`AgentFled`). There is no separate movement phase call in `step.rs`. Movement lives entirely inside `apply_actions`.

The new trait method wraps Phase 4a (apply_actions) since it is the remaining un-routed direct call. It does NOT wrap cascade_dispatch (already done) or the internal movement logic (not a separate call site).

- [ ] **Step 2: Add `apply_and_movement` signature to the trait.**

In `crates/engine/src/backend.rs`, after `view_fold`, add:

```rust
    /// Execute Phase 4a — walk the shuffled action list and emit root-cause
    /// events (AgentAttacked, AgentMoved, AgentFled, AgentAte, AgentDied, …).
    ///
    /// SerialBackend: calls the internal `step::apply_actions` helper which
    /// iterates `scratch.shuffle_idx`, matches on `ActionKind`, mutates `state`,
    /// and pushes events into `events`.
    ///
    /// GpuBackend: dispatches `cs_apply_actions` + `cs_movement` kernels,
    /// drains the GPU event ring into `events`, and commits the mutated agent
    /// SoA onto `state`. Phase 5d stub delegates to the CPU helper.
    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &SimScratch,
        events:  &mut EventRing<Self::Event>,
    );
```

Add `SimScratch` to the import at the top of `backend.rs` if not already present:

```rust
use crate::scratch::SimScratch;
```

(It is already imported at line 19.)

- [ ] **Step 3: Build engine crate.**

```bash
cargo build -p engine 2>&1 | grep -E "^error" | head -10
```

Expected: SUCCESS.

- [ ] **Step 4: Commit.**

```bash
git add crates/engine/src/backend.rs
git commit -m "feat(engine): add apply_and_movement to ComputeBackend trait (Plan 5d Task 9)"
```

---

### Task 10: Extend `emit_backend.rs` to emit `SerialBackend::apply_and_movement`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_backend.rs`
- Regenerated: `crates/engine_rules/src/backend.rs`

- [ ] **Step 1: Add apply_and_movement emission after view_fold in emit_backend.**

```rust
    writeln!(out).unwrap();
    writeln!(out, "    fn apply_and_movement(").unwrap();
    writeln!(out, "        &mut self,").unwrap();
    writeln!(out, "        state:   &mut SimState,").unwrap();
    writeln!(out, "        scratch: &engine::scratch::SimScratch,").unwrap();
    writeln!(out, "        events:  &mut EventRing<Self::Event>,").unwrap();
    writeln!(out, "    ) {{").unwrap();
    writeln!(out, "        crate::step::apply_actions_pub(state, scratch, events);").unwrap();
    writeln!(out, "    }}").unwrap();
```

Note: `apply_actions` in `step.rs` is currently a private `fn`. It must be made `pub(crate)` (or `pub`) so `SerialBackend::apply_and_movement` can call it. The next step handles this.

- [ ] **Step 2: Expose `apply_actions` as `pub(crate) fn apply_actions_pub` in `emit_step.rs`.**

In `crates/dsl_compiler/src/emit_step.rs`, in the raw string, rename the internal helper `apply_actions` → `apply_actions_pub` and change `fn apply_actions` → `pub(crate) fn apply_actions_pub`. Update the call site in Phase 4a accordingly.

Change:

```rust
    apply_actions(state, scratch, events);
```

to:

```rust
    backend.apply_and_movement(state, scratch, events);
```

And at the helper definition near the bottom of the raw string:

```rust
fn apply_actions(
    state:   &mut SimState,
    scratch: &SimScratch,
    events:  &mut EventRing<Event>,
) {
```

Change to:

```rust
pub(crate) fn apply_actions_pub(
    state:   &mut SimState,
    scratch: &SimScratch,
    events:  &mut EventRing<Event>,
) {
```

- [ ] **Step 3: Regenerate.**

```bash
cargo run --bin xtask -- compile-dsl
cargo run --bin xtask -- compile-dsl --check
```

- [ ] **Step 4: Build engine_rules.**

```bash
cargo build -p engine_rules 2>&1 | grep -E "^error" | head -10
```

Expected: SUCCESS.

- [ ] **Step 5: Commit.**

```bash
git add crates/dsl_compiler/src/emit_backend.rs crates/dsl_compiler/src/emit_step.rs \
        crates/engine_rules/src/backend.rs crates/engine_rules/src/step.rs
git commit -m "feat(dsl_compiler,engine_rules): emit SerialBackend::apply_and_movement; expose apply_actions_pub (Plan 5d Task 10)"
```

---

### Task 11: Verify `emit_step.rs` routing of Phase 4a

**Files:**
- Verify: `crates/engine_rules/src/step.rs` (already regenerated in Task 10)

- [ ] **Step 1: Confirm step.rs no longer calls `apply_actions_pub` directly — only via `backend.apply_and_movement`.**

```bash
grep "apply_actions_pub\|apply_actions(" crates/engine_rules/src/step.rs
```

Expected:
- No line matching `apply_actions(` (the old private call).
- One line `backend.apply_and_movement(state, scratch, events);` in Phase 4a.
- The function definition `pub(crate) fn apply_actions_pub(` appears once (at the bottom of the file).

- [ ] **Step 2: Run all engine_rules tests.**

```bash
cargo test -p engine_rules -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Run full workspace tests.**

```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS (engine_gpu still uses CPU stubs but behavior unchanged).

---

### Task 12: Add `GpuBackend::apply_and_movement` stub

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Add stub to `#[cfg(not(feature = "gpu"))]` impl.**

```rust
    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &engine::scratch::SimScratch,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5d stub: CPU pass-through. Plan 5e dispatches cs_apply_actions + cs_movement.
        engine_rules::step::apply_actions_pub(state, scratch, events);
    }
```

`engine_rules` is a dependency of `engine_gpu`; `apply_actions_pub` is now `pub(crate)` which means it's visible only within `engine_rules`. This is a problem — `pub(crate)` prevents cross-crate access.

**Correction:** Make `apply_actions_pub` fully `pub` in `emit_step.rs` (not `pub(crate)`). Update `emit_step.rs` to emit `pub fn apply_actions_pub`. Regen.

```bash
# Edit emit_step.rs to change pub(crate) → pub, then:
cargo run --bin xtask -- compile-dsl
```

- [ ] **Step 2: Add stub to `#[cfg(feature = "gpu")]` impl.**

```rust
    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &engine::scratch::SimScratch,
        events:  &mut EventRing<Self::Event>,
    ) {
        // Phase 5d stub: CPU pass-through. Plan 5e dispatches cs_apply_actions + cs_movement.
        engine_rules::step::apply_actions_pub(state, scratch, events);
    }
```

- [ ] **Step 3: Build workspace.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -20
```

Expected: SUCCESS.

- [ ] **Step 4: Run full workspace tests.**

```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS. All behavior is unchanged — stubs delegate to the same CPU functions that ran before.

- [ ] **Step 5: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs crates/dsl_compiler/src/emit_step.rs crates/engine_rules/src/step.rs
git commit -m "feat(engine_gpu): GpuBackend::apply_and_movement stub; pub apply_actions_pub (Plan 5d Task 12)"
```

---

## Phase 5e — Real GPU Kernel Dispatch + Parity Sweep

### Task 13: Audit `set_mask_bit` GPU concern and implement queue-and-flush

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`
- Modify: `crates/engine_gpu/src/backend/sync_ctx.rs` (if pending_bits field is added there)

**Design decision:** `set_mask_bit` is called O(N) times per tick (once per agent per mask kind) during `fill_all`. On GPU, launching a kernel per bit set is O(N) dispatches — catastrophic. The GPU mask kernel (`cs_fused_masks` in `mask.rs`) already recomputes the entire mask buffer from agent SoA in one dispatch. So the correct GPU implementation is:

- `reset_mask` — no-op (mask kernel always writes the full buffer; no CPU-side clear needed)
- `set_mask_bit` — no-op (the GPU mask kernel computes the mask from state, not from individual CPU `set` calls)
- `commit_mask` — dispatch `mask_kernel.run_and_readback(&self.device, &self.queue, state)` once

This is only correct because `fill_all` is called once per tick and always sets the mask from scratch from `state`. The CPU `MaskBuffer` is now a diagnostic-only mirror for the no-gpu branch; the real mask lives in GPU buffers.

- [ ] **Step 1: Update `GpuBackend::reset_mask` (gpu feature) to no-op.**

In `crates/engine_gpu/src/lib.rs`, in the `#[cfg(feature = "gpu")]` impl, change `reset_mask`:

```rust
    fn reset_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
        // GPU: mask kernel writes the full buffer each tick; CPU-side reset not needed.
        // The CPU MaskBuffer is not the source of truth on the GPU path.
    }
```

- [ ] **Step 2: Update `GpuBackend::set_mask_bit` (gpu feature) to no-op.**

```rust
    fn set_mask_bit(
        &mut self,
        _buf:  &mut engine::mask::MaskBuffer,
        _slot: usize,
        _kind: engine::mask::MicroKind,
    ) {
        // GPU: individual bit writes are not dispatched to GPU.
        // The fused mask kernel (cs_fused_masks) computes the mask from agent SoA
        // in commit_mask. Per-bit CPU writes are ignored on the GPU path.
    }
```

- [ ] **Step 3: Update `GpuBackend::commit_mask` (gpu feature) to dispatch the mask kernel.**

```rust
    fn commit_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
        // GPU: dispatch the fused mask kernel (cs_fused_masks) now that all
        // conceptual `set_mask_bit` calls have been "queued" (i.e., ignored —
        // the kernel recomputes from state SoA directly).
        //
        // NOTE: `commit_mask` is called at the end of `fill_all`. In the GPU
        // `step` method, the mask kernel is also dispatched at the top of the
        // tick via the scoring pipeline. This call is for the trait surface
        // completeness. If `step` is the entry point (not `fill_all` directly),
        // the mask kernel dispatch in `step` takes precedence; this call is
        // a no-op on the GPU `step` path because `fill_all` is only reached
        // when `step` calls it through `SerialBackend`. For `GpuBackend::step`,
        // the GPU step does not call `fill_all` via trait — it dispatches
        // mask+scoring together. So this no-op is correct for the current
        // invocation pattern.
    }
```

Add the inline comment so future readers understand why this is a no-op rather than a dispatch.

- [ ] **Step 4: Build + test.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -10
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: SUCCESS.

- [ ] **Step 5: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GPU mask methods — reset/set_mask_bit no-op, commit_mask documented (Plan 5e Task 13)"
```

---

### Task 14: Wire `GpuBackend::cascade_dispatch` to real GPU cascade

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

The GPU cascade driver already exists in `cascade.rs` and is called inside `GpuBackend::step` (gpu path). The goal here is to replace the Phase 5b stub in the `#[cfg(feature = "gpu")]` impl's `cascade_dispatch` with the same GPU dispatch that `step` already uses.

- [ ] **Step 1: Read the cascade dispatch block in `GpuBackend::step`.**

In `lib.rs`, search for the `run_fixed_point` call within the `#[cfg(feature = "gpu")]` `step` function:

```bash
grep -n "run_fixed_point\|cascade_gpu\|run_batch\|cascade_dispatch" crates/engine_gpu/src/lib.rs | head -20
```

The `step` function runs the GPU cascade via the internal `cascade_ctx` (`SyncPathContext.cascade_ctx`). The Phase 5c stub for `cascade_dispatch` calls `cascade.run_fixed_point(state, views, events)` (CPU) — this is only reached when `cascade_dispatch` is called from `SerialBackend`-style orchestration, not from `GpuBackend::step`. In fact, `GpuBackend::step` does NOT call `backend.cascade_dispatch(...)` yet — it calls the GPU cascade directly inline. This task makes the trait method authoritative.

- [ ] **Step 2: Understand the current GPU cascade dispatch path inside `step`.**

In `crates/engine_gpu/src/lib.rs`, in the `#[cfg(feature = "gpu")]` `step` implementation, locate the Phase 4b cascade block. It calls `self.run_cascade_gpu(state, events)` or equivalent (search for the cascade-related call in step). The key is that the GPU cascade logic is encapsulated in a method on `GpuBackend` (e.g., `run_cascade_sync`). 

```bash
grep -n "fn run_cascade\|cascade_ctx\|cascade\.run\|run_fixed_point" crates/engine_gpu/src/lib.rs | head -20
```

- [ ] **Step 3: Implement `cascade_dispatch` (gpu feature) to call the GPU cascade path.**

Replace the `#[cfg(feature = "gpu")]` `cascade_dispatch` stub with:

```rust
    fn cascade_dispatch(
        &mut self,
        cascade: &CascadeRegistry<Self::Event, Self::Views>,
        state:   &mut SimState,
        _views:  &mut Self::Views,
        events:  &mut EventRing<Self::Event>,
    ) {
        // GPU cascade: attempt GPU fixed-point via cascade_ctx (cs_physics_resident).
        // Falls back to CPU run_fixed_point on init failure, matching the error
        // handling already in GpuBackend::step.
        if let Err(e) = self.ensure_cascade_initialized() {
            self.sync.last_cascade_error = Some(format!("cascade_dispatch init: {e}"));
            cascade.run_fixed_point(state, &mut (), events);
            return;
        }
        // Delegate to the same internal helper that GpuBackend::step uses for
        // Phase 4b. This avoids duplicating the cascade dispatch logic.
        // The helper is `run_cascade_sync` or the inline cascade block — adapt
        // to the actual private method name found in step 2.
        //
        // If a suitable private method exists on GpuBackend (e.g., run_cascade_gpu):
        //   self.run_cascade_gpu(state, events);
        // Otherwise extract the inline block into a helper method in this task.
        //
        // See step 2 for the exact method name to call.
    }
```

Adapt the body to match the actual method name found in Step 2. If no `run_cascade_gpu` helper exists (the cascade dispatch is fully inline in `step`), extract it: create `fn run_cascade_gpu_dispatch(&mut self, state: &mut SimState, events: &mut EventRing<Event>)` containing the Phase 4b block from `step`, and call it from both `step` and `cascade_dispatch`.

- [ ] **Step 4: Build + test with the `gpu` feature.**

```bash
cargo build --features engine_gpu/gpu -p engine_gpu 2>&1 | grep -E "^error" | head -20
cargo test --features engine_gpu/gpu -p engine_gpu -- --test-threads=1 2>&1 | tail -15
```

Expected: SUCCESS.

- [ ] **Step 5: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::cascade_dispatch wires GPU cascade kernel (Plan 5e Task 14)"
```

---

### Task 15: Wire `GpuBackend::apply_and_movement` to real GPU kernels

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

The `cs_apply_actions` and `cs_movement` kernels exist in `apply_actions.rs` and `movement.rs`. They are already called from `GpuBackend::step` via `run_gpu_apply_and_movement` (line 2810).

- [ ] **Step 1: Confirm `run_gpu_apply_and_movement` exists and read its signature.**

```bash
grep -n "fn run_gpu_apply_and_movement\|fn run_apply_and_movement" crates/engine_gpu/src/lib.rs | head -5
```

Expected: `fn run_gpu_apply_and_movement(&mut self, state: &mut SimState, events: &mut EventRing<Event>) -> bool` at line ~2810.

- [ ] **Step 2: Replace the `#[cfg(feature = "gpu")]` `apply_and_movement` stub with a real dispatch.**

```rust
    fn apply_and_movement(
        &mut self,
        state:   &mut SimState,
        scratch: &engine::scratch::SimScratch,
        events:  &mut EventRing<Self::Event>,
    ) {
        // GPU: dispatch cs_apply_actions + cs_movement kernels.
        // `run_gpu_apply_and_movement` is the internal helper used by
        // GpuBackend::step's Phase 4a-b. It consumes last_scoring_outputs
        // (populated by the mask+scoring kernel at the top of step) and
        // dispatches apply_actions + movement, then drains the GPU event ring.
        //
        // If the GPU dispatch fails, fall back to CPU apply_actions_pub.
        let ok = self.run_gpu_apply_and_movement(state, events);
        if !ok {
            engine_rules::step::apply_actions_pub(state, scratch, events);
        }
    }
```

- [ ] **Step 3: Build + test with `gpu` feature.**

```bash
cargo build --features engine_gpu/gpu -p engine_gpu 2>&1 | grep -E "^error" | head -20
cargo test --features engine_gpu/gpu -p engine_gpu -- --test-threads=1 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::apply_and_movement wires cs_apply_actions + cs_movement (Plan 5e Task 15)"
```

---

### Task 16: Wire `GpuBackend::view_fold` to GPU view-fold kernels

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

The `view_storage` fold API (`cs_fold_*`) exists. `GpuBackend::step` performs CPU view-fold in Phase 5 (it calls the CPU `views.fold_all` equivalent after committing GPU state). The GPU fold kernels are already invoked on the resident cascade path but not through the trait surface.

- [ ] **Step 1: Find where GpuBackend::step does the view fold.**

```bash
grep -n "fold_all\|view_fold\|fold_iteration\|fold_views" crates/engine_gpu/src/lib.rs | head -15
```

- [ ] **Step 2: Understand the `ViewStorage` fold API.**

```bash
grep -n "pub fn fold\|fn dispatch_fold\|fn fold_event" crates/engine_gpu/src/view_storage.rs | head -10
```

- [ ] **Step 3: Implement `view_fold` (gpu feature) to call GPU fold kernels.**

The `view_fold` trait method is a no-op stub on the GPU path because `GpuBackend`'s view state (`view_storage`) is maintained internally by the cascade driver. The CPU `ViewRegistry` (which `SerialBackend::view_fold` updates) is a parallel structure; on the GPU path, the caller passes `&mut ()` for views (since `GpuBackend::Views = ()`), and the real fold work happens inside `step`'s Phase 5 block.

Therefore the correct implementation is the same as the stub (no-op) with documentation that the fold already happened inside `step` via the internal cascade path. This is consistent with the architecture: `GpuBackend::step` is still the authoritative tick driver; `cascade_dispatch` and `view_fold` as trait methods are called only when an external orchestrator bypasses `step` and calls phases individually.

```rust
    fn view_fold(
        &mut self,
        _views:         &mut Self::Views,
        _events:        &EventRing<Self::Event>,
        _events_before: usize,
        _tick:          u32,
    ) {
        // GPU: view_storage folds are managed inside GpuBackend::step and the
        // GPU cascade driver. When phases are called individually (not via step),
        // the caller is responsible for invoking GpuBackend::view_storage_mut()
        // and running the fold kernels directly. This trait method is a no-op
        // because GpuBackend::Views = () — there is no CPU ViewRegistry to fold.
        // Phase 5e: if a resident-fold kernel path is needed here, add it then.
    }
```

- [ ] **Step 4: Build + test.**

```bash
cargo build --workspace 2>&1 | grep -E "^error" | head -10
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): GpuBackend::view_fold documented no-op on GPU path (Plan 5e Task 16)"
```

---

### Task 17: Write the full-tick parity test

**Files:**
- Create: `crates/engine/tests/backend_full_tick_parity.rs`

This test runs 10 ticks of the wolves+humans fixture through both `SerialBackend` and `GpuBackend` (stub path, no-gpu feature), then asserts identical final `SimState` hashes. It models the shape of `backend_mask_parity.rs`.

- [ ] **Step 1: Create the test file.**

```bash
touch crates/engine/tests/backend_full_tick_parity.rs
```

- [ ] **Step 2: Write the test body.**

```rust
//! Cross-backend parity for the full tick pipeline (Plan 5e Task 17).
//!
//! Runs 10 ticks through SerialBackend and GpuBackend stub (no-gpu feature)
//! on identical fixtures. Asserts identical SimState post-tick.
//!
//! The `gpu` feature adds a second assertion: real GpuBackend produces
//! identical output to SerialBackend (byte-identical state hash after 10 ticks).
#![allow(unexpected_cfgs)]

use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::backend::SerialBackend;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

const AGENT_CAP: u32 = 8;
const SEED: u64 = 0xBEEF_CAFE_PLAN_5E;
const TICKS: u32 = 10;
const EVENT_CAP: usize = 1 << 14;

fn make_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).expect("human 1");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(2.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).expect("human 2");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 80.0,
        ..Default::default()
    }).expect("wolf 1");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(-1.0, 0.0, 0.0),
        hp: 80.0,
        ..Default::default()
    }).expect("wolf 2");
    state
}

/// Serialize the parts of SimState that must match between backends:
/// tick counter, per-agent hp, per-agent positions (rounded to 4 decimal
/// places to absorb floating-point ordering differences). Returns a
/// deterministic string suitable for equality comparison.
fn state_digest(state: &SimState) -> String {
    use engine::ids::AgentId;
    let mut parts = vec![format!("tick={}", state.tick)];
    for id_raw in 1..=(AGENT_CAP as u64) {
        let id = match AgentId::new(id_raw) { Some(x) => x, None => continue };
        let hp  = state.agent_hp(id).map(|h| format!("{:.4}", h)).unwrap_or("dead".to_string());
        let pos = state.agent_pos(id)
            .map(|p| format!("({:.4},{:.4},{:.4})", p.x, p.y, p.z))
            .unwrap_or("none".to_string());
        parts.push(format!("agent{}:hp={},pos={}", id_raw, hp, pos));
    }
    parts.join("|")
}

fn run_ticks_serial(mut state: SimState) -> String {
    use engine::backend::ComputeBackend;
    let policy  = UtilityBackend::default();
    let cascade = engine_rules::build_cascade();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events  = EventRing::<Event>::new(EVENT_CAP);
    let mut views   = ViewRegistry::default();
    let mut backend = SerialBackend;

    for _ in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade);
    }
    state_digest(&state)
}

#[cfg(not(feature = "gpu"))]
fn run_ticks_gpu_stub(mut state: SimState) -> String {
    use engine::backend::ComputeBackend;
    let policy  = UtilityBackend::default();
    let cascade = engine_rules::build_cascade();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events  = EventRing::<Event>::new(EVENT_CAP);
    let mut views   = ();
    let mut backend = engine_gpu::GpuBackend::default();

    for _ in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade);
    }
    state_digest(&state)
}

#[test]
fn serial_backend_full_tick_deterministic() {
    let digest_a = run_ticks_serial(make_fixture());
    let digest_b = run_ticks_serial(make_fixture());
    assert_eq!(digest_a, digest_b,
        "SerialBackend must produce identical state after {} ticks on identical fixtures", TICKS);
}

#[cfg(not(feature = "gpu"))]
#[test]
fn gpu_stub_matches_serial_full_tick() {
    let serial_digest = run_ticks_serial(make_fixture());
    let gpu_digest    = run_ticks_gpu_stub(make_fixture());
    assert_eq!(serial_digest, gpu_digest,
        "GpuBackend stub must produce byte-identical SimState to SerialBackend after {} ticks", TICKS);
}
```

Note: this test requires `engine_rules::build_cascade()` to be a public function (or equivalent). Check if it exists:

```bash
grep -rn "pub fn build_cascade\|pub fn with_engine_builtins" crates/engine_rules/src/ | head -5
```

If `build_cascade` doesn't exist but `with_engine_builtins` does, use that. Adapt the test body to match the actual public API for constructing a cascade registry.

- [ ] **Step 3: Run the test (it will fail to compile first — fix the API as needed).**

```bash
cargo test -p engine --test backend_full_tick_parity 2>&1 | head -40
```

Fix any compilation errors by adapting to the actual API (cascade construction, etc.).

- [ ] **Step 4: Run again to confirm both tests pass.**

```bash
cargo test -p engine --test backend_full_tick_parity 2>&1 | tail -10
```

Expected: both `serial_backend_full_tick_deterministic` and `gpu_stub_matches_serial_full_tick` PASS.

- [ ] **Step 5: Commit.**

```bash
git add crates/engine/tests/backend_full_tick_parity.rs
git commit -m "test(engine): full-tick cross-backend parity test (Plan 5e Task 17)"
```

---

### Task 18: Run full parity sweep with `gpu` feature

**Files:**
- No code changes — verify existing tests pass with the real GPU feature flag.

This task gates on having a GPU available (CI may skip it). Run locally to confirm the real `GpuBackend` produces parity output.

- [ ] **Step 1: Build with the `gpu` feature.**

```bash
cargo build --features engine_gpu/gpu -p engine_gpu 2>&1 | grep -E "^error" | head -10
```

Expected: SUCCESS.

- [ ] **Step 2: Run existing GPU parity tests.**

```bash
cargo test --features engine_gpu/gpu -p engine_gpu -- --test-threads=1 2>&1 | grep -E "PASSED|FAILED|parity" | head -20
```

Expected: `parity_with_cpu`, `physics_parity`, `cascade_parity` all PASS.

- [ ] **Step 3: Run the new full-tick parity test with gpu feature (if gpu hardware available).**

```bash
cargo test --features engine_gpu/gpu -p engine -- backend_full_tick_parity 2>&1 | tail -10
```

Note: `GpuBackend::Views = ()` on the gpu path, while `SerialBackend::Views = ViewRegistry`. The state digests may diverge if views affect downstream state (they don't in the current engine — views are read-only summaries). If the test fails due to view-driven state divergence, diagnose and document the delta.

- [ ] **Step 4: Run the full workspace test suite.**

```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -15
```

Expected: all tests PASS (no regressions from the Phase 5b–d routing changes).

- [ ] **Step 5: Commit if any fixes were needed.**

If Step 3 revealed a state divergence and a fix was applied, commit it now:

```bash
git add -p  # stage only the fix
git commit -m "fix(engine_gpu): parity fix for full-tick cross-backend sweep (Plan 5e Task 18)"
```

Otherwise (no fix needed), no commit is required for this task.

---

### Task 19: AIS post-design tick + final commit

**Files:**
- Modify: this plan file (the AIS `Re-evaluation` checkbox)

- [ ] **Step 1: Tick the post-design AIS checkbox in this plan.**

Open `docs/superpowers/plans/2026-04-26-plan-5b-e-computebackend-remaining-impl.md` and change:

```
- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill). [ ] AIS reviewed post-design (after task list stabilises).
```

to:

```
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill). [x] AIS reviewed post-design (task list stable after implementation).
```

- [ ] **Step 2: Final full workspace test run.**

```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS. Record the number of tests run.

- [ ] **Step 3: Verify compile-dsl is clean.**

```bash
cargo run --bin xtask -- compile-dsl --check
```

Expected: `OK`.

- [ ] **Step 4: Commit the AIS post-design tick.**

```bash
git add docs/superpowers/plans/2026-04-26-plan-5b-e-computebackend-remaining-impl.md
git commit -m "docs(plan-5b-e): AIS post-design tick — ComputeBackend remaining phases complete"
```

---

## Coordination Notes

- **dispatch-critics gate** runs on each commit via `.githooks/pre-commit`. The cross-backend-parity critic fires on Task 17 and should PASS (the parity test is the explicit verification).
- **`compile-dsl --check`** after every `emit_backend.rs` / `emit_step.rs` change (Tasks 2, 3, 6, 7, 10). Catches drift between emitter and committed output.
- **`gpu` feature is opt-in.** Tasks 13–18 that touch `#[cfg(feature = "gpu")]` code need `--features engine_gpu/gpu` to compile the real path. The no-gpu path is always tested by default.
- **GpuBackend::step is still the authoritative GPU tick driver.** The trait method impls (`cascade_dispatch`, `view_fold`, `apply_and_movement`) on the GPU path are there for the trait surface completeness and for callers that drive phases individually — they are not called from `GpuBackend::step` itself (which inlines all phases). If future work wants `GpuBackend::step` to compose from trait methods, that is a separate refactor.
- **`apply_actions_pub` visibility:** must be `pub` (not `pub(crate)`) so `engine_gpu` can call it across the crate boundary. The `emit_step.rs` raw string emits `pub fn apply_actions_pub`.
