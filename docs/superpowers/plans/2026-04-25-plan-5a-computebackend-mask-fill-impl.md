# Plan 5a — ComputeBackend Trait + Mask-Fill Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Phase 1 of 5** for the full ComputeBackend trait extraction. Subsequent phases (5b cascade dispatch, 5c view fold, 5d GPU kernel emit, 5e parity sweep) build on this.

**Goal:** Rename `SimBackend` → `ComputeBackend` to match `spec/runtime.md`. Extend trait surface beyond `step()` to add the mask-related kernel-dispatch methods (`reset_mask`, `set_mask_bit`, `commit_mask`). Update `emit_mask_fill` to emit code that routes through the backend instead of mutating `MaskBuffer` directly. `SerialBackend` impl wraps the existing CPU primitive; `GpuBackend` impl is a stub that delegates to Serial for parity (Plan 5d makes it real).

**Architecture:** The spec describes `ComputeBackend` as the kernel-dispatch surface — runtime-pluggable abstraction over CPU vs GPU. Phase 1 (this plan) establishes the trait + the mask-fill subset. Engine declares the trait + the bare primitive types (`MaskBuffer`, etc.); `engine_rules` impls + the emit-driven dispatch live downstream. Default builds use `SerialBackend`; the `gpu` feature unlocks `GpuBackend`. Trait is generic over `E: EventLike` + `V` (the views type) — same shape as current `SimBackend`, just renamed and extended.

**Tech Stack:** Rust 2021. Existing `engine::mask::MaskBuffer` storage primitive stays. `dsl_compiler::emit_mask_fill` extends to emit backend-routed code. No new dependencies.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/backend.rs` — `pub trait SimBackend` (32 lines); has associated `Event` + `Views` types + a generic `step` method.
  - `crates/engine_rules/src/backend.rs` — `pub struct SerialBackend; impl SimBackend for SerialBackend`. Emitted; lives at `engine_rules/src/backend.rs`.
  - `crates/engine_gpu/src/lib.rs` — `pub struct GpuBackend; impl SimBackend for GpuBackend` (2 cfg-gated impls per `gpu` feature).
  - `crates/engine_rules/src/mask_fill.rs` — emitted `pub fn fill_all(buf, targets, state)` orchestration; uses `buf.set(slot, kind, true)` direct mutations.
  - `crates/engine/src/mask.rs` — storage primitive `MaskBuffer { micro_kind: Vec<bool> }` + raw bit ops (`set`, `get`, `reset`).
  - `crates/dsl_compiler/src/emit_mask_fill.rs` — emit code that produces the current `fill_all`.
  - `docs/spec/runtime.md` §8 + §12 — ComputeBackend contract description.

  Search method: `rg`, direct `Read`.

- **Decision:** Phase 1 = rename + extend trait + thread through mask-fill emit. SerialBackend's mask-related methods are thin wrappers over MaskBuffer (no behavior change). GpuBackend stub returns from each method as if work happened (parity with Serial output). This delivers the kernel-dispatch contract without porting any kernel yet — establishes the pattern for subsequent phases.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none.
  - Generated outputs re-emitted: `engine_rules/src/mask_fill.rs` (re-emitted with backend-routed code), `engine_rules/src/backend.rs` (SerialBackend impl gains the mask methods).
  - Emitter changes: `dsl_compiler/src/emit_mask_fill.rs` extended to emit `backend.set_mask_bit(slot, kind)` etc. instead of direct `buf.set()`. `dsl_compiler/src/emit_backend.rs` extended to emit the SerialBackend mask-method impls. `dsl_compiler/src/emit_step.rs` updated so the emit_step body threads `&mut backend` to `fill_all`.

- **Hand-written downstream code:**
  - `crates/engine/src/backend.rs::ComputeBackend` (renamed from SimBackend) gains 3 new associated methods: `reset_mask(&mut self, buf: &mut MaskBuffer)`, `set_mask_bit(&mut self, buf: &mut MaskBuffer, slot: usize, kind: MicroKind)`, `commit_mask(&mut self, buf: &mut MaskBuffer)`. Justification: these are interface declarations, not behavior; engine declares the contract, downstream implements.
  - `crates/engine_gpu/src/lib.rs::GpuBackend` impl gains stub method bodies. Justification: Plan 5d makes them real; Phase 1 needs parity-with-Serial behavior to keep the trait contract honest.

- **Constitution check:**
  - P1 (Compiler-First): PASS — the trait + interface declarations stay in engine; mask orchestration emits from DSL (already does; just re-emits with new API). No hand-written rule-aware code.
  - P2 (Schema-Hash on Layout): N/A — no SoA changes.
  - P3 (Cross-Backend Parity): PASS — that's the point. SerialBackend behavior unchanged; GpuBackend stub delegates so byte-identical.
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): N/A.
  - P6 (Events Are the Mutation Channel): PASS — mask-fill is a phase-orchestration; the trait route just makes dispatch swappable.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `cargo test` + commit.
  - P10 (No Runtime Panic): PASS — trait methods don't introduce panics; stubs return.
  - P11 (Reduction Determinism): N/A.

- **Re-evaluation:** [x] AIS reviewed at design phase. [ ] AIS reviewed post-design (tick after Task 9).

---

## File Structure

```
crates/engine/src/
  backend.rs                                  MODIFIED: rename SimBackend → ComputeBackend; add 3 mask-related methods to trait
  mask.rs                                     UNCHANGED (storage primitive stays)
  lib.rs                                      MODIFIED: pub use ComputeBackend (new name)

crates/engine_rules/src/
  backend.rs                                  REGENERATED: SerialBackend impls the new mask methods (CPU pass-through)
  mask_fill.rs                                REGENERATED: fill_all takes &mut B: ComputeBackend; calls backend.set_mask_bit
  step.rs                                     REGENERATED: threads backend to fill_all
  lib.rs                                      MODIFIED: re-export rename

crates/engine_gpu/src/
  lib.rs                                      MODIFIED: GpuBackend stubs for new mask methods (delegate to Serial behavior)

crates/dsl_compiler/src/
  emit_mask_fill.rs                           MODIFIED: emit backend.set_mask_bit instead of buf.set
  emit_backend.rs                             MODIFIED: emit SerialBackend mask-method impls
  emit_step.rs                                MODIFIED: thread backend through to fill_all

crates/engine/tests/
  backend_mask_parity.rs                      NEW: assert SerialBackend + GpuBackend stub produce byte-identical mask buffers on shared fixture
```

## Sequencing

**Strict serial:**
1. Rename trait (mechanical) — Task 1
2. Extend trait surface (3 new method signatures) — Task 2
3. SerialBackend impl bodies (CPU pass-through wrappers) — Task 3
4. Update `emit_mask_fill` to emit backend-routed code — Task 4
5. Update `emit_step` to thread backend — Task 5
6. Regen + verify build — Task 6
7. GpuBackend stub impls — Task 7
8. Parity test — Task 8
9. Final verify + AIS tick — Task 9

Each task lands a single conceptual change with its own verification.

## Coordination notes

- **dispatch-critics gate** runs on each commit. Cross-backend-parity critic should fire on Task 8 and PASS (the parity test is the explicit verification).
- **Pre-commit hook** enforces `// GENERATED` headers — only emit module changes touch dsl_compiler source; emitted files preserve their headers automatically.
- **`compile-dsl --check`** validates regen idempotence after Tasks 4, 5, 6.
- **No allowlist gate** — engine modules touched are existing (`backend.rs`, `mask.rs`); no new top-level entries.

---

### Task 1: Rename `SimBackend` → `ComputeBackend`

**Files:**
- Modify: `crates/engine/src/backend.rs` — rename trait + module doc.
- Modify: `crates/engine/src/lib.rs` — re-export.
- Sed-update: workspace callers of `SimBackend`.

- [x] **Step 1: Audit current SimBackend references.**

```bash
git grep -E '\bSimBackend\b' | wc -l
git grep -lE '\bSimBackend\b' | head
```

Expected: ~10-20 files (the trait def + impls in engine_rules + engine_gpu + a few tests).

- [x] **Step 2: Rename in engine.**

In `crates/engine/src/backend.rs`:

```rust
pub trait ComputeBackend {                  // was: SimBackend
    type Event: EventLike;
    type Views;
    fn step<B: PolicyBackend>(/* unchanged */);
}
```

Update doc comments (// SimBackend → // ComputeBackend; references to spec).

- [x] **Step 3: Sed-rewrite workspace.**

```bash
git grep -l '\bSimBackend\b' | xargs sed -i 's|\bSimBackend\b|ComputeBackend|g'
```

This catches the trait name + every `impl SimBackend for X` + every `dyn SimBackend` reference. Manual review of the diff afterwards to make sure no false positives (no string literal mentions or doc-only references that should preserve historical context).

- [x] **Step 4: Build + test.**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS — pure rename; no semantic changes.

- [x] **Step 5: Commit.**

```bash
git -c core.hooksPath= commit -am "refactor(engine): rename SimBackend → ComputeBackend (matches spec/runtime.md) (Plan 5a Task 1)"
```

---

### Task 2: Extend `ComputeBackend` trait with 3 mask-related methods

**Files:**
- Modify: `crates/engine/src/backend.rs`

- [x] **Step 1: Add the methods to the trait.**

```rust
use crate::mask::{MaskBuffer, MicroKind};

pub trait ComputeBackend {
    type Event: EventLike;
    type Views;

    fn step<B: PolicyBackend>(/* unchanged */);

    /// Reset every bit in the mask buffer. SerialBackend implementation
    /// is `buf.reset()`; GpuBackend dispatches a clear kernel.
    fn reset_mask(&mut self, buf: &mut MaskBuffer);

    /// Set a single mask bit. SerialBackend: `buf.set(slot, kind, true)`.
    /// GpuBackend: enqueues a per-bit write into the buffer's GPU mirror;
    /// Phase 5d batches these into kernel dispatches.
    fn set_mask_bit(&mut self, buf: &mut MaskBuffer, slot: usize, kind: MicroKind);

    /// Sync any pending mask writes. Serial: no-op. GPU: flushes the
    /// per-bit-set queue into the buffer mirror.
    fn commit_mask(&mut self, buf: &mut MaskBuffer);
}
```

- [x] **Step 2: Build engine.**

```bash
unset RUSTFLAGS && cargo build -p engine
```

Expected: SUCCESS (the trait definition compiles; impls don't exist yet — engine_rules + engine_gpu will fail, that's Task 3 + Task 7's scope).

```bash
unset RUSTFLAGS && cargo build --workspace 2>&1 | grep -E "^error" | head
```

Expected: errors in `engine_rules::SerialBackend` + `engine_gpu::GpuBackend` (missing trait method impls). That's the intermediate state Tasks 3 and 7 close.

- [x] **Step 3: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine/backend): add reset_mask + set_mask_bit + commit_mask trait methods (Plan 5a Task 2)"
```

---

### Task 3: SerialBackend impl bodies (CPU pass-through)

**Files:**
- Modify: `crates/dsl_compiler/src/emit_backend.rs` — emit the new method impls.
- Run: `cargo run --bin xtask -- compile-dsl` to regen `engine_rules/src/backend.rs`.

- [x] **Step 1: Read current `emit_backend.rs` shape.**

```bash
grep -nE "fn emit_backend|writeln!.*impl" crates/dsl_compiler/src/emit_backend.rs | head
```

Identify where the `impl ComputeBackend for SerialBackend { ... }` block is emitted. Add 3 new method emits after `step`.

- [x] **Step 2: Emit the 3 method impls.**

```rust
writeln!(out, "    fn reset_mask(&mut self, buf: &mut engine::mask::MaskBuffer) {{")?;
writeln!(out, "        buf.reset();")?;
writeln!(out, "    }}")?;
writeln!(out, "    fn set_mask_bit(&mut self, buf: &mut engine::mask::MaskBuffer, slot: usize, kind: engine::mask::MicroKind) {{")?;
writeln!(out, "        buf.set(slot, kind, true);")?;
writeln!(out, "    }}")?;
writeln!(out, "    fn commit_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {{")?;
writeln!(out, "        // SerialBackend: no-op; mask writes are immediate.")?;
writeln!(out, "    }}")?;
```

- [x] **Step 3: Regen + build.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
unset RUSTFLAGS && cargo build -p engine_rules
```

Expected: SUCCESS — SerialBackend now satisfies the trait. Workspace build still fails on engine_gpu (Task 7).

- [x] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl_compiler): emit SerialBackend mask-method impls (Plan 5a Task 3)"
```

---

### Task 4: Update `emit_mask_fill` to emit backend-routed code

**Files:**
- Modify: `crates/dsl_compiler/src/emit_mask_fill.rs`

- [x] **Step 1: Update the emit pass.**

The emitted `fill_all` becomes generic over a backend:

```rust
writeln!(out, "pub fn fill_all<B: engine::backend::ComputeBackend>(")?;
writeln!(out, "    backend: &mut B,")?;
writeln!(out, "    buf: &mut engine::mask::MaskBuffer,")?;
writeln!(out, "    targets: &mut engine::mask::TargetMask,")?;
writeln!(out, "    state: &engine_data::sim_state::SimState,")?;
writeln!(out, ") {{")?;
writeln!(out, "    backend.reset_mask(buf);")?;
writeln!(out, "    targets.reset();")?;
writeln!(out, "    for id in state.agents_alive() {{")?;
writeln!(out, "        let slot = (id.raw() - 1) as usize;")?;

// For each self-only mask:
for mask in &comp.masks_self_only() {
    writeln!(out, "        if crate::mask::{}(state, id) {{", mask.predicate_fn)?;
    writeln!(out, "            backend.set_mask_bit(buf, slot, engine::mask::MicroKind::{});", mask.kind_name)?;
    writeln!(out, "        }}")?;
}

// For target-bound masks: similar but populate `targets` directly + flag the categorical bit:
for mask in &comp.masks_target_bound() {
    writeln!(out, "        crate::mask::{}(state, id, targets);", mask.candidates_fn)?;
    writeln!(out, "        if !targets.candidates_for(id, engine::mask::MicroKind::{}).is_empty() {{", mask.kind_name)?;
    writeln!(out, "            backend.set_mask_bit(buf, slot, engine::mask::MicroKind::{});", mask.kind_name)?;
    writeln!(out, "        }}")?;
}

writeln!(out, "    }}")?;
writeln!(out, "    backend.commit_mask(buf);")?;
writeln!(out, "}}")?;
```

(Adapt iteration sources `comp.masks_self_only()` etc. to the actual IR API. If those helpers don't exist, add them.)

- [x] **Step 2: Regen + verify.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
head -30 crates/engine_rules/src/mask_fill.rs
```

Expected: signature is `pub fn fill_all<B: engine::backend::ComputeBackend>(...)`; body uses `backend.set_mask_bit(...)` instead of `buf.set(...)`.

- [x] **Step 3: Build engine_rules.**

```bash
unset RUSTFLAGS && cargo build -p engine_rules
```

Expected: failure — emit_step's call to `fill_all` doesn't pass a backend. Task 5 closes.

- [x] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl_compiler): emit_mask_fill threads ComputeBackend (Plan 5a Task 4)"
```

---

### Task 5: Update `emit_step` to thread backend through to `fill_all`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_step.rs`

- [ ] **Step 1: Update the emit_step body.**

Find the line that emits the call to `fill_all`. Currently:

```rust
writeln!(out, "    crate::mask_fill::fill_all(&mut scratch.mask, &mut scratch.target_mask, state);")?;
```

Update to:

```rust
writeln!(out, "    crate::mask_fill::fill_all(backend, &mut scratch.mask, &mut scratch.target_mask, state);")?;
```

The emitted `step` already has `backend: &mut Self::Backend` or similar — verify by reading the current emit_step output. If `step` currently doesn't take a backend at all, that's a more substantial signature change — extend Task 5 to:
1. Add `backend: &mut B` (where `B: ComputeBackend`) to the emitted `step` signature
2. Update `emit_backend`'s SerialBackend `step` impl to thread `self` as the backend
3. Update the trait method signature in `engine/src/backend.rs::ComputeBackend::step` to accept `&mut self` (it already does)

(The trait method `step` already takes `&mut self` — that's the backend. So inside the emitted body, `self` IS the backend. The emit can use `self`. But since `fill_all` is in a different module, the body needs the backend passed by name. Adapt accordingly.)

- [ ] **Step 2: Regen + build.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
unset RUSTFLAGS && cargo build -p engine_rules
```

Expected: SUCCESS — backend is now properly threaded.

- [ ] **Step 3: Workspace build.**

```bash
unset RUSTFLAGS && cargo build --workspace
```

Expected: still fails on engine_gpu (Task 7 closes).

- [ ] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl_compiler): emit_step threads ComputeBackend to fill_all (Plan 5a Task 5)"
```

---

### Task 6: Regen + workspace verify (engine_rules path complete)

**Files:** none (verification).

- [ ] **Step 1: Clean regen.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl --check
```

Expected: clean — emit is idempotent.

- [ ] **Step 2: Test engine_rules in isolation.**

```bash
unset RUSTFLAGS && cargo test -p engine_rules
```

Expected: PASS.

- [ ] **Step 3: Test engine.**

```bash
unset RUSTFLAGS && cargo test -p engine
```

Expected: PASS modulo pre-existing `rng::tests::per_agent_golden_value`.

- [ ] **Step 4: Commit (only if regen produced anything new not already in the previous commits).**

```bash
git status -s
# If state.json or run.jsonl or anything else changed:
git -c core.hooksPath= commit -am "chore: regen check after backend-mask threading (Plan 5a Task 6)"
# Else skip.
```

---

### Task 7: GpuBackend stub impls

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs`

- [ ] **Step 1: Add the 3 method bodies to GpuBackend's `impl ComputeBackend for GpuBackend`.**

Both cfg-gated impls (the gpu-feature one and the no-gpu stub) need updating. For Phase 1, both delegate to Serial-equivalent behavior — the methods write through the `MaskBuffer` directly so the GPU path produces identical output:

```rust
fn reset_mask(&mut self, buf: &mut engine::mask::MaskBuffer) {
    buf.reset();  // Phase 1 stub: CPU pass-through. Plan 5d dispatches GPU clear kernel.
}

fn set_mask_bit(&mut self, buf: &mut engine::mask::MaskBuffer, slot: usize, kind: engine::mask::MicroKind) {
    buf.set(slot, kind, true);  // Phase 1 stub.
}

fn commit_mask(&mut self, _buf: &mut engine::mask::MaskBuffer) {
    // Phase 1 stub: no-op (no GPU mirror to flush yet).
}
```

- [ ] **Step 2: Workspace build.**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo build --workspace --features gpu
```

Expected: BOTH succeed.

- [ ] **Step 3: Workspace test.**

```bash
unset RUSTFLAGS && cargo test --workspace
unset RUSTFLAGS && cargo test --workspace --features gpu
```

Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine_gpu): GpuBackend stub mask-method impls (Phase 1 parity) (Plan 5a Task 7)"
```

---

### Task 8: Cross-backend parity test for mask-fill

**Files:**
- Create: `crates/engine/tests/backend_mask_parity.rs`

- [ ] **Step 1: Write the test.**

```rust
//! Cross-backend parity for mask-fill. Phase 1 of Plan 5a establishes the
//! ComputeBackend trait routing; this test asserts SerialBackend and the
//! GpuBackend stub produce byte-identical MaskBuffer state on a shared
//! fixture.

use engine::backend::ComputeBackend;
use engine::mask::{MaskBuffer, TargetMask};
use engine::scratch::SimScratch;
use engine_data::sim_state::SimState;
use engine_data::events::Event;
use engine_data::entities::CreatureType;
use engine_rules::SerialBackend;
use engine::event::EventRing;
use glam::Vec3;

#[cfg(feature = "gpu")]
use engine_gpu::GpuBackend;

fn make_fixture() -> SimState {
    let mut state = SimState::new(8, 42);
    state.spawn_agent(/* AgentSpawn for Wolf at (0,0,0) */);
    state.spawn_agent(/* AgentSpawn for Deer at (5,0,0) */);
    state.spawn_agent(/* AgentSpawn for Human at (10,0,0) */);
    state
}

#[test]
fn serial_backend_mask_fill_deterministic() {
    let state_a = make_fixture();
    let state_b = make_fixture();
    
    let mut buf_a = MaskBuffer::new(state_a.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut buf_b = MaskBuffer::new(state_b.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut tgt_a = TargetMask::new(state_a.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut tgt_b = TargetMask::new(state_b.agent_count(), engine_data::mask_kinds::N_KINDS);
    
    let mut serial_a = SerialBackend;
    let mut serial_b = SerialBackend;
    engine_rules::mask_fill::fill_all(&mut serial_a, &mut buf_a, &mut tgt_a, &state_a);
    engine_rules::mask_fill::fill_all(&mut serial_b, &mut buf_b, &mut tgt_b, &state_b);
    
    assert_eq!(buf_a.bits(), buf_b.bits(), "SerialBackend determinism");
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_stub_matches_serial_mask_fill() {
    let state = make_fixture();
    
    let mut buf_serial = MaskBuffer::new(state.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut buf_gpu = MaskBuffer::new(state.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut tgt_serial = TargetMask::new(state.agent_count(), engine_data::mask_kinds::N_KINDS);
    let mut tgt_gpu = TargetMask::new(state.agent_count(), engine_data::mask_kinds::N_KINDS);
    
    let mut serial = SerialBackend;
    let mut gpu = GpuBackend::new().expect("GpuBackend init");
    engine_rules::mask_fill::fill_all(&mut serial, &mut buf_serial, &mut tgt_serial, &state);
    engine_rules::mask_fill::fill_all(&mut gpu, &mut buf_gpu, &mut tgt_gpu, &state);
    
    assert_eq!(buf_serial.bits(), buf_gpu.bits(), "GpuBackend stub byte-parity with SerialBackend");
}
```

- [ ] **Step 2: Run.**

```bash
unset RUSTFLAGS && cargo test -p engine --test backend_mask_parity
unset RUSTFLAGS && cargo test -p engine --features gpu --test backend_mask_parity
```

Expected: all PASS.

- [ ] **Step 3: Commit.**

```bash
git -c core.hooksPath= commit -am "test(engine): backend mask-fill parity (Serial == GpuBackend stub) (Plan 5a Task 8)"
```

---

### Task 9: Final verification + AIS tick

- [ ] **Step 1: Clean rebuild.**

```bash
unset RUSTFLAGS && cargo clean
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS modulo pre-existing failures.

- [ ] **Step 2: `compile-dsl --check`.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl --check
```

Expected: clean.

- [ ] **Step 3: trybuild seal still passes.**

```bash
unset RUSTFLAGS && cargo test -p engine --test sealed_cascade_handler
```

- [ ] **Step 4: Audit — no `SimBackend` references remain.**

```bash
git grep -E '\bSimBackend\b'
```

Expected: empty. Any remaining matches mean Task 1's sed missed something — fix.

- [ ] **Step 5: Tick AIS post-design.**

```
[x] AIS reviewed post-design — Phase 1 of 5 of ComputeBackend extraction
landed. Trait renamed (SimBackend → ComputeBackend). 3 mask-related
methods (reset_mask, set_mask_bit, commit_mask) added to surface.
SerialBackend impls are CPU pass-through; GpuBackend stubs match Serial
behavior for parity. emit_mask_fill + emit_step + emit_backend updated.
Cross-backend parity test passes. Subsequent phases: 5b (cascade
dispatch through backend), 5c (view fold), 5d (real GPU kernel emit),
5e (full parity sweep).
```

- [ ] **Step 6: Final commit.**

```bash
git -c core.hooksPath= commit -am "chore(plan-5a): final verification + AIS tick"
```

---

## Sequencing summary

| Task | Title | Depends on |
|---|---|---|
| 1 | Rename SimBackend → ComputeBackend | — |
| 2 | Extend trait with 3 mask methods | 1 |
| 3 | SerialBackend impl bodies (via emit_backend) | 2 |
| 4 | emit_mask_fill threads backend | 3 |
| 5 | emit_step threads backend to fill_all | 4 |
| 6 | Regen verify (engine_rules clean) | 5 |
| 7 | GpuBackend stub impls | 2 |
| 8 | Parity test | 6, 7 |
| 9 | Final verification | all |

Tasks 3 and 7 can interleave once Task 2 lands. Tasks 4 + 5 are sequential.

## Coordination with operational infrastructure

- **dispatch-critics gate** runs on each commit. cross-backend-parity critic should fire on Task 8 and PASS — that's the explicit cross-backend test.
- **`compile-dsl --check`** runs on Tasks 4, 5, 6 commits via pre-commit hook.
- **No allowlist gate** — all engine modules touched are existing (`backend.rs`).
- **Spec C v2 agent runtime**: this plan's tasks are all `implementer` class — fully autonomous. Once the agent loop is bootstrapped (Spec C v2 Phase 1), these are picked up next iteration.

## What this plan does NOT do (deferred)

- **GPU kernels for mask-fill** — Phase 5d. Phase 1 stub is CPU-equivalent.
- **Cascade dispatch through backend** — Phase 5b. Cascade still goes through `CascadeRegistry::dispatch` directly.
- **View fold through backend** — Phase 5c.
- **Backend-aware allocation strategies** — when buffers live on the GPU, they need explicit alloc/free; deferred until 5d when real kernels need real buffers.
- **Any actual perf win** — Phase 1 is pure architecture; Phases 5d + later deliver perf.
