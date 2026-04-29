# `step_batch` Runtime Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `engine_gpu::step_batch` actually executable at runtime. Today every call panics inside the per-tick "CPU forward" because it invokes `engine::step::step` which is `unimplemented!()` (Plan B1' Task 11 deleted the body pending an emitted replacement). The replacement exists at `engine_rules::step::step` with a different signature — the migration was never wired through to `engine_gpu`.

**Architecture:** `engine_rules::step::step` requires `CB: ComputeBackend<Event = Event, Views = ViewRegistry>` plus a `views: &mut ViewRegistry` parameter and `debug: &DebugConfig`. `GpuBackend` currently declares `type Views = ();` — trait-level contract mismatch with the `Views = ViewRegistry` constraint. This plan changes `GpuBackend::Views` to `ViewRegistry`, threads `views` through `step_batch`'s signature and the `ComputeBackend::step` impl, and replaces the 3 `engine::step::step` call sites with `engine_rules::step::step` invocations passing `&mut engine_rules::backend::SerialBackend` as the inner CB (the CPU reference path the GPU forward stays in lockstep with).

**Tech Stack:** Rust, wgpu 26.0.1, engine_rules step + cascade + view-fold infrastructure.

**Architectural Impact Statement (P8):**

- **P1 (compiler-first):** untouched — no new hand-written rule logic. Wires the existing emitted `engine_rules::step::step` into `engine_gpu`.
- **P2 (schema-hash):** untouched — no SoA / event variant / scoring contract change.
- **P3 (cross-backend parity):** **THIS PLAN IS WHAT MAKES P3 VERIFIABLE.** Without runtime execution, Stream C's parity tests can't run; any "parity" claim is vacuous. After this plan, `parity_with_cpu` becomes meaningful.
- **P5 (deterministic RNG):** untouched at the wiring level. Once runtime works, the RNG cross-backend test becomes verifiable for the first time post-T16.
- **P6 (events as mutation channel):** preserved — `engine_rules::step::step` is the canonical event-folded step.
- **P10 (no runtime panic):** **THIS PLAN RESOLVES A LATENT P10 VIOLATION.** Calling `step_batch` today hits `unimplemented!()` — that's a deterministic-path panic that should never have shipped. T15+T16+Stream A landed clean compiles but never proved runtime. Closing this is non-negotiable.
- **P11 (reduction determinism):** untouched at the wiring level.
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

**Sequencing:** This plan is the prerequisite to Stream C (GPU test port). After it lands, Stream C Task 1 (`parity_with_cpu`) becomes feasible — the CPU forward inside `step_batch` actually runs, so byte-equality holds against a CPU `engine_rules::step::step` reference.

---

## What's broken (precise inventory)

`engine::step::step` was deleted in Plan B1' Task 11 and replaced with an `unimplemented!()` stub:

```rust
// crates/engine/src/step.rs:25
pub fn step<B: PolicyBackend, V>(...) {
    unimplemented!("engine::step::step is DELETED (Plan B1' Task 11). \
                   Re-enable after engine_rules::step::step is emitted.")
}
```

`engine_gpu::lib.rs` calls this stub at three sites:

- **Line 279** — `#[cfg(not(feature = "gpu"))]` `ComputeBackend::step` stub. CPU-only build crashes if anything calls `step()`.
- **Line 555** — `step_batch`'s resident-init-failure fallback. Crashes if the resident init ever fails.
- **Line 588** — `step_batch`'s per-tick CPU forward (added in T15+ to keep state authoritative until each emitted WGSL body lands). Crashes on every tick.

`engine_rules::step::step` is the live replacement (in `crates/engine_rules/src/step.rs:41`) with this signature:

```rust
pub fn step<CB, B>(
    backend: &mut CB,
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing<Event>,
    views:   &mut ViewRegistry,
    policy:  &B,
    cascade: &CascadeRegistry<Event, ViewRegistry>,
    debug:   &DebugConfig,
) where
    CB: ComputeBackend<Event = Event, Views = ViewRegistry>,
    B:  PolicyBackend,
```

`GpuBackend`'s ComputeBackend impl declares `type Views = ()` (lib.rs:308 for the no-gpu impl, lib.rs:1081 for the gpu impl). Trait-level mismatch.

---

## File Structure

- Modify: `crates/engine_gpu/src/lib.rs` — `ComputeBackend::Views` type alias, `step_batch` signature, 3 `engine::step::step` call sites, the `ComputeBackend::step` impl that routes through `step_batch`.
- (Maybe) Modify: `crates/engine_gpu/src/snapshot.rs` and `crates/engine_gpu/src/backend/*.rs` if anything depends on `Self::Views = ()`.
- (Maybe) Modify: external callers of `step_batch` (xtask scenario runners, etc.) if the signature change is breaking.

---

### Task 1: Survey the breakage before touching code

**Files:**
- Read-only inventory.

The signature change is invasive. Before editing, identify every site that depends on `GpuBackend::Views = ()` so the migration is mechanical.

- [ ] **Step 1: Find every `GpuBackend` `step_batch` call site**

Run: `grep -rn 'step_batch(' crates/ --include='*.rs' | grep -v 'crates/engine_gpu/src/lib.rs'`

Expected: `xtask` scenario runners + any test harnesses. Capture the count.

- [ ] **Step 2: Find every reference to `GpuBackend::Views`**

Run: `grep -rn 'GpuBackend.*Views\|::Views = ()' crates/ --include='*.rs'`

- [ ] **Step 3: Find every `engine::step::step` call in the workspace**

Run: `grep -rn 'engine::step::step' crates/ --include='*.rs' | grep -v -E 'crates/engine/src/step\.rs|comments|//'`

- [ ] **Step 4: Confirm `engine_rules::step::step` works**

Run: `cargo test -p engine --test debug_agent_history --no-run` — confirm an existing caller compiles. The test bodies show the working call shape.

- [ ] **Step 5: Document the inventory**

Write the counts + sites to a scratch comment in this plan's "Survey results" section (below). Do not commit yet.

#### Survey results

(Filled in by Task 1 execution.)

---

### Task 2: Migrate `GpuBackend::Views` from `()` to `ViewRegistry`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs:308` (no-gpu-feature `impl ComputeBackend for GpuBackend`)
- Modify: `crates/engine_gpu/src/lib.rs:1081` (gpu-feature `impl ComputeBackend for GpuBackend`)

- [ ] **Step 1: Add `engine_rules::views::ViewRegistry` import to lib.rs**

At the top of `crates/engine_gpu/src/lib.rs`, add (next to the existing `use engine::{...}` block):

```rust
use engine_rules::views::ViewRegistry;
```

(Verify the path — if it's at `engine::view::ViewRegistry` instead, use that. The existing test caller in `crates/engine/tests/debug_agent_history.rs:3` uses `engine_rules::views::ViewRegistry`.)

- [ ] **Step 2: Change both `ComputeBackend` impls**

In both impl blocks (no-gpu at line 306, gpu at line 1078), change:

```rust
type Views = ();
```

to:

```rust
type Views = ViewRegistry;
```

- [ ] **Step 3: Update the `step()` impl signatures**

The trait `ComputeBackend::step` takes `views: &mut Self::Views`. Now that `Self::Views = ViewRegistry`, the param is `&mut ViewRegistry` in both impls. Change `_views: &mut Self::Views` → `views: &mut Self::Views` (drop the underscore — we're going to use it).

- [ ] **Step 4: Compile-check (expect failures we'll fix in Task 3)**

Run: `cargo build --workspace`

Expected: failures at sites that depend on `Views = ()`. Capture the error inventory; Task 3 addresses each.

If the only error is "no_gpu impl's step() body still calls `engine::step::step`", proceed. If errors propagate to xtask or other crates, list them in this task's "open errors" section before continuing.

#### Open errors discovered in Step 4

(Filled in by Task 2 execution.)

---

### Task 3: Replace `engine::step::step` calls with `engine_rules::step::step`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` — 3 call sites (~lines 279, 555, 588)

The new call shape per `crates/engine/tests/debug_agent_history.rs:62`:

```rust
engine_rules::step::step(
    &mut engine_rules::backend::SerialBackend,
    state,
    scratch,
    events,
    views,                    // &mut ViewRegistry — threaded through
    policy,
    cascade,                  // &CascadeRegistry<Event, ViewRegistry>
    &engine::debug::DebugConfig::default(),
);
```

Note: `cascade` parameter type changes from `&CascadeRegistry<Event, ()>` to `&CascadeRegistry<Event, ViewRegistry>`. This is a downstream signature shift — `step_batch` and `ComputeBackend::step` must take the same generic.

- [ ] **Step 1: Update `step_batch` signature**

`crates/engine_gpu/src/lib.rs:541-548`:

```rust
pub fn step_batch<B: PolicyBackend>(
    &mut self,
    state:    &mut SimState,
    scratch:  &mut SimScratch,
    events:   &mut EventRing<Event>,
    views:    &mut ViewRegistry,                                   // NEW
    policy:   &B,
    cascade:  &CascadeRegistry<Event, ViewRegistry>,               // CHANGED from <Event, ()>
    n_ticks:  u32,
) {
```

- [ ] **Step 2: Update the gpu-feature `ComputeBackend::step` body (line ~1083)**

The Stream A commit (6b081acf) routes `step()` through `step_batch(n=1)`. Update the call to pass `views`:

```rust
self.step_batch(state, scratch, events, views, policy, cascade, 1);
```

(Trait method already has `views: &mut Self::Views = &mut ViewRegistry` from Task 2 Step 3.)

- [ ] **Step 3: Replace line 588 (per-tick CPU forward)**

Inside `step_batch`'s tick body, change:

```rust
engine::step::step(state, scratch, events, policy, cascade);
```

to:

```rust
engine_rules::step::step(
    &mut engine_rules::backend::SerialBackend,
    state,
    scratch,
    events,
    views,
    policy,
    cascade,
    &engine::debug::DebugConfig::default(),
);
```

Note: this passes `&mut engine_rules::backend::SerialBackend` — the **CPU reference path** runs alongside the GPU dispatcher. The GPU's SCHEDULE-loop has already recorded its dispatches into `encoder` (writes to GPU memory); the CPU step then re-runs the same tick on the same `state` and overwrites whatever the GPU wrote, keeping CPU as authoritative. This is the "CPU forward" the plan documented — now actually executable.

- [ ] **Step 4: Replace line 555 (resident-init-failure fallback)**

```rust
for _ in 0..n_ticks {
    engine_rules::step::step(
        &mut engine_rules::backend::SerialBackend,
        state, scratch, events, views, policy, cascade,
        &engine::debug::DebugConfig::default(),
    );
}
```

- [ ] **Step 5: Replace line 279 (no-gpu-feature stub)**

In the `#[cfg(not(feature = "gpu"))]` `ComputeBackend::step` impl, replace `engine::step::step(...)` with the same `engine_rules::step::step(...)` invocation. The no-gpu `GpuBackend` is just a CPU forwarder; the SerialBackend inner CB is correct for it.

- [ ] **Step 6: Compile-check**

Run: `cargo build --workspace`
Run: `cargo build -p engine_gpu --features gpu`

Both must be clean.

If errors remain, capture in "open errors" — likely candidates: `cascade` generic-arg mismatch in xtask scenario callers, `step_batch` external callers needing the new `views` arg.

#### Open errors discovered in Step 6

(Filled in by Task 3 execution.)

---

### Task 4: Update external `step_batch` callers

**Files:**
- Likely modify: `crates/xtask/src/...` — scenario runners that call `step_batch`
- Possibly modify: any test harness that called `step_batch` directly (the cfg-gated tests don't count; they stay gated)

- [ ] **Step 1: For every site identified in Task 1 Step 1**

Add a `let mut views = ViewRegistry::default();` at the call site (or thread one in from the caller's caller) and pass `&mut views` as the new `views` argument.

- [ ] **Step 2: Compile-check**

Run: `cargo build --workspace`
Expected: clean.

- [ ] **Step 3: Confirm `cargo test --workspace` (default features) is no worse than the pre-T16 baseline**

Run: `cargo test --workspace 2>&1 | grep -E 'FAILED|test result' | head -20`

The 3 pre-existing engine probe-harness failures (Plan B1' Task 11 — different unimplemented stub) stay. Other tests should pass at least as well as on `b9163e66`.

If new failures appear, that's a bug in Task 3's wiring — diagnose and fix before continuing.

---

### Task 5: Runtime smoke test — prove `step_batch` actually executes

This is the gate that proves the integration works. Without this, the plan is "compile clean, runtime unverified" — exactly the failure mode this plan exists to fix.

**Files:**
- Create: `crates/engine_gpu/tests/step_batch_runtime_smoke.rs` (new test, NOT cfg-gated, runs on default features)

```rust
//! Runtime gate for engine_gpu::step_batch. Must pass for any future
//! GPU work to be meaningful — without this, "step_batch works" is
//! vacuously true at compile time but false at runtime (Plan
//! 2026-04-28-step-batch-runtime-fix discovery).

use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_data::config::Config;
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

#[test]
fn step_batch_n1_runs_without_panic() {
    let mut state = SimState::new(8, 42);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 0.0),
        ..Default::default()
    });

    let mut scratch = SimScratch::new(state.agent_cap());
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::default();

    let mut gpu = match engine_gpu::GpuBackend::new() {
        // gpu-feature OFF: GpuBackend is the Phase 0 stub, step_batch
        // forwards via the no-gpu impl which now goes through
        // engine_rules::step::step. Should not panic.
        backend => backend,
    };

    let tick0 = state.tick;
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 1);
    assert_eq!(state.tick, tick0 + 1, "step_batch(1) advanced tick");
}

#[test]
fn step_batch_n5_advances_tick_count() {
    let mut state = SimState::new(8, 42);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 0.0),
        ..Default::default()
    });

    let mut scratch = SimScratch::new(state.agent_cap());
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::default();

    let mut gpu = engine_gpu::GpuBackend::new();
    let tick0 = state.tick;
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 5);
    assert_eq!(state.tick, tick0 + 5, "step_batch(5) advanced tick by 5");
}
```

- [ ] **Step 1: Verify imports**

The exact paths above (`engine::policy::utility::UtilityBackend`, `engine_data::entities::CreatureType`, `glam::Vec3`, etc.) are likely the right paths — verify each via `grep -rn` before committing. If `UtilityBackend` lives elsewhere or `AgentSpawn` has different field names, adjust.

- [ ] **Step 2: For the gpu-feature ON case, gate gracefully**

`GpuBackend::new()` in the gpu-feature ON build returns `Result<Self, GpuInitError>`. The test as-written assumes the no-gpu-feature path. If running under `--features gpu`, wrap with:

```rust
let mut gpu = match engine_gpu::GpuBackend::new() {
    Ok(g) => g,
    Err(_) => { eprintln!("skipping: no gpu adapter"); return; }
};
```

(`#[cfg(feature = "gpu")]` to discriminate.)

- [ ] **Step 3: Run the test under both feature configs**

```bash
cargo test -p engine_gpu --test step_batch_runtime_smoke
cargo test -p engine_gpu --features gpu --test step_batch_runtime_smoke
```

Expected: BOTH pass. Default-features runs the no-gpu CPU path (engine_rules::step). gpu-features either runs the SCHEDULE-loop dispatcher (if adapter available) or skips gracefully.

If either fails: a runtime bug. Diagnose:
- Panic in `engine_rules::step`? Look at the exact phase that panics.
- Panic in the SCHEDULE-loop dispatch? Look at which kernel's `record()` blew up.
- Panic in `views.fold_all`? `ViewRegistry::default()` may not be empty-safe — wrap with `cascade.run_fixed_point` semantics.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/tests/step_batch_runtime_smoke.rs crates/engine_gpu/src/lib.rs <other-callers>
git commit -m "fix(engine_gpu): step_batch actually runs end-to-end

Migrates engine_gpu::step_batch from the panicking engine::step::step
(unimplemented! stub from Plan B1' Task 11) to the live
engine_rules::step::step. Threads &mut ViewRegistry through
ComputeBackend::Views, step_batch, and the cascade generic arg.

Closes a latent P10 violation: post-T16, step_batch's per-tick CPU
forward called engine::step::step which is unimplemented!() —
runtime panic on every call. T15+T16+Stream A landed clean compiles
but never proved runtime. The new step_batch_runtime_smoke test is
the gate that catches this regression class.

Unblocks Stream C Task 1 (parity_with_cpu) — now feasible because
the CPU forward inside step_batch actually runs.
"
```

---

### Task 6: Closeout

- [ ] **Step 1: Update `pending-decisions.md`**

Replace the Stream C entry's "awaiting user" status with "step_batch runtime fix landed (commits …); Stream C unblocked." Add this plan's outcome to the dispatch-emit retrospective entry — T15+T16+Stream A premature-done was a real failure of the close-criteria. Future plan close-out must include a runtime smoke test, not just a compile pass.

- [ ] **Step 2: Update the dispatch-emit plan's AIS retrospective section**

Open `docs/superpowers/plans/2026-04-26-kernel-dispatch-emit-impl.md` and add a "Lessons learned" section noting:
- Critics passed on T16 / Stream A close because they check architectural compliance, not buildability or runtime behavior.
- Compile-clean ≠ runtime-clean. Future plans MUST include at least one runtime smoke test before close.
- The close-criteria template needs a "runtime gate" line; update `docs/architecture/plan-template-ais.md` accordingly.

- [ ] **Step 3: Update plan-template-ais.md**

Add a "Runtime gate" close-criterion to the template so future plans default to including one.

- [ ] **Step 4: Final commit**

```bash
git add docs/
git commit -m "docs: capture step_batch runtime-fix retrospective + plan-template runtime-gate
"
```

---

## Final verification

After Tasks 1-6 the following invariants hold:

1. `cargo test -p engine_gpu --test step_batch_runtime_smoke` passes (default features).
2. `cargo test -p engine_gpu --features gpu --test step_batch_runtime_smoke` passes (or gracefully skips on no-adapter CI).
3. `cargo build --workspace` clean.
4. `cargo build -p engine_gpu --features gpu` clean.
5. The 3 pre-existing engine probe failures remain (Plan B1' Task 11 work — different stub).
6. **Stream C Task 1 (`parity_with_cpu`) is now feasible.** The CPU forward inside `step_batch` runs, so byte-equality holds against a CPU-stepped reference.
7. `pending-decisions.md` reflects the unblock; the dispatch-emit retrospective documents the lesson.

---

## What this plan deliberately does NOT do

- **Does NOT remove the CPU forward inside `step_batch`'s tick body.** The CPU forward stays authoritative until the GPU bodies (Stream B's remaining work) are verified by parity tests (Stream C). Removing it here would silently break simulation state.
- **Does NOT touch the SCHEDULE-loop dispatcher.** The dispatch encoder still records every emitted kernel; their writes are simply overwritten by the CPU forward that follows.
- **Does NOT port the cfg-gated tests.** That's Stream C's job.
- **Does NOT refactor `step_batch`'s shape.** Minimal-diff: signature gets one new arg (`views`), three call sites get rewritten, one `Views` type alias flips. Anything beyond is scope creep.
