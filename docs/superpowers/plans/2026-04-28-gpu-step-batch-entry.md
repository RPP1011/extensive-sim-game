# GPU step → step_batch entry point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `engine_gpu::step_batch` to be the `ComputeBackend::step` entry point so every CPU consumer of `step()` exercises the SCHEDULE-loop dispatcher.

**Architecture:** Today `ComputeBackend::step` (gpu-feature ON, at `crates/engine_gpu/src/lib.rs:1083`) is a CPU forward — left honest after T16's hand-written-kernel deletion. `step_batch` (line 541) runs the SCHEDULE loop via `dispatch()` then CPU-forwards inside its tick body to keep state in sync until each emitted WGSL body lands. Stream A reroutes `step()` through `step_batch(..., n_ticks = 1)` so the dispatcher gets exercised on every tick. Simulation correctness is unchanged (the CPU forward inside `step_batch`'s loop body still authoritatively advances state); what changes is that the GPU command encoder runs every tick, surfacing dispatch-side regressions immediately instead of only under batched callers.

**Tech Stack:** Rust, wgpu 26.0.1, engine_gpu_rules SCHEDULE module.

**Architectural Impact Statement (P8):**
- **P1 (compiler-first):** untouched — no new hand-written rule logic. Routes through the existing emitted dispatch.
- **P2 (schema-hash):** untouched — no SoA / event / scoring contract change.
- **P3 (cross-backend parity):** preserved — CPU SerialBackend untouched. GPU side runs the same CPU forward inside `step_batch` it would have run inside `step` directly. Byte equality maintained.
- **P5 (deterministic RNG):** untouched.
- **P6 (events as mutation channel):** untouched.
- **P10 (no runtime panic):** preserved — `step_batch`'s CPU fallback (`if let Err(e) = self.ensure_resident_init(state)`) keeps the deterministic path panic-free.
- **P11 (reduction determinism):** untouched.
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

---

## File Structure

- Modify: `crates/engine_gpu/src/lib.rs` — `ComputeBackend::step` (gpu-feature ON) routes through `step_batch`.

That is the only production-code change. Documentation and follow-up notes (this plan's narrative) live alongside.

---

### Task 1: Make `step()` route through `step_batch(n_ticks=1)`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs:1083-1101` (the gpu-feature ON `impl ComputeBackend for GpuBackend::step` body)

The signature mismatch is small: `step` takes `_views: &mut Self::Views`; `step_batch` doesn't. `Self::Views = ()`, so `_views` is already a unit and dropping it is a no-op. `step_batch` takes `n_ticks: u32`; pass `1`.

- [ ] **Step 1: Read the current step() body to confirm shape**

Run: `sed -n '1080,1102p' crates/engine_gpu/src/lib.rs`

Expected to see (verbatim):

```rust
    type Event = Event;
    type Views = ();

    fn step<B: PolicyBackend>(
        &mut self,
        state:    &mut SimState,
        scratch:  &mut SimScratch,
        events:   &mut EventRing<Self::Event>,
        _views:   &mut Self::Views,
        policy:   &B,
        cascade:  &CascadeRegistry<Self::Event, Self::Views>,
    ) {
        // GPU sync-path was retired in T16 (commit 4474566c) along
        // with every hand-written kernel that orchestrated mask /
        // scoring / cascade / apply / movement. The SCHEDULE-loop in
        // step_batch is the authoritative GPU path going forward.
        // Until step_batch becomes the ComputeBackend::step entry
        // point, this method honestly forwards to the CPU step so the
        // tick still advances correctly. P10-clean: no panic, no
        // unimplemented!(), no stub state.
        engine::step::step(state, scratch, events, policy, cascade);
    }
```

If the body has drifted, stop and surface — the rest of this task is built on this exact shape.

- [ ] **Step 2: Replace the CPU-forward body with a step_batch call**

Replace the body of `step()` (lines 1092-1100) with:

```rust
        // SCHEDULE-loop entry point. Forwards to step_batch(n_ticks=1)
        // so every per-tick consumer of ComputeBackend::step exercises
        // the emitted dispatcher. step_batch internally CPU-forwards
        // inside its tick body until each emitted WGSL body lands
        // (Stream B); when those bodies replace the CPU forward, this
        // path becomes GPU-authoritative without any further change
        // here.
        self.step_batch(state, scratch, events, policy, cascade, 1);
```

The `_views` parameter is dropped — it's `&mut ()` and `step_batch` has no `views` parameter.

Use the Edit tool with the exact `engine::step::step(state, scratch, events, policy, cascade);` line + the full preceding comment block as `old_string`, the new comment + `self.step_batch(...)` call as `new_string`.

- [ ] **Step 3: Run default-features build to confirm no regression**

Run: `cargo build --workspace`

Expected: clean build (no errors, no new warnings).

- [ ] **Step 4: Run gpu-feature build to confirm step() still type-checks**

Run: `cargo build -p engine_gpu --features gpu`

Expected: clean build. The signature compatibility is the only thing this verifies — the runtime behavior is exercised by Step 5.

- [ ] **Step 5: Run default-features test suite to verify SerialBackend unaffected**

Run: `cargo test -p engine`

Expected: same pass count as HEAD before this task. SerialBackend doesn't touch GpuBackend; this verifies we didn't accidentally break a shared dependency.

Run: `cargo test -p engine_gpu`

Expected: pass (no gpu-feature tests run by default; the cfg-gated tests stay gated).

- [ ] **Step 6: Sanity-check the dispatch path is exercised**

Run: `grep -n 'pub fn step_batch' crates/engine_gpu/src/lib.rs`

Confirm `step_batch` is still public (consumers may still call it directly with `n_ticks > 1`).

Run: `grep -n 'self.step_batch(' crates/engine_gpu/src/lib.rs`

Expected: at least one hit — the call we just inserted.

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/src/lib.rs
git commit -m "feat(engine_gpu): ComputeBackend::step routes through step_batch(n=1)

Stream A of GPU execution recovery. step() previously CPU-forwarded
honestly after T16 deleted the hand-written sync-path kernels. Now
forwards to step_batch(n_ticks=1) so every per-tick consumer of
ComputeBackend::step exercises the SCHEDULE-loop dispatcher.

Simulation correctness is unchanged: step_batch's tick body still
CPU-forwards via engine::step::step until each emitted WGSL body
lands (Stream B). When those bodies replace step_batch's CPU
forward, this path becomes GPU-authoritative without any further
change at the step() call site.

Tests: cargo build --workspace clean; cargo build -p engine_gpu
--features gpu clean; cargo test -p engine + cargo test -p engine_gpu
pass with no regression."
```

---

## Final verification

After Task 1 completes, the following invariants hold:

1. `ComputeBackend::step` (gpu-feature ON) routes through `step_batch(state, scratch, events, policy, cascade, 1)`.
2. `cargo build --workspace` is clean.
3. `cargo build -p engine_gpu --features gpu` is clean.
4. `cargo test -p engine` passes with no regression.
5. `cargo test -p engine_gpu` passes (default features) with no regression.
6. The `step_batch` symbol remains `pub` so callers passing `n_ticks > 1` continue to work.
7. Stream A is closed. Stream B (emitted WGSL body fill) and Stream C (test port) become unblocked and parallelizable.

---

## What this plan deliberately does NOT do

- **Drop the CPU forward inside `step_batch`'s tick body.** That's Stream B's responsibility — when each emitted kernel's WGSL body authoritatively mutates state on the GPU, the CPU forward gets replaced with a GPU-resident commit. Doing it here would silently break simulation state until Stream B lands.
- **Re-enable the cfg-gated GPU tests.** That's Stream C. Those tests reference deleted kernel-method surfaces and need rewriting against `BindingSources` / `Kernel::record`.
- **Touch the gpu-feature-OFF `step()`** (lib.rs:270). It's the Phase 0 stub and is already correct.
- **Remove `step_batch` as a public method.** External callers (xtask, scenario harnesses) may still call it with `n_ticks > 1`. Keeping it public costs nothing and preserves their ergonomics.
