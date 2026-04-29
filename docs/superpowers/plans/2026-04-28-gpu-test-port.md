# GPU Test Port Implementation Plan (Stream C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable the 93 cfg-gated tests under `crates/engine_gpu/tests/` by rewriting their bodies against the post-T16 SCHEDULE-loop dispatcher surface. Reactivated parity tests become the gates that let Stream B's WGSL bodies be verified for correctness as they land.

**Architecture:** Tests are file-scope-gated via `#![cfg(any())]` after the engine_gpu repair (commit `984a6e5a`). They reference deleted symbols (`crate::mask::cpu_mask_bitmap`, `crate::physics::run_batch_resident`, `state.views.*`, `ChronicleRing` layout, `crate::scoring::cpu_score_outputs`). The SCHEDULE-loop surface exposes a smaller, clearer API: `engine_gpu_rules::<kernel>::Kernel::new(&device)` + `BindingSources` aggregate + `kernel.bind(&sources, &cfg_buf).record(...)` + readback. End-to-end parity tests get even simpler — just call `gpu_backend.step_batch(...)` against a CPU-stepped reference.

**Tech Stack:** Rust, wgpu 26.0.1, engine_gpu_rules SCHEDULE/Kernel API.

**Sequencing:** This plan is the gate for Stream B. Execute Task 1 (parity helper + `parity_with_cpu` port) FIRST so subsequent Stream B WGSL-fill tasks can verify against a live parity test. Tasks 2-5 then interleave with Stream B per the join points listed in **Stream B/C interleave** below.

**Architectural Impact Statement (P8):**
- **P1 (compiler-first):** untouched — tests don't add rule logic.
- **P2 (schema-hash):** untouched.
- **P3 (cross-backend parity):** **this plan IS the verification gate.** Stream B fills GPU bodies; Stream C's parity tests prove they match CPU.
- **P5 (deterministic RNG):** the rng_cross_backend test reactivates here, asserting `per_agent_u32` and `per_agent_u32_glsl` produce identical streams.
- **P6, P10, P11:** untouched.
- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

---

## Test inventory (93 tests across 29 files)

| Tier | Files | Purpose | Priority |
|---|---|---|---|
| 1 (parity) | parity_with_cpu, physics_parity, cascade_parity, view_parity, topk_view_parity, spatial_parity | byte-equal CPU vs GPU on canonical fixtures | **gate** |
| 1 (rng) | (file under engine, not engine_gpu — verify) | per_agent_u32 host vs GLSL byte-equal | gate |
| 2 (smoke) | step_batch_smoke, alive_bitmap_pack, async_smoke, snapshot_double_buffer, tick_advance_is_gpu_resident | "no panic, sane output" | high |
| 3 (correctness) | cold_state_gold_transfer, cold_state_memory, cold_state_standing, indirect_cascade_converges, event_ring_parity, batch_iter_cap_convergence, physics_run_batch_resident_smoke, chronicle_batch_path, chronicle_isolated_smoke, chronicle_batch_probe | specific behaviors | high |
| 4 (perf) | gpu_step_perf, perf_n100, chronicle_batch_perf_n100k, chronicle_batch_perf_n200k, chronicle_batch_stress_n20k, chronicle_drain_perf, spatial_resident | timing benchmarks | low |

`spatial_resident` may be Tier 3 — verify when reading.

---

## File Structure

- New: `crates/engine_gpu/tests/common/mod.rs` — shared test helpers (`assert_cpu_gpu_parity`, `make_gpu_backend`, `read_back_agents`, etc.). Tests opt in via `mod common;`.
- Rewrite (don't replace) each test file: remove the `#![cfg(any())]` banner, restore body using the SCHEDULE-loop API.

---

## Stream B/C interleave

To prevent Stream B from running blind, Stream C lands these gates ahead of the matching Stream B tasks:

| Stream C task | Lands before Stream B task | Why |
|---|---|---|
| C1 (parity_with_cpu) | B1 (physics.wgsl) | end-to-end gate catches any kernel regression |
| C2 (per-kernel Tier 1) | B4-B7 (Bucket R kernels) | localizes which kernel diverged |
| C3 (smoke) | B8 (Stream B closeout) | "GPU compiles + runs without panic" gate |
| C4-C5 | after Stream B closeout | correctness + perf are downstream of bodies-correct |

In practice this means executing in the order: **C1, B1, B2, B3, C2, B4, B5, B6, B7, C3, B8, C4, C5**.

---

### Task 1: Build `parity_helper` + port `parity_with_cpu`

**Files:**
- Create: `crates/engine_gpu/tests/common/mod.rs` (test-only helper module)
- Modify: `crates/engine_gpu/tests/parity_with_cpu.rs` (remove `#![cfg(any())]`, rewrite body)

The parity helper:

```rust
// crates/engine_gpu/tests/common/mod.rs
use engine::backend::ComputeBackend;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::policy::PolicyBackend;
use engine::state::SimState;
use engine::step::SimScratch;
use engine_data::events::Event;

/// Runs `n_ticks` of stepping on two cloned states — one through the
/// CPU `engine::step::step` reference, one through `gpu_backend.step_batch`.
/// Asserts the post-step `SimState` SoA + emitted `EventRing` are byte-
/// equal.
pub fn assert_cpu_gpu_parity<P: PolicyBackend>(
    state: &SimState,
    policy: &P,
    cascade: &CascadeRegistry<Event, ()>,
    n_ticks: u32,
) {
    let mut cpu_state = state.clone();
    let mut cpu_scratch = SimScratch::new(state.agent_cap());
    let mut cpu_events = EventRing::<Event>::with_cap(4096);
    for _ in 0..n_ticks {
        engine::step::step(&mut cpu_state, &mut cpu_scratch, &mut cpu_events, policy, cascade);
    }

    let mut gpu_backend = engine_gpu::GpuBackend::new()
        .expect("gpu init — skip test if no adapter");
    let mut gpu_state = state.clone();
    let mut gpu_scratch = SimScratch::new(state.agent_cap());
    let mut gpu_events = EventRing::<Event>::with_cap(4096);
    gpu_backend.step_batch(&mut gpu_state, &mut gpu_scratch, &mut gpu_events, policy, cascade, n_ticks);

    assert_eq!(cpu_state.agents_soa(), gpu_state.agents_soa(),
        "agent SoA diverged after {n_ticks} ticks");
    assert_eq!(cpu_events.events_replayable_iter().collect::<Vec<_>>(),
               gpu_events.events_replayable_iter().collect::<Vec<_>>(),
        "event stream diverged after {n_ticks} ticks");
}

/// Build a gpu backend that gracefully skips the test if no adapter
/// is available (CI on software adapters).
pub fn try_gpu_backend() -> Option<engine_gpu::GpuBackend> {
    engine_gpu::GpuBackend::new().ok()
}
```

(Exact accessors may need adjustment — `agents_soa()` may be named differently; `events_replayable_iter` may be the wrong method. Trace via `grep` while writing.)

- [ ] **Step 1: Read the original parity_with_cpu.rs body**

Run: `cat crates/engine_gpu/tests/parity_with_cpu.rs | head -80`

Capture the test setup pattern (which scenario it builds, how many ticks, what assertions).

- [ ] **Step 2: Read engine::step::step signature and SimState API**

Run: `grep -n 'pub fn step<\|pub fn agents_soa\|pub fn events_replayable_iter\|pub fn agents_alive' crates/engine/src/{step,state}.rs crates/engine/src/event/ring.rs 2>&1 | head -10`

Note the actual method names. The helper above uses placeholders; use the real ones.

- [ ] **Step 3: Write `crates/engine_gpu/tests/common/mod.rs`**

Use the helper above as a template, with the real method names from Step 2. Pay attention to:
- `SimScratch::new(agent_cap)` — verify the constructor.
- `EventRing::with_cap(N)` — confirm the API.
- `events_replayable_iter` vs alternatives — pick the closest equivalent.

- [ ] **Step 4: Rewrite parity_with_cpu.rs body**

Remove the `#![cfg(any())]` line. Replace the entire test body with calls to the parity helper. Preserve the original test names (e.g., `parity_with_cpu_n4_t10`).

Pattern for each test:
```rust
mod common;
use common::assert_cpu_gpu_parity;

#[test]
fn parity_with_cpu_n4_t10() {
    let Some(_gpu) = common::try_gpu_backend() else {
        eprintln!("skipping: no gpu adapter");
        return;
    };
    let (state, policy, cascade) = build_n4_fixture();  // existing helper, find via grep
    assert_cpu_gpu_parity(&state, &policy, &cascade, 10);
}
```

- [ ] **Step 5: Run the test (it should fail — Stream B hasn't filled bodies yet)**

Run: `cargo test -p engine_gpu --features gpu --test parity_with_cpu`

Expected outcome at THIS point in the sequence: **the test compiles** but **fails** because the GPU WGSL bodies are still stubs. That's the right state — Stream B Task 1 lands physics.wgsl and the test starts gating that.

If the test panics during compile or before executing assertions, that's a real bug in the rewrite — fix it.

- [ ] **Step 6: If GPU init fails (no adapter), confirm graceful skip**

The `try_gpu_backend` pattern above returns `None` on no-adapter. The test should print `skipping: no gpu adapter` and return, NOT fail. Verify on a software-adapter CI shape (or simulate via `WGPU_BACKEND=empty cargo test ...` if available).

- [ ] **Step 7: Commit**

```bash
git add crates/engine_gpu/tests/common/ crates/engine_gpu/tests/parity_with_cpu.rs
git commit -m "test(engine_gpu): port parity_with_cpu to SCHEDULE-loop API (Stream C Task 1)"
```

---

### Task 2: Port Tier 1 per-kernel parity tests

**Files:** Modify (remove `#![cfg(any())]`, rewrite):
- `crates/engine_gpu/tests/physics_parity.rs`
- `crates/engine_gpu/tests/cascade_parity.rs`
- `crates/engine_gpu/tests/view_parity.rs`
- `crates/engine_gpu/tests/topk_view_parity.rs`
- `crates/engine_gpu/tests/spatial_parity.rs`

These per-kernel tests pre-T16 ran ONE kernel, read back its output buffer, and compared against a CPU reference computed inline. Post-T16 they need:
1. Build the kernel via `engine_gpu_rules::<kernel>::<Kernel>::new(&device)`.
2. Construct `BindingSources` aggregate (resident, pingpong, pool, transient, external) — see `step_batch`'s body for the construction pattern.
3. Build `cfg` via `kernel.build_cfg(state)`, upload to a uniform buffer.
4. Call `kernel.bind(&sources, &cfg_buf)` then `kernel.record(&device, &mut encoder, &bindings, agent_cap)`.
5. Submit the encoder and readback the relevant buffer.
6. Compute CPU reference on the same `state` and compare.

A second helper makes sense:

```rust
// In tests/common/mod.rs, add:

/// Construct a minimal BindingSources for a SINGLE-kernel test.
/// All buffers but the ones the test cares about point at a 16-byte
/// placeholder buffer — the test asserts on whichever output buffer
/// the kernel writes into.
pub struct SingleKernelHarness { ... }
impl SingleKernelHarness {
    pub fn new(device: &wgpu::Device, state: &SimState) -> Self { ... }
    pub fn binding_sources(&self) -> BindingSources<'_> { ... }
    pub fn agent_cap(&self) -> u32 { ... }
}
```

- [ ] **Step 1: Read each test pre-T16 to understand what it asserted**

Run: `git show 4474566c~1:crates/engine_gpu/tests/physics_parity.rs | head -60` for each.

- [ ] **Step 2: Add `SingleKernelHarness` to common/mod.rs**

Match the construction pattern in `step_batch`. Use a SHARED `engine_gpu::GpuBackend` and reach into its `resident.path_ctx` / `resident.pingpong_ctx` / `resident.pool` for the `BindingSources` fields — that gives realistic buffers without re-implementing them.

- [ ] **Step 3: Port physics_parity.rs**

The test should: construct state, call PhysicsKernel for one tick, readback agents buffer, compare to `engine::step::step_phase_4a` (the CPU phase) on the same state.

- [ ] **Step 4: Port cascade_parity.rs**

Cascade uses the FixedPoint dispatch op. The test should call `dispatch()` with `DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 }` and verify event-ring convergence matches CPU `cascade::run_fixed_point` output.

- [ ] **Step 5: Port view_parity.rs and topk_view_parity.rs**

These run a single FoldKernel and compare its output buffer to the CPU view-fold (in `engine::view`).

- [ ] **Step 6: Port spatial_parity.rs**

Three kernels in sequence (hash → kin → engagement). Compare each output to CPU spatial reference.

- [ ] **Step 7: Run all 5 tests**

Run: `cargo test -p engine_gpu --features gpu --test physics_parity --test cascade_parity --test view_parity --test topk_view_parity --test spatial_parity`

Expected: the tests compile and FAIL (Stream B Tasks 4-7 haven't yet filled the corresponding WGSL bodies). FAIL is the correct state — these tests will pass as Stream B's R-bucket tasks land.

- [ ] **Step 8: Commit**

```bash
git add crates/engine_gpu/tests/{common,physics_parity,cascade_parity,view_parity,topk_view_parity,spatial_parity}.rs
git commit -m "test(engine_gpu): port 5 Tier-1 per-kernel parity tests (Stream C Task 2)"
```

---

### Task 3: Port Tier 2 smoke tests

**Files:** Remove `#![cfg(any())]`, rewrite as step_batch smoke:
- `crates/engine_gpu/tests/step_batch_smoke.rs`
- `crates/engine_gpu/tests/alive_bitmap_pack.rs`
- `crates/engine_gpu/tests/async_smoke.rs`
- `crates/engine_gpu/tests/snapshot_double_buffer.rs`
- `crates/engine_gpu/tests/tick_advance_is_gpu_resident.rs`

These are "exercise the path, assert no panic + tick counter advances" tests. The new shape is small:

```rust
mod common;

#[test]
fn step_batch_advances_tick_n10() {
    let Some(mut gpu) = common::try_gpu_backend() else { return };
    let mut state = make_smoke_fixture(/* n=4 */);
    let mut scratch = SimScratch::new(state.agent_cap());
    let mut events = EventRing::<Event>::with_cap(4096);
    let policy = engine::policy::DefaultPolicy::default();
    let cascade = engine::cascade::CascadeRegistry::default();

    let tick0 = state.tick;
    gpu.step_batch(&mut state, &mut scratch, &mut events, &policy, &cascade, 10);
    assert_eq!(state.tick, tick0 + 10);
}
```

- [ ] **Step 1: For each smoke test, identify what specifically it smokes**

Read the original via `git show 4474566c~1:crates/engine_gpu/tests/<name>.rs`.

- [ ] **Step 2: Rewrite each as a step_batch + tick-advance assertion**

Strip per-kernel buffer plumbing — smoke is end-to-end. Replace the original assertions with `step_batch` invocation and minimal output checks.

- [ ] **Step 3: Run all 5**

Run: `cargo test -p engine_gpu --features gpu --test step_batch_smoke --test alive_bitmap_pack --test async_smoke --test snapshot_double_buffer --test tick_advance_is_gpu_resident`

Expected: PASS even before Stream B completes — the CPU forward inside step_batch keeps tick state correct.

- [ ] **Step 4: Commit**

```bash
git add crates/engine_gpu/tests/{step_batch_smoke,alive_bitmap_pack,async_smoke,snapshot_double_buffer,tick_advance_is_gpu_resident}.rs
git commit -m "test(engine_gpu): port 5 Tier-2 smoke tests (Stream C Task 3)"
```

---

### Task 4: Port Tier 3 correctness tests

**Files:** Remove `#![cfg(any())]`, rewrite:
- `crates/engine_gpu/tests/cold_state_gold_transfer.rs`
- `crates/engine_gpu/tests/cold_state_memory.rs`
- `crates/engine_gpu/tests/cold_state_standing.rs`
- `crates/engine_gpu/tests/indirect_cascade_converges.rs`
- `crates/engine_gpu/tests/event_ring_parity.rs`
- `crates/engine_gpu/tests/batch_iter_cap_convergence.rs`
- `crates/engine_gpu/tests/physics_run_batch_resident_smoke.rs`
- `crates/engine_gpu/tests/chronicle_batch_path.rs`
- `crates/engine_gpu/tests/chronicle_isolated_smoke.rs`
- `crates/engine_gpu/tests/chronicle_batch_probe.rs`
- `crates/engine_gpu/tests/gpu_prefix_scan.rs`

Per test:

- [ ] **Step 1-N: Per-test rewrite**

For cold_state_*: use parity helper but with fixtures that exercise the cold-state replay path.

For indirect_cascade_converges, batch_iter_cap_convergence: test that step_batch's FixedPoint Physics arm converges within the iter cap.

For event_ring_parity: assert GPU-emitted events match CPU-emitted events byte-equal (subset of parity_with_cpu).

For chronicle_batch_*: chronicles are non-replayable telemetry; the assertions should be on tail/length rather than content.

For gpu_prefix_scan: this might be testing a primitive that no longer exists post-T16. If the prefix-scan was inside the deleted physics.rs, mark this test as `#[ignore = "prefix-scan kernel retired in T16; reactivate if reintroduced"]` rather than rewriting.

- [ ] **Step 2: Run all**

Run: `cargo test -p engine_gpu --features gpu --test cold_state_gold_transfer --test cold_state_memory --test cold_state_standing --test indirect_cascade_converges --test event_ring_parity --test batch_iter_cap_convergence --test physics_run_batch_resident_smoke --test chronicle_batch_path --test chronicle_isolated_smoke --test chronicle_batch_probe --test gpu_prefix_scan`

Expected: pass (after Stream B has landed bodies they exercise).

- [ ] **Step 3: Commit**

```bash
git add crates/engine_gpu/tests/{cold_state_gold_transfer,cold_state_memory,cold_state_standing,indirect_cascade_converges,event_ring_parity,batch_iter_cap_convergence,physics_run_batch_resident_smoke,chronicle_batch_path,chronicle_isolated_smoke,chronicle_batch_probe,gpu_prefix_scan}.rs
git commit -m "test(engine_gpu): port 11 Tier-3 correctness tests (Stream C Task 4)"
```

---

### Task 5: Port Tier 4 perf tests + final closeout

**Files:** Remove `#![cfg(any())]`, rewrite:
- `crates/engine_gpu/tests/gpu_step_perf.rs`
- `crates/engine_gpu/tests/perf_n100.rs`
- `crates/engine_gpu/tests/chronicle_batch_perf_n100k.rs`
- `crates/engine_gpu/tests/chronicle_batch_perf_n200k.rs`
- `crates/engine_gpu/tests/chronicle_batch_stress_n20k.rs`
- `crates/engine_gpu/tests/chronicle_drain_perf.rs`
- `crates/engine_gpu/tests/spatial_resident.rs`

Per-test:

- [ ] **Step 1-N: Rewrite each as a step_batch perf measurement**

Use `std::time::Instant` directly; the tests don't need a dedicated benchmark framework. Rewrite to `gpu.step_batch(..., n_ticks=K)` and measure wall-clock. `#[ignore]` the perf tests by default so they don't run in CI; opt-in via `cargo test -- --ignored`.

- [ ] **Step 2: Run with --ignored**

Run: `cargo test -p engine_gpu --features gpu -- --ignored --test gpu_step_perf` (etc.)
Expected: tests run; numbers are informational, not gating.

- [ ] **Step 3: Update pending-decisions.md**

Edit `docs/superpowers/dag/pending-decisions.md`. Replace the "Stream C" section's status from "awaiting user" to "Stream C closed (commits …)".

If Stream B has also closed by this point, the entire "GPU execution recovery" entry can be removed (the work is done; git history retains the deliberation).

- [ ] **Step 4: Final commit**

```bash
git add crates/engine_gpu/tests/{gpu_step_perf,perf_n100,chronicle_batch_perf_n100k,chronicle_batch_perf_n200k,chronicle_batch_stress_n20k,chronicle_drain_perf,spatial_resident}.rs docs/superpowers/dag/pending-decisions.md
git commit -m "test(engine_gpu): port 7 Tier-4 perf tests + close Stream C (Stream C Task 5)"
```

---

## Final verification

After all 5 tasks (and their interleaved Stream B counterparts), the following invariants hold:

1. No `#![cfg(any())]` banners remain in `crates/engine_gpu/tests/*.rs`.
2. `cargo test -p engine_gpu --features gpu --no-run` compiles all 29 test binaries.
3. `cargo test -p engine_gpu --features gpu` passes Tier-1 parity gates, Tier-2 smoke, Tier-3 correctness (perf tests gated `#[ignore]`).
4. `parity_with_cpu` passes — the canonical CPU-vs-GPU byte-equality gate.
5. `cargo build --workspace` clean. `cargo build -p engine_gpu --features gpu` clean.
6. The CPU forward inside `step_batch`'s tick body MAY be removable — once parity passes, a follow-up commit drops it. (NOT scope of this plan; that's a Stream-B-tail or new "step_batch GPU-authoritative" plan.)

---

## What this plan deliberately does NOT do

- **Does NOT rewrite tests against pre-T16 APIs.** They reference deleted symbols; the rewrite uses the SCHEDULE-loop surface.
- **Does NOT remove the CPU forward in `step_batch`.** That's a follow-up after parity passes.
- **Does NOT skip parity by stubbing `assert_eq!`.** If a parity test fails, that's the signal Stream B's WGSL body has a bug — fix the body, not the test.
- **Does NOT port `engine::probe::run_probe` tests.** Those failures are pre-existing Plan B1' Task 11 work, unrelated.
