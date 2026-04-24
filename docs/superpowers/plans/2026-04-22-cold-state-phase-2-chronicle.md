# GPU Cold-State Replay — Phase 2: Chronicle Stubs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 8 DSL chronicle rules (`chronicle_death`, `chronicle_attack`, `chronicle_engagement`, `chronicle_wound`, `chronicle_break`, `chronicle_rout`, `chronicle_flee`, `chronicle_rally`) to fire on the batch path, with emitted `ChronicleEntry` events flowing through `snapshot()` to the observer.

**Architecture:** Phase 1 investigation revealed all 8 rules are already structured DSL emit-to-chronicle-ring physics rules (no narrative text in DSL bodies — text rendering lives CPU-side in `chronicle.rs::templates` as a post-drain consumer). The scope is NOT per-rule porting; it's (1) verify/enable `@phase(post)` GPU execution in cascade_resident, (2) wire the existing `CascadeResidentCtx::chronicle_ring` into the snapshot pipeline, (3) integration-test end-to-end.

**Tech Stack:** `crates/dsl_compiler` (post-phase emit check), `crates/engine_gpu` (cascade_resident post-phase integration, snapshot chronicle wiring).

**Spec reference:** `docs/superpowers/specs/2026-04-22-gpu-cold-state-replay-design.md` §Phase 2.

---

## Prior investigation (this plan's scope revisions vs spec)

- **Spec said**: "8 chronicle rules — some structured, some narrative. Narrative rules stay `@cpu_only`."
- **Reality**: all 8 are structured `physics @phase(post) { on <Event> { emit ChronicleEntry { ... } } }`. Text rendering is CPU-external (not in DSL). `@cpu_only` annotation isn't relevant to this subsystem's actual work.
- **Spec said**: "5-7 tasks, ~1 week."
- **Revised estimate**: 4-5 tasks, likely less than 1 week given most rules are already GPU-emittable.

## File structure

### Modified

- `crates/engine_gpu/src/cascade_resident.rs` — wire post-phase physics execution if not already running on GPU.
- `crates/engine_gpu/src/snapshot.rs` — add `chronicle_since_last: Vec<ChronicleRecord>` field on `GpuSnapshot`; staging copies the chronicle slice.
- `crates/engine_gpu/src/lib.rs` — `snapshot()` method reads chronicle ring alongside event ring.

### New tests

- `crates/engine_gpu/tests/chronicle_batch_path.rs` — cast → chronicle_attack emits → snapshot reads chronicle entry.

---

## Task 2.1: Audit `@phase(post)` GPU execution — DONE (classification A at the WGSL level; batch-path runtime bug surfaced in Task 2.4)

**Result:** All `@phase(post)` rules are correctly *emitted* into the physics WGSL dispatcher for both sync and resident shaders. The audit's paper-classification is (A). However, Task 2.4's end-to-end integration test later surfaced a runtime bug: the resident physics kernel does not emit chronicle records when dispatched through `GpuBackend::step_batch`'s full flow, despite emitting correctly when driven by a local harness (see `tests/physics_run_batch_resident_smoke.rs` which confirms `run_batch_resident` writes chronicle records from a seeded AgentAttacked event). Task 2.4's two ignored tests guard this bug; root-causing it is tracked in the task list under "step_batch chronicle-emit bug".

**Evidence:**
- `crates/dsl_compiler/src/parser.rs:804` — parser accepts `@phase(event)` / `@phase(post)` annotations.
- `crates/dsl_compiler/src/ir.rs:395-402` — `PhysicsIR` only stores `cpu_only`. **No phase field exists on the IR**; the annotation is dropped at lowering.
- `crates/dsl_compiler/src/emit_physics_wgsl.rs:328-362` — `applicable_rules()` filters only by `p.cpu_only`. All rules regardless of phase land in the generated `physics_dispatch(event_idx)`.
- `crates/dsl_compiler/src/emit_physics_wgsl.rs:276-319` — `emit_physics_dispatcher_wgsl` emits a single kind-keyed switch with no phase branching.
- `crates/engine_gpu/src/physics.rs:721-811` — resident physics shader reuses the same dispatcher (via `build_physics_shader_with_chronicle`); the sync path shares the exact same dispatcher function. Both paths call `physics_dispatch(i)` on each event.
- `crates/engine_gpu/src/cascade_resident.rs:1126-1177` — `run_batch_resident` dispatches the unified physics kernel each cascade iteration. Since events emitted by event-phase rules land in the next iteration's `events_in`, chronicle rules observe them and fire.
- Existing tests already observe chronicle emission on the batch path: `chronicle_drain_perf.rs`, `physics_parity.rs:10-279`, `event_ring_parity.rs:511`.

**Implication:** Task 2.2 is a no-op. Phase 2 collapses to Tasks 2.3 (snapshot wiring) + 2.4 (integration test) + 2.5 (closeout).

**Caveat** (deferred to subsystem 2 Phase 3 / future): the cascade dispatches `@phase(post)` rules *during* the cascade iteration that carries their trigger events, not strictly after cascade convergence. For chronicle rules — which are pure observers that only emit to the chronicle ring, never mutate agent state or feed back into the cascade — this is semantically equivalent to a strict post-cascade pass. If a future `@phase(post)` rule needs to read settled post-cascade state (e.g. final tick HP, not intermediate), it will need a real post-phase dispatch split. The emitter is currently missing phase machinery; adding it is a straightforward follow-up (filter in `applicable_rules`, add `emit_physics_post_dispatcher_wgsl`, dispatch after the cascade loop in `run_batch_resident`).

---

## Task 2.2: Enable `@phase(post)` rules on batch path — NO-OP per Task 2.1 finding (A)

Skipped. The dispatcher is phase-agnostic and both batch + sync paths already run all rules. Proceed to Task 2.3.

---

## Task 2.3: Wire `chronicle_ring` into `snapshot()`

**Files:**
- Modify: `crates/engine_gpu/src/snapshot.rs` — add `ChronicleRecord` public type + `chronicle_since_last: Vec<ChronicleRecord>` on `GpuSnapshot`.
- Modify: `crates/engine_gpu/src/snapshot.rs` — extend `GpuStaging` with chronicle staging buffer; `kick_copy` copies `chronicle_ring[last_read..tail]`.
- Modify: `crates/engine_gpu/src/lib.rs` — `snapshot()` reads `cascade_ctx.chronicle_ring` tail via readback, tracks watermark, passes slice into staging.
- Modify: `crates/engine_gpu/src/backend/snapshot_ctx.rs` — rename `snapshot_chronicle_ring_read: u64` (currently dead-code'd); remove the `#[allow(dead_code)]` attribute.

### Goal

Observer calling `snapshot()` receives `snap.chronicle_since_last` containing the `ChronicleEntry` records emitted since the previous snapshot, matching the structural pattern used for `events_since_last`.

### Steps

1. **Define `ChronicleRecord`** in `snapshot.rs`:
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub struct ChronicleRecord {
       pub template_id: u32,
       pub agent:       AgentId,
       pub target:      AgentId,
       pub tick:        u32,
   }
   ```
   Adjust to match the existing `ChronicleEntry` WGSL layout. Grep `crates/engine_gpu/src/event_ring.rs` for `GpuChronicleRing` layout.

2. **Extend `GpuSnapshot`**:
   ```rust
   pub struct GpuSnapshot {
       pub tick: u32,
       pub agents: Vec<GpuAgentSlot>,
       pub events_since_last: Vec<EventRecord>,
       pub chronicle_since_last: Vec<ChronicleRecord>,  // new
   }
   ```

3. **Extend `GpuStaging`** with a third pair of staging buffers (front + back × 3):
   - `chronicle_staging: wgpu::Buffer` (adds one per side to the existing agents + events pair).
   - `chronicle_len_bytes: u64` watermark.

4. **Modify `kick_copy` + `take_snapshot`** to handle chronicle copies in addition to events. Pattern mirrors the existing events-ring copy.

5. **Modify `snapshot()` in `lib.rs`**:
   - Read `chronicle_ring.tail()` via `readback_typed::<u32>` (mirrors the existing apply_event_ring.tail readback).
   - Update `snapshot_chronicle_ring_read` watermark.
   - Pass the `[last_read..tail]` chronicle slice into `kick_copy`.

6. **Build + run existing snapshot tests**:
   ```
   cargo test --release --features gpu -p engine_gpu --test async_smoke --test snapshot_double_buffer
   ```
   Expected: all pass. Existing tests don't assert on chronicle; they'll continue to work with the added staging (slice is empty when chronicle ring is unused).

7. **Commit**:
   ```bash
   git commit -m "feat(engine_gpu): snapshot().chronicle_since_last — double-buffered chronicle readback"
   ```

---

## Task 2.4: Integration test — chronicle emits through batch path

**Files:**
- Create: `crates/engine_gpu/tests/chronicle_batch_path.rs`

### Goal

End-to-end test: run `step_batch(n)` with a fixture that provokes at least one chronicle-triggering event (attack, engagement, etc.), then `snapshot()` and assert the expected `ChronicleRecord` appears in `snap.chronicle_since_last` with the right template_id.

### Steps

1. **Write the test**:
   ```rust
   //! End-to-end: chronicle_attack fires on the batch path. Verifies
   //! the full flow: @phase(post) physics rule emits ChronicleEntry
   //! → chronicle ring accumulates → snapshot reads it back → observer
   //! sees the record with the right template_id.

   #![cfg(feature = "gpu")]

   use engine::backend::SimBackend;
   use engine::cascade::CascadeRegistry;
   use engine::creature::CreatureType;
   use engine::event::EventRing;
   use engine::policy::UtilityBackend;
   use engine::state::{AgentSpawn, SimState};
   use engine::step::SimScratch;
   use engine_gpu::GpuBackend;
   use glam::Vec3;

   const CHRONICLE_TEMPLATE_ATTACK: u32 = 2; // per chronicle_attack rule

   #[test]
   fn chronicle_attack_fires_on_batch_path() {
       let mut gpu = GpuBackend::new().expect("gpu init");
       let mut state = SimState::new(8, 0xCA0CA0);
       // Two combatants at attack range; attack fires on tick 1.
       state.spawn_agent(AgentSpawn {
           creature_type: CreatureType::Human,
           pos: Vec3::new(0.0, 0.0, 0.0),
           hp: 100.0,
           ..Default::default()
       }).unwrap();
       state.spawn_agent(AgentSpawn {
           creature_type: CreatureType::Wolf,
           pos: Vec3::new(1.0, 0.0, 0.0),
           hp: 100.0,
           ..Default::default()
       }).unwrap();
       let mut scratch = SimScratch::new(state.agent_cap() as usize);
       let mut events = EventRing::with_cap(256);
       let cascade = CascadeRegistry::with_engine_builtins();

       gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
       gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 5);

       // Triple-snapshot dance (first empty, second kicks, third returns).
       let _e = gpu.snapshot().expect("empty");
       let _k = gpu.snapshot().expect("kick");
       let snap = gpu.snapshot().expect("read");

       let attack_records: Vec<_> = snap.chronicle_since_last.iter()
           .filter(|r| r.template_id == CHRONICLE_TEMPLATE_ATTACK)
           .collect();
       assert!(
           !attack_records.is_empty(),
           "expected at least one chronicle_attack record in snapshot"
       );
   }
   ```

2. **Run**:
   ```
   cargo test --release --features gpu -p engine_gpu --test chronicle_batch_path
   ```
   Expected: PASS.

3. **If it fails**, diagnose. Likely failure modes:
   - Task 2.2 didn't actually enable post-phase execution → zero chronicle records in snap.
   - Task 2.3's watermark is off-by-one → records emitted but not observed.
   - The `step` warmup tick pre-populated the chronicle ring before batch started, so the first snapshot's slice includes the warmup record (test may pass "accidentally"). Consider resetting the chronicle ring after warmup OR asserting specifically that records come from batch ticks (check `record.tick > tick_after_warmup`).

4. **Commit**:
   ```bash
   git commit -m "test(engine_gpu): chronicle_attack fires on batch path (Phase 2 integration)"
   ```

---

## Task 2.5: Phase 2 closeout — PARTIAL (blocked on step_batch chronicle-emit bug)

Task 2.3 (snapshot wiring) is complete and regression-clean. Task 2.4 landed two ignored end-to-end tests that will start passing once the step_batch physics chronicle-emit bug is root-caused and fixed (tracked separately). Phase 2 is best-considered "snapshot plumbing done; end-to-end flow requires upstream engine fix."

### Original Task 2.5 scope (no longer gating Phase 2)

- [ ] **Step 1:** Full regression:
   ```
   cargo test --release --features gpu -p engine_gpu
   cargo test --release -p dsl_compiler
   cargo test --release -p engine
   ```
   Expected: no new failures beyond the 3 pre-existing engine failures (rng golden, wolves x2).

- [ ] **Step 2:** Perf sweep to confirm no regression in batch path:
   ```
   cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -15
   ```
   Expected: N=2048 batch µs/tick within 10% of baseline (~5500-5700 µs). The added post-phase dispatch adds a small constant overhead per tick.

- [ ] **Step 3:** Push. No commit needed — verification.

---

## Notes

- **Task 2.1 may reveal the batch path already runs post-phase rules**. In that case, 2.2 is a no-op and the whole phase collapses to Tasks 2.3 + 2.4. That's the happy path.
- **Chronicle ring overflow protection**: the ring is fixed-capacity (per `GpuChronicleRing::new` at ~1M records). Long batches at high N may overflow before a snapshot drains them. Out of scope for Phase 2 — subsystem (3) or a follow-up addresses with larger caps or auto-drain. Add a comment in `cascade_resident.rs` noting the ring size assumption.
- **Pre-existing 3 engine failures** are NOT in scope — stay below them in any diagnostic noise.
