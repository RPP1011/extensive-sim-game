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

## Task 2.1: Audit `@phase(post)` GPU execution

**Files:**
- Read-only: `crates/dsl_compiler/src/emit_physics_wgsl.rs`, `crates/engine_gpu/src/cascade_resident.rs`, `crates/engine_gpu/src/physics.rs`

### Goal

Determine whether `@phase(post)` physics rules are:
- (A) Already emitted to the physics WGSL kernel and run in the cascade iterations.
- (B) Filtered out by the emitter (e.g. only `@phase(event)` rules run on GPU).
- (C) Emitted but not dispatched by cascade_resident's driver.

### Steps

1. **Grep the WGSL emitter for phase handling**:
   ```
   grep -rn "phase\|Phase\|@phase" crates/dsl_compiler/src/emit_physics_wgsl.rs | head -20
   ```

2. **Read how the main physics kernel handles phase-tagged rules**: check if there's separate `event` vs `post` dispatch, or if all rules run each iter regardless of phase.

3. **Check cascade_resident's physics dispatch**: does it invoke only the event-phase kernel, or does it also run post?

4. **Classify the situation** (A/B/C) in the report.

5. **No code changes** — verification task. The finding shapes Task 2.2's scope.

### Commit (if any)

If classification documents useful engineering notes, commit a short doc:
```bash
git commit -m "docs(engine_gpu): @phase(post) execution audit on batch path"
```

Otherwise no commit — pure read-only investigation feeding Task 2.2.

### Report

Status: DONE (verification task). Include: which of (A/B/C) describes current state + exact code refs.

---

## Task 2.2: Enable `@phase(post)` rules on batch path (if Task 2.1 reveals they're off)

**Files (conditional on 2.1 finding):**
- If (B): `crates/dsl_compiler/src/emit_physics_wgsl.rs` — include post-phase rules in WGSL emit.
- If (C): `crates/engine_gpu/src/cascade_resident.rs` — dispatch a post-phase physics kernel after event-phase iterations.
- If (A): task is a no-op; skip to Task 2.3.

### Goal

Ensure the 8 chronicle rules execute on the batch path. Post-phase rules may need their own kernel dispatch (distinct from the event-phase cascade) because they run ONCE after the event-phase converges, not inside the cascade loop.

### Steps

1. If WGSL emitter filters `@phase(post)`: remove the filter, regenerate, verify naga parse passes.

2. If cascade_resident needs post-phase dispatch: add one after `run_cascade_resident`'s event-phase loop, before snapshot handshake. Use the same indirect-dispatch pattern as event-phase if applicable, or direct dispatch (simpler since post runs once).

3. Run existing parity tests to ensure no regression:
   ```
   cargo test --release --features gpu -p engine_gpu --test parity_with_cpu --test physics_parity --test cascade_parity
   ```

4. Commit:
   ```bash
   git commit -m "feat(engine_gpu): run @phase(post) physics rules on batch path"
   ```

### Scope discipline

- Only the specific files needed based on Task 2.1's classification.
- Do NOT wire snapshot chronicle yet (Task 2.3).
- Do NOT add new tests yet (Task 2.4).

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

## Task 2.5: Phase 2 closeout

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
