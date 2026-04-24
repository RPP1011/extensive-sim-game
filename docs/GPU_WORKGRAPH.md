# GPU Workgraph Reference — Wolves & Humans Simulation Engine

**Audience:** GPU-compute-literate engineers joining the project who have read `docs/technical_overview.md` but lack project-specific knowledge of the GPU paths.

**Date:** 2026-04-23

---

## 1. Overview: Two Execution Paths

The engine exposes two simulation ticks on the GPU backend (`crates/engine_gpu/src/lib.rs`):

- **Sync path** (`SimBackend::step`): CPU → CPU-driven cascade with per-dispatch GPU kernels; one fence per iteration. Authoritative for determinism. See `/lib.rs:308-319`.
- **Batch/resident path** (`step_batch`): GPU-resident cascade with indirect dispatch; one fence per N ticks. Observation-mode fast-path. See `/lib.rs:1459+`.

Both paths use the same underlying compute kernels (mask, scoring, apply_actions, movement, physics). The resident path adds housekeeping kernels (unpack, seed, indirect-args manipulation) to eliminate per-tick CPU/GPU fences.

---

## 2. Sync Path Pipeline

```
step(state, events, cascade) -> state'
├─ 1. mask.run_batch()
│     (reads: agent pos/alive/creature_type; writes: 7 × bitmap buffers)
├─ 2. scoring.run_batch()
│     (reads: bitmaps, agent fields, view_storage; writes: ScoreOutput SoA)
├─ 3. apply_actions.run_batch()
│     (reads: ScoreOutput, agent slots; writes: hp/shield/alive, event_ring)
├─ 4. movement.run_batch()
│     (reads: ScoreOutput, agent slots; writes: pos, event_ring)
├─ 5. cascade.run_cascade()
│     ├─ for iter = 0..MAX_CASCADE_ITERATIONS:
│     │   ├─ physics.run_batch(events_in) -> events_out
│     │   ├─ fold_iteration_events(events_out) -> view_storage
│     │   └─ if events_out.is_empty(): break
│     └─ return final_agent_slots, all_events, iterations, converged
├─ 6. cold_state_replay() [CPU]
│     (record gold, standing, memory mutations from drained events)
└─ 7. finalize() [CPU]
     (tick++, invariant checks, telemetry)
```

**Files:**
- Entry: `crates/engine_gpu/src/lib.rs:308-319` (stub forwarding to CPU)
- Mask kernel: `crates/engine_gpu/src/mask.rs`
- Scoring kernel: `crates/engine_gpu/src/scoring.rs`
- Apply+Movement: `crates/engine_gpu/src/apply_actions.rs`, `movement.rs`
- Cascade driver: `crates/engine_gpu/src/cascade.rs:188`
- View folds: `crates/engine_gpu/src/view_storage.rs`

---

## 3. Resident/Batch Path Pipeline

```
step_batch(n) -> ()
├─ ensure_resident_init()
│  (allocate resident_agents_buf, sim_cfg_buf, cascade_resident_ctx on first call)
└─ for tick = 0..n:
    ├─ 1. fused_unpack_kernel.run()
    │     (reads: resident_agents_buf; writes: mask SoA + scoring agent_data_buf)
    ├─ 2. mask_resident.run()
    │     (reads: mask SoA; writes: bitmap buffers)
    ├─ 3. scoring_resident.run()
    │     (reads: bitmaps, agent_data, view_storage; writes: ScoreOutput)
    ├─ 4. apply_actions.run_resident()
    │     (reads: ScoreOutput, resident_agents_buf; writes: hp/shield/alive, batch_events_ring)
    ├─ 5. movement.run_resident()
    │     (reads: ScoreOutput, resident_agents_buf; writes: pos, batch_events_ring)
    ├─ 6. append_events()
    │     (reads: batch_events_ring tail; writes: apply_event_ring)
    ├─ 7. seed_kernel.run()
    │     (atomicAdd(sim_cfg.tick, 1); seed indirect_args[0]; clear num_events[1..N])
    └─ 8. for iter = 0..MAX_CASCADE_ITERATIONS:
         ├─ physics.run_batch_resident()
         │   (reads: apply_event_ring[iter]; writes: resident_agents_buf, physics_ring_[iter], indirect_args[iter+1])
         └─ if indirect_args[iter+1] == (0,1,1): break [convergence encoded in args]
```

**Files:**
- Entry & context init: `crates/engine_gpu/src/lib.rs:1459+`
- Resident context: `crates/engine_gpu/src/backend/resident_ctx.rs`
- Resident cascade driver: `crates/engine_gpu/src/cascade_resident.rs`
- Resident unpack kernels: `crates/engine_gpu/src/mask.rs` (FusedAgentUnpackKernel)

---

## 4. Snapshot (Double-Buffer Observer Sync Point)

```
snapshot() -> GpuSnapshot { tick, agents, events_since_last, chronicle_since_last }
├─ poll front staging buffer [non-blocking from previous snapshot]
├─ copy_buffer_to_buffer(live GPU buffers -> back staging)
└─ swap front/back
```

**Note:** One-frame lag; the returned snapshot contains data from the tick *before* the current one. The double-buffer prevents GPU→CPU synchronization on the hot path.

**Files:**
- Snapshot API: `crates/engine_gpu/src/snapshot.rs`
- Snapshot context: `crates/engine_gpu/src/backend/snapshot_ctx.rs`

---

## 5. Compute Kernels

### 5.1 Mask Kernel

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/mask.rs` |
| **Struct** | `FusedMaskKernel` (sync), `MaskUnpackKernel` (resident unpack) |
| **Entry points** | `cs_fused_masks` (sync/batch); `cs_mask_unpack` (resident unpack) |
| **Workgroup size** | 64 |
| **Bind group layout** | Sync: agents (pos, alive, creature_type), 7 bitmap outputs, cfg. Resident unpack: resident_agents_buf → mask SoA. |
| **Inputs** | Agent position, alive status, creature_type; ConfigUniform (movement radius) |
| **Outputs** | 7 bitmap arrays: Attack, MoveToward, Hold, Flee, Eat, Drink, Rest (atomic u32 per agent). |
| **Runs on** | Sync: every `step`. Resident: every batch tick. |
| **Notes** | Cast mask skipped (requires view + ability storage not yet GPU-resident). Fused dispatch writes all 7 in one kernel call. Resident unpack rewrites mask SoA from resident_agents_buf each tick (no per-tick CPU upload). |

---

### 5.2 Scoring Kernel

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/scoring.rs` |
| **Struct** | `ScoringKernel` (sync), `ScoringUnpackKernel` (resident, legacy) |
| **Entry point** | `cs_scoring` (sync/batch) |
| **Workgroup size** | 64 |
| **Bind group layout** | agent_data (SoA), bitmaps (read), view_storage (atomic reads), cfg, sim_cfg (task 2.4), spatial queries (read) |
| **Inputs** | 7 mask bitmaps; agent fields (hp_pct, engaged_with, …); view_storage atomic reads (my_enemies, threat_level, kin_fear, pack_focus, rally_boost). |
| **Outputs** | `ScoreOutput[agent_cap]` — one struct per agent: chosen_action, chosen_target, score. |
| **Runs on** | Sync: every `step`. Resident: every batch tick. |
| **Notes** | View reads are stub 0.0 in Phase 2; task 190 wires view_storage atomics. Spatial query reads are read-only on precomputed results (kin list, nearest hostile). |

---

### 5.3 Apply Actions Kernel

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/apply_actions.rs` |
| **Struct** | `ApplyActionsKernel` |
| **Entry points** | `cs_apply_actions` (sync), `cs_apply_actions_resident` (batch) |
| **Workgroup size** | 64 |
| **Bind group layout** | agents (read_write), scoring (read), event_ring (read_write), event_ring_tail (atomic), cfg, sim_cfg (shared) |
| **Inputs** | ScoreOutput; agent slots (hp, shield, alive). |
| **Outputs** | Mutated agent hp/shield/alive; events: AgentAttacked, AgentDied, AgentAte, AgentDrank, AgentRested. |
| **Runs on** | Sync & resident (two separate entry points). |
| **Scope gaps** | Opportunity attacks, engagement slow on MoveToward, announce/communicate — all in CPU path. |

**Attack event emission:** reads target hp, applies damage, emits AgentAttacked, kills target if hp ≤ 0 and emits AgentDied.

---

### 5.4 Movement Kernel

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/movement.rs` |
| **Struct** | `MovementKernel` |
| **Entry points** | `cs_movement` (sync), `cs_movement_resident` (batch) |
| **Workgroup size** | 64 |
| **Bind group layout** | agents (read_write), scoring (read), event_ring (read_write), event_ring_tail (atomic), cfg, sim_cfg (shared) |
| **Inputs** | ScoreOutput; agent slots (pos, slow_factor_q8). |
| **Outputs** | Updated agent pos; events: AgentMoved, AgentFled. |
| **Runs on** | Sync & resident. |
| **Scope gaps** | Kin-flee-bias (herding), effect slow multiplier — deferred. |

**Movement math:** MoveToward = `self.pos + normalize(target.pos - self.pos) * move_speed`. Flee (pure-away) = `self.pos + normalize(self.pos - threat.pos) * move_speed`.

---

### 5.5 Physics Kernel (Event Processor)

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/physics.rs` |
| **Struct** | `PhysicsKernel` |
| **Entry points** | `cs_physics` (sync), `cs_physics_resident` (batch) |
| **Workgroup size** | 64 |
| **Bind group layout** (sync) | agents (SoA, read_write), event_ring_in (read), event_ring_out (read_write), event_ring_tail (atomic), view_storage (atomic), spatial (kin/hostile, read), abilities (read), cfg, sim_cfg (shared), chronicle_ring (sync), chronicle_ring_tail (sync). |
| **Bind group layout** (resident) | Same as sync, plus indirect_args (slots 13), num_events_buf (slot 14), resident_cfg (slot 15). |
| **Inputs** | Event batch (one per thread). Agent SoA. Pre-computed kin lists & nearest-hostile results. Ability registry. SimCfg (tick, kin_radius). |
| **Outputs** | Mutated agent state (hp, shield, stun, slow, engaged_with, alive). New events into event_ring. Chronicle entries into chronicle_ring (sync) or resident chronicle_ring (batch). |
| **Runs on** | Sync: per cascade iteration; resident: per cascade iteration (indirect dispatch). |
| **Rule coverage** | 12/23 rules fully implemented; 8 chronicle rules + gold/standing/memory are stubs (no-ops). |

**Determinism:** Physics emits events in non-deterministic order (atomic tail racing). Host drain sorts by `(tick, kind, payload[0])` pre-fold.

**Chronicle rings (KEY DISTINCTION):**
  - Sync path: `PhysicsKernel::chronicle_ring` (bindings 11-12 in sync BGL). Written to per tick; drained separately by `GpuBackend::drain_chronicle_ring()`.
  - Resident/batch path: `CascadeResidentCtx::chronicle_ring` (caller-owned). Written to per tick; snapshot reads it via watermark.

---

### 5.6 Fold Kernels (View Materialization)

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/view_storage.rs` |
| **Entry points** | `cs_fold_<view_name>` (one per view: engaged_with, my_enemies, threat_level, kin_fear, pack_focus, rally_boost) |
| **Workgroup size** | 64 |
| **Bind group layout** | fold_inputs (read), view_storage (read_write atomic), cfg |
| **Inputs** | FoldInput batch (one per event-cell pair): observer_id, other_id, delta, anchor_tick. |
| **Outputs** | View storage atomic updates (CAS loop). |
| **Runs on** | Sync cascade only (post physics iteration). Resident fold is deferred (Task D4). |

**Determinism:** All folds use constant delta `+= 1.0`; atomic CAS order doesn't matter for commutative sums.

---

### 5.7 Spatial Hash Query Kernels (Resident Path)

| Property | Value |
|----------|-------|
| **Module** | `crates/engine_gpu/src/spatial_gpu.rs` |
| **Entry points** | `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query` |
| **Inputs** | Agent positions; two radii (kin=12m, engagement=2m). |
| **Outputs** | Per-agent query results: nearby agents (within), kin-species membership (kin), nearest hostile/kin (nearest, one u32 per agent). |
| **Runs on** | Resident batch path (two query radii per tick). Sync path uses CPU spatial hash. |

---

## 6. Major Buffers and Ownership

| Buffer | Owner | Size (N=100k) | Purpose | Sync Path | Resident Path |
|--------|-------|---|---------|-----------|---|
| `resident_agents_buf` | `ResidentPathContext` | ~16 MB | Agent SoA (GpuAgentSlot). Persistent across batch. | — | Read/write each tick |
| `sim_cfg_buf` | `ResidentPathContext` | 256 B | SimCfg: tick (atomic), attack_range, kin_radius, move_speed, world seed. | Per-call upload | Persistent; tick atomically incremented |
| `apply_event_ring` | `CascadeCtx` (sync) | ~24 MB | Seed ring: apply_actions events → physics iter 0. Cleared per-tick. | Read by physics iter 0 | Via append_events per tick |
| `physics_ring_a`/`_b` | `CascadeResidentCtx` | ~24 MB each | Ping-pong resident cascade rings. Iter N writes to ring (N+1) % 2. | — | Read/write per iteration |
| `batch_events_ring` | `CascadeResidentCtx` | ~24 MB | Append-only accumulator of all apply_actions events (batch). Exposed to snapshot. | — | Appended to each tick |
| `chronicle_ring` | `PhysicsKernel` (sync) | ~24 MB | Narrative records (ChronicleEntry). Accumulated across ticks. | Write, then drain separately | — |
| `chronicle_ring` | `CascadeResidentCtx` (resident) | ~24 MB | Same, but caller-owned. Exposed via snapshot watermark. | — | Write + snapshot readback |
| `indirect_args` | `ResidentPathContext` | 32 B × (MAX_CASCADE_ITERATIONS+1) | Dispatch args: (workgroup_x, workgroup_y, workgroup_z) per iteration. Slot 0 seeded; iter N updates slot N+1. | — | Read by indirect dispatch; written by physics |
| `num_events_buf` | `CascadeResidentCtx` | 4 B × (MAX_CASCADE_ITERATIONS+1) | Event counts per iteration (diagnostic). | — | Written by seed kernel, physics |
| `view_storage` | `ViewStorage` | ~144 MB (6 views @ N=100k) | Materialized per-view state: engaged_with (slot_map), my_enemies/threat_level/kin_fear/pack_focus/rally_boost (pair_maps, top-K). | Read by scoring; written by fold kernels | Read by scoring (sync fold only) |
| Spatial query outputs | `SpatialOutputs` (kin, engagement) | ~80 MB each | kin_within/kin_kin/kin_nearest, eng_within/eng_kin/eng_nearest. | CPU rebuild + upload | Resident buffers; rebuilt each batch tick |
| Ability registry | `PackedAbilityRegistry` | ~256 KB | abilities_known[], abilities_cooldown[], abilities_effects_count[], abilities_effects[]. | Per-call upload via CPU | Resident buffers with content-addressed upload |

**Ownership summary:**
- **SyncPathContext owns:** mask_kernel, scoring_kernel, view_storage, (optional) cascade_ctx, last_mask_bitmaps, last_scoring_outputs.
- **ResidentPathContext owns:** resident_agents_buf, resident_indirect_args, resident_cascade_ctx, sim_cfg_buf, fused_unpack_kernel.
- **CascadeResidentCtx owns:** physics_ring_a/b, batch_events_ring, chronicle_ring, spatial buffers, ability buffers, num_events_buf.

---

## 7. The Cascade: Fixed-Point Loop via Indirect Dispatch

### Sync Cascade (CPU-Driven)

```
for iter = 0..MAX_CASCADE_ITERATIONS:
  physics.run_batch(events_in) -> (agents, events_out)
  for view in [engaged_with, my_enemies, threat_level, kin_fear, pack_focus, rally_boost]:
    view_fold.run_batch(events_out) -> view_storage[view]
  if events_out.len() == 0:
    break  // converged
  events_in = events_out
return all_events_across_all_iters
```

**Determinism:** Each `run_batch` call is a separate dispatch; atomicAdd on the event tail is serialized per-call by the host's fence.

**Files:** `crates/engine_gpu/src/cascade.rs:188` (`run_cascade`), `:376` (`fold_iteration_events`).

---

### Resident Cascade (GPU-Resident Indirect Dispatch)

```
seed_kernel: indirect_args[0] = (ceil(num_events[0] / WGSIZE), 1, 1)

for iter = 0..MAX_CASCADE_ITERATIONS:
  indirect_dispatch(indirect_args[iter]) cs_physics_resident
    -> mutate(agents), write(physics_ring[iter % 2]), append(num_events[iter+1]), write(indirect_args[iter+1])
  if indirect_args[iter+1] == (0, 1, 1):  // convergence: no new events
    break
return agents  // all mutations accumulated across iterations
```

**Convergence signaling:** Physics kernel writes `indirect_args[iter+1] = (0,1,1)` if no events emitted on this iteration. The GPU/host boundary is still crossed (one encode per tick, not per iteration), but the actual physics iterations run GPU-resident without intermediate readbacks.

**Files:** `crates/engine_gpu/src/cascade_resident.rs:1039+` (`run_cascade_resident`), indirect buffer setup `gpu_util/indirect.rs`.

---

## 8. DSL → WGSL Lowering

### High-Level Flow

1. **Parse + IR generation:** `dsl_compiler` reads `assets/sim/masks.sim`, `physics.sim`, `views.sim`, etc.
2. **Emit to WGSL:** Per-subsystem emitters produce WGSL modules:
   - `emit_mask_wgsl::emit_masks_wgsl_fused` → one fused module with per-mask bitmap writes.
   - `emit_physics_wgsl::emit_physics_wgsl` → one module with `physics_dispatch(event)` switch.
   - `emit_view_wgsl::emit_view_fold_wgsl` → one fold kernel per view.
3. **Assemble + compile:** Host assembles fragments, pipes to `naga`/wgpu, produces `RenderPipeline` or `ComputePipeline`.

### Physics Shader Assembly

**Sync shader (build_physics_shader_with_chronicle):**

```wgsl
// Per-kernel boilerplate
struct EventRecord { kind: u32, tick: u32, payload: array<u32, 8> }
struct AgentSlot { hp: f32, ... }
struct SimCfg { tick: atomic<u32>, ... }

// Emitted physics rules (12 rules × their match/emit bodies)
fn rule_damage(event: EventRecord) { ... }
fn rule_stun(event: EventRecord) { ... }
... (repeat for all 12 rules)

// Dispatcher (emitter output)
fn physics_dispatch(event_idx: u32) {
  let event = event_ring_in[event_idx];
  switch(event.kind) {
    case AgentAttacked: rule_damage(); rule_opportunity_attack(); break;
    case AgentDied: rule_fear_spread(); break;
    ... (etc.)
  }
}

@compute @workgroup_size(64)
fn cs_physics(
  @builtin(global_invocation_id) gid: vec3u,
  @group(0) @binding(0) agents: array<read_write, AgentSlot>,
  @group(0) @binding(1) event_ring_in: array<EventRecord>,
  @group(0) @binding(2) event_ring_out: array<read_write, EventRecord>,
  @group(0) @binding(3) event_ring_out_tail: atomic<u32>,
  ...
  @group(0) @binding(11) chronicle_ring: array<read_write, ChronicleRecord>,
  @group(0) @binding(12) chronicle_tail: atomic<u32>,
) {
  if gid.x < num_events_in {
    physics_dispatch(gid.x);
  }
}
```

**Resident shader (build_physics_shader_resident):**

Same as sync, plus:
- Bindings 13–15: indirect_args, num_events_buf, resident_cfg.
- `cs_physics_resident` entry point uses indirect dispatch.
- Chronicle ring is caller-supplied (same binding structure, different buffer).

**Files:**
- `crates/engine_gpu/src/physics.rs:416` (`build_physics_shader`).
- `crates/engine_gpu/src/physics.rs:434` (`build_physics_shader_with_chronicle`).
- `crates/engine_gpu/src/physics.rs:721` (`build_physics_shader_resident`).
- Emitter (DSL compiler): `dsl_compiler/src/emit_physics_wgsl.rs`.
- Chronicle WGSL snippet: `crates/engine_gpu/src/event_ring.rs:CHRONICLE_RING_WGSL`.

---

## 9. Known Issues and TODOs (as of 2026-04-23)

### Critical Bug: Resident Physics Chronicle-Emit Regression

**Status:** REGRESSION (BLOCKING).

**Symptom:** The resident `step_batch` path does not emit chronicle records despite both:
1. The WGSL emitter correctly including chronicle rules in `physics_dispatch` for both sync + resident shaders.
2. The isolated `PhysicsKernel::run_batch_resident` call (when seeded locally) emitting chronicle records correctly.

**Symptom trace:** After `step_batch(5)` on a Human+Wolf fixture producing 8 AgentAttacked events, the resident `chronicle_ring` tail remains 0, and records are all-zero.

**Root cause:** TBD. Diagnostics so far rule out:
- Bind-group wrong buffer (physics writes to caller-supplied ring in isolation).
- sim_cfg.tick staleness (GPU tick advances correctly).
- Cascade falling back to sync (sync chronicle tail unaffected).
- Pipeline validation (passes).

**Suspected area:** apply_event_ring → events_in binding flow on the batch path.

**Guard:** `crates/engine_gpu/tests/chronicle_batch_path.rs` (currently `#[ignore]`d). Re-enable when fixed.

**Impact:** Batch-mode chronicle observability is broken. Sync-path and direct physics-kernel tests both pass.

---

### Performance Gaps

- **Per-event fold dispatch (N=100k, heavy combat):** View fold kernels dispatch once per event, not once per view. Batching via segmented-reduction would 8–10× speedup. See `docs/technical_overview.md:156`.
- **Resident fold kernels:** Task D4 deferred. Currently only the sync path folds; resident cascade assumes views stay stale across batch ticks. Wiring fold kernels into the resident path requires careful binding-group re-architecture (fold kernels currently bind view_storage read_write, and the physics kernel also needs write access during the same iteration).

---

### Architectural Gaps

- **GPU agent SoA missing fields:** No gold, standing, memory ring on GPU side. Cold-state rules (transfer_gold, modify_standing, record_memory) stay on CPU. Porting requires expanding GpuAgentSlot or adding parallel storage.
- **Kin-flee-bias (herding):** Movement kernel has a stub for `flee_direction_with_kin_bias` but doesn't consume kin_centroid. Deferred; current fixtures (wolves+humans) don't herd.
- **Cast mask on GPU:** Requires ability-registry + cooldown storage. Planned for Phase 4+.
- **Opportunity attacks on resident path:** apply_actions kernel omits the engagement-aware OpportunityAttackTriggered emit. Deferred alongside engagement-slow.

---

### Test Coverage

- **Sync path:** Full parity tests passing at N=50 (canonical fixture). Statistical parity (alive counts, event multisets) at N=100k.
- **Resident path:** Isolated kernel smoke tests passing. `chronicle_batch_path.rs` regression guard (currently ignored) awaiting chronicle-emit fix.
- **Snapshot double-buffer:** `snapshot_double_buffer.rs` passing for agents + batch_events_ring. Chronicle snapshot readback also passing (pre-chronicle-emit fix discovery).

---

## 10. File Index

**Core modules:**
- `crates/engine_gpu/src/lib.rs` — GpuBackend struct, SimBackend impl, phase timings.
- `crates/engine_gpu/src/backend/sync_ctx.rs` — SyncPathContext (mask + scoring + view_storage).
- `crates/engine_gpu/src/backend/resident_ctx.rs` — ResidentPathContext (agents + indirect_args + cascade_ctx).
- `crates/engine_gpu/src/backend/snapshot_ctx.rs` — SnapshotContext (double-buffer staging).

**Kernels:**
- `crates/engine_gpu/src/mask.rs` — FusedMaskKernel, MaskUnpackKernel, FusedAgentUnpackKernel.
- `crates/engine_gpu/src/scoring.rs` — ScoringKernel, ScoringUnpackKernel.
- `crates/engine_gpu/src/apply_actions.rs` — ApplyActionsKernel.
- `crates/engine_gpu/src/movement.rs` — MovementKernel.
- `crates/engine_gpu/src/physics.rs` — PhysicsKernel, build_physics_shader*, rule implementations.

**Cascade drivers:**
- `crates/engine_gpu/src/cascade.rs` — Sync cascade (CPU-driven fixed-point loop).
- `crates/engine_gpu/src/cascade_resident.rs` — Resident cascade (GPU-resident indirect dispatch).

**Supporting systems:**
- `crates/engine_gpu/src/view_storage.rs` — ViewStorage, fold kernels.
- `crates/engine_gpu/src/spatial_gpu.rs` — GpuSpatialHash, resident spatial queries.
- `crates/engine_gpu/src/event_ring.rs` — GpuEventRing, GpuChronicleRing, EventKindTag, pack/unpack.
- `crates/engine_gpu/src/snapshot.rs` — GpuSnapshot, GpuStaging (double-buffer).
- `crates/engine_gpu/src/gpu_util/indirect.rs` — IndirectArgsBuffer, indirect dispatch helpers.
- `crates/engine_gpu/src/gpu_util/readback.rs` — GPU→CPU staging + mapped reads.
- `crates/engine_gpu/src/sim_cfg.rs` — SimCfg (shared GPU state), AtomicSimCfg layout.

**Tests:**
- `crates/engine_gpu/tests/chronicle_batch_path.rs` — Regression guard (ignored pending fix).
- `crates/engine_gpu/tests/snapshot_double_buffer.rs` — Snapshot + batch_events_ring.
- `crates/engine_gpu/tests/physics_run_batch_*.rs` — Isolated physics kernel tests.

---

## 11. Quick Reference: Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_CASCADE_ITERATIONS` | 8 | `cascade.rs:89` |
| `PHYSICS_WORKGROUP_SIZE` | 64 | `physics.rs:109` |
| `MAX_EFFECTS` (per ability program) | 8 | `physics.rs:128` |
| `MAX_ABILITIES` | 256 | `physics.rs:134` |
| `DEFAULT_CAPACITY` (event ring) | 655,360 | `event_ring.rs:92` |
| `DEFAULT_CHRONICLE_CAPACITY` | 1,000,000 | `event_ring.rs:106` |
| `PAYLOAD_WORDS` (per event) | 8 | `event_ring.rs:84` |
| `K` (spatial query cap) | 32 | `spatial_gpu.rs` |
| `FOLD_WORKGROUP_SIZE` | 64 | `view_storage.rs` |
| `SIM_CFG_BINDING` | 16 | `physics.rs:122` |

---

**Next reading:** See `docs/technical_overview.md` sections 8–10 for determinism guarantees, cascade semantics, and performance decomposition. For DSL → WGSL specifics, consult `dsl_compiler/src/emit_*.rs`.
