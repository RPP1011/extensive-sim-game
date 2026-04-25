# GPU Backend Specification

GPU is a co-equal engine backend alongside `SerialBackend`. Both expose the same `SimBackend` trait; both produce game-equivalent simulation. This file specifies the GPU-resident execution model, sim-state mirroring, cold-state handler dispatch, ability evaluation, and the kernel pipeline that ties them together.

Companion to `runtime.md` (engine contract), `compiler.md` (DSL → backend lowering), and `state.md` (field catalog). Where sections cross-reference engine semantics that apply to both backends, this file calls out only the GPU-specific behaviour.

---

## 1. Overview

The GPU backend exposes two execution modes:

- **Sync mode** — `SimBackend::step(state, intents, dt_ms)`. CPU-driven cascade with one fence per kernel dispatch. Authoritative for parity tests; matches `SerialBackend` semantics.
- **Batch mode** — `step_batch(state, n, cascade)`. GPU-resident cascade with indirect dispatch. One submit per N ticks, observation via non-blocking `snapshot()`.

The **Resident Cascade** (§2) is the foundational pattern. It eliminates per-tick CPU/GPU fences by binding each kernel's outputs as the next kernel's inputs and replacing CPU-driven loops with GPU indirect dispatch.

**Layered subsystems** (each builds on the previous):

| Layer | Adds |
|---|---|
| Resident cascade (§2) | GPU-resident pipeline, indirect dispatch, double-buffered snapshot |
| Sim state mirroring (§3) | `SimCfg` shared buffer, GPU-side tick advance, `@cpu_only` annotation |
| Cold-state replay (§4) | Per-event-kind handler dispatch, gold/standing/memory on GPU |
| Ability evaluation (§5) | `pick_ability` kernel, `ability::tag` scoring primitive, `per_ability` row |

§6 is a kernel/buffer reference. §7 covers cross-cutting concerns (determinism, schema hash, parity contract).

**Non-goals**

- Byte-exact GPU↔CPU parity in batch mode. Atomic-tail event ordering is non-commutative on GPU; same-seed runs may diverge.
- Cross-GPU reproducibility. Same hardware + same seed reproduces; different vendors may diverge.
- Replacing the sync path. Sync stays load-bearing for parity tests, deterministic chronicle output, and `SerialBackend` cross-checks.

---

## 2. Resident cascade

### 2.1 Principle

In sync mode, every kernel's outputs are copied to CPU and the next kernel's inputs are re-uploaded. The resident cascade binds output buffers directly as the next kernel's inputs; CPU only observes when explicitly asked.

### 2.2 Public surface

```rust
impl GpuBackend {
    /// Run N ticks GPU-resident. One command buffer, one submit, one
    /// poll at end. Non-deterministic in event order. Does not populate
    /// the caller's EventRing — observe state via snapshot().
    pub fn step_batch(&mut self, state: &mut SimState, n: u32,
                      cascade: &CascadeRegistry) -> Result<(), BatchError>;

    /// Cheap non-blocking observation via double-buffered staging.
    /// First call returns an empty snapshot. Subsequent calls return
    /// the state as-of the previous snapshot call (one frame lag).
    pub fn snapshot(&mut self) -> Result<GpuSnapshot, SnapshotError>;
}

pub struct GpuSnapshot {
    pub tick: u32,
    pub agents: Vec<GpuAgentSlot>,
    pub events_since_last: Vec<EventRecord>,
    pub chronicle_since_last: Vec<ChronicleEntry>,
}
```

### 2.3 One tick on the batch path

All GPU-resident — no CPU round-trips inside the tick:

```
agents_buf ──▶ [mask] ──▶ mask_bitmaps_buf ──▶ [scoring] ──▶ scoring_buf
                                                                 │
                                                                 ▼
agents_buf ◀── [movement] ◀── [apply_actions] ◀─────────────────┘
      │             │                │
      │             │                ▼
      │             │          apply_event_ring_buf
      ▼             │                │
[spatial: count → GPU-scan → scatter → sort → query]
      │                                           │
      ▼                                           │
kin_buf, nearest_buf ──▶ [cascade: N× indirect physics dispatch]
                                      │
                                      ▼
                            physics_event_ring_buf, updated agents_buf
                                      │
                                      ▼
                            [fold_iteration kernels ──▶ view_storage_buf]
                                      │
                                      ▼
                            events accumulate in main_event_ring_buf (GPU)
                            chronicle entries → chronicle_ring_buf (GPU)
```

### 2.4 Indirect dispatch for cascade iterations

No per-iteration readback of a "converged?" flag. Instead:

- End-of-iter, the physics kernel writes indirect dispatch args `(workgroup_count, 1, 1)` to a small GPU buffer, where `workgroup_count = ceil(num_events_next_iter / PHYSICS_WORKGROUP_SIZE)` clamped to `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)`.
- When there are no follow-on events, the kernel writes `(0, 1, 1)`. Subsequent indirect dispatches are GPU no-ops (microseconds).
- `run_cascade_resident` pre-records `MAX_CASCADE_ITERATIONS` indirect dispatches into one encoder.
- Iteration count surfaces as an inferred value from the args buffer, read alongside `snapshot()`.

### 2.5 Submit shape

One command encoder records N ticks: each tick = mask + scoring + spatial + apply + movement + cascade-indirect × `MAX_CASCADE_ITERATIONS` + fold + tick-counter increment. One `queue.submit`. One `device.poll(Wait)` at end of batch.

Per-tick scalars that change (RNG seed, tick counter) live in the GPU-side `SimCfg` buffer (§3) updated by a tiny end-of-tick kernel.

### 2.6 Snapshot — double-buffered staging

Three staging-buffer pairs: `{agents, events, chronicle}` × `{front, back}`.

On call:
1. Encode `copy_buffer_to_buffer` for current `agents_buf` + `event_ring[last_read..tail]` + `chronicle_ring[last_read..tail]` into the **back** staging buffers. Update `last_read` watermarks.
2. `queue.submit`.
3. `map_async(Read)` on the **front** staging buffers (filled by the previous `snapshot()` call).
4. `device.poll(Wait)` — drives pending map callbacks.
5. Decode front staging → `GpuSnapshot`, unmap, swap front/back pointers.

First call returns `GpuSnapshot::empty()`. The one-frame lag is acceptable because the rendering layer interpolates via a delta value.

### 2.7 Additivity

- `SimBackend::step()` and its `GpuBackend` impl are unmodified by the resident path.
- Caller-provided `EventRing` is populated only by the sync path. Batch events are observable only via `snapshot().events_since_last`.
- Existing parity tests, perf tests, scenario tests run against the sync path.

### 2.8 Error cases

| Failure | Detection | Behaviour |
|---|---|---|
| Indirect args corruption | Kernel clamps `workgroup_count ≤ ceil(agent_cap / WGSIZE)` before write; snapshot validates | Logged warning; bounded dispatch |
| GPU ring overflow | Kernel writes `overflowed` flag; `snapshot()` reads it | `Err(SnapshotError::RingOverflow { tick, events_dropped })` |
| Cascade non-convergence | Indirect args still non-zero at last iteration | Warning on snapshot; correctness unaffected (subsequent ticks pick up) |
| Staging map failure | `map_async` callback returns `Err` | `Err(SnapshotError)`; first call returns `Ok(empty)` |
| Kernel dispatch failure | `wgpu` validation / device-lost | `Err(BatchError)`; **no CPU fallback** — caller re-issues via sync `step()` for graceful degradation |

### 2.9 Performance contract

- `chronicle --perf-sweep --batch-ticks N` is the gate.
- Target: batch mean µs/tick `< 0.8×` sync mean µs/tick at N ≥ 512.
- Looser at smaller N where per-tick overhead dominates.
- `PhaseTimings` records `batch_submit_us` and `batch_poll_us` so the sweep distinguishes encoding cost from GPU execution cost.

### 2.10 Backend factoring

`GpuBackend` is a thin composite of three sub-structs:

```rust
pub struct GpuBackend {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,
    backend_label: String,
    sync:     SyncPathContext,      // mask, scoring, view_storage, sync cascade
    resident: ResidentPathContext,  // resident_agents_buf, indirect_args, sim_cfg, unpack
    snapshot: SnapshotContext,      // front/back staging, watermarks
}
```

Each sub-struct owns its own lazy-init. `step()`, `step_batch()`, `snapshot()` delegate.

---

## 3. Sim state on GPU

### 3.1 `SimCfg` — shared GPU storage buffer

One storage buffer owns sim-wide scalars previously duplicated across per-kernel cfg uniforms:

```wgsl
struct SimCfg {
    tick:                          atomic<u32>,
    world_seed_lo:                 u32,
    world_seed_hi:                 u32,
    _pad0:                         u32,
    engagement_range:              f32,
    attack_damage:                 f32,
    attack_range:                  f32,
    move_speed:                    f32,
    move_speed_mult:               f32,
    kin_radius:                    f32,
    cascade_max_iterations:        u32,
    rules_registry_generation:     u32,
    abilities_registry_generation: u32,
    _reserved:                     array<u32, 4>,
}
```

- `tick` is `atomic<u32>` (storage binding, not uniform) because GPU writes it; all other kernels read it.
- `_reserved` is 16 bytes of headroom for future sim-wide scalars without bumping layout.
- `rules_registry_generation` / `abilities_registry_generation` are u32 cache-invalidation counters, incremented when CPU changes registry shape. Kernels key uploaded caches on equality.

Kernel-local cfg uniforms (workgroup size, slot indices, per-kernel thresholds) stay in their own small uniforms. `SimCfg` holds only sim-wide scalars.

### 3.2 Tick advance

The seed-indirect kernel that runs once per tick to seed cascade iter 0 grows one line:

```wgsl
atomicAdd(&sim_cfg.tick, 1u);
```

at end-of-tick. No new kernel, no new dispatch.

### 3.3 RNG

`per_agent_u32(agent_id, purpose_tag)` is a pure hash of `(world_seed, tick, agent_id, purpose_tag)`. No GPU atomic state; the function stays pure.

- `world_seed` uploaded once at `ensure_resident_init` to `SimCfg.world_seed_{lo,hi}`.
- `tick` read from `SimCfg.tick` instead of per-kernel cfg.

### 3.4 CPU `state.tick` during batch

Stale across the whole batch by design. `step_batch` does not advance `state.tick`. End of batch, `state.tick` is unchanged from start-of-batch. The next `snapshot()` reads `SimCfg.tick` and reports it via `GpuSnapshot.tick`. Callers wanting CPU `state.tick` to reflect reality call `snapshot()` first.

The sync `step()` path is unaffected — it advances `state.tick` on CPU each call (sync path doesn't yet use `SimCfg`).

### 3.5 `@cpu_only` annotation

DSL annotation marking rules that stay CPU-resident by design (chronicle narrative formatting, debug-only side effects, anything requiring strings or unbounded allocation).

```
@cpu_only physics chronicle_render_attack_narrative @phase(event) {
    on AgentAttacked { ... build String ... push chronicle text ... }
}
```

Compiler behaviour:

1. Records the annotation on the rule's IR node.
2. **Emits the CPU handler** as today (Rust function, registered in `CascadeRegistry::with_engine_builtins()`).
3. **Skips WGSL emission**. No entry in the GPU physics kernel's event dispatch table; no entry in the per-event-kind GPU dispatcher table (§4).
4. **Relaxes the GPU-emittable validator** — primitives it would otherwise reject (strings, unbounded alloc) are accepted inside `@cpu_only` rule bodies.

The implicit `@gpu_emittable` default is unchanged. Existing rules don't need annotations.

### 3.6 SimState mirroring summary

| State | CPU resident | GPU resident | Mirroring |
|---|---|---|---|
| Hot agent fields (pos, hp, alive, …) | `SimState.agents` | `resident_agents_buf` (`GpuAgentSlot`) | Upload at `ensure_resident_init`; snapshot copies back |
| Tick | `state.tick` | `SimCfg.tick` (atomic u32) | GPU advances; CPU reads via snapshot |
| World seed | `state.world_seed` | `SimCfg.world_seed_{lo,hi}` | Upload once at init |
| World scalars (engagement_range, …) | `state.config.combat.*` | `SimCfg.*` | Upload once at init |
| RNG state | (derived) | (derived) | Stateless: `hash(seed, tick, agent_id, purpose)` |
| Cold-state (gold, standing, memory) | `SimState.cold_*` | Side buffers (§4) | Upload at init; snapshot reads back |
| Event ring | Caller-provided | `event_ring_buf` | Sync: populated each tick. Batch: only via snapshot |
| Chronicle | (CPU `Vec`) | `chronicle_ring_buf` | Sync: drained per call. Batch: snapshot watermark |

### 3.7 Field-layout invariants

WGSL struct layout (alignment, padding) must match Rust struct layout. Compile-time assertions compare field offsets against hand-written WGSL constants. Drift panics at startup.

`SimCfg` storage binding is **storage**, not **uniform**, because `tick` is mutated by atomic. All other kernels declare it `read_only`.

---

## 4. Cold-state replay on GPU

### 4.1 Dispatch framework

One WGSL handler kernel per GPU-resident event kind. End of each tick's cascade iteration, the cascade driver scans observed events by kind and dispatches the matching kernel(s).

**Emission is automatic.** Every DSL `physics rule @phase(event) { on EventKind { ... } }` not marked `@cpu_only` produces (a) a WGSL kernel and (b) an entry in a generated dispatch-by-event-kind manifest. No hand-written dispatcher kernel — the driver is Rust glue reading the compiler-emitted manifest.

**Per kind, not per event.** A tick with 100 `AgentAttacked` events dispatches the rule kernel once; the kernel processes all 100 events in the slice. Event kinds with zero observations skip dispatch (CPU reads the per-kind event count, as it already does for the cascade seed kernel).

Placement: end-of-tick, after cascade converges, before next tick starts. All inside one command encoder — no new submits.

### 4.2 Event-kind dispatch table (generated)

| Event kind | Handler rule(s) | Storage written | Notes |
|---|---|---|---|
| `AgentAttacked` | `damage`, `opportunity_attack`, structured chronicle | `agents_buf.hp/shield`, chronicle ring | GPU-native |
| `AgentDied` | `fear_spread`, structured chronicle | `view_storage.kin_fear` | GPU-native |
| `EffectGoldTransfer` | `transfer_gold` | `gold_buf` (atomic add/sub) | §4.3 |
| `EffectStandingDelta` | `modify_standing` view fold | `standing` view storage | §4.4 |
| `RecordMemory` | `record_memory` view fold | `memory` view storage | §4.5 |
| `AgentCast` | `cast` physics rule (existing) | `agents_buf`, event ring | GPU-native |
| `chronicle_*_narrative` (text) | (none on GPU) | — | `@cpu_only` |

Registry refresh: when a rule is added to DSL, the dispatch manifest regenerates. Compile-time check: every non-`@cpu_only` rule appears in the manifest.

### 4.3 Gold

Storage:

```wgsl
struct GoldBuf {
    per_agent: array<atomic<i32>>, // length = agent_cap
}
```

- Dedicated buffer, not part of `GpuAgentSlot` — touched by one handler, no reason to bloat every kernel's view.
- ~8 KB at N=2048; scales linearly.
- Written by `transfer_gold` GPU handler (atomic add/sub); read by scoring rules that gate on gold (`gold > threshold`).
- `ensure_resident_init` uploads from `SimState.cold_inventory.gold`.
- `snapshot()` copies `gold_buf` back into `SimState.cold_inventory.gold`.

`transfer_gold` exists today as a DSL rule (`agents.sub_gold(from, a); agents.add_gold(to, a)`). After the gold-narrowing commit aligned i32 types, the existing DSL body auto-lowers to GPU; no body changes.

**Overflow.** i32 amounts; accumulating +2.1B of deltas wraps. Acceptable — gameplay caps are enforced at ability-design level, not at the kernel.

### 4.4 Standing

Reborn as a DSL `@materialized` view with the new `@symmetric_pair_topk(K=8)` annotation:

```
view standing
    @materialized
    @symmetric_pair_topk(K = 8)
    // pair key is (min(a,b), max(a,b)); value is i16 clamped to [-1000, 1000]
{
    on EffectStandingDelta { a, b, delta } {
        state[min(a,b), max(a,b)] += delta
    }
}
```

Compiler work:
1. Parse `@symmetric_pair_topk(K)`.
2. Emit CPU fold code (replaces hand-written `SparseStandings::adjust`).
3. Emit GPU fold kernel (analogous to `kin_fear` but: pair-symmetric storage, no decay, i16 clamp).
4. Delete hand-written `SparseStandings`. Consumers migrate to `state.views.standing.*`.

Per-agent storage: `N × (8 × 6 bytes + u32 count)` ≈ 56 KB at N=2048, 2.5 MB at N=100k. Bounded.

**K-budget overflow.** When 8 slots are full, the lowest `|standing|` entry is evicted (near-zero standing has no gameplay effect). Logged in debug builds.

### 4.5 Memory

Reborn as a DSL `@materialized` view with the new `@per_entity_ring(K=64)` annotation:

```
view memory
    @materialized
    @per_entity_ring(K = 64)
{
    on RecordMemory { observer, source, fact_payload, confidence } {
        push into state[observer] as MemoryEvent { source, fact_payload, confidence, tick }
    }
}
```

Compiler work:
1. Parse `@per_entity_ring(K)`.
2. Emit CPU fold (FIFO ring with eviction at K).
3. Emit GPU fold:
   - Per-agent ring with `cursor: u32`.
   - Push: `atomicAdd(cursor, 1)`, write at `ring[agent][cursor % K]`.
   - Reads return all K slots in cursor-relative order.
4. Delete hand-written memory smallvec. Consumers migrate to `state.views.memory.*`.

Per-agent storage: `N × (64 × 24 bytes + u32 cursor)` ≈ 3 MB at N=2048, ~150 MB at N=100k. At N=100k this is non-trivial; if it matters in profiling, K shrinks. For now, accept the cost.

**Ring overflow.** By design — oldest entry evicted. Not an error.

### 4.6 Chronicle

Two tiers, decided per-rule:

- **Structured chronicle** — fixed-layout `ChronicleEntry` (template id + fixed payload). Emitted GPU-side via the existing chronicle-emission DSL primitive into the existing GPU chronicle ring. Snapshot exposes via `chronicle_since_last`.
- **Narrative chronicle** — multi-sentence prose with string interpolation. Marked `@cpu_only`. Runs async off the batch snapshot's event stream (CPU consumer iterates `events_since_last`).

No new DSL grammar for chronicle — the existing chronicle-emission primitive handles structured entries; `@cpu_only` (§3.5) handles narrative.

### 4.7 Snapshot handshake

```
gold_buf       → copy into SimState.cold_inventory.gold
standing view  → copy into SimState.cold_standing (or expose via state.views)
memory view    → copy into SimState.cold_memory (or expose via state.views)
chronicle_ring → snapshot.chronicle_since_last (existing watermark)
```

`GpuSnapshot` gains optional fields gated by snapshot-flag config (caller doesn't pay for what they don't read).

### 4.8 Engine-core vs DSL-lowered

| Concern | Owner |
|---|---|
| Side-buffer allocation + binding (gold, standing, memory) | Engine-core (hand-written Rust + WGSL) |
| Dispatch table driver | Engine-core (Rust glue reads compiler manifest) |
| Snapshot handshake (readback into `SimState.cold_*`) | Engine-core |
| `ensure_resident_init` upload | Engine-core |
| Rule bodies (`transfer_gold`, `chronicle_*`, `modify_standing`, `record_memory`) | DSL-lowered |
| View fold bodies (standing, memory) | DSL-lowered |
| Annotation processing (`@symmetric_pair_topk`, `@per_entity_ring`, `@cpu_only`) | DSL-lowered |

---

## 5. Ability evaluation on GPU

### 5.1 Position in pipeline

```
mask → scoring → pick_ability → apply_actions → movement → spatial → cascade(N iters) → fold → cold_state_dispatch
                 ^^^^^^^^^^^^
                 new
```

Scoring runs first, producing per-agent action scores (attack/move/flee/hold). `pick_ability` runs next with its own output buffer. `apply_actions` reads both — if an ability is chosen, emit `AgentCast`; otherwise, emit the scoring kernel's chosen action.

### 5.2 `pick_ability` kernel

Compiler-emitted from DSL (`pick_ability.wgsl`). Per agent per tick:

- Iterates abilities in the agent's known set.
- Evaluates each ability's `guard`. Skips on false.
- Evaluates each ability's `score`. Tracks argmax.
- Picks target via the ability's `target:` clause.
- Writes `chosen_ability_buf[agent]` = packed `(ability_slot: u8, target_agent_id: u32, sentinel-for-no-cast)`.

If no ability's guard passes, writes the no-cast sentinel.

### 5.3 New scoring grammar — `ability::tag(TAG)` primitive

Reads the tag value for the ability being scored in the current row. Returns `f32`, or `0` if the ability lacks the tag.

```
score = ability::tag(PHYSICAL) * (1 - target.hp_frac)
      + ability::tag(CROWD_CONTROL) * engaged_with_kin
```

Also `ability::hint` — a string-enum compare against the ability's coarse category (`damage` / `defense` / `crowd_control` / `utility`):

```
score = (if ability::hint == damage { 1.0 } else { 0.0 }) * ...
```

`ability::range` — pulls from `Area::SingleTarget { range }` (other `Area` variants deferred).

`ability::on_cooldown(ability)` — predicate, reads from per-slot cooldown buffer.

### 5.4 New scoring grammar — `per_ability` row type

Today scoring rows run once per agent (one score for "attack nearest," "flee," etc.). `per_ability` rows iterate over an agent's abilities and produce one score per (agent, ability):

```
row pick_ability per_ability {
    guard:    !ability::on_cooldown(ability)
    score:    ability::tag(PHYSICAL) * (1 - target.hp_frac)
            + ability::tag(CROWD_CONTROL) * (if engaged_with_hostile { 0.5 } else { 0.0 })
            + ability::tag(DEFENSE) * (if self.hp_frac < 0.3 { 1.0 } else { 0.0 })
    target:   nearest_hostile_in_range(ability::range)
}
```

Output per agent: `(ability_slot, target_agent_id)` of the highest-scoring ability whose guard passes; sentinel if none.

### 5.5 Tag registry

Each `.ability` file carries:

- `hint:` — coarse category enum (`damage` | `defense` | `crowd_control` | `utility`). One per ability.
- Per-effect `[TAG: value]` — numeric power ratings, multiple per effect line.

Tag names are a fixed enum (each tag is a known buffer index; lower-cost than a symbol table). User-extensible string tags are deferred.

`AbilityDef.tags` serialise into the `PackedAbilityRegistry` consumed by kernels. The tag table is bound to `pick_ability` alongside the ability registry.

### 5.6 Target selection

`nearest_hostile_in_range(range)` uses the existing GPU spatial-hash output (kin + hostile result buffers from §6 spatial kernel). Pointer-attention / learned targeting / lowest-HP heuristics are deferred — this subsystem ports the existing CPU heuristic only.

### 5.7 Side buffer

```wgsl
struct ChosenAbilityBuf {
    per_agent: array<u64>, // packed: (ability_slot: u8, target_agent_id: u32, _pad: u24)
}
```

One u64 per agent. Sentinel value indicates no-cast. Consumed by `apply_actions`:

```
if chosen_ability_buf[agent] is non-sentinel and cooldowns pass:
    emit AgentCast { caster, ability, target }
    apply cooldown
else:
    emit score_output's chosen action
```

The `cast` physics rule (already GPU-native) handles the downstream effects.

### 5.8 Engine-core vs DSL-lowered

| Concern | Owner |
|---|---|
| `chosen_ability_buf` allocation + binding | Engine-core |
| `pick_ability` kernel dispatch in `step_batch` | Engine-core |
| `apply_actions` extension to read `chosen_ability_buf` and emit `AgentCast` | Engine-core |
| Tag table serialisation into ability registry (if missing) | Engine-core |
| `pick_ability.wgsl` (compiler output from `per_ability` row) | DSL-lowered |
| CPU `pick_ability` handler (compiler output, replaces hand-tuned `evaluate_hero_ability`) | DSL-lowered |
| Tag reads + scoring arithmetic | DSL-lowered (`scoring.sim`) |

### 5.9 Failure modes

| Failure | Behaviour |
|---|---|
| Ability tag missing | `ability::tag(UNKNOWN)` returns 0 silently (tags are sparse per ability) |
| All abilities on cooldown | `pick_ability` writes sentinel; `apply_actions` falls through to score_output |
| Tag value overflow | f32 in WGSL; designer-visible. Not a runtime error |
| Cooldown buffer not initialised | Verified at `ensure_resident_init`; panic on missing |
| Per-agent ability count variance | Empty slots score 0; ~5 wasted evaluations per agent per tick at MAX_ABILITIES=8 |

---

## 6. Pipeline reference

### 6.1 Sync path

```
step(state, events, cascade) -> state'
├─ 1. mask.run_batch()
│     reads:  agent pos/alive/creature_type
│     writes: 7 × bitmap buffers
├─ 2. scoring.run_batch()
│     reads:  bitmaps, agent fields, view_storage
│     writes: ScoreOutput SoA
├─ 3. apply_actions.run_batch()
│     reads:  ScoreOutput, agent slots
│     writes: hp/shield/alive, event_ring
├─ 4. movement.run_batch()
│     reads:  ScoreOutput, agent slots
│     writes: pos, event_ring
├─ 5. cascade.run_cascade()
│     for iter in 0..MAX_CASCADE_ITERATIONS:
│         physics.run_batch(events_in) -> events_out
│         fold_iteration_events(events_out) -> view_storage
│         break if events_out.is_empty()
├─ 6. cold_state_replay() [CPU]
│     gold, standing, memory mutations from drained events
└─ 7. finalize() [CPU]
     tick++, invariant checks, telemetry
```

### 6.2 Resident path

```
step_batch(n)
├─ ensure_resident_init()
│     allocate resident_agents_buf, sim_cfg_buf, cascade_resident_ctx (first call)
└─ for tick in 0..n:
    ├─ 1. fused_unpack_kernel
    │     reads:  resident_agents_buf
    │     writes: mask SoA + scoring agent_data_buf
    ├─ 2. mask_resident
    │     reads:  mask SoA
    │     writes: bitmap buffers
    ├─ 3. scoring_resident
    │     reads:  bitmaps, agent_data, view_storage
    │     writes: ScoreOutput
    ├─ 4. pick_ability               (§5)
    │     reads:  agent_data, ability_registry, tag_table, cooldowns_buf, spatial outputs, sim_cfg
    │     writes: chosen_ability_buf
    ├─ 5. apply_actions.run_resident
    │     reads:  ScoreOutput, chosen_ability_buf, resident_agents_buf
    │     writes: hp/shield/alive, batch_events_ring
    ├─ 6. movement.run_resident
    │     reads:  ScoreOutput, resident_agents_buf
    │     writes: pos, batch_events_ring
    ├─ 7. append_events
    │     reads:  batch_events_ring tail
    │     writes: apply_event_ring
    ├─ 8. seed_kernel
    │     atomicAdd(sim_cfg.tick, 1); seed indirect_args[0]; clear num_events[1..N]
    ├─ 9. for iter in 0..MAX_CASCADE_ITERATIONS:
    │     physics.run_batch_resident (indirect)
    │         reads:  apply_event_ring[iter]
    │         writes: resident_agents_buf, physics_ring[iter+1], indirect_args[iter+1]
    │     break if indirect_args[iter+1] == (0,1,1)
    └─ 10. cold_state_dispatch       (§4)
          per event kind K with count > 0: dispatch K's handler kernel
          view fold kernels (engaged_with, my_enemies, threat_level, kin_fear,
                             pack_focus, rally_boost, standing, memory)
```

### 6.3 Snapshot

```
snapshot() -> GpuSnapshot
├─ poll front staging buffer (non-blocking from previous snapshot)
├─ copy_buffer_to_buffer(live GPU buffers -> back staging)
└─ swap front/back
```

One-frame lag. The returned snapshot contains data from the tick before the current one. The double-buffer prevents GPU→CPU sync on the hot path.

### 6.4 Kernel inventory

#### 6.4.1 Mask kernel

| Property | Value |
|---|---|
| Struct | `FusedMaskKernel` (sync), `MaskUnpackKernel` (resident unpack) |
| Entry points | `cs_fused_masks` (sync/batch); `cs_mask_unpack` (resident unpack) |
| Workgroup size | 64 |
| Bind group | Sync: agents, 7 bitmap outputs, cfg. Resident unpack: resident_agents_buf → mask SoA |
| Inputs | Agent position, alive, creature_type; ConfigUniform (movement radius) |
| Outputs | 7 bitmap arrays: Attack, MoveToward, Hold, Flee, Eat, Drink, Rest (atomic u32 per agent) |
| Notes | Cast mask covered by `pick_ability` (§5); fused dispatch writes all 7 in one call |

#### 6.4.2 Scoring kernel

| Property | Value |
|---|---|
| Struct | `ScoringKernel` (sync), `ScoringUnpackKernel` (resident) |
| Entry point | `cs_scoring` |
| Workgroup size | 64 |
| Bind group | agent_data SoA, bitmaps, view_storage (atomic reads), cfg, sim_cfg, spatial queries |
| Inputs | 7 mask bitmaps; agent fields; view_storage atomic reads (my_enemies, threat_level, kin_fear, pack_focus, rally_boost) |
| Outputs | `ScoreOutput[agent_cap]` — per-agent struct: chosen_action, chosen_target, score |
| Notes | Spatial query reads are read-only on precomputed kin / nearest-hostile results |

#### 6.4.3 Pick ability kernel

| Property | Value |
|---|---|
| Struct | (compiler-emitted) |
| Entry point | `cs_pick_ability` |
| Workgroup size | 64 |
| Bind group | agent_data, ability_registry, tag_table, ability_cooldowns_buf, spatial outputs, sim_cfg, chosen_ability_buf (write) |
| Inputs | Agent state, packed ability registry, cooldowns, spatial-query results |
| Outputs | `chosen_ability_buf[agent_cap]` — packed `(slot, target, sentinel)` u64 |
| Notes | Compiler-emitted from `per_ability` rows in `scoring.sim` (§5.4) |

#### 6.4.4 Apply actions kernel

| Property | Value |
|---|---|
| Struct | `ApplyActionsKernel` |
| Entry points | `cs_apply_actions` (sync), `cs_apply_actions_resident` (batch) |
| Workgroup size | 64 |
| Bind group | agents (rw), scoring (r), chosen_ability_buf (r, batch), event_ring (rw), event_ring_tail (atomic), cfg, sim_cfg |
| Inputs | ScoreOutput; agent slots; chosen_ability_buf (batch only) |
| Outputs | Mutated hp/shield/alive; events: AgentAttacked, AgentDied, AgentCast (batch), AgentAte, AgentDrank, AgentRested |
| Scope gaps | Opportunity attacks, engagement slow on MoveToward, announce/communicate — sync CPU path |

#### 6.4.5 Movement kernel

| Property | Value |
|---|---|
| Struct | `MovementKernel` |
| Entry points | `cs_movement` (sync), `cs_movement_resident` (batch) |
| Workgroup size | 64 |
| Bind group | agents (rw), scoring (r), event_ring (rw), event_ring_tail (atomic), cfg, sim_cfg |
| Inputs | ScoreOutput; agent slots (pos, slow_factor_q8) |
| Outputs | Updated pos; events: AgentMoved, AgentFled |
| Math | MoveToward: `pos + normalize(target - pos) * move_speed`. Flee: `pos + normalize(pos - threat) * move_speed` |
| Scope gaps | Kin-flee-bias (herding), effect slow multiplier — deferred |

#### 6.4.6 Physics kernel (event processor)

| Property | Value |
|---|---|
| Struct | `PhysicsKernel` |
| Entry points | `cs_physics` (sync), `cs_physics_resident` (batch) |
| Workgroup size | 64 |
| Bind group (sync) | agents SoA (rw), event_ring_in (r), event_ring_out (rw), event_ring_tail (atomic), view_storage (atomic), spatial (kin/hostile, r), abilities (r), cfg, sim_cfg, chronicle_ring (rw), chronicle_ring_tail (atomic) |
| Bind group (resident) | Same as sync, plus indirect_args, num_events_buf, resident_cfg |
| Inputs | Event batch (one per thread); agent SoA; pre-computed kin / nearest-hostile; ability registry; SimCfg |
| Outputs | Mutated agent state (hp, shield, stun, slow, engaged_with, alive); new events; chronicle entries |
| Determinism | Emits events in non-deterministic order (atomic tail racing). Host drain sorts by `(tick, kind, payload[0])` pre-fold |

Chronicle rings:
- Sync path: `PhysicsKernel::chronicle_ring` (bindings 11–12). Drained separately by `GpuBackend::drain_chronicle_ring()`.
- Resident path: `CascadeResidentCtx::chronicle_ring` (caller-owned). Snapshot reads via watermark.

#### 6.4.7 Fold kernels (view materialization)

| Property | Value |
|---|---|
| Entry points | `cs_fold_<view_name>` — one per view (engaged_with, my_enemies, threat_level, kin_fear, pack_focus, rally_boost, **standing**, **memory**) |
| Workgroup size | 64 |
| Bind group | fold_inputs (r), view_storage (rw atomic), cfg |
| Inputs | FoldInput batch (observer_id, other_id, delta, anchor_tick) |
| Outputs | View storage atomic updates (CAS loop) |
| Determinism | Commutative folds (`+= 1.0`) — atomic CAS order doesn't matter for the sums |

#### 6.4.8 Spatial hash kernels (resident path)

| Property | Value |
|---|---|
| Entry points | `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query` |
| Inputs | Agent positions; two radii (kin=12m, engagement=2m) |
| Outputs | Per-agent query results: nearby agents (within), kin-species membership, nearest hostile/kin (one u32 per agent) |
| Notes | Resident path uses GPU spatial hash; sync path uses CPU spatial hash |

### 6.5 Buffer ownership

| Buffer | Owner | Size (N=100k) | Purpose |
|---|---|---|---|
| `resident_agents_buf` | `ResidentPathContext` | ~16 MB | Agent SoA, persistent across batch |
| `sim_cfg_buf` | `ResidentPathContext` | 256 B | `SimCfg` (§3.1) — atomic tick + world scalars |
| `apply_event_ring` | `CascadeCtx` (sync) | ~24 MB | Seeds physics iter 0; cleared per-tick |
| `physics_ring_a` / `_b` | `CascadeResidentCtx` | ~24 MB each | Ping-pong resident cascade rings |
| `batch_events_ring` | `CascadeResidentCtx` | ~24 MB | Append-only accumulator across batch ticks; exposed to snapshot |
| `chronicle_ring` (sync) | `PhysicsKernel` | ~24 MB | Narrative records; drained separately |
| `chronicle_ring` (resident) | `CascadeResidentCtx` | ~24 MB | Caller-owned; snapshot watermark |
| `indirect_args` | `ResidentPathContext` | 32 B × (`MAX_CASCADE_ITERATIONS` + 1) | Indirect dispatch args per iteration |
| `num_events_buf` | `CascadeResidentCtx` | 4 B × (`MAX_CASCADE_ITERATIONS` + 1) | Event counts per iteration (diagnostic) |
| `view_storage` | `ViewStorage` | ~144 MB (6 views @ N=100k) | Materialised view state (incl. standing, memory in §4) |
| `gold_buf` | `ResidentPathContext` | ~400 KB | Per-agent atomic i32 gold (§4.3) |
| Spatial query outputs | `SpatialOutputs` | ~80 MB each | kin / engagement query results |
| `PackedAbilityRegistry` | engine | ~256 KB | Ability metadata; resident with content-addressed upload |
| `chosen_ability_buf` | `ResidentPathContext` | 8 B × N | Per-agent `(slot, target, sentinel)` u64 (§5.7) |

### 6.6 Key constants

| Constant | Value |
|---|---|
| `MAX_CASCADE_ITERATIONS` | 8 |
| `PHYSICS_WORKGROUP_SIZE` | 64 |
| `MAX_EFFECTS` (per ability program) | 8 |
| `MAX_ABILITIES` | 256 |
| `DEFAULT_CAPACITY` (event ring) | 655,360 |
| `DEFAULT_CHRONICLE_CAPACITY` | 1,000,000 |
| `PAYLOAD_WORDS` (per event) | 8 |
| `K` (spatial query cap) | 32 |
| `FOLD_WORKGROUP_SIZE` | 64 |

### 6.7 DSL → WGSL lowering

Per-subsystem emitters in `dsl_compiler` produce WGSL modules:

- `emit_mask_wgsl` → fused module with per-mask bitmap writes.
- `emit_physics_wgsl` → one module with `physics_dispatch(event)` switch over event kinds.
- `emit_view_wgsl` → one fold kernel per view.
- `emit_pick_ability_wgsl` → `pick_ability.wgsl` from `per_ability` rows (§5).

The host assembles fragments, pipes through `naga`/wgpu, produces compute pipelines.

Physics shader assembly produces a dispatch table:

```wgsl
fn physics_dispatch(event_idx: u32) {
    let event = event_ring_in[event_idx];
    switch(event.kind) {
        case AgentAttacked: rule_damage(); rule_opportunity_attack(); break;
        case AgentDied: rule_fear_spread(); break;
        // ...
    }
}
```

Resident shader is the same as sync, plus bindings for `indirect_args`, `num_events_buf`, `resident_cfg`, and a `cs_physics_resident` entry point that uses indirect dispatch. Chronicle ring is caller-supplied (same binding structure, different buffer).

---

## 7. Cross-cutting concerns

### 7.1 Determinism contract

| Path | Determinism |
|---|---|
| Sync GPU step | Same-seed reproducible on same hardware. Event tail atomic ordering serialised by per-call host fence. Host drain sorts events by `(tick, kind, payload[0])` pre-fold |
| Batch GPU step | **Non-deterministic in event order.** Atomic tail racing inside the resident cascade does not serialise. Same-seed runs may diverge in fold order. Statistical parity (alive counts, event multisets, conservation laws) is the contract |
| Cross-GPU | Same hardware + same seed reproduces. Different vendors may diverge — non-commutative GPU folds |
| RNG | Pure hash `(world_seed, tick, agent_id, purpose_tag)`. Stateless. Identical across CPU and GPU given identical inputs |

Determinism tests run against `SerialBackend` and against the GPU sync path. The batch path is explicitly excluded from byte-exact tests.

### 7.2 Schema hash — GPU surface

The `SCORING_HASH` and engine schema hash include:

- `SimCfg` field layout (offsets, sizes).
- Event-kind dispatch manifest (rule names, kinds, registration order).
- Ability registry packed format (tag table layout, slot count).
- `chosen_ability_buf` packing format.
- View storage layouts for `@symmetric_pair_topk` and `@per_entity_ring` annotated views.

CI fence: changes to any of the above bump `crates/engine/.schema_hash`. Drift fails the parity tests at startup.

`rules_registry_generation` and `abilities_registry_generation` (in `SimCfg`) are u32 cache-invalidation counters — independent of the schema hash, used for kernel-local upload caches.

### 7.3 Parity test contract

| Test | Path | Asserts |
|---|---|---|
| `parity_with_cpu` | Sync GPU vs `SerialBackend` | Byte-exact agent state + event multiset at N=50 |
| `physics_parity`, `cascade_parity` | Sync GPU vs `SerialBackend` | Byte-exact per-iteration cascade outputs |
| `perf_n100` | Sync GPU | Alive count within ±25% of `SerialBackend` at N=100 |
| `async_smoke` | Batch path | Tick advance, agent count, alive count within ±25%, events present, chronicle present |
| `snapshot_double_buffer` | Batch path | First call empty; subsequent calls non-overlapping watermark windows |
| `cold_state_4*` | Batch path | Gold conservation, standing pair-symmetric clamp, memory FIFO eviction |
| `pick_ability_*` | Batch path | Cooldown respected, range-gated targeting, cast events emitted |
| Cross-path parity (sync ↔ batch) | **Excluded.** Non-deterministic by design |
| Statistical parity (sync ↔ batch) | Allowed | Cast cadence within ±25%, ability-type distributions, gold conservation laws |

### 7.4 Telemetry

`PhaseTimings` extends with:

- `batch_submit_us` — encoder record + `queue.submit` cost.
- `batch_poll_us` — `device.poll(Wait)` cost (GPU execution).
- Per-iteration cascade dispatch counts (for non-convergence diagnosis).
- Per-event-kind dispatch counts (cold-state replay).

`chronicle --perf-sweep --batch-ticks N` runs the sweep at varying batch sizes; CI gates at N=2048; local runs at N=100k.

### 7.5 Failure surface summary

| Failure mode | Detection | Behaviour |
|---|---|---|
| Indirect args corruption (§2.8) | Kernel clamp + snapshot validate | Logged warning |
| Ring overflow (event / chronicle) | Kernel sets flag; snapshot reads | `Err(SnapshotError::RingOverflow)` |
| Cascade non-convergence | Final-iter args still non-zero | Warning; subsequent ticks pick up |
| Staging map failure | `map_async` callback `Err` | `Err(SnapshotError)` |
| Kernel dispatch failure | wgpu validation | `Err(BatchError)`; **no CPU fallback** |
| `SimCfg` field drift | Compile-time offset assertion | Startup panic |
| Missing dispatch entry (rule added, manifest stale) | Compile-time check | Build fail |
| Standing topk overflow | K=8 budget full | Lowest `\|standing\|` evicted; logged in debug |
| Memory ring overflow | By design | Oldest entry evicted; not an error |
| Gold atomic overflow | i32 wrap | Acceptable; cap at ability-design layer |
| Tag missing on ability | `ability::tag(MISSING)` | Returns 0 silently |
| All abilities on cooldown | `pick_ability` writes sentinel | Falls through to scoring kernel's chosen action |
| Cooldown buffer not initialised | `ensure_resident_init` check | Startup panic |

### 7.6 Engine-core boundary

The standing rule "engine core = hand-written; game logic = DSL" applies on GPU:

- **Engine-core (hand-written Rust + WGSL)**: buffer allocation, bind-group layouts, dispatch driver, indirect-args plumbing, snapshot handshake, `ensure_resident_init`, `SimCfg` struct, sub-struct factoring.
- **DSL-lowered (compiler output)**: all rule bodies, view fold bodies, `pick_ability` kernel, scoring expressions, annotation processing (`@cpu_only`, `@symmetric_pair_topk`, `@per_entity_ring`).

