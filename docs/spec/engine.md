# Engine Specification

Runtime contract for the engine the DSL compiler targets. The engine is the **unified runtime for all compiler output**. Whatever the compiler emits — scalar Rust, GPU dispatch code, SPIR-V shader source — lands in the engine. The engine ships **two first-class backends** that implement the same interface:

- **`SerialBackend`** — host-resident state, scalar Rust execution. The **reference implementation**. Determinism oracle for every test, every parity check against the GPU backend, every verified-correct port.
- **`GpuBackend`** — GPU-resident state via `voxel_engine::GpuHarness`, compiled SPIR-V kernels for gather + apply phases. Performance path for large N.

An implementation is correct if and only if it satisfies §§1–13 AND **produces byte-identical `replayable_sha256()` across both backends on the same seed**.

Companion to `dsl.md` (language reference + compiler contract), `state.md` (field catalog), and the per-subsystem status in `docs/engine/status.md`.

---

## 1. Scope

The engine is a Rust library crate (`crates/engine/`) that provides runtime primitives the compiler targets, plus the two concrete backends.

**Owned by the engine:**

- Generic `Pool<T>` with `NonZeroU32` IDs, freelist, slot reuse.
- Event ring with byte-stable SHA-256 over the replayable subset.
- 3D spatial index with voxel-chunk-keyed dispatch (CPU BTreeMap + z-sort for Serial; GPU hash for Vulkan).
- Per-world RNG with shader-derivable per-agent keyed sub-streams; constants pinned across CPU/GPU.
- Universal `MicroKind` (18 variants) and `MacroKind` (4 variants) with built-in execution for both backends.
- `MaskBuffer` layout + per-predicate dispatch (backend selects how to execute).
- `PolicyBackend` trait — abstracts over host-scalar and GPU-kernel implementations.
- `MaterializedView` / `LazyView` / `TopKView` traits with backend-local storage implementations.
- `TrajectoryWriter` + `TrajectoryReader` over safetensors; input comes from whichever backend is active.
- State snapshot + load with schema-hash versioning; covers both backends' state formats.
- `Invariant`, `Probe`, `TelemetrySink` traits — backend-agnostic where state access is abstracted; CPU-side for dispatch/sink mechanics.
- Debug & trace runtime (`trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, `snapshot`) — downloads from GPU as needed; serial-backend access is direct.
- SPIR-V bytecode for engine-universal kernels (embedded via `include_bytes!`).
- `ComputeBackend` trait + `SerialBackend` + `GpuBackend` concrete impls; runtime backend selection.

**Not owned by the engine** (deferred to compiler / domain):

- DSL parser and codegen.
- Verb desugaring, `Read → Ask(doc, AboutAll)` lowering.
- Domain types (which items exist, which abilities exist, which group kinds exist).
- Cascade *rules* (registered at init; engine provides the dispatch runtime for both backends — compiler emits CPU closures for `SerialBackend` and SPIR-V kernels for `GpuBackend`).
- Chronicle prose templates.
- Curriculum pipelines.

---

## 2. Backends abstraction

The engine has a **backend-agnostic tick pipeline** plus two backend implementations.

```
┌─────────────────────────────────────────────────────────────┐
│              Tick Pipeline (backend-agnostic)               │
│                                                             │
│  phase1: mask_build(backend, state, mask)                  │
│  phase2: policy_eval(backend, state, mask, actions)        │
│  phase3: shuffle_actions(actions, seed, tick)              │
│  phase4: apply_actions+cascade(backend, actions, state,    │
│                                 events, cascade_registry)  │
│  phase5: view_fold(backend, events, views)                 │
│  phase6: invariants + telemetry (host)                     │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐           ┌──────────────────┐
│  SerialBackend   │           │    GpuBackend    │
│                  │           │                  │
│  state: Vec<T>   │           │  state:          │
│  events: Vec     │           │    FieldHandle   │
│  mask: Vec<bool> │           │  events: GPU ring│
│                  │           │  mask: FieldH    │
│  exec: scalar    │           │  exec: SPIR-V    │
│   Rust loops     │           │   compute kernels│
└──────────────────┘           └──────────────────┘
```

**Both backends implement the same public API** — `agent_pos(id)`, `event_ring()`, `mask_buffer()` — so test code is backend-agnostic.

**Host orchestrator responsibilities** (same for both backends):
- Own the scheduler (`step()`, `step_full()`)
- Own user-facing API and debug runtime
- Pump telemetry (sinks are always host-side)
- Drive save/load serialization
- Run invariants (which may trigger a GPU→host sync for state access)

**Backend-local responsibilities:**
- State storage (where and how)
- Kernel dispatch (what compute runs where)
- Event emission (how writes happen)
- Cascade execution (CPU closures vs SPIR-V kernels)

The GPU backend exposes two execution modes:

- **Sync mode** — `SimBackend::step(state, intents, dt_ms)`. CPU-driven cascade with one fence per kernel dispatch. Authoritative for parity tests; matches `SerialBackend` semantics.
- **Batch mode** — `step_batch(state, n, cascade)`. GPU-resident cascade with indirect dispatch. One submit per N ticks, observation via non-blocking `snapshot()`.

---

## 3. SimState shape contract

Both backends expose the **same public SimState API**: `agent_pos(id) -> Option<Vec3>`, `set_agent_hp(id, hp)`, `agents_alive() -> impl Iterator<AgentId>`, etc. Callers don't know or care where the data lives. For the field catalog, see `state.md`.

### SerialBackend residency

- All state lives in host-resident `Vec<T>` per SoA hot field.
- Per-agent scalar access is direct indexing.
- No sync semantics — every read and write is immediate.

### GpuBackend residency

- **Hot fields are GPU-authoritative**, stored in `FieldHandle` per field (`pos`, `hp`, `max_hp`, `alive`, `movement_mode`).
- **Host maintains per-field dirty-tracked mirror**. Mirrors are always present but may be stale.
- Read path: `agent_pos(id)` syncs the `hot_pos` field from GPU if marked stale, then reads from mirror.
- Write path: `set_agent_pos(id, p)` writes to mirror AND marks the field dirty for upload at next tick boundary (or immediately if in an apply phase that needs the update visible to subsequent kernels within the same tick).
- Cold fields (`creature_type`, `channels`, `spawn_tick`) are host-only — GPU kernels don't need them during gather phases.

### Sync semantics

A **sync point** is any place where host and GPU ownership must align. Fixed sync points per tick:

| Phase | Sync | What happens |
|---|---|---|
| Before phase 1 | `upload_dirty` | Host→GPU push of any fields marked dirty since last tick |
| After phase 4 (apply) | `download_events` | GPU→host drain of events pushed during apply |
| After phase 4 (apply) | `download_state_if_needed` | Optional — only if invariants / trajectory / debug require host-readable state |
| After phase 5 (views) | `download_view_results` | GPU→host pull of view buffers that invariants or telemetry read |

Between sync points, **GPU is authoritative for hot fields**, mirror is stale.

### Snapshot semantics

For save/load (§11) and trajectory emission, the engine **forces a full sync** — all dirty fields uploaded, all GPU-authoritative state downloaded, mirror becomes clean.

Implementation: `crates/engine/src/state/` holds the backend-agnostic API; `crates/engine/src/backend/serial.rs` and `crates/engine/src/backend/gpu.rs` provide the residency layers.

### `SimCfg` — GPU shared storage buffer

One storage buffer owns sim-wide scalars on the GPU-resident path:

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

- `tick` is `atomic<u32>` because GPU writes it; all other kernels read it.
- `_reserved` is 16 bytes of headroom for future sim-wide scalars.
- `rules_registry_generation` / `abilities_registry_generation` are u32 cache-invalidation counters, independent of the schema hash.
- Kernel-local cfg uniforms (workgroup size, slot indices, per-kernel thresholds) stay in their own small uniforms.

WGSL struct layout (alignment, padding) must match Rust struct layout. Compile-time assertions compare field offsets against hand-written WGSL constants. Drift panics at startup.

---

## 4. Per-phase contracts

### 4.1 Tick pipeline overview

Six phases; each dispatched through the current `ComputeBackend`.

| Phase | Host-side vs backend-dispatched | Both backends |
|---|---|---|
| 1. Mask build | Backend-dispatched | Serial: scalar predicates; GPU: kernel dispatch |
| 2. Policy eval | Backend-dispatched | Serial: PolicyBackend scalar; GPU: PolicyBackend kernel |
| 3. Action shuffle | **Host-side** | Shuffle operates on the ActionBuffer after download (GPU) or in-place (Serial); seed-deterministic |
| 4. Apply + cascade | Backend-dispatched | Serial: Rust closures; GPU: SPIR-V kernels |
| 5. View fold | Backend-dispatched | Serial: scalar fold; GPU: sorted-key reduction |
| 6. Invariant + telemetry | **Host-side** | Backends download state snapshot as needed; invariants read mirror |

Phase 3 (shuffle) is host-side because Fisher-Yates is inherently sequential and the download-shuffle-upload cost is negligible. Phase 6 is host-side because invariants + telemetry are inherently small, low-frequency, and pragmatic to run on the host.

Full signature:

```rust
pub fn step_full<B: PolicyBackend>(
    backend:    &mut dyn ComputeBackend,
    state:      &mut SimState,
    scratch:    &mut SimScratch,
    events:     &mut EventRing,
    policy:     &B,
    cascade:    &CascadeRegistry,
    views:      &mut [&mut dyn MaterializedView],
    invariants: &InvariantRegistry,
    telemetry:  &dyn TelemetrySink,
);
```

### 4.2 Mask build (phase 1)

**Serial:** `Vec<bool>` per head. Predicates mutate via `impl MaskBuffer { fn mark_hold_allowed(&mut self, state: &SimState) }`.

**GPU:** FieldHandle per head (each bit stored as u32, one thread per bit). Predicates are SPIR-V kernels that write per-slot bits based on state queries. Universal predicates (hold, move-allowed, flee-if-threat, attack-in-range, needs, domain-hook) have engine-shipped GPU kernels. Domain-specific predicates come from the compiler.

```rust
pub trait Mask {
    fn reset(&mut self, backend: &mut dyn ComputeBackend);
    fn mark_hold_allowed(&mut self, backend: &mut dyn ComputeBackend, state: &SimState);
    fn mark_move_allowed_if_others_exist(&mut self, backend: &mut dyn ComputeBackend, state: &SimState);
    // ... etc.
}
```

**Mask validity invariant** (§4.7): GPU backend downloads the relevant mask slice before invariant check.

### 4.3 Scoring / policy eval (phase 2)

`PolicyBackend` trait abstracts over execution mode:

```rust
pub trait PolicyBackend: Send + Sync {
    fn evaluate(
        &self,
        backend: &mut dyn ComputeBackend,
        state:   &SimState,
        mask:    &MaskBuffer,
        actions: &mut ActionBuffer,
    );
}
```

`ActionBuffer` is backend-local: host `Vec<Action>` for Serial, FieldHandle for GPU.

- **`UtilityBackend`** — both variants. Serial: scalar argmax over masked score table. GPU: parallel argmax kernel per agent. Both produce the same argmax (integer tie-break on MicroKind ordinal).
- **`NeuralBackend`** — GPU-only in the MVP. Runs compiler-emitted matmul kernels for forward pass. Serial impl is a `todo!()` stub.

Cross-backend parity: for every policy + seed + state, `SerialBackend::evaluate` and `GpuBackend::evaluate` produce byte-identical `ActionBuffer` contents (after download).

### 4.4 Action select / shuffle (phase 3)

Per-tick agent action ordering is shuffled via deterministic Fisher-Yates keyed on `per_agent_u32(seed, AgentId(1), tick << 16, b"shuffle")`. Both backends apply the SAME shuffle sequence.

### 4.5 Apply actions (phase 4)

**Action space — MicroKind:**

```rust
#[repr(u8)]
pub enum MicroKind {
    Hold = 0, MoveToward, Flee,
    Attack, Cast, UseItem,
    Harvest, Eat, Drink, Rest,
    PlaceTile, PlaceVoxel, HarvestVoxel,
    Converse, ShareStory,
    Communicate, Ask,
    Remember,
}
```

**Execution per backend:**

| MicroKind | Serial execution | GPU execution |
|---|---|---|
| Hold | No-op | No-op |
| MoveToward | CPU scalar pos update | `apply_move_toward` kernel |
| Flee | CPU scalar pos update | `apply_flee` kernel |
| Attack | CPU scalar HP deduction + conditional kill | `apply_attack` kernel w/ atomic write to alive buffer on kill |
| Eat/Drink/Rest | CPU scalar need-restoration | `apply_needs_restore` kernel |
| Cast/UseItem/etc. | Emit event only (no state mutation in engine) | Same; compiler-registered cascade handles the effect |

The apply kernel's job: read `scratch.actions` FieldHandle, filter by MicroKind, write mutations to state fields + emit events into the GPU event ring via atomic append.

**MacroKind** — four variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`) + `NoOp`. The Announce cascade runs as:

- Serial: `for obs in state.agents_alive() { ... }` loop with **3D Euclidean** distance check, emits RecordMemory per recipient.
- GPU: `apply_announce` kernel — parallel **3D distance** check per agent, atomic-append to event ring for each match, bounded by `MAX_ANNOUNCE_RECIPIENTS` via an atomic counter early-exit.

**Distance semantics:** 3D Euclidean across both backends. Agents at different elevations are evaluated by full 3D distance, not planar (XZ-only). Confirmed 2026-04-26 (status.md Q#1 resolved).

### 4.6 Cascade dispatch (phase 4, continued)

Cascade is GPU-dispatchable, with Serial as the reference.

**Serial:** `CascadeRegistry` holds `Box<dyn CascadeHandler>`. `run_fixed_point(state, events)` walks new events, dispatches handlers in lane order, bounded by `MAX_CASCADE_ITERATIONS = 8`.

**GPU:** `CascadeRegistry` holds a catalog of SPIR-V kernels (one per `(EventKindId, Lane)` pair). Compiler emits these kernels when lowering DSL `physics` rules. `run_fixed_point` on GPU is a host-orchestrated loop:

```
for iter in 0..MAX_CASCADE_ITERATIONS:
    total_before = gpu_events.counter()
    dispatch all handlers whose trigger event-kind was pushed this iteration
    total_after = gpu_events.counter()
    if total_after == total_before: break  # fixed point reached
```

No state download between iterations — cascade mutates GPU-resident state directly.

**Compiler-emitted cascade handlers** — a DSL rule such as:

```
physics damage_on_attack @phase(event) {
  on AgentAttacked { attacker: _, target: t, damage: d } {
    state.hp[t] -= d
    if state.hp[t] <= 0 {
      emit AgentDied { agent_id: t }
    }
  }
}
```

Compiles to a SPIR-V kernel (reads the `AgentAttacked` event buffer, updates `state.hp[t]`, atomic-appends `AgentDied`) AND a Rust closure for `SerialBackend`. Both have bit-identical observable behavior; determinism tests verify.

**Cross-entity walks:** `for t in quest.eligible_acceptors` becomes a kernel that reads a GPU-resident `AggregatePool<Quest>` entry, iterates its embedded `[AgentId; N]` array. Bounded by the pool's fixed-size array capacity.

Implementation: `crates/engine/src/cascade/`, with `backend/serial/cascade.rs` and `backend/gpu/cascade.rs`.

### 4.7 View fold (phase 5)

Three storage modes × two backends:

| Mode | Serial impl | GPU impl |
|---|---|---|
| `MaterializedView` | Scalar fold over event iterator | Sorted-by-target-id + parallel reduction kernel |
| `LazyView` | Compute on demand with staleness flag | Same shape; compute dispatched as kernel, cached in FieldHandle |
| `TopKView` | Per-target Vec with sort | Per-target small-array with parallel merge-and-truncate |

Determinism for materialized views under GPU requires **sorting events by their stable per-tick sequence number before reduction** — otherwise float associativity breaks parity. Commutative integer reductions (counts) don't need sorting.

### 4.8 Tick end — invariants and telemetry (phase 6)

Invariants run on the host, against the **host mirror post-sync**. The step pipeline forces a sync at phase-6 entry for any invariant that declares `requires_state: true`.

Built-in invariants:
- `mask_validity` — checks post-apply actions/mask pair; ActionBuffer already downloaded at phase 3; no additional sync cost.
- `pool_non_overlap` — reads host Pool freelist/alive vec; GPU backend must have uploaded the Pool post-apply.
- `event_hash_stable` (dev-only) — re-hash replayable events; host-resident already.

Telemetry sinks are host-side. Both backends emit metrics via the same sink API. GPU-specific metrics go through the same sink under the `engine.gpu.*` prefix.

Built-in metrics:

| Metric | Serial | GPU |
|---|---|---|
| `engine.tick_ms` | wall-clock | wall-clock |
| `engine.event_count` | host-side | after-sync count |
| `engine.agent_alive` | direct | after-sync |
| `engine.cascade_iterations` | counted | counted |
| `engine.mask_true_frac` | computed | downloaded mask slice |
| `engine.gpu.upload_ms` | `0.0` | per-tick sum |
| `engine.gpu.download_ms` | `0.0` | per-tick sum |
| `engine.gpu.kernel_ms` | `0.0` | per-kernel histogram |

---

## 5. Cascade primitive

### CascadeRegistry and lanes

Cascade handlers declare a lane: `Validation | Effect | Reaction | Audit`. Lanes run in order; within a lane, handlers run in lexicographic mod-id. Multiple handlers per lane coexist (additive). `MAX_CASCADE_ITERATIONS = 8` bounds fixed-point convergence. A cascade rule using self-emission must be annotated `@terminating_in(N)`.

### Event log

The `EventRing` abstraction is backend-local in storage but identical in contract.

**Serial:** `VecDeque<Entry>` with fixed capacity. `push(event) -> EventId` monotonic. `replayable_sha256()` iterates host entries and hashes.

**GPU:** GPU-resident ring buffer + atomic append counter. At phase-4 end, the host drains the GPU ring to its host mirror. The **replayable-hash is always computed on the host mirror** — both backends emit the same byte-packed representation. Events are sorted by their stable per-tick sequence number before hashing.

Cascade events carry `cause: Option<EventId>`. In GPU mode, the cause is written into the event record by the kernel that emitted the cascade. Host reconstruction of the causal tree works identically.

Capacity rules:
- Serial: `with_cap(n)` — ring drops oldest on overflow.
- GPU: pre-allocated FieldHandle; overflow is logged; subsequent pushes in the overflowing tick are dropped deterministically.

Event byte-packing uses `f32::to_bits().to_le_bytes()` plus explicit variant tags. Debug formatting is forbidden in hash input.

---

## 6. Determinism and replayability

### Determinism contract

The engine promises: **same seed + same compile-time `schema_hash` + same agent spawn order + same action sequence ⇒ bit-exact SHA-256 over the replayable-subset event log**, on every supported platform, **and identical across both backends**.

Parity test (mandatory in CI):

```rust
let seed = 42;
let hash_cpu = run_n_ticks(SerialBackend::new(), seed, 1000);
let hash_gpu = run_n_ticks(GpuBackend::new()?, seed, 1000);
assert_eq!(hash_cpu, hash_gpu, "cross-backend parity broken");
```

Implementation obligations:
- All randomness derives from `WorldRng` (PCG-XSH-RR) with fixed-keyed sub-streams. The shader implementation uses byte-identical constants.
- `HashMap` iteration is forbidden in hot paths on both backends. Use `BTreeMap` (serial) or sorted indices (GPU).
- Float reductions are either integer-fixed-point OR sorted-key to avoid associativity drift.
- GPU atomic ops in kernels are commutative-and-associative by construction OR must sort before reduction.
- The `replayable_sha256()` result is computed from events downloaded to host regardless of backend.

**Batch GPU path determinism:** Non-deterministic in event order by design. Atomic tail racing inside the resident cascade does not serialise. Statistical parity (alive counts, event multisets, conservation laws) is the contract for batch mode. Determinism tests run against `SerialBackend` and the GPU sync path only.

See `crates/engine/tests/parity_backends.rs` for the mandatory cross-backend determinism test.

### Spatial index

Two backends; same query API: `within_radius(center, r)`, `nearest_k(center, k)`, `in_column_z_range(col, z_range)`.

**Serial:** 2D-column BTreeMap + per-column sorted z-list + `movement_mode` sidecar. Cell-size = 16m (voxel-chunk edge). Planar queries walk 9 columns (3×3). Volumetric queries walk 9 columns and binary-search the z-range. Agents with `movement_mode != Walk` live in a separate `in_transit` list that every spatial query scans linearly. Deterministic by BTreeMap ordered iteration.

**GPU:** Voxel-chunk-keyed spatial hash residing in FieldHandles. Queries are compute-kernel dispatches (`within_radius_kernel`, `nearest_k_kernel`). For host-side queries, GPU backend falls back to a synced CPU-side copy. Insert/remove: both backends rebuild on `spawn_agent` / `kill_agent` / `set_agent_pos`. GPU rebuild is a batched kernel per tick.

Implementation: `crates/engine/src/spatial/`.

### RNG streams

Identical PCG-XSH-RR algorithm on host and in shaders. The four keyed-hash constants are `pub const`s in `rng.rs` AND are embedded in shader source:

```rust
// crates/engine/src/rng.rs
pub const RNG_KEY_1: u64 = 0xA5A5_A5A5_A5A5_A5A5;
pub const RNG_KEY_2: u64 = 0x5A5A_5A5A_5A5A_5A5A;
pub const RNG_KEY_3: u64 = 0xDEAD_BEEF_CAFE_F00D;
pub const RNG_KEY_4: u64 = 0x0123_4567_89AB_CDEF;
```

Shader GLSL:

```glsl
#define RNG_KEY_1 0xA5A5A5A5A5A5A5A5ul
#define RNG_KEY_2 0x5A5A5A5A5A5A5A5Aul
#define RNG_KEY_3 0xDEADBEEFCAFEF00Dul
#define RNG_KEY_4 0x0123456789ABCDEFul
```

**Cross-backend golden test** asserts `per_agent_u32(42, AgentId(1), 100, b"action")` returns the same value when computed on host Rust AND inside a test shader.

Implementation: `crates/engine/src/rng.rs`, `crates/engine/shaders/rng.glsl`, `crates/engine/tests/rng_cross_backend.rs`.

---

## 7. Cross-backend parity contract

Every DSL mask predicate / cascade handler / view gets both a scalar Rust implementation (registered with `SerialBackend`) AND a SPIR-V kernel (registered with `GpuBackend`).

### Parity test suite

| Test | Path | Asserts |
|---|---|---|
| `parity_with_cpu` | Sync GPU vs `SerialBackend` | Byte-exact agent state + event multiset at N=50 |
| `physics_parity`, `cascade_parity` | Sync GPU vs `SerialBackend` | Byte-exact per-iteration cascade outputs |
| `perf_n100` | Sync GPU | Alive count within ±25% of `SerialBackend` at N=100 |
| `async_smoke` | Batch path | Tick advance, agent count, alive count within ±25%, events present, chronicle present |
| `snapshot_double_buffer` | Batch path | First call empty; subsequent calls non-overlapping watermark windows |
| `cold_state_4*` | Batch path | Gold conservation, standing pair-symmetric clamp, memory FIFO eviction |
| `pick_ability_*` | Batch path | Cooldown respected, range-gated targeting, cast events emitted |
| Cross-path parity (sync ↔ batch) | **Excluded** — non-deterministic by design | |
| Statistical parity (sync ↔ batch) | Allowed | Cast cadence within ±25%, ability-type distributions, gold conservation laws |

### Aggregate residency

`AggregatePool<T>` for non-spatial entities:

- **Host-only aggregates** (default): T has no `Pod` constraint; works only with `SerialBackend` OR requires explicit download for GPU access.
- **GPU-eligible aggregates**: `T: Pod`; storage is `FieldHandle` when `GpuBackend` is active. Required when cascade handlers running on GPU need to read aggregate fields.

For MVP, `Quest` and `Group` use the GPU-eligible shape — their fields are Pod (`Option<AgentId> = u32`, fixed-size arrays via `[AgentId; N]`).

---

## 8. Schema hash

SHA-256 covers the **engine schema** + **kernel catalog hashes** + **RNG constants** + **all layout-relevant types**:

```rust
h.update(b"SoA-layout: <field names + types>");
h.update(b"MicroKind: <variants>");
h.update(b"MacroKind: <variants>");
h.update(b"EventKindId: <variants + ordinals>");
h.update(b"Lane: Validation,Effect,Reaction,Audit");
h.update(b"MAX_CASCADE_ITERATIONS=8");
h.update(b"RNG_KEY_1=..., RNG_KEY_2=..., ...");
for k in &catalog.kernel_hashes {
    h.update(k);
}
```

A kernel recompile that changes bytecode bumps the engine schema hash automatically. Checkpoint load (§11) rejects on mismatch.

The GPU schema surface additionally includes:
- `SimCfg` field layout (offsets, sizes).
- Event-kind dispatch manifest (rule names, kinds, registration order).
- Ability registry packed format (tag table layout, slot count).
- `chosen_ability_buf` packing format.
- View storage layouts for `@symmetric_pair_topk` and `@per_entity_ring` annotated views.

CI fence: changes to any of the above bump `crates/engine/.schema_hash`. Drift fails parity tests at startup.

---

## 9. GPU annex: resident cascade (batch mode)

### 9.1 Principle

The **Resident Cascade** eliminates per-tick CPU/GPU fences by binding each kernel's outputs as the next kernel's inputs and replacing CPU-driven loops with GPU indirect dispatch.

**Layered subsystems (each builds on the previous):**

| Layer | Adds |
|---|---|
| Resident cascade | GPU-resident pipeline, indirect dispatch, double-buffered snapshot |
| Sim state mirroring | `SimCfg` shared buffer, GPU-side tick advance, `@cpu_only` annotation |
| Cold-state replay | Per-event-kind handler dispatch, gold/standing/memory on GPU |
| Ability evaluation | `pick_ability` kernel, `ability::tag` scoring primitive, `per_ability` row |

**Non-goals for batch mode:**
- Byte-exact GPU↔CPU parity. Atomic-tail event ordering is non-commutative.
- Cross-GPU reproducibility. Same hardware + same seed reproduces; different vendors may diverge.
- Replacing the sync path. Sync stays load-bearing for parity tests, deterministic chronicle output, and `SerialBackend` cross-checks.

### 9.2 Public surface

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

### 9.3 One tick on the batch path

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

**Submit shape:** One command encoder records N ticks: each tick = mask + scoring + spatial + apply + movement + cascade-indirect × `MAX_CASCADE_ITERATIONS` + fold + tick-counter increment. One `queue.submit`. One `device.poll(Wait)` at end of batch.

**Indirect dispatch for cascade iterations:** End-of-iter, the physics kernel writes indirect dispatch args `(workgroup_count, 1, 1)` to a small GPU buffer. When there are no follow-on events, the kernel writes `(0, 1, 1)` — subsequent indirect dispatches are GPU no-ops. `run_cascade_resident` pre-records `MAX_CASCADE_ITERATIONS` indirect dispatches into one encoder.

**Performance contract:** batch mean µs/tick `< 0.8×` sync mean µs/tick at N ≥ 512.

### 9.4 Snapshot — double-buffered staging

Three staging-buffer pairs: `{agents, events, chronicle}` × `{front, back}`.

On call:
1. Encode `copy_buffer_to_buffer` for current live buffers into the **back** staging buffers. Update watermarks.
2. `queue.submit`.
3. `map_async(Read)` on the **front** staging buffers (filled by the previous call).
4. `device.poll(Wait)`.
5. Decode front staging → `GpuSnapshot`, unmap, swap front/back pointers.

First call returns `GpuSnapshot::empty()`. One-frame lag is acceptable.

**Backend factoring:**

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

**Error cases:**

| Failure | Detection | Behaviour |
|---|---|---|
| Indirect args corruption | Kernel clamps `workgroup_count ≤ ceil(agent_cap / WGSIZE)` before write; snapshot validates | Logged warning |
| GPU ring overflow | Kernel writes `overflowed` flag; `snapshot()` reads it | `Err(SnapshotError::RingOverflow { tick, events_dropped })` |
| Cascade non-convergence | Final-iter args still non-zero | Warning; subsequent ticks pick up |
| Staging map failure | `map_async` callback `Err` | `Err(SnapshotError)` |
| Kernel dispatch failure | wgpu validation / device-lost | `Err(BatchError)`; **no CPU fallback** |
| `SimCfg` field drift | Compile-time offset assertion | Startup panic |
| Missing dispatch entry | Compile-time check | Build fail |

---

## 10. GPU annex: sim state mirroring and cold-state replay

### 10.1 Sim state mirroring

| State | CPU resident | GPU resident | Mirroring |
|---|---|---|---|
| Hot agent fields (pos, hp, alive, …) | `SimState.agents` | `resident_agents_buf` (`GpuAgentSlot`) | Upload at `ensure_resident_init`; snapshot copies back |
| Tick | `state.tick` | `SimCfg.tick` (atomic u32) | GPU advances; CPU reads via snapshot |
| World seed | `state.world_seed` | `SimCfg.world_seed_{lo,hi}` | Upload once at init |
| World scalars (engagement_range, …) | `state.config.combat.*` | `SimCfg.*` | Upload once at init |
| RNG state | (derived) | (derived) | Stateless: `hash(seed, tick, agent_id, purpose)` |
| Cold-state (gold, standing, memory) | `SimState.cold_*` | Side buffers (§10.2) | Upload at init; snapshot reads back |
| Event ring | Caller-provided | `event_ring_buf` | Sync: populated each tick. Batch: only via snapshot |
| Chronicle | (CPU `Vec`) | `chronicle_ring_buf` | Sync: drained per call. Batch: snapshot watermark |

Tick advance: the seed-indirect kernel runs once per tick and appends `atomicAdd(&sim_cfg.tick, 1u)` at end-of-tick. CPU `state.tick` is stale across the whole batch by design.

**`@cpu_only` annotation:** DSL annotation marking rules that stay CPU-resident (chronicle narrative formatting, debug-only side effects, anything requiring strings or unbounded allocation):

```
@cpu_only physics chronicle_render_attack_narrative @phase(event) {
    on AgentAttacked { ... build String ... push chronicle text ... }
}
```

Compiler behaviour: emits the CPU handler, skips WGSL emission, relaxes the GPU-emittable validator for constructs inside the rule body.

### 10.2 Cold-state replay on GPU

One WGSL handler kernel per GPU-resident event kind. End of each tick's cascade iteration, the cascade driver scans observed events by kind and dispatches the matching kernel(s). Emission is automatic — every DSL `physics rule @phase(event) { on EventKind { ... } }` not marked `@cpu_only` produces (a) a WGSL kernel and (b) an entry in a generated dispatch-by-event-kind manifest.

**Event-kind dispatch table (generated):**

| Event kind | Handler rule(s) | Storage written | Notes |
|---|---|---|---|
| `AgentAttacked` | `damage`, `opportunity_attack`, structured chronicle | `agents_buf.hp/shield`, chronicle ring | GPU-native |
| `AgentDied` | `fear_spread`, structured chronicle | `view_storage.kin_fear` | GPU-native |
| `EffectGoldTransfer` | `transfer_gold` | `gold_buf` (atomic add/sub) | |
| `EffectStandingDelta` | `modify_standing` view fold | `standing` view storage | |
| `RecordMemory` | `record_memory` view fold | `memory` view storage | |
| `AgentCast` | `cast` physics rule (existing) | `agents_buf`, event ring | GPU-native |
| `chronicle_*_narrative` (text) | (none on GPU) | — | `@cpu_only` |

**Gold:** Dedicated `gold_buf` with `per_agent: array<atomic<i32>>`. Written by `transfer_gold` GPU handler; read by scoring rules that gate on gold. `snapshot()` copies back into `SimState.cold_inventory.gold`. i32 overflow wraps — gameplay caps enforced at ability-design level.

**Standing:** Reborn as a DSL `@materialized` view with `@symmetric_pair_topk(K = 8)`. Pair key is `(min(a,b), max(a,b))`; value is i16 clamped to `[-1000, 1000]`. When 8 slots are full, the lowest `|standing|` entry is evicted. Per-agent storage: ~56 KB at N=2048.

**Memory:** Reborn as a DSL `@materialized` view with `@per_entity_ring(K = 64)`. Per-agent ring with `cursor: u32`; push via `atomicAdd(cursor, 1)`, write at `ring[agent][cursor % K]`. Ring overflow by design — oldest entry evicted. Per-agent storage: ~3 MB at N=2048.

**Chronicle tiers:**
- **Structured chronicle** — fixed-layout `ChronicleEntry` emitted GPU-side into the GPU chronicle ring.
- **Narrative chronicle** — multi-sentence prose with string interpolation. Marked `@cpu_only`. Runs async off the batch snapshot's event stream.

**Engine-core vs DSL-lowered:**

| Concern | Owner |
|---|---|
| Side-buffer allocation + binding (gold, standing, memory) | Engine-core |
| Dispatch table driver | Engine-core |
| Snapshot handshake | Engine-core |
| `ensure_resident_init` upload | Engine-core |
| Rule bodies, view fold bodies, annotation processing | DSL-lowered |

---

## 11. GPU annex: ability evaluation kernel

### 11.1 Position in pipeline

```
mask → scoring → pick_ability → apply_actions → movement → spatial → cascade → fold → cold_state_dispatch
                 ^^^^^^^^^^^^
```

Scoring runs first, producing per-agent action scores. `pick_ability` runs next with its own output buffer. `apply_actions` reads both — if an ability is chosen, emit `AgentCast`; otherwise, emit the scoring kernel's chosen action.

### 11.2 `pick_ability` kernel (compiler-emitted)

Per agent per tick:
- Iterates abilities in the agent's known set.
- Evaluates each ability's `guard`. Skips on false.
- Evaluates each ability's `score`. Tracks argmax.
- Picks target via the ability's `target:` clause.
- Writes `chosen_ability_buf[agent]` = packed `(ability_slot: u8, target_agent_id: u32, sentinel-for-no-cast)`.

### 11.3 `ability::tag` scoring primitive

Reads the tag value for the ability being scored. Returns `f32`, or `0` if the ability lacks the tag.

```
score = ability::tag(PHYSICAL) * (1 - target.hp_frac)
      + ability::tag(CROWD_CONTROL) * engaged_with_kin
```

Also: `ability::hint` (string-enum compare against coarse category: `damage` / `defense` / `crowd_control` / `utility`); `ability::range` (pulls from `Area::SingleTarget { range }`); `ability::on_cooldown(ability)` (predicate, reads per-slot cooldown buffer).

### 11.4 `per_ability` row type

Iterates over an agent's abilities and produces one score per (agent, ability):

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

### 11.5 Tag registry

Each `.ability` file carries a `hint:` (coarse category: `damage` | `defense` | `crowd_control` | `utility`) and per-effect `[TAG: value]` numeric power ratings. Tag names are a fixed enum. `AbilityDef.tags` serialise into the `PackedAbilityRegistry` consumed by kernels.

### 11.6 Side buffer

```wgsl
struct ChosenAbilityBuf {
    per_agent: array<u64>, // packed: (ability_slot: u8, target_agent_id: u32, _pad: u24)
}
```

Failure modes:
- Ability tag missing → `ability::tag(UNKNOWN)` returns 0 silently.
- All abilities on cooldown → writes sentinel; `apply_actions` falls through to score_output.
- Cooldown buffer not initialised → verified at `ensure_resident_init`; startup panic.

**Engine-core vs DSL-lowered:**

| Concern | Owner |
|---|---|
| `chosen_ability_buf` allocation + binding | Engine-core |
| `pick_ability` kernel dispatch in `step_batch` | Engine-core |
| `apply_actions` extension to read `chosen_ability_buf` and emit `AgentCast` | Engine-core |
| `pick_ability.wgsl` (compiler output from `per_ability` row) | DSL-lowered |
| CPU `pick_ability` handler | DSL-lowered |
| Tag reads + scoring arithmetic | DSL-lowered |

---

## 12. GPU annex: kernel and buffer reference

### 12.1 Sync path

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

### 12.2 Resident path

```
step_batch(n)
├─ ensure_resident_init()
│     allocate resident_agents_buf, sim_cfg_buf, cascade_resident_ctx (first call)
└─ for tick in 0..n:
    ├─ 1. fused_unpack_kernel
    ├─ 2. mask_resident → bitmap buffers
    ├─ 3. scoring_resident → ScoreOutput
    ├─ 4. pick_ability (§11)
    │     reads:  agent_data, ability_registry, tag_table, cooldowns_buf, spatial outputs, sim_cfg
    │     writes: chosen_ability_buf
    ├─ 5. apply_actions.run_resident → hp/shield/alive, batch_events_ring
    ├─ 6. movement.run_resident → pos, batch_events_ring
    ├─ 7. append_events → apply_event_ring
    ├─ 8. seed_kernel: atomicAdd(sim_cfg.tick, 1); seed indirect_args[0]; clear num_events
    ├─ 9. for iter in 0..MAX_CASCADE_ITERATIONS:
    │     physics.run_batch_resident (indirect)
    │         reads:  apply_event_ring[iter]
    │         writes: resident_agents_buf, physics_ring[iter+1], indirect_args[iter+1]
    │     break if indirect_args[iter+1] == (0,1,1)
    └─ 10. cold_state_dispatch (§10.2)
          per event kind K with count > 0: dispatch K's handler kernel
          view fold kernels (engaged_with, my_enemies, threat_level, kin_fear,
                             pack_focus, rally_boost, standing, memory)
```

### 12.3 Kernel inventory

**Mask kernel:**

| Property | Value |
|---|---|
| Struct | `FusedMaskKernel` (sync), `MaskUnpackKernel` (resident unpack) |
| Entry points | `cs_fused_masks`; `cs_mask_unpack` |
| Workgroup size | 64 |
| Outputs | 7 bitmap arrays: Attack, MoveToward, Hold, Flee, Eat, Drink, Rest |
| Notes | Cast mask covered by `pick_ability` (§11) |

**Scoring kernel:**

| Property | Value |
|---|---|
| Entry point | `cs_scoring` |
| Workgroup size | 64 |
| Bind group | agent_data SoA, bitmaps, view_storage, cfg, sim_cfg, spatial queries |
| Outputs | `ScoreOutput[agent_cap]` — per-agent: chosen_action, chosen_target, score |

**Pick ability kernel:**

| Property | Value |
|---|---|
| Entry point | `cs_pick_ability` |
| Workgroup size | 64 |
| Outputs | `chosen_ability_buf[agent_cap]` — packed `(slot, target, sentinel)` u64 |
| Notes | Compiler-emitted from `per_ability` rows in `scoring.sim` |

**Apply actions kernel:**

| Property | Value |
|---|---|
| Entry points | `cs_apply_actions` (sync), `cs_apply_actions_resident` (batch) |
| Workgroup size | 64 |
| Outputs | Mutated hp/shield/alive; events: AgentAttacked, AgentDied, AgentCast (batch), AgentAte, AgentDrank, AgentRested |
| Scope gaps | Opportunity attacks, engagement slow on MoveToward, announce/communicate — sync CPU path |

**Movement kernel:**

| Property | Value |
|---|---|
| Entry points | `cs_movement` (sync), `cs_movement_resident` (batch) |
| Workgroup size | 64 |
| Outputs | Updated pos; events: AgentMoved, AgentFled |
| Math | MoveToward: `pos + normalize(target - pos) * move_speed`. Flee: `pos + normalize(pos - threat) * move_speed` |

**Physics kernel (event processor):**

| Property | Value |
|---|---|
| Entry points | `cs_physics` (sync), `cs_physics_resident` (batch) |
| Workgroup size | 64 |
| Outputs | Mutated agent state; new events; chronicle entries |
| Determinism | Non-deterministic order (atomic tail). Host drain sorts by `(tick, kind, payload[0])` pre-fold |

**Fold kernels (view materialization):**

| Property | Value |
|---|---|
| Entry points | `cs_fold_<view_name>` — one per view (engaged_with, my_enemies, threat_level, kin_fear, pack_focus, rally_boost, standing, memory) |
| Workgroup size | 64 |

**Spatial hash kernels (resident path):**

| Property | Value |
|---|---|
| Entry points | `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query` |
| Inputs | Agent positions; two radii (kin=12m, engagement=2m) |
| Outputs | Per-agent: nearby agents, kin-species membership, nearest hostile/kin |

### 12.4 Buffer ownership

| Buffer | Owner | Size (N=100k) | Purpose |
|---|---|---|---|
| `resident_agents_buf` | `ResidentPathContext` | ~16 MB | Agent SoA, persistent across batch |
| `sim_cfg_buf` | `ResidentPathContext` | 256 B | `SimCfg` — atomic tick + world scalars |
| `apply_event_ring` | `CascadeCtx` (sync) | ~24 MB | Seeds physics iter 0; cleared per-tick |
| `physics_ring_a` / `_b` | `CascadeResidentCtx` | ~24 MB each | Ping-pong resident cascade rings |
| `batch_events_ring` | `CascadeResidentCtx` | ~24 MB | Append-only accumulator across batch ticks |
| `chronicle_ring` | `CascadeResidentCtx` | ~24 MB | Caller-owned; snapshot watermark |
| `indirect_args` | `ResidentPathContext` | 32 B × (`MAX_CASCADE_ITERATIONS` + 1) | Indirect dispatch args |
| `num_events_buf` | `CascadeResidentCtx` | 4 B × (`MAX_CASCADE_ITERATIONS` + 1) | Event counts per iteration |
| `view_storage` | `ViewStorage` | ~144 MB (6 views @ N=100k) | Materialised view state |
| `gold_buf` | `ResidentPathContext` | ~400 KB | Per-agent atomic i32 gold |
| Spatial query outputs | `SpatialOutputs` | ~80 MB each | kin / engagement query results |
| `PackedAbilityRegistry` | engine | ~256 KB | Ability metadata; content-addressed upload |
| `chosen_ability_buf` | `ResidentPathContext` | 8 B × N | Per-agent `(slot, target, sentinel)` u64 |

### 12.5 Key constants

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

### 12.6 DSL → WGSL lowering

Per-subsystem emitters in `dsl_compiler` produce WGSL modules:

- `emit_mask_wgsl` → fused module with per-mask bitmap writes.
- `emit_physics_wgsl` → one module with `physics_dispatch(event)` switch over event kinds.
- `emit_view_wgsl` → one fold kernel per view.
- `emit_pick_ability_wgsl` → `pick_ability.wgsl` from `per_ability` rows.

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

---

## 13. Debug and trace surface

Six components: `trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, `snapshot`. All host-side.

When state is GPU-resident, debug tools trigger downloads on demand:
- `trace_mask` — syncs observation snapshot + mask for the target tick.
- `causal_tree` — events are already synced to host each tick.
- `tick_stepper` — stops between phases; can request phase-specific downloads.
- `tick_profile` — adds kernel-scoped timing for GPU backend.

GPU stepping: each phase exposes a `debug_readback` toggle that forces additional syncs for inspection.

---

## 14. Save / load

Backend-agnostic snapshot format. On save, the current backend forces a full sync, downloads state to host mirror, and serializes the mirror. On load, the file is deserialized into host mirror; the backend's `upload_from_mirror()` restores GPU-resident state.

Format:

| Block | Content |
|---|---|
| Header (64 B) | Magic, engine schema_hash, kernel catalog hash, tick, seed |
| SoA hot field mirrors | One block per field, little-endian |
| SoA cold field mirrors | Option<T> encoded with present-bit |
| Pool freelist | alive Vec<bool> + freelist Vec<u32> |
| Event ring tail | Host-mirror snapshot of replay-continuity events |

Loading rejects a snapshot whose kernel catalog hash differs from the current engine's.

---

## 15. Backend selection and feature flags

```rust
pub enum BackendKind { Serial, Gpu }

pub fn new_backend(preferred: BackendKind) -> Box<dyn ComputeBackend>;
pub fn new_backend_auto() -> Box<dyn ComputeBackend>;
```

Fallback sequence for `new_backend_auto()`:
1. Try `GpuBackend::new()`. Success → use it.
2. On `GpuInitError`, log via `tracing::warn!`, fall back to `SerialBackend::new()`.
3. SerialBackend never fails.

Feature flags:

```toml
[features]
default = ["serial", "gpu"]
serial = []
gpu = ["dep:voxel_engine"]

[dependencies]
voxel_engine = { path = "/home/ricky/Projects/voxel_engine", optional = true }
```

CI has a Vulkan-capable container (Mesa lavapipe for software Vulkan when no device). Parity tests run both backends and compare.

---

## 16. What's NOT in the engine

- **DSL parser and codegen.** Compiler concern.
- **Verb desugaring / Read → Ask lowering.** Compiler concern.
- **Domain types** (item catalog, ability list, group kinds). Compiler-generated.
- **Cascade RULES**. Compiler emits kernel bytecode + Rust closures; engine provides the dispatch runtime.
- **Chronicle prose templates.** Host-side text generation.
- **Curriculum pipelines.** External.
- **LLM backend implementation.** Separate downstream crate.

---

## Implementation map

See **`docs/engine/status.md`** — per-subsystem state (Serial vs. GPU), associated plan, tests, weak-test risks, and visual-check criteria. The Serial column is the ground truth; GPU lands one section at a time with cross-backend parity tests.

Implementation: `crates/engine/src/` — Rust implementation; `crates/engine/shaders/` — SPIR-V bytecode + GLSL source; `crates/engine/tests/parity_backends.rs` — mandatory cross-backend determinism test.
