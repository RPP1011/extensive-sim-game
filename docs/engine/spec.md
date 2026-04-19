# World Sim Engine Specification

Runtime contract for the engine the compiler targets. Companion to `../dsl/spec.md` (language reference) and `../compiler/spec.md` (text-to-engine lowering).

The engine is the **unified runtime for all compiler output**. The compiler has one target; whatever it emits — scalar Rust, kernel dispatch code, SPIR-V shader source — lands in the engine. The engine ships **two first-class backends** that implement the same interface:

- **`SerialBackend`** — host-resident state, scalar Rust execution. The **reference implementation**. Determinism oracle for every test, for every parity check against the GPU backend, for every verified-correct port.
- **`GpuBackend`** — GPU-resident state via `voxel_engine::GpuHarness`, compiled SPIR-V kernels for gather + apply phases. Performance path for large N.

An implementation is correct if and only if it satisfies §§2–26 AND **produces byte-identical `replayable_sha256()` across both backends on the same seed**. Cross-backend parity is a first-class correctness invariant, not an aspiration.

---

## 1. Scope

The engine is a Rust library crate (`crates/engine/`) that provides runtime primitives the compiler targets, PLUS two concrete backends that execute those primitives.

**Owned by the engine:**

- Generic `Pool<T>` with `NonZeroU32` IDs, freelist, slot reuse.
- Event ring with byte-stable SHA-256 over the replayable subset.
- 3D spatial index with voxel-chunk-keyed dispatch (CPU BTreeMap + z-sort for Serial; GPU hash for Vulkan).
- Per-world RNG with shader-derivable per-agent keyed sub-streams; constants pinned across CPU/GPU.
- Universal `MicroKind` (18 variants) and `MacroKind` (4 variants) with built-in execution for both backends.
- `MaskBuffer` layout + per-predicate dispatch (backend selects how to execute).
- `PolicyBackend` trait — abstracts over both host-scalar and GPU-kernel implementations.
- `MaterializedView` / `LazyView` / `TopKView` traits with backend-local storage implementations.
- `TrajectoryWriter` + `TrajectoryReader` over safetensors; input comes from whichever backend is active.
- State snapshot + load with schema-hash versioning; covers both backends' state formats.
- `Invariant`, `Probe`, `TelemetrySink` traits — backend-agnostic where state access is abstracted; CPU-side for dispatch/sink mechanics.
- Debug & trace runtime (trace_mask, causal_tree, tick_stepper, tick_profile, agent_history, snapshot) — downloads from GPU as needed; serial-backend access is direct.
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

## 2. Determinism contract

The engine promises: **same seed + same compile-time `schema_hash` + same agent spawn order + same action sequence ⇒ bit-exact SHA-256 over the replayable-subset event log**, on every supported platform, **and identical across both backends**.

Parity test (mandatory in CI):

```rust
let seed = 42;
let hash_cpu = run_n_ticks(SerialBackend::new(), seed, 1000);
let hash_gpu = run_n_ticks(GpuBackend::new()?, seed, 1000);
assert_eq!(hash_cpu, hash_gpu, "cross-backend parity broken");
```

Implementation obligations:

- All randomness derives from `WorldRng` (PCG-XSH-RR) with fixed-keyed sub-streams. **The shader implementation uses byte-identical constants** — golden tests assert `per_agent_u32` returns the same value when run in host Rust and in a compute shader.
- `HashMap` iteration is forbidden in hot paths on both backends. Use `BTreeMap` (serial) or sorted indices (GPU).
- Float reductions are either integer-fixed-point OR sorted-key to avoid associativity drift. This applies to both backends; GPU must sort events by `target_id` before atomic accumulation into view buffers.
- Per-tick agent action ordering is shuffled via deterministic Fisher-Yates keyed on `per_agent_u32(seed, AgentId(1), tick << 16, b"shuffle")`. Both backends apply the SAME shuffle sequence.
- Event byte-packing uses `f32::to_bits().to_le_bytes()` plus explicit variant tags. Debug formatting is forbidden in hash input.
- The `replayable_sha256()` result is computed from events downloaded to host regardless of backend — GPU backend drains its GPU-resident event buffer to host once per tick; the hash is computed on the host copy.
- GPU atomic ops in kernels are commutative-and-associative by construction OR must sort before reduction.

See `crates/engine/tests/parity_backends.rs` for the mandatory cross-backend determinism test.

---

## 3. Runtime architecture

The engine has a **backend-agnostic tick pipeline** + **two backend implementations**.

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

**Both backends implement the same public API** — `agent_pos(id)`, `event_ring()`, `mask_buffer()` — so test code is backend-agnostic. Internally they are profoundly different.

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

---

## 4. State residency

Both backends expose the **same public SimState API**: `agent_pos(id) -> Option<Vec3>`, `set_agent_hp(id, hp)`, `agents_alive() -> impl Iterator<AgentId>`, etc. Callers don't know or care where the data lives.

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

A **sync point** is any place where host and GPU ownership must align. The engine defines fixed sync points per tick:

| Phase | Sync | What happens |
|---|---|---|
| Before phase 1 | `upload_dirty` | Host→GPU push of any fields marked dirty since last tick |
| After phase 4 (apply) | `download_events` | GPU→host drain of events pushed during apply |
| After phase 4 (apply) | `download_state_if_needed` | Optional — only if invariants / trajectory / debug require host-readable state |
| After phase 5 (views) | `download_view_results` | GPU→host pull of view buffers that invariants or telemetry read |

Between sync points, **GPU is authoritative for hot fields**, mirror is stale. Callers that query state outside a sync point (debug tools, invariants) trigger an on-demand sync.

### Snapshot semantics

For save/load (§18) and trajectory emission (§17), the engine **forces a full sync** — all dirty fields uploaded, all GPU-authoritative state downloaded, mirror becomes clean. Snapshots always capture the host mirror post-sync.

Implementation: `crates/engine/src/state/` holds the backend-agnostic API; `crates/engine/src/backend/serial.rs` and `crates/engine/src/backend/gpu.rs` provide the residency layers.

---

## 5. Event log

The `EventRing` abstraction is backend-local in storage but identical in contract.

### Serial

`VecDeque<Entry>` with fixed capacity. `push(event) -> EventId` monotonic. `replayable_sha256()` iterates host entries and hashes.

### GPU

GPU-resident ring buffer + atomic append counter. Events emitted by kernels (mask predicate violations, cascade handlers, apply actions) use `atomicAdd` on the counter to claim a slot, then write the event record.

At phase 4 end, the host drains the GPU ring to its host mirror via `GpuHarness::download`. The **replayable-hash is always computed on the host mirror** — this ensures byte-identity with the `SerialBackend` hash, because:

1. Both backends emit the same events in the same logical order per tick (modulo GPU atomic-append race, which is resolved by sorting events by their stable per-tick sequence number before hashing — see §2).
2. Both backends emit the same byte-packed representation (byte-level format is spec'd, not backend-specific).

Cascade events carry `cause: Option<EventId>`. In GPU mode, the cause is written into the event record by the kernel that emitted the cascade. Host reconstruction of the causal tree works identically.

Capacity rules:

- Serial: `with_cap(n)` — ring drops oldest on overflow.
- GPU: pre-allocated FieldHandle sized for worst-case events-per-tick × ticks-retained; overflow is an error (logged; subsequent pushes in the overflowing tick are dropped; determinism is preserved because drop decisions are deterministic).

See `../dsl/spec.md` §2.2 / §7.3. Implementation: `crates/engine/src/event/`, `crates/engine/src/backend/*/event.rs`.

---

## 6. Spatial index

Two backends; same query API: `within_radius(center, r)`, `nearest_k(center, k)`, `in_column_z_range(col, z_range)`.

### Serial

2D-column BTreeMap + per-column sorted z-list + `movement_mode` sidecar. As previously specified. Deterministic by BTreeMap ordered iteration.

### GPU

Voxel-chunk-keyed spatial hash residing in FieldHandles. Cell-size matches voxel chunks (16 m) per compiler spec §1.2. Queries are compute-kernel dispatches:

- `within_radius_kernel` — fills a GPU result buffer with matching agent IDs
- `nearest_k_kernel` — parallel top-K reduction

For host-side queries via the public `SpatialIndex` API (debug tools, cascade handlers that happen to run CPU), GPU backend falls back to a synced CPU-side copy. Cost: one dispatch + download per query. Acceptable for low-frequency use; not for hot-loop callers.

Insert/remove: both backends rebuild on `spawn_agent` / `kill_agent` / `set_agent_pos`. GPU rebuild is a batched kernel per tick.

See `../dsl/spec.md` §9 D25. Implementation: `crates/engine/src/spatial/`.

---

## 7. RNG streams

Identical PCG-XSH-RR algorithm on host and in shaders. The four keyed-hash constants (`K1..K4`) are `pub const`s in `rng.rs` AND are embedded in shader source via specialization constants or preprocessor defines.

```rust
// crates/engine/src/rng.rs
pub const RNG_KEY_1: u64 = 0xA5A5_A5A5_A5A5_A5A5;
pub const RNG_KEY_2: u64 = 0x5A5A_5A5A_5A5A_5A5A;
pub const RNG_KEY_3: u64 = 0xDEAD_BEEF_CAFE_F00D;
pub const RNG_KEY_4: u64 = 0x0123_4567_89AB_CDEF;
```

Every shader that calls `per_agent_u32` reads the same constants. Shader GLSL:

```glsl
#define RNG_KEY_1 0xA5A5A5A5A5A5A5A5ul
#define RNG_KEY_2 0x5A5A5A5A5A5A5A5Aul
#define RNG_KEY_3 0xDEADBEEFCAFEF00Dul
#define RNG_KEY_4 0x0123456789ABCDEFul
```

**Cross-backend golden test** asserts `per_agent_u32(42, AgentId(1), 100, b"action")` returns the same value when computed on host Rust AND when computed inside a test shader that returns the value via a result buffer.

See `../dsl/spec.md` §9 D12. Implementation: `crates/engine/src/rng.rs`, `crates/engine/shaders/rng.glsl` (header), `crates/engine/tests/rng_cross_backend.rs`.

---

## 8. GPU runtime integration

The engine depends on `voxel_engine = { path = "/home/ricky/Projects/voxel_engine" }` to provide `VulkanContext` + `GpuHarness`.

### Initialization

```rust
pub struct GpuBackend {
    ctx:      VulkanContext,
    harness:  GpuHarness,
    kernels:  KernelCatalog,
    state:    GpuState,
    events:   GpuEventRing,
    // ... per-phase FieldHandles ...
}

impl GpuBackend {
    pub fn new() -> Result<Self, GpuInitError>;
}
```

`GpuInitError` covers no-device, no-compute-queue, out-of-memory-at-init. Callers must handle — backend selection (§25) uses `Result::ok()` to fall back to `SerialBackend`.

### Kernel catalog

The engine ships SPIR-V bytecode for universal kernels, embedded via `include_bytes!`:

```rust
pub struct KernelCatalog {
    pub mask_hold:                  Kernel,
    pub mask_move_allowed:          Kernel,
    pub mask_flee_allowed:          Kernel,
    pub mask_attack_allowed:        Kernel,
    pub mask_needs_allowed:         Kernel,
    pub mask_domain_hook_allowed:   Kernel,
    pub policy_utility_argmax:      Kernel,
    pub apply_move_toward:          Kernel,
    pub apply_flee:                 Kernel,
    pub apply_attack:               Kernel,
    pub apply_needs_restore:        Kernel,
    pub view_damage_taken_reduce:   Kernel,
    pub spatial_insert:             Kernel,
    pub spatial_within_radius:      Kernel,
    // ... etc.
}
```

Bytecode lives at `crates/engine/shaders/*.spv` — pre-compiled from GLSL source in `crates/engine/shaders/src/*.glsl` via a one-time `cargo xtask compile-shaders` command. Source and bytecode are both committed. Rebuilding requires `shaderc` (dev-time tool; not a runtime dep).

### SPIR-V versioning

Each shipped kernel has a content-hash recorded in `KernelCatalog::kernel_hashes`. The schema hash (§22) includes this set; a kernel recompile that changes bytecode bumps the engine's schema hash.

### Compiler-emitted kernels

In addition to universal kernels, the **compiler emits domain-specific kernels** (per-DSL-program mask predicates, cascade handlers, view reductions). These are loaded into the `GpuBackend` at init via `load_kernel_from_spirv(bytes)`. The compiler's output includes both Rust code (calling engine APIs) and SPIR-V files (loaded at runtime).

Implementation: `crates/engine/src/backend/gpu/`, `crates/engine/shaders/`.

---

## 9. Action space — MicroKind

Same 18-variant closed enum as before:

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

### Execution per backend

| MicroKind | Serial execution | GPU execution |
|---|---|---|
| Hold | No-op | No-op |
| MoveToward | CPU scalar pos update | `apply_move_toward` kernel |
| Flee | CPU scalar pos update | `apply_flee` kernel |
| Attack | CPU scalar HP deduction + conditional kill | `apply_attack` kernel w/ atomic write to alive buffer on kill |
| Eat/Drink/Rest | CPU scalar need-restoration | `apply_needs_restore` kernel |
| Cast/UseItem/etc. | Emit event only (no state mutation in engine) | Same; compiler-registered cascade handles the effect |

The apply kernel's job is: read `scratch.actions` FieldHandle, filter by MicroKind, write mutations to state fields + emit events into the GPU event ring via atomic append. Each MicroKind gets its own parallel dispatch; actions of other kinds in the same batch are no-ops for that kernel.

See `../dsl/spec.md` §3.3 and Appendix A.

---

## 10. Macro mechanisms

Same 4 variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`) + `NoOp`.

Announce cascade (the only universal macro that mutates state) runs as:

- Serial: `for obs in state.agents_alive() { ... }` loop with distance check, emits RecordMemory per recipient.
- GPU: `apply_announce` kernel — parallel distance check per agent, atomic-append to event ring for each match, bounded by `MAX_ANNOUNCE_RECIPIENTS` via an atomic counter early-exit.

Both backends produce the same events in the same logical order (per-tick seq-sorted).

---

## 11. Physics cascade runtime

Cascade is **GPU-dispatchable** (full GPU-resident target), with Serial as the reference.

### Serial

`CascadeRegistry` holds `Box<dyn CascadeHandler>`. `run_fixed_point(state, events)` walks new events, dispatches handlers in lane order, bounded by `MAX_CASCADE_ITERATIONS = 8`.

### GPU

`CascadeRegistry` holds a catalog of **SPIR-V kernels** (one per `(EventKindId, Lane)` pair). Compiler emits these kernels when lowering DSL `physics` rules. `run_fixed_point` on GPU is a host-orchestrated loop:

```
for iter in 0..MAX_CASCADE_ITERATIONS:
    total_before = gpu_events.counter()
    dispatch all handlers whose trigger event-kind was pushed this iteration
    total_after = gpu_events.counter()
    if total_after == total_before: break  # fixed point reached
```

Each iteration re-dispatches handler kernels over new events. **No state download between iterations** — cascade mutates GPU-resident state directly.

### Compiler-emitted cascade handlers

DSL rule:

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

Compiles to a SPIR-V kernel that:
1. Reads the `AgentAttacked` event buffer
2. For each event, reads `state.hp[t]`, subtracts damage, writes back
3. If new hp ≤ 0, atomic-appends `AgentDied` to the event buffer

The same DSL rule also compiles to a Rust closure for `SerialBackend`. Both have bit-identical observable behavior; determinism tests verify.

### Cross-entity walks

`for t in quest.eligible_acceptors` becomes a kernel that reads a GPU-resident `AggregatePool<Quest>` entry, iterates its embedded `[AgentId; N]` array, dispatches inner work per agent. Bounded by the pool's fixed-size array capacity.

Implementation: `crates/engine/src/cascade/`, with `backend/serial/cascade.rs` and `backend/gpu/cascade.rs`.

---

## 12. Mask buffer

Same layout; backend-local storage.

### Serial

`Vec<bool>` per head. Predicates mutate via `impl MaskBuffer { fn mark_hold_allowed(&mut self, state: &SimState) }`.

### GPU

FieldHandle per head (each bit stored as u32, one thread per bit). Predicates are SPIR-V kernels that write per-slot bits based on state queries.

Universal predicates (hold, move-allowed, flee-if-threat, attack-in-range, needs, domain-hook) have engine-shipped GPU kernels. Domain-specific predicates come from the compiler.

### Dispatch

```rust
pub trait Mask {
    fn reset(&mut self, backend: &mut dyn ComputeBackend);
    fn mark_hold_allowed(&mut self, backend: &mut dyn ComputeBackend, state: &SimState);
    fn mark_move_allowed_if_others_exist(&mut self, backend: &mut dyn ComputeBackend, state: &SimState);
    // ... etc.
}
```

Backend chooses scalar vs kernel dispatch internally.

**Mask validity invariant** (§20) checks every chosen action's bit. GPU backend downloads the relevant mask slice before invariant check.

---

## 13. Policy backend

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

- **`UtilityBackend`** — has both variants. Serial: scalar argmax over masked score table. GPU: parallel argmax kernel per agent. Both produce the same argmax (integer tie-break on MicroKind ordinal).
- **`NeuralBackend`** — GPU-only in the MVP. Runs compiler-emitted matmul kernels for forward pass. Serial impl is a `todo!()` stub for now (lands when compiler emits Rust matmul code paths).

Cross-backend parity: for every policy + seed + state, `SerialBackend::evaluate` and `GpuBackend::evaluate` produce byte-identical `ActionBuffer` contents (after download).

Implementation: `crates/engine/src/policy/`, with backend-local variants in `crates/engine/src/backend/*/policy.rs`.

---

## 14. Tick pipeline

Six phases; each dispatched through the current `ComputeBackend`.

| Phase | Host-side vs backend-dispatched | Both backends |
|---|---|---|
| 1. Mask build | Backend-dispatched | Serial: scalar predicates; GPU: kernel dispatch |
| 2. Policy eval | Backend-dispatched | Serial: PolicyBackend scalar; GPU: PolicyBackend kernel |
| 3. Action shuffle | **Host-side** | Shuffle operates on the ActionBuffer after download (GPU) or in-place (Serial); seed-deterministic |
| 4. Apply + cascade | Backend-dispatched | Serial: Rust closures; GPU: SPIR-V kernels |
| 5. View fold | Backend-dispatched | Serial: scalar fold; GPU: sorted-key reduction |
| 6. Invariant + telemetry | **Host-side** | Backends download state snapshot as needed; invariants read mirror |

Phase 3 (shuffle) is host-side because Fisher-Yates is inherently sequential and the cost of downloading-to-shuffle-then-uploading is negligible relative to one shuffle per tick. The ActionBuffer is sync'd at the phase-2 boundary.

Phase 6 is host-side because invariants + telemetry are inherently small, low-frequency, and pragmatic to run on the host.

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

`ComputeBackend` abstracts the storage + kernel dispatch surface. `step_full` doesn't know whether it's talking to CPU or GPU.

---

## 15. Views

Three storage modes × two backends.

| Mode | Serial impl | GPU impl |
|---|---|---|
| `MaterializedView` | Scalar fold over event iterator | Sorted-by-target-id + parallel reduction kernel |
| `LazyView` | Compute on demand with staleness flag | Same shape; compute dispatched as kernel, cached in FieldHandle |
| `TopKView` | Per-target Vec with sort | Per-target small-array with parallel merge-and-truncate |

Determinism for materialized views under GPU requires **sorting events by their stable per-tick sequence number before reduction** — otherwise float associativity breaks parity. Commutative integer reductions (counts) don't need sorting.

---

## 16. Aggregates

`AggregatePool<T>` for non-spatial entities. Two variants:

- **Host-only aggregates** (default): T has no `Pod` constraint; works only with `SerialBackend` OR requires explicit download for GPU access. Good for quest metadata that rarely needs GPU access.
- **GPU-eligible aggregates**: `T: Pod`; storage is `FieldHandle` when `GpuBackend` is active. Required when cascade handlers running on GPU need to read aggregate fields.

For MVP, `Quest` and `Group` use the GPU-eligible shape — their fields are Pod (Option<AgentId> = u32, fixed-size arrays via `[AgentId; N]` instead of `SmallVec`).

---

## 17. Trajectory emission

Same safetensors output across backends. GPU backend downloads per-tick snapshot to the host-side `TrajectoryWriter` buffer; Serial backend writes directly.

Register-extensible tensor schema unchanged from current engine. Cross-backend test: both backends produce byte-identical safetensors on the same seed.

---

## 18. Save / load

Backend-agnostic snapshot format. On save, the current backend forces a full sync, downloads state to host mirror, and serializes the mirror. On load, the file is deserialized into host mirror; the backend's `upload_from_mirror()` restores GPU-resident state.

Format:

| Block | Content |
|---|---|
| Header (64 B) | Magic, engine schema_hash, kernel catalog hash, tick, seed |
| SoA hot field mirrors | One block per field, little-endian |
| SoA cold field mirrors | Option<T> encoded with present-bit |
| Pool freelist | alive Vec<bool> + freelist Vec<u32> (mirror — GPU reconstructs) |
| Event ring tail | Host-mirror snapshot of replay-continuity events |

Loading rejects a snapshot whose kernel catalog hash differs from the current engine's (beyond the standard schema hash check). This prevents "sim loaded with different kernel semantics than it was saved with" bugs.

---

## 19. Invariant runtime

Invariants run on the host, against the **host mirror post-sync**. The step pipeline forces a sync at phase-6 entry for any invariant that declares `requires_state: true` (a method on the Invariant trait).

Built-ins:

- `mask_validity` — checks post-apply actions/mask pair; ActionBuffer already downloaded at phase 3; no additional sync cost.
- `pool_non_overlap` — reads host Pool freelist/alive vec; GPU backend must have uploaded the Pool post-apply.
- `event_hash_stable` (dev-only) — re-hash replayable events; host-resident already.

Invariants may be GPU-expressible in theory but the engine runs them host-side for MVP — the check frequency (once per tick) makes the sync cost negligible.

---

## 20. Probe harness

Backend-agnostic. A `Probe` spawns a fixed initial state, runs N ticks through `step_full`, asserts on events / views / state. Probes run on whichever backend is active. A probe CAN be marked `#[probe(backend = "serial")]` to pin the backend, but the default is to run on both backends and assert parity.

---

## 21. Telemetry sink

Host-side only. Both backends emit metrics via the same sink API. GPU-specific metrics (kernel wall-time via timestamps, upload/download ms) go through the same sink under the `engine.gpu.*` prefix.

Built-in metrics (engine-emitted):

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

## 22. Schema hash

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

A kernel recompile that changes bytecode bumps the engine schema hash automatically. Checkpoint load (§18) rejects on mismatch.

---

## 23. Observation packing

`ObsPacker` builds `[n × feature_dim]` f32 for policy input.

- Serial: iterates agents in host Rust.
- GPU: parallel kernel writes directly to a GPU-resident `obs_field`.

Feature source traits are backend-aware; each source registers its pack fn (Rust closure) AND its SPIR-V pack kernel (for GPU).

---

## 24. Debug & trace runtime

Six components: `trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, `snapshot`. All host-side.

When state is GPU-resident, debug tools trigger downloads on demand:

- `trace_mask` — syncs observation snapshot + mask for the target tick
- `causal_tree` — events are already synced to host each tick
- `tick_stepper` — stops between phases; can request phase-specific downloads
- `tick_profile` — adds kernel-scoped timing for GPU backend

GPU stepping: each phase exposes a `debug_readback` toggle that forces additional syncs for inspection, at the cost of performance.

---

## 25. Backend selection + fallback policy

The engine exposes a single `new_backend()` entry point:

```rust
pub enum BackendKind { Serial, Gpu }

pub fn new_backend(preferred: BackendKind) -> Box<dyn ComputeBackend>;

// Default: prefer GPU, fall back to Serial:
pub fn new_backend_auto() -> Box<dyn ComputeBackend>;
```

Fallback sequence for `new_backend_auto()`:

1. Try `GpuBackend::new()`. Success → use it.
2. On `GpuInitError`, log via `tracing::warn!`, fall back to `SerialBackend::new()`.
3. SerialBackend never fails — `new()` returns `Box<dyn ComputeBackend>` directly (no `Result`).

Runtime fallback is deterministic per process: backend selection happens once at init; no mid-run swaps.

**CI strategy:** CI has a Vulkan-capable container (Mesa lavapipe for software Vulkan when no device). Parity tests run both backends and compare. Non-Vulkan CI (unlikely) falls back to Serial-only, with a warning.

**Feature flags:**

- `default-backends = ["serial", "gpu"]` — both compiled in.
- `["serial"]` — Serial-only build (for embedded or CI without GPU).
- `["gpu"]` — GPU-only build (for shipped game runtime on known-GPU targets).

Engine crate's Cargo.toml:

```toml
[features]
default = ["serial", "gpu"]
serial = []
gpu = ["dep:voxel_engine"]

[dependencies]
voxel_engine = { path = "/home/ricky/Projects/voxel_engine", optional = true }
```

---

## 26. What's NOT in the engine

- **DSL parser and codegen.** Compiler concern.
- **Verb desugaring / Read → Ask lowering.** Compiler concern.
- **Domain types** (item catalog, ability list, group kinds). Compiler-generated.
- **Cascade RULES** (the DSL rules themselves). Compiler emits the kernel bytecode + Rust closures; engine provides the dispatch runtime for both.
- **Chronicle prose templates.** Host-side text generation; not in engine.
- **Curriculum pipelines.** External.
- **LLM backend implementation.** Separate downstream crate.

The engine provides **both runtimes** (Serial + GPU). Neither is "downstream"; neither is "a future plan." Both are first-class as of the 2026-04-19 spec rewrite.

---

## Implementation map

Status as of 2026-04-19 (rewrite day).

| Section | Serial | GPU | Notes |
|---|---|---|---|
| §3 State model | ✅ | ❌ | FieldHandle residency not yet built |
| §4 Event log | ✅ | ❌ | GPU ring w/ atomic append TBD |
| §5 Spatial index | ✅ | ❌ | Voxel-chunk-keyed TBD |
| §6 RNG streams | ✅ | ❌ | Shader RNG TBD; CPU-side done |
| §7 MicroKind | ✅ | ❌ | Apply kernels TBD |
| §8 MacroKind | ✅ | ❌ | Apply kernels TBD |
| §9 Cascade runtime | ✅ | ❌ | SPIR-V handler dispatch TBD |
| §10 Mask buffer | ✅ | ❌ | Kernel predicates TBD |
| §11 Policy backend | ✅ | ❌ | GPU argmax kernel TBD |
| §12 Tick pipeline | ✅ 6-phase | ❌ | Backend trait not yet extracted |
| §13 Views | ✅ | ❌ | Parallel reductions TBD |
| §14 Aggregates | ✅ | ⚠️ | Need T: Pod discipline |
| §15 Trajectory | ✅ | ❌ | GPU download path TBD |
| §16 Save/load | ❌ | ❌ | Both TBD |
| §17 Invariants | ✅ | ⚠️ | Post-sync invariants TBD |
| §18 Probes | ❌ | ❌ | Both TBD |
| §19 Telemetry | ✅ | ⚠️ | GPU metrics TBD |
| §20 Schema hash | ✅ | ⚠️ | Kernel-hash inclusion TBD |
| §21 Obs packing | ❌ | ❌ | Both TBD |
| §22 Debug & trace | ❌ | ❌ | Both TBD |
| §25 Backend selection | ❌ | ❌ | Trait + selection TBD |

The **Serial column is the ground truth**. GPU implementations land one section at a time, each verified bit-for-bit against Serial on fixed seeds. The existing 150-test suite is the determinism oracle; adding GPU implementations adds new tests but never changes the expected outputs.

**Implementation plan sequencing:**

- **Plan 3** — persistence + obs packer + probes (Serial first, per existing plan). Adds §16, §18, §21, §22 Serial ✅.
- **Plan 4** — debug & trace runtime (Serial first). Completes Serial backend fully.
- **Plan 5** — `ComputeBackend` trait extraction + SerialBackend refactor. Existing code moves behind the trait with no semantic change. New parity test infrastructure.
- **Plan 6** — GpuBackend foundation: VulkanContext init, state residency, first kernel (mask_hold). Cross-backend parity verified.
- **Plan 7+** — GPU kernel porting, one section at a time. Mask predicates → policy argmax → apply kernels → cascade handler SPIR-V → view reductions → obs packing. Each kernel ships with a cross-backend parity test.

---

## References

- `../dsl/spec.md` — language reference (grammar, type system, worked example, settled decisions)
- `../compiler/spec.md` — compiler contract (emission modes — CPU code vs GPU kernel dispatch — for the unified engine target)
- `../dsl/stories.md` — per-batch user-story investigations
- `../dsl/decisions.md` — per-decision rationale log
- `crates/engine/src/` — Rust implementation (Serial complete through Plan 2; GPU starting Plan 6)
- `crates/engine/shaders/` — SPIR-V bytecode + GLSL source (landing with Plan 6+)
- `crates/engine/tests/parity_backends.rs` — mandatory cross-backend determinism test (landing with Plan 5)
