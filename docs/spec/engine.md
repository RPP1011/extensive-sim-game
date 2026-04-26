# Engine Specification

Unified runtime contract for all compiler output. Merges the former `runtime.md` (engine contract) and `gpu.md` (GPU backend) into one canonical reference.

The engine ships **two first-class backends** that implement the same interface:

- **`SerialBackend`** — host-resident state, scalar Rust execution. The **reference implementation**. Determinism oracle for every test, for every parity check against the GPU backend, for every verified-correct port.
- **`GpuBackend`** — GPU-resident state via `voxel_engine::GpuHarness`, compiled SPIR-V kernels for gather + apply phases. Performance path for large N.

An implementation is correct if and only if it satisfies §§2–26 AND **produces byte-identical `replayable_sha256()` across both backends on the same seed**.

Companion documents: `dsl.md` (language reference + compiler contract), `state.md` (field catalog).

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

> ⚠️ **Audit 2026-04-26:** Cross-backend determinism is mandated here but **no end-to-end parity test exists** — `crates/engine/tests/parity_backends.rs` (the contract test) is absent; only Serial-only determinism tests + GPU mask/scoring parity tests exist.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** `step_full` is `unimplemented!()` in `crates/engine/src/step.rs`; the actual six-phase pipeline lives in `engine_rules::step::step` (different crate). `ComputeBackend` is non-object-safe (generic over `B: PolicyBackend`) — callers cannot use `dyn ComputeBackend` interchangeably as the architecture diagram implies. Crate split is `engine` / `engine_gpu` / `engine_rules` / `engine_data`, not the single-crate layout the spec describes.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** GPU residency model as described (per-field dirty mirror, `FieldHandle`, `upload_dirty`, sync-point table) is **not implemented**. The actual `GpuBackend` packs full `SimState` into GPU buffers via `pack_agent_slots` / `unpack_agent_slots` each tick — a synchronous copy, not a lazy dirty-tracked mirror.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

See `dsl.md` §7.3 (replay scope). Implementation: `crates/engine/src/event/`, `crates/engine/src/backend/*/event.rs`.

---

## 6. Spatial index

> ⚠️ **Audit 2026-04-26:** `in_column_z_range` query primitive is absent from both CPU and GPU spatial implementations despite being part of the spec's query API. GPU spatial hash exists but is not yet wired into the fused scoring/physics kernels.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

Two backends; same query API: `within_radius(center, r)`, `nearest_k(center, k)`, `in_column_z_range(col, z_range)`.

### Serial

2D-column BTreeMap + per-column sorted z-list + `movement_mode` sidecar. As previously specified. Deterministic by BTreeMap ordered iteration.

### GPU

Voxel-chunk-keyed spatial hash residing in FieldHandles. Cell-size matches voxel chunks (16 m) per compiler spec §1.2. Queries are compute-kernel dispatches:

- `within_radius_kernel` — fills a GPU result buffer with matching agent IDs
- `nearest_k_kernel` — parallel top-K reduction

For host-side queries via the public `SpatialIndex` API (debug tools, cascade handlers that happen to run CPU), GPU backend falls back to a synced CPU-side copy. Cost: one dispatch + download per query. Acceptable for low-frequency use; not for hot-loop callers.

Insert/remove: both backends rebuild on `spawn_agent` / `kill_agent` / `set_agent_pos`. GPU rebuild is a batched kernel per tick.

See `dsl.md` §9.2 (D25). Implementation: `crates/engine/src/spatial/`.

---

## 7. RNG streams

> 🤔 **Audit 2026-04-26 (spec mismatch):** Implementation uses `ahash` (a hash function), not PCG-XSH-RR, for `per_agent_u32`/`per_agent_u64`. The four constants (`RNG_KEY_1..4`) match but are AHasher seeds, not PCG stream keys. GPU uses WGSL not GLSL — `crates/engine/shaders/rng.glsl` does not exist. The cross-backend golden test (`tests/rng_cross_backend.rs`) does not exist.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

See `dsl.md` §9.2 (D12). Implementation: `crates/engine/src/rng.rs`, `crates/engine/shaders/rng.glsl` (header), `crates/engine/tests/rng_cross_backend.rs`.

---

## 8. GPU runtime integration

> 🤔 **Audit 2026-04-26 (spec mismatch):** Implementation uses `wgpu` (portable WebGPU abstraction), not `voxel_engine::VulkanContext + GpuHarness` — `voxel_engine` is not a dependency of `crates/engine_gpu`. There is no `KernelCatalog` struct; kernels are built from WGSL strings at runtime via `naga`, not loaded from pre-compiled SPIR-V `.spv` files. Feature flags also diverge (no `serial` feature; `default = []` in `engine_gpu`).
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` and `docs/superpowers/notes/2026-04-26-audit-gpu.md` for detail.

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

See `dsl.md` §3.3 and Appendix A.

---

## 10. Macro mechanisms

Same 4 variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`) + `NoOp`.

Announce cascade (the only universal macro that mutates state) runs as:

- Serial: `for obs in state.agents_alive() { ... }` loop with **3D Euclidean** distance check (`spatial.within_radius(center, r)` evaluated in 3D), emits RecordMemory per recipient.
- GPU: `apply_announce` kernel — parallel **3D distance** check per agent, atomic-append to event ring for each match, bounded by `MAX_ANNOUNCE_RECIPIENTS` via an atomic counter early-exit.

**Distance semantics:** 3D Euclidean across both backends. Agents at different elevations are evaluated by full 3D distance, not planar (XZ-only). Confirmed 2026-04-26 (status.md Q#1 resolved).

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

> ⚠️ **Audit 2026-04-26:** `NeuralBackend` is not implemented (no type, no `todo!()` stub). `PolicyBackend::evaluate` signature differs from spec — actual takes `(&self, state, mask, target_mask, out: &mut Vec<Action>)` with no `ComputeBackend` or `ActionBuffer` parameter. GPU scoring outputs go into a separate `Vec<ScoreOutput>` readback, not a shared `ActionBuffer`.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** `step_full` (the full-signature entry point with views/invariants/telemetry) is `unimplemented!()` in `crates/engine/src/step.rs`. Real step logic lives in `engine_rules::step::step` with a `&DebugConfig` for invariant/telemetry hookup — close to but not the spec's `step_full` signature. GPU phases 3–6 still run CPU-side.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** `Quest` and `Group` use `SmallVec` (not `[AgentId; N]` Pod arrays) and are not GPU-eligible. The two-variant "host-only vs GPU-eligible" `AggregatePool` design described here is absent.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** Snapshot header does not include `kernel_catalog_hash` (because `KernelCatalog` doesn't exist as described). GPU `upload_from_mirror()` is not implemented; `crates/engine_gpu/src/snapshot.rs` implements a different `GpuSnapshot` (observer API), not the §18 save/load contract.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ⚠️ **Audit 2026-04-26:** `requires_state` is not on the actual `Invariant` trait (which has only `name`, `failure_mode`, `check`). The `event_hash_stable` dev-only built-in is not implemented as such.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

Invariants run on the host, against the **host mirror post-sync**. The step pipeline forces a sync at phase-6 entry for any invariant that declares `requires_state: true` (a method on the Invariant trait).

Built-ins:

- `mask_validity` — checks post-apply actions/mask pair; ActionBuffer already downloaded at phase 3; no additional sync cost.
- `pool_non_overlap` — reads host Pool freelist/alive vec; GPU backend must have uploaded the Pool post-apply.
- `event_hash_stable` (dev-only) — re-hash replayable events; host-resident already.

Invariants may be GPU-expressible in theory but the engine runs them host-side for MVP — the check frequency (once per tick) makes the sync cost negligible.

---

## 20. Probe harness

> ❌ **Audit 2026-04-26:** `run_probe` is `unimplemented!()`. Probe infrastructure is a skeleton; tests using it (`probe_determinism.rs`, `probe_harness.rs`) are `#[ignore]`d. The `#[probe(backend = "serial")]` attribute does not exist.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

Backend-agnostic. A `Probe` spawns a fixed initial state, runs N ticks through `step_full`, asserts on events / views / state. Probes run on whichever backend is active. A probe CAN be marked `#[probe(backend = "serial")]` to pin the backend, but the default is to run on both backends and assert parity.

---

## 21. Telemetry sink

> ⚠️ **Audit 2026-04-26:** `engine.gpu.upload_ms`, `engine.gpu.download_ms`, `engine.gpu.kernel_ms` are **not defined** in `metrics.rs` and not emitted by `GpuBackend`. `PhaseTimings` exists for internal use but is not pushed through the `TelemetrySink` API.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> 🤔 **Audit 2026-04-26 (spec mismatch):** No `kernel_catalog_hash` is included in `schema_hash()` because `KernelCatalog`/SPIR-V files do not exist. Implementation hash covers SoA layout, enums, event kinds, lane ordinals, RNG constants, builtin metrics/invariants, snapshot format version, feature sources — broader than spec minimum but missing the kernel-content input.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

The GPU surface adds the following to the schema hash (see §GPU-4 for GPU-specific details):

- `SimCfg` field layout (offsets, sizes).
- Event-kind dispatch manifest (rule names, kinds, registration order).
- Ability registry packed format (tag table layout, slot count).
- `chosen_ability_buf` packing format.
- View storage layouts for `@symmetric_pair_topk` and `@per_entity_ring` annotated views.

---

## 23. Observation packing

> ⚠️ **Audit 2026-04-26:** GPU-parallel observation packing is not present. `ObsPacker` is CPU-only; no GPU `obs_field` or SPIR-V pack kernel. Consistent with "Serial complete, GPU pending" status.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

> ❌ **Audit 2026-04-26:** `new_backend()` / `new_backend_auto()` / `BackendKind` enum **do not exist**. Callers construct backends directly. Feature-flag layout differs (no `serial` feature). Spec's `default = ["serial", "gpu"]` pattern is absent.
> See `docs/superpowers/notes/2026-04-26-audit-runtime.md` for detail.

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

## GPU-1. GPU execution modes

The GPU backend exposes two execution modes:

- **Sync mode** — `SimBackend::step(state, intents, dt_ms)`. CPU-driven cascade with one fence per kernel dispatch. Authoritative for parity tests; matches `SerialBackend` semantics.
- **Batch mode** — `step_batch(state, n, cascade)`. GPU-resident cascade with indirect dispatch. One submit per N ticks, observation via non-blocking `snapshot()`.

The **Resident Cascade** (§GPU-2) is the foundational pattern. It eliminates per-tick CPU/GPU fences by binding each kernel's outputs as the next kernel's inputs and replacing CPU-driven loops with GPU indirect dispatch.

**Layered subsystems** (each builds on the previous):

| Layer | Adds |
|---|---|
| Resident cascade (§GPU-2) | GPU-resident pipeline, indirect dispatch, double-buffered snapshot |
| Sim state mirroring (§GPU-3) | `SimCfg` shared buffer, GPU-side tick advance, `@cpu_only` annotation |
| Cold-state replay (§GPU-4) | Per-event-kind handler dispatch, gold/standing/memory on GPU |
| Ability evaluation (§GPU-5) | `pick_ability` kernel, `ability::tag` scoring primitive, `per_ability` row |

§GPU-6 is a kernel/buffer reference. §GPU-7 covers cross-cutting concerns (determinism, schema hash, parity contract).

**Non-goals**

- Byte-exact GPU↔CPU parity in batch mode. Atomic-tail event ordering is non-commutative on GPU; same-seed runs may diverge.
- Cross-GPU reproducibility. Same hardware + same seed reproduces; different vendors may diverge.
- Replacing the sync path. Sync stays load-bearing for parity tests, deterministic chronicle output, and `SerialBackend` cross-checks.

---

## GPU-2. Resident cascade

> ⚠️ **Audit 2026-04-26:** `step_batch` returns `()` not `Result<(), BatchError>` — `BatchError` does not exist; mid-batch kernel failures `panic!()` instead of degrading to sync. View fold kernels on the batch path are still CPU-resident (`fold_iteration_events`); resident fold kernel pending. `device`/`queue` are `wgpu::Device`/`Queue` (not `Arc<...>`).
> See `docs/superpowers/notes/2026-04-26-audit-gpu.md` for detail.

### GPU-2.1 Principle

In sync mode, every kernel's outputs are copied to CPU and the next kernel's inputs are re-uploaded. The resident cascade binds output buffers directly as the next kernel's inputs; CPU only observes when explicitly asked.

### GPU-2.2 Public surface

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

### GPU-2.3 One tick on the batch path

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

### GPU-2.4 Indirect dispatch for cascade iterations

No per-iteration readback of a "converged?" flag. Instead:

- End-of-iter, the physics kernel writes indirect dispatch args `(workgroup_count, 1, 1)` to a small GPU buffer, where `workgroup_count = ceil(num_events_next_iter / PHYSICS_WORKGROUP_SIZE)` clamped to `ceil(agent_cap / PHYSICS_WORKGROUP_SIZE)`.
- When there are no follow-on events, the kernel writes `(0, 1, 1)`. Subsequent indirect dispatches are GPU no-ops (microseconds).
- `run_cascade_resident` pre-records `MAX_CASCADE_ITERATIONS` indirect dispatches into one encoder.
- Iteration count surfaces as an inferred value from the args buffer, read alongside `snapshot()`.

### GPU-2.5 Submit shape

One command encoder records N ticks: each tick = mask + scoring + spatial + apply + movement + cascade-indirect × `MAX_CASCADE_ITERATIONS` + fold + tick-counter increment. One `queue.submit`. One `device.poll(Wait)` at end of batch.

Per-tick scalars that change (RNG seed, tick counter) live in the GPU-side `SimCfg` buffer (§GPU-3) updated by a tiny end-of-tick kernel.

### GPU-2.6 Snapshot — double-buffered staging

Three staging-buffer pairs: `{agents, events, chronicle}` × `{front, back}`.

On call:
1. Encode `copy_buffer_to_buffer` for current `agents_buf` + `event_ring[last_read..tail]` + `chronicle_ring[last_read..tail]` into the **back** staging buffers. Update `last_read` watermarks.
2. `queue.submit`.
3. `map_async(Read)` on the **front** staging buffers (filled by the previous `snapshot()` call).
4. `device.poll(Wait)` — drives pending map callbacks.
5. Decode front staging → `GpuSnapshot`, unmap, swap front/back pointers.

First call returns `GpuSnapshot::empty()`. The one-frame lag is acceptable because the rendering layer interpolates via a delta value.

### GPU-2.7 Additivity

- `SimBackend::step()` and its `GpuBackend` impl are unmodified by the resident path.
- Caller-provided `EventRing` is populated only by the sync path. Batch events are observable only via `snapshot().events_since_last`.
- Existing parity tests, perf tests, scenario tests run against the sync path.

### GPU-2.8 Error cases

| Failure | Detection | Behaviour |
|---|---|---|
| Indirect args corruption | Kernel clamps `workgroup_count ≤ ceil(agent_cap / WGSIZE)` before write; snapshot validates | Logged warning; bounded dispatch |
| GPU ring overflow | Kernel writes `overflowed` flag; `snapshot()` reads it | `Err(SnapshotError::RingOverflow { tick, events_dropped })` |
| Cascade non-convergence | Indirect args still non-zero at last iteration | Warning on snapshot; correctness unaffected (subsequent ticks pick up) |
| Staging map failure | `map_async` callback returns `Err` | `Err(SnapshotError)`; first call returns `Ok(empty)` |
| Kernel dispatch failure | `wgpu` validation / device-lost | `Err(BatchError)`; **no CPU fallback** — caller re-issues via sync `step()` for graceful degradation |

### GPU-2.9 Performance contract

- `chronicle --perf-sweep --batch-ticks N` is the gate.
- Target: batch mean µs/tick `< 0.8×` sync mean µs/tick at N ≥ 512.
- Looser at smaller N where per-tick overhead dominates.
- `PhaseTimings` records `batch_submit_us` and `batch_poll_us` so the sweep distinguishes encoding cost from GPU execution cost.

### GPU-2.10 Backend factoring

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

## GPU-3. Sim state on GPU

### GPU-3.1 `SimCfg` — shared GPU storage buffer

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

### GPU-3.2 Tick advance

The seed-indirect kernel that runs once per tick to seed cascade iter 0 grows one line:

```wgsl
atomicAdd(&sim_cfg.tick, 1u);
```

at end-of-tick. No new kernel, no new dispatch.

### GPU-3.3 RNG

`per_agent_u32(agent_id, purpose_tag)` is a pure hash of `(world_seed, tick, agent_id, purpose_tag)`. No GPU atomic state; the function stays pure.

- `world_seed` uploaded once at `ensure_resident_init` to `SimCfg.world_seed_{lo,hi}`.
- `tick` read from `SimCfg.tick` instead of per-kernel cfg.

### GPU-3.4 CPU `state.tick` during batch

Stale across the whole batch by design. `step_batch` does not advance `state.tick`. End of batch, `state.tick` is unchanged from start-of-batch. The next `snapshot()` reads `SimCfg.tick` and reports it via `GpuSnapshot.tick`. Callers wanting CPU `state.tick` to reflect reality call `snapshot()` first.

The sync `step()` path is unaffected — it advances `state.tick` on CPU each call (sync path doesn't yet use `SimCfg`).

### GPU-3.5 `@cpu_only` annotation

DSL annotation marking rules that stay CPU-resident by design (chronicle narrative formatting, debug-only side effects, anything requiring strings or unbounded allocation).

```
@cpu_only physics chronicle_render_attack_narrative @phase(event) {
    on AgentAttacked { ... build String ... push chronicle text ... }
}
```

Compiler behaviour:

1. Records the annotation on the rule's IR node.
2. **Emits the CPU handler** as today (Rust function, registered in `CascadeRegistry::with_engine_builtins()`).
3. **Skips WGSL emission**. No entry in the GPU physics kernel's event dispatch table; no entry in the per-event-kind GPU dispatcher table (§GPU-4).
4. **Relaxes the GPU-emittable validator** — primitives it would otherwise reject (strings, unbounded alloc) are accepted inside `@cpu_only` rule bodies.

The implicit `@gpu_emittable` default is unchanged. Existing rules don't need annotations.

### GPU-3.6 SimState mirroring summary

| State | CPU resident | GPU resident | Mirroring |
|---|---|---|---|
| Hot agent fields (pos, hp, alive, …) | `SimState.agents` | `resident_agents_buf` (`GpuAgentSlot`) | Upload at `ensure_resident_init`; snapshot copies back |
| Tick | `state.tick` | `SimCfg.tick` (atomic u32) | GPU advances; CPU reads via snapshot |
| World seed | `state.world_seed` | `SimCfg.world_seed_{lo,hi}` | Upload once at init |
| World scalars (engagement_range, …) | `state.config.combat.*` | `SimCfg.*` | Upload once at init |
| RNG state | (derived) | (derived) | Stateless: `hash(seed, tick, agent_id, purpose)` |
| Cold-state (gold, standing, memory) | `SimState.cold_*` | Side buffers (§GPU-4) | Upload at init; snapshot reads back |
| Event ring | Caller-provided | `event_ring_buf` | Sync: populated each tick. Batch: only via snapshot |
| Chronicle | (CPU `Vec`) | `chronicle_ring_buf` | Sync: drained per call. Batch: snapshot watermark |

### GPU-3.7 Field-layout invariants

WGSL struct layout (alignment, padding) must match Rust struct layout. Compile-time assertions compare field offsets against hand-written WGSL constants. Drift panics at startup.

`SimCfg` storage binding is **storage**, not **uniform**, because `tick` is mutated by atomic. All other kernels declare it `read_only`.

---

## GPU-4. Cold-state replay on GPU

> ⚠️ **Audit 2026-04-26:** The §GPU-4.1 "one WGSL handler kernel per GPU-resident event kind + compiler-emitted manifest + Rust dispatch driver" architecture is **not implemented**. Implementation instead routes gold/standing/memory mutations through the physics kernel's WGSL handler (atomics in `GoldBuf`, `standing_storage`, `memory_storage`) bound at slots 17/18/19/20/21. There is no generated dispatch manifest; the compile-time manifest check (§GPU-4.2) does not exist. Outcome equivalent for the three implemented cold-state types, but architecture diverges from spec's extensibility contract.
> See `docs/superpowers/notes/2026-04-26-audit-gpu.md` for detail.

### GPU-4.1 Dispatch framework

One WGSL handler kernel per GPU-resident event kind. End of each tick's cascade iteration, the cascade driver scans observed events by kind and dispatches the matching kernel(s).

**Emission is automatic.** Every DSL `physics rule @phase(event) { on EventKind { ... } }` not marked `@cpu_only` produces (a) a WGSL kernel and (b) an entry in a generated dispatch-by-event-kind manifest. No hand-written dispatcher kernel — the driver is Rust glue reading the compiler-emitted manifest.

**Per kind, not per event.** A tick with 100 `AgentAttacked` events dispatches the rule kernel once; the kernel processes all 100 events in the slice. Event kinds with zero observations skip dispatch (CPU reads the per-kind event count, as it already does for the cascade seed kernel).

Placement: end-of-tick, after cascade converges, before next tick starts. All inside one command encoder — no new submits.

### GPU-4.2 Event-kind dispatch table (generated)

| Event kind | Handler rule(s) | Storage written | Notes |
|---|---|---|---|
| `AgentAttacked` | `damage`, `opportunity_attack`, structured chronicle | `agents_buf.hp/shield`, chronicle ring | GPU-native |
| `AgentDied` | `fear_spread`, structured chronicle | `view_storage.kin_fear` | GPU-native |
| `EffectGoldTransfer` | `transfer_gold` | `gold_buf` (atomic add/sub) | §GPU-4.3 |
| `EffectStandingDelta` | `modify_standing` view fold | `standing` view storage | §GPU-4.4 |
| `RecordMemory` | `record_memory` view fold | `memory` view storage | §GPU-4.5 |
| `AgentCast` | `cast` physics rule (existing) | `agents_buf`, event ring | GPU-native |
| `chronicle_*_narrative` (text) | (none on GPU) | — | `@cpu_only` |

Registry refresh: when a rule is added to DSL, the dispatch manifest regenerates. Compile-time check: every non-`@cpu_only` rule appears in the manifest.

### GPU-4.3 Gold

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

### GPU-4.4 Standing

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

### GPU-4.5 Memory

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

### GPU-4.6 Chronicle

Two tiers, decided per-rule:

- **Structured chronicle** — fixed-layout `ChronicleEntry` (template id + fixed payload). Emitted GPU-side via the existing chronicle-emission DSL primitive into the existing GPU chronicle ring. Snapshot exposes via `chronicle_since_last`.
- **Narrative chronicle** — multi-sentence prose with string interpolation. Marked `@cpu_only`. Runs async off the batch snapshot's event stream (CPU consumer iterates `events_since_last`).

No new DSL grammar for chronicle — the existing chronicle-emission primitive handles structured entries; `@cpu_only` (§GPU-3.5) handles narrative.

### GPU-4.7 Snapshot handshake

```
gold_buf       → copy into SimState.cold_inventory.gold
standing view  → copy into SimState.cold_standing (or expose via state.views)
memory view    → copy into SimState.cold_memory (or expose via state.views)
chronicle_ring → snapshot.chronicle_since_last (existing watermark)
```

`GpuSnapshot` gains optional fields gated by snapshot-flag config (caller doesn't pay for what they don't read).

### GPU-4.8 Engine-core vs DSL-lowered

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

## GPU-5. Ability evaluation on GPU

> ❌ **Audit 2026-04-26:** **Subsystem 4 is entirely absent.** No `cs_pick_ability` kernel exists. `per_ability` rows parse and resolve into `ScoringIR.per_ability_rows` but **neither CPU emitter nor WGSL emitter iterates over them** — silent no-op bug. `chosen_ability_buf` is not allocated. `apply_actions` does not read it. `PackedAbilityRegistry.hints`/`tag_values` are populated but bound to nothing. No `pick_ability_*` tests exist. Cast decisions fall to CPU oracle / student model.
> See `docs/superpowers/notes/2026-04-26-audit-gpu.md` for detail.

### GPU-5.1 Position in pipeline

```
mask → scoring → pick_ability → apply_actions → movement → spatial → cascade(N iters) → fold → cold_state_dispatch
                 ^^^^^^^^^^^^
                 new
```

Scoring runs first, producing per-agent action scores (attack/move/flee/hold). `pick_ability` runs next with its own output buffer. `apply_actions` reads both — if an ability is chosen, emit `AgentCast`; otherwise, emit the scoring kernel's chosen action.

### GPU-5.2 `pick_ability` kernel

Compiler-emitted from DSL (`pick_ability.wgsl`). Per agent per tick:

- Iterates abilities in the agent's known set.
- Evaluates each ability's `guard`. Skips on false.
- Evaluates each ability's `score`. Tracks argmax.
- Picks target via the ability's `target:` clause.
- Writes `chosen_ability_buf[agent]` = packed `(ability_slot: u8, target_agent_id: u32, sentinel-for-no-cast)`.

If no ability's guard passes, writes the no-cast sentinel.

### GPU-5.3 New scoring grammar — `ability::tag(TAG)` primitive

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

### GPU-5.4 New scoring grammar — `per_ability` row type

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

### GPU-5.5 Tag registry

Each `.ability` file carries:

- `hint:` — coarse category enum (`damage` | `defense` | `crowd_control` | `utility`). One per ability.
- Per-effect `[TAG: value]` — numeric power ratings, multiple per effect line.

Tag names are a fixed enum (each tag is a known buffer index; lower-cost than a symbol table). User-extensible string tags are deferred.

`AbilityDef.tags` serialise into the `PackedAbilityRegistry` consumed by kernels. The tag table is bound to `pick_ability` alongside the ability registry.

### GPU-5.6 Target selection

`nearest_hostile_in_range(range)` uses the existing GPU spatial-hash output (kin + hostile result buffers from §GPU-6 spatial kernel). Pointer-attention / learned targeting / lowest-HP heuristics are deferred — this subsystem ports the existing CPU heuristic only.

### GPU-5.7 Side buffer

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

### GPU-5.8 Engine-core vs DSL-lowered

| Concern | Owner |
|---|---|
| `chosen_ability_buf` allocation + binding | Engine-core |
| `pick_ability` kernel dispatch in `step_batch` | Engine-core |
| `apply_actions` extension to read `chosen_ability_buf` and emit `AgentCast` | Engine-core |
| Tag table serialisation into ability registry (if missing) | Engine-core |
| `pick_ability.wgsl` (compiler output from `per_ability` row) | DSL-lowered |
| CPU `pick_ability` handler (compiler output, replaces hand-tuned `evaluate_hero_ability`) | DSL-lowered |
| Tag reads + scoring arithmetic | DSL-lowered (`scoring.sim`) |

### GPU-5.9 Failure modes

| Failure | Behaviour |
|---|---|
| Ability tag missing | `ability::tag(UNKNOWN)` returns 0 silently (tags are sparse per ability) |
| All abilities on cooldown | `pick_ability` writes sentinel; `apply_actions` falls through to score_output |
| Tag value overflow | f32 in WGSL; designer-visible. Not a runtime error |
| Cooldown buffer not initialised | Verified at `ensure_resident_init`; panic on missing |
| Per-agent ability count variance | Empty slots score 0; ~5 wasted evaluations per agent per tick at MAX_ABILITIES=8 |

---

## GPU-6. Pipeline reference

### GPU-6.1 Sync path

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

### GPU-6.2 Resident path

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
    ├─ 4. pick_ability               (§GPU-5)
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
    └─ 10. cold_state_dispatch       (§GPU-4)
          per event kind K with count > 0: dispatch K's handler kernel
          view fold kernels (engaged_with, my_enemies, threat_level, kin_fear,
                             pack_focus, rally_boost, standing, memory)
```

### GPU-6.3 Snapshot

```
snapshot() -> GpuSnapshot
├─ poll front staging buffer (non-blocking from previous snapshot)
├─ copy_buffer_to_buffer(live GPU buffers -> back staging)
└─ swap front/back
```

One-frame lag. The returned snapshot contains data from the tick before the current one. The double-buffer prevents GPU→CPU sync on the hot path.

### GPU-6.4 Kernel inventory

#### GPU-6.4.1 Mask kernel

| Property | Value |
|---|---|
| Struct | `FusedMaskKernel` (sync), `MaskUnpackKernel` (resident unpack) |
| Entry points | `cs_fused_masks` (sync/batch); `cs_mask_unpack` (resident unpack) |
| Workgroup size | 64 |
| Bind group | Sync: agents, 7 bitmap outputs, cfg. Resident unpack: resident_agents_buf → mask SoA |
| Inputs | Agent position, alive, creature_type; ConfigUniform (movement radius) |
| Outputs | 7 bitmap arrays: Attack, MoveToward, Hold, Flee, Eat, Drink, Rest (atomic u32 per agent) |
| Notes | Cast mask covered by `pick_ability` (§GPU-5); fused dispatch writes all 7 in one call |

#### GPU-6.4.2 Scoring kernel

| Property | Value |
|---|---|
| Struct | `ScoringKernel` (sync), `ScoringUnpackKernel` (resident) |
| Entry point | `cs_scoring` |
| Workgroup size | 64 |
| Bind group | agent_data SoA, bitmaps, view_storage (atomic reads), cfg, sim_cfg, spatial queries |
| Inputs | 7 mask bitmaps; agent fields; view_storage atomic reads (my_enemies, threat_level, kin_fear, pack_focus, rally_boost) |
| Outputs | `ScoreOutput[agent_cap]` — per-agent struct: chosen_action, chosen_target, score |
| Notes | Spatial query reads are read-only on precomputed kin / nearest-hostile results |

#### GPU-6.4.3 Pick ability kernel

| Property | Value |
|---|---|
| Struct | (compiler-emitted) |
| Entry point | `cs_pick_ability` |
| Workgroup size | 64 |
| Bind group | agent_data, ability_registry, tag_table, ability_cooldowns_buf, spatial outputs, sim_cfg, chosen_ability_buf (write) |
| Inputs | Agent state, packed ability registry, cooldowns, spatial-query results |
| Outputs | `chosen_ability_buf[agent_cap]` — packed `(slot, target, sentinel)` u64 |
| Notes | Compiler-emitted from `per_ability` rows in `scoring.sim` (§GPU-5.4) |

#### GPU-6.4.4 Apply actions kernel

| Property | Value |
|---|---|
| Struct | `ApplyActionsKernel` |
| Entry points | `cs_apply_actions` (sync), `cs_apply_actions_resident` (batch) |
| Workgroup size | 64 |
| Bind group | agents (rw), scoring (r), chosen_ability_buf (r, batch), event_ring (rw), event_ring_tail (atomic), cfg, sim_cfg |
| Inputs | ScoreOutput; agent slots; chosen_ability_buf (batch only) |
| Outputs | Mutated hp/shield/alive; events: AgentAttacked, AgentDied, AgentCast (batch), AgentAte, AgentDrank, AgentRested |
| Scope gaps | Opportunity attacks, engagement slow on MoveToward, announce/communicate — sync CPU path |

#### GPU-6.4.5 Movement kernel

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

#### GPU-6.4.6 Physics kernel (event processor)

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

#### GPU-6.4.7 Fold kernels (view materialization)

| Property | Value |
|---|---|
| Entry points | `cs_fold_<view_name>` — one per view (engaged_with, my_enemies, threat_level, kin_fear, pack_focus, rally_boost, **standing**, **memory**) |
| Workgroup size | 64 |
| Bind group | fold_inputs (r), view_storage (rw atomic), cfg |
| Inputs | FoldInput batch (observer_id, other_id, delta, anchor_tick) |
| Outputs | View storage atomic updates (CAS loop) |
| Determinism | Commutative folds (`+= 1.0`) — atomic CAS order doesn't matter for the sums |

#### GPU-6.4.8 Spatial hash kernels (resident path)

| Property | Value |
|---|---|
| Entry points | `cs_spatial_hash`, `cs_kin_query`, `cs_engagement_query` |
| Inputs | Agent positions; two radii (kin=12m, engagement=2m) |
| Outputs | Per-agent query results: nearby agents (within), kin-species membership, nearest hostile/kin (one u32 per agent) |
| Notes | Resident path uses GPU spatial hash; sync path uses CPU spatial hash |

### GPU-6.5 Buffer ownership

| Buffer | Owner | Size (N=100k) | Purpose |
|---|---|---|---|
| `resident_agents_buf` | `ResidentPathContext` | ~16 MB | Agent SoA, persistent across batch |
| `sim_cfg_buf` | `ResidentPathContext` | 256 B | `SimCfg` (§GPU-3.1) — atomic tick + world scalars |
| `apply_event_ring` | `CascadeCtx` (sync) | ~24 MB | Seeds physics iter 0; cleared per-tick |
| `physics_ring_a` / `_b` | `CascadeResidentCtx` | ~24 MB each | Ping-pong resident cascade rings |
| `batch_events_ring` | `CascadeResidentCtx` | ~24 MB | Append-only accumulator across batch ticks; exposed to snapshot |
| `chronicle_ring` (sync) | `PhysicsKernel` | ~24 MB | Narrative records; drained separately |
| `chronicle_ring` (resident) | `CascadeResidentCtx` | ~24 MB | Caller-owned; snapshot watermark |
| `indirect_args` | `ResidentPathContext` | 32 B × (`MAX_CASCADE_ITERATIONS` + 1) | Indirect dispatch args per iteration |
| `num_events_buf` | `CascadeResidentCtx` | 4 B × (`MAX_CASCADE_ITERATIONS` + 1) | Event counts per iteration (diagnostic) |
| `view_storage` | `ViewStorage` | ~144 MB (6 views @ N=100k) | Materialised view state (incl. standing, memory in §GPU-4) |
| `gold_buf` | `ResidentPathContext` | ~400 KB | Per-agent atomic i32 gold (§GPU-4.3) |
| Spatial query outputs | `SpatialOutputs` | ~80 MB each | kin / engagement query results |
| `PackedAbilityRegistry` | engine | ~256 KB | Ability metadata; resident with content-addressed upload |
| `chosen_ability_buf` | `ResidentPathContext` | 8 B × N | Per-agent `(slot, target, sentinel)` u64 (§GPU-5.7) |

### GPU-6.6 Key constants

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

### GPU-6.7 DSL → WGSL lowering

Per-subsystem emitters in `dsl_compiler` produce WGSL modules:

- `emit_mask_wgsl` → fused module with per-mask bitmap writes.
- `emit_physics_wgsl` → one module with `physics_dispatch(event)` switch over event kinds.
- `emit_view_wgsl` → one fold kernel per view.
- `emit_pick_ability_wgsl` → `pick_ability.wgsl` from `per_ability` rows (§GPU-5).

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

## GPU-7. GPU cross-cutting concerns

### GPU-7.1 Determinism contract

| Path | Determinism |
|---|---|
| Sync GPU step | Same-seed reproducible on same hardware. Event tail atomic ordering serialised by per-call host fence. Host drain sorts events by `(tick, kind, payload[0])` pre-fold |
| Batch GPU step | **Non-deterministic in event order.** Atomic tail racing inside the resident cascade does not serialise. Same-seed runs may diverge in fold order. Statistical parity (alive counts, event multisets, conservation laws) is the contract |
| Cross-GPU | Same hardware + same seed reproduces. Different vendors may diverge — non-commutative GPU folds |
| RNG | Pure hash `(world_seed, tick, agent_id, purpose_tag)`. Stateless. Identical across CPU and GPU given identical inputs |

Determinism tests run against `SerialBackend` and against the GPU sync path. The batch path is explicitly excluded from byte-exact tests.

### GPU-7.2 Schema hash — GPU surface

The `SCORING_HASH` and engine schema hash include:

- `SimCfg` field layout (offsets, sizes).
- Event-kind dispatch manifest (rule names, kinds, registration order).
- Ability registry packed format (tag table layout, slot count).
- `chosen_ability_buf` packing format.
- View storage layouts for `@symmetric_pair_topk` and `@per_entity_ring` annotated views.

CI fence: changes to any of the above bump `crates/engine/.schema_hash`. Drift fails the parity tests at startup.

`rules_registry_generation` and `abilities_registry_generation` (in `SimCfg`) are u32 cache-invalidation counters — independent of the schema hash, used for kernel-local upload caches.

### GPU-7.3 Parity test contract

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

### GPU-7.4 Telemetry

> ⚠️ **Audit 2026-04-26:** `PhaseTimings` does **not** have typed `batch_submit_us` / `batch_poll_us` fields. `ResidentPathContext::last_batch_phase_us: Vec<(&str, u64)>` provides batch phase µs but as a generic vec, not the named struct fields the spec requires. Per-event-kind dispatch counts: absent (driver doesn't exist).
> See `docs/superpowers/notes/2026-04-26-audit-gpu.md` for detail.

`PhaseTimings` extends with:

- `batch_submit_us` — encoder record + `queue.submit` cost.
- `batch_poll_us` — `device.poll(Wait)` cost (GPU execution).
- Per-iteration cascade dispatch counts (for non-convergence diagnosis).
- Per-event-kind dispatch counts (cold-state replay).

`chronicle --perf-sweep --batch-ticks N` runs the sweep at varying batch sizes; CI gates at N=2048; local runs at N=100k.

### GPU-7.5 Failure surface summary

| Failure mode | Detection | Behaviour |
|---|---|---|
| Indirect args corruption (§GPU-2.8) | Kernel clamp + snapshot validate | Logged warning |
| Ring overflow (event / chronicle) | Kernel sets flag; snapshot reads | `Err(SnapshotError::RingOverflow)` |
| Cascade non-convergence | Final-iter args still non-zero | Warning; subsequent ticks pick up |
| Staging map failure | `map_async` callback `Err` | `Err(SnapshotError)` |
| Kernel dispatch failure | wgpu validation | `Err(BatchError)`; **no CPU fallback** |
| `SimCfg` field drift | Compile-time offset assertion | Startup panic |
| Missing dispatch entry (rule added, manifest stale) | Compile-time check | Build fail |
| Standing topk overflow | K=8 budget full | Lowest `|standing|` evicted; logged in debug |
| Memory ring overflow | By design | Oldest entry evicted; not an error |
| Gold atomic overflow | i32 wrap | Acceptable; cap at ability-design layer |
| Tag missing on ability | `ability::tag(MISSING)` | Returns 0 silently |
| All abilities on cooldown | `pick_ability` writes sentinel | Falls through to scoring kernel's chosen action |
| Cooldown buffer not initialised | `ensure_resident_init` check | Startup panic |

### GPU-7.6 Engine-core boundary

The standing rule "engine core = hand-written; game logic = DSL" applies on GPU:

- **Engine-core (hand-written Rust + WGSL)**: buffer allocation, bind-group layouts, dispatch driver, indirect-args plumbing, snapshot handshake, `ensure_resident_init`, `SimCfg` struct, sub-struct factoring.
- **DSL-lowered (compiler output)**: all rule bodies, view fold bodies, `pick_ability` kernel, scoring expressions, annotation processing (`@cpu_only`, `@symmetric_pair_topk`, `@per_entity_ring`).

---

## Implementation map

See **`../engine/status.md`** — per-subsystem state (Serial vs. GPU), associated plan, tests, weak-test risks, and visual-check criteria are tracked there. The **Serial column is the ground truth**; GPU lands one section at a time with cross-backend parity tests starting Plan 5+.

---

## References

- `dsl.md` — language reference (grammar, type system, worked example, settled decisions) + compiler contract (emission modes, schema hash emission, lowering passes)
- `state.md` — field catalog (every SoA field, who reads, who writes)
- `crates/engine/src/` — Rust implementation (Serial complete through Plan 2; GPU starting Plan 6)
- `crates/engine/shaders/` — SPIR-V bytecode + GLSL source (landing with Plan 6+)
- `crates/engine/tests/parity_backends.rs` — mandatory cross-backend determinism test (landing with Plan 5)
