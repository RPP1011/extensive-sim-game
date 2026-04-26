# Runtime Spec Audit (2026-04-26)

> Audit of `docs/spec/runtime.md` against implementation in `crates/`.

## Summary

- Total sections audited: 26 (§1 scope/§26 non-impl are non-code; 24 substantive)
- ✅ Implemented: 11
- ⚠️ Partial: 9 (most-impactful gaps listed below)
- ❌ Not implemented: 4
- 🤔 Spec mismatch: 2
- ❓ Needs judgment: 0

## Top gaps (ranked by impact)

**1. No cross-backend determinism test at all (§2 / §14).** The spec mandates `crates/engine/tests/parity_backends.rs` with a `run_n_ticks(SerialBackend) == run_n_ticks(GpuBackend)` SHA-256 assertion. That file does not exist. The closest thing — `crates/engine_gpu/tests/parity_with_cpu.rs` — compares individual GPU mask bitmaps and scoring outputs against CPU references, not the full per-tick event-log hash. The fundamental contract in §2 is untested end-to-end.

**2. `step_full` is a `unimplemented!()` stub; `run_probe` is `unimplemented!()` (§14 / §20 / §3).** `crates/engine/src/step.rs` is explicitly deleted as of Plan B1' Task 11. `step_full`, `step`, `apply_actions`, `finalize_tick`, and `shuffle_actions_in_place` all panic at runtime. The canonical replacement (`engine_rules::step::step`) exists and is used, but the `step_full` entry-point that drives the six-phase pipeline with views/invariants/telemetry is missing. The probe harness (`run_probe`) also panics for the same reason.

**3. GpuBackend does not implement the spec's `ComputeBackend` public API (§3 / §4 / §25).** `crates/engine_gpu` implements a bespoke `GpuBackend::step` that forwards to `engine::step::step` (CPU) and additionally runs GPU mask/scoring/cascade kernels in parallel for comparison/sidecar purposes. The spec describes a GpuBackend that is the authoritative tick driver (GPU-resident state, GPU-authoritative hot fields, per-field dirty mirror). The actual GpuBackend packs agent data from CPU `SimState`, dispatches GPU kernels, and unpacks back to CPU — it is a sync-path hybrid, not the GPU-authoritative residency model described in §4. The `new_backend()` / `new_backend_auto()` / `BackendKind` API from §25 does not exist.

**4. RNG cross-backend golden test missing; GLSL/WGSL `rng.glsl` header absent (§7).** The spec requires `crates/engine/tests/rng_cross_backend.rs` asserting `per_agent_u32` returns the same value on host Rust and in a test shader. That file does not exist. The RNG constants (`RNG_KEY_1..4`) appear in `crates/engine/src/rng.rs` but use `ahash::RandomState::with_seeds` (not PCG-XSH-RR with those constants as a shader header). No `crates/engine/shaders/rng.glsl` exists; GPU shaders use WGSL not GLSL. The spec's GLSL header example (§7) does not match the implementation's approach.

**5. KernelCatalog / SPIR-V bytecode + pre-compiled `.spv` files absent (§8 / §22).** The spec describes `KernelCatalog` with named kernel fields and pre-compiled SPIR-V embedded via `include_bytes!`. Implementation uses WGSL strings built at runtime via `naga`, not pre-compiled SPIR-V. No `.spv` files exist anywhere in `crates/`. The kernel-catalog-hash that §22 requires to be included in `schema_hash` is therefore absent from `crates/engine/src/schema_hash.rs`.

## Per-section findings

### §1 Scope
- Status: ✅
- Evidence: `crates/engine/src/lib.rs`, `crates/engine/src/pool/mod.rs`, `crates/engine/src/event/ring.rs`, `crates/engine/src/spatial.rs`, `crates/engine/src/rng.rs`, `crates/engine/src/cascade/`, `crates/engine/src/mask.rs`, `crates/engine/src/view/`, `crates/engine/src/trajectory.rs`, `crates/engine/src/snapshot/`, `crates/engine/src/invariant/`, `crates/engine/src/debug/`, `crates/engine/src/backend.rs`
- Notes: The scope items are all present as modules. The split "not owned" items (DSL parser etc.) are correctly in `crates/dsl_compiler/`. Minor deviation: the crate split is `engine` / `engine_gpu` / `engine_rules` / `engine_data` rather than a single `crates/engine/` — the spec says "Rust library crate (`crates/engine/`)". This is implementation detail; the decomposition is more granular but compatible.

### §2 Determinism contract
- Status: ⚠️ Partial
- Evidence:
  - Serial determinism: `crates/engine/tests/proptest_event_hash.rs`, `crates/engine/tests/acceptance_plan2_deterministic.rs`, `crates/engine/tests/wolves_and_humans_parity.rs` — all test Serial-only determinism.
  - Cross-backend hash parity: no test file. The spec mandates `crates/engine/tests/parity_backends.rs` — **does not exist**.
  - `replayable_sha256()`: implemented at `crates/engine/src/event/ring.rs:170`.
  - Fisher-Yates shuffle: implemented in `crates/engine_rules/src/step.rs` (search `shuffle_actions_in_place`).
  - `HashMap` banned in hot paths: confirmed; `SpatialHash` uses `FxHashMap` (fixed-seeded rustc-hash), not `AHasher`.
  - Float reduction determinism: integer counts + sorted-key GPU events before hash — partially enforced. GPU event ring drain sorts by `(tick, kind, payload[0])` in `crates/engine_gpu/src/lib.rs`.
- Notes: Missing: the mandatory end-to-end cross-backend `replayable_sha256` parity test.

### §3 Runtime architecture
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/backend.rs` defines `ComputeBackend` trait. `crates/engine_rules/src/backend.rs` provides `SerialBackend`. `crates/engine_gpu/src/lib.rs` provides `GpuBackend`.
- Notes: The six-phase pipeline diagram is partially embodied in `engine_rules::step::step` (phases 1–6 present). However `step_full` (the full-signature entry-point with views/invariants/telemetry) is `unimplemented!()` in `crates/engine/src/step.rs`. GpuBackend does not fully match the spec's backend-agnostic API — it does not expose `agent_pos(id)`, `event_ring()`, `mask_buffer()` as described; callers cannot use it interchangeably with SerialBackend via `dyn ComputeBackend` because the trait is non-object-safe (generic over `B: PolicyBackend`).

### §4 State residency
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/state/` provides `SimState` with SoA hot fields (Vec<f32>, Vec<bool>, etc.) — this is the Serial residency model.
- Notes: GPU residency model as described (per-field dirty-tracked mirror, `FieldHandle`, `upload_dirty`, `download_events` per sync point) is **not implemented**. The actual GpuBackend uses `pack_agent_slots` / `unpack_agent_slots` to copy full CPU `SimState` into/out of GPU buffers each tick (`crates/engine_gpu/src/physics.rs`). There are no per-field dirty flags, no incremental upload, no explicit sync-point table. The "cold fields are host-only" distinction is also not present in this form — GPU kernels receive a flat `GpuAgentSlot` that contains a subset of fields.

### §5 Event log
- Status: ✅
- Evidence: `crates/engine/src/event/ring.rs` — `VecDeque<Entry<E>>` with capacity, `push`, `push_caused_by`, `replayable_sha256`, monotonic `EventId`. GPU event ring in `crates/engine_gpu/src/event_ring.rs` (GPU-resident ring with drain + sort). `cause: Option<EventId>` on `Entry`.
- Notes: Serial impl exactly matches spec. GPU ring uses `wgpu` buffers and atomic-append semantics; drain sorts by `(tick, kind, payload)` before pushing to host ring — satisfies the spec's sort requirement. Causal tree is host-side (`crates/engine/src/debug/causal_tree.rs`).

### §6 Spatial index
- Status: ⚠️ Partial
- Evidence:
  - Serial: `crates/engine/src/spatial.rs` — `SpatialHash` with `FxHashMap<(i32,i32), CellBucket>`, `within_radius`, cell-size 16 m. `CELL_SIZE = 16.0` matches spec.
  - GPU: `crates/engine_gpu/src/spatial_gpu.rs` — `GpuSpatialHash` with per-tick full rebuild, `within_radius`, `nearest_hostile_to`, `nearby_kin`.
- Notes: Three query primitives (`within_radius`, `nearest_k`, `in_column_z_range`) — `in_column_z_range` is absent from both the CPU and GPU spatial implementations. The spec §6 explicitly lists it as part of the query API. The GPU spatial hash is not yet wired into the fused scoring/physics kernels (noted as "follow-on" in `spatial_gpu.rs` header); instead scoring reads spatial candidates from a CPU-side pass.

### §7 RNG streams
- Status: 🤔 Spec mismatch
- Evidence: `crates/engine/src/rng.rs` — `per_agent_u32` / `per_agent_u64` use `ahash::RandomState::with_seeds(0xA5A5..., 0x5A5A..., 0xDEAD_BEEF..., 0x0123...)`. The four constants match the spec's `RNG_KEY_1..4` values.
- Notes: The spec says "PCG-XSH-RR algorithm on host and in shaders" with those constants as PCG stream keys. The actual implementation uses `ahash` (a hash function, not PCG) with those same constants as AHasher seeds. `WorldRng` is a real PCG-XSH-RR but is used independently; `per_agent_u32` does not use PCG at all. The cross-backend golden test (`crates/engine/tests/rng_cross_backend.rs`) does not exist. No `crates/engine/shaders/rng.glsl` exists; GPU uses WGSL. The spec's GLSL sample is inapplicable — this is a spec/impl terminology mismatch that should be resolved by updating the spec to reflect `ahash` for `per_agent_u32` and WGSL for shaders.

### §8 GPU runtime integration
- Status: ⚠️ Partial
- Evidence: `crates/engine_gpu/src/lib.rs` — `GpuBackend` exists, uses `wgpu::Device` / `wgpu::Queue` (not `VulkanContext` / `GpuHarness`). `GpuBackend::new()` returns `Result<Self, E>` (implicitly via `?` operators in construction). Feature gate `gpu = ["dep:wgpu", ...]` in `crates/engine_gpu/Cargo.toml`.
- Notes: The spec describes `voxel_engine::VulkanContext + GpuHarness` as the GPU layer. The actual implementation uses `wgpu` (a portable WebGPU abstraction) — `voxel_engine` is not a dependency of `engine_gpu`. `KernelCatalog` struct with named kernel fields does not exist; kernels are built as `wgpu::ShaderModule` from WGSL strings, not from pre-compiled SPIR-V. No `.spv` files; no `crates/engine/shaders/src/*.glsl`. The feature-flag structure (`default = []`, `gpu = ["dep:wgpu", ...]`) diverges from spec's `default = ["serial", "gpu"]` with separate `serial` / `gpu` features — `serial` feature does not exist.

### §9 Action space — MicroKind
- Status: ✅
- Evidence: `crates/engine/src/mask.rs` — `MicroKind` enum with exactly 18 variants, same ordinals as spec (Hold=0 through Remember=17). `#[repr(u8)]` present.
- Notes: MacroKind execution (§10) is wired; both have tests in `crates/engine/tests/`.

### §10 Macro mechanisms
- Status: ⚠️ Partial
- Evidence: `crates/engine_rules/src/step.rs` — `PostQuest`, `AcceptQuest`, `Bid`, `Announce` handled in apply-actions phase. `MacroKind` in `crates/engine/src/policy/macro_kind.rs` has `NoOp`, `PostQuest`, `AcceptQuest`, `Bid`, `Announce`.
- Notes: Announce uses `spatial.within_radius` which is XY-planar (the `SpatialHash` is a 2D column grid). The spec §10 states 3D Euclidean distance with a "Confirmed 2026-04-26" annotation. Actual `within_radius` in `crates/engine/src/spatial.rs` does a 3D distance check via `Vec3::distance` in its inner filter (`sp.distance(tp) <= range` at line 379 of step.rs — using full Vec3). Serial parity is satisfied. GPU announce (`apply_announce` kernel) is **not implemented** — no kernel exists for it; the GPU backend does not handle `MacroKind::Announce` in its step dispatch.

### §11 Physics cascade runtime
- Status: ⚠️ Partial
- Evidence:
  - Serial: `crates/engine/src/cascade/dispatch.rs` — `CascadeRegistry`, `run_fixed_point`, `MAX_CASCADE_ITERATIONS = 8`, lane-ordered dispatch, compiler-emitted `KindDispatcher` support. Fully matches spec.
  - GPU: `crates/engine_gpu/src/cascade.rs` / `crates/engine_gpu/src/cascade_resident.rs` — GPU cascade exists with WGSL kernels emitted by `dsl_compiler::emit_physics_wgsl`.
- Notes: GPU cascade is present but the fixed-point loop described (host-orchestrated, counter-compare) is partially implemented — the resident path runs a bounded loop but the exact "no state download between iterations" contract is not explicitly verified by any test. Cross-entity walks (quest `eligible_acceptors` over GPU-resident `AggregatePool`) are not implemented; `Quest` uses `SmallVec` (not `[AgentId; N]` Pod array) and is not GPU-eligible.

### §12 Mask buffer
- Status: ✅
- Evidence: `crates/engine/src/mask.rs` — `MaskBuffer` with `Vec<bool>` per head, `reset`, `set`, `get`. `ComputeBackend::reset_mask`, `set_mask_bit`, `commit_mask` in backend trait. GPU fused mask kernel in `crates/engine_gpu/src/mask.rs`. GPU backend dispatches `FusedMaskKernel` for seven mask heads.
- Notes: `Mask` trait with methods like `mark_hold_allowed(&mut self, backend, state)` is not the actual API shape — the implementation uses `mask_fill::fill_all(backend, mask, target_mask, state)` (generated by dsl_compiler). The spec's `Mask` trait does not exist as written; the concept is there but the surface differs. The Cast mask head is explicitly skipped in the GPU path.

### §13 Policy backend
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/policy/mod.rs` — `PolicyBackend` trait exists. `UtilityBackend` implemented (both Serial and used by GPU via CPU-side evaluate). `crates/engine/src/policy/utility.rs` — scalar argmax over scoring table. GPU scoring kernel in `crates/engine_gpu/src/scoring.rs` for parallel argmax.
- Notes: `NeuralBackend` is not implemented (no type, no `todo!()` stub found). The `PolicyBackend` trait signature differs from spec: actual is `evaluate(&self, state, mask, target_mask, out: &mut Vec<Action>)` — takes no `ComputeBackend` or `ActionBuffer` parameter. The `ActionBuffer` as a FieldHandle for GPU is not implemented; GPU scoring outputs are a separate `Vec<ScoreOutput>` readback, not a shared `ActionBuffer`. Cross-backend parity for policy evaluation is partially covered by `parity_with_cpu.rs` scoring tests.

### §14 Tick pipeline
- Status: ⚠️ Partial
- Evidence: `crates/engine_rules/src/step.rs` — six phases present for Serial (mask build, policy eval, Fisher-Yates shuffle, apply+cascade, view fold, tick advance). Phase labels and logic match spec.
- Notes: `step_full` (the full-signature version with views/invariants/telemetry passed as slices) is `unimplemented!()` in `crates/engine/src/step.rs`. The actual `step` function in `engine_rules` takes `debug: &DebugConfig` for invariant/telemetry hookup, which is close but not identical to the spec's `step_full` signature. Phase 3 (shuffle) is host-side as specified. GPU phases 1–2 run GPU kernels; phases 3–6 still CPU-side. GpuBackend's `step` does not drive all six phases through GPU dispatch as implied by the spec's architecture diagram.

### §15 Views
- Status: ✅
- Evidence: `crates/engine/src/view/materialized.rs` — `MaterializedView` trait + `DamageTaken`. `crates/engine/src/view/lazy.rs` — `LazyView` trait + `NearestEnemyLazy`. `crates/engine/src/view/topk.rs` — `TopKView` trait + `MostHostileTopK`. Tests in `crates/engine/tests/view_*.rs`.
- Notes: GPU implementations of views exist via `crates/engine_gpu/src/view_storage.rs` (materialized), `view_storage_symmetric_pair.rs`, `view_storage_per_entity_ring.rs`. The spec's note about sorting events by per-tick sequence number for GPU materialized views is addressed by the GPU event ring drain sorting.

### §16 Aggregates
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/aggregate/pool.rs` — `AggregatePool<T>` with Pool + slots. `crates/engine/src/aggregate/quest.rs` — `Quest`. `crates/engine/src/aggregate/group.rs` — `Group`. Tests in `crates/engine/tests/aggregate_pool.rs`.
- Notes: The spec requires `Quest` and `Group` to be GPU-eligible (using `T: Pod` with `[AgentId; N]` fixed-size arrays instead of `SmallVec`). The actual `Quest` uses `SmallVec<[AgentId; 4]>` and `Group` uses `SmallVec<[AgentId; 8]>` — neither implements `Pod`. There is no GPU-eligible variant of `AggregatePool`. The "host-only vs GPU-eligible" two-variant design is absent.

### §17 Trajectory emission
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/trajectory.rs` — `TrajectoryWriter` (safetensors output), `TrajectoryReader`. Test in `crates/engine/tests/trajectory_roundtrip.rs`.
- Notes: Serial implementation complete. GPU backend trajectory path: no evidence of a "download per-tick snapshot to `TrajectoryWriter`" path in `crates/engine_gpu/`. The cross-backend byte-identical safetensors test mentioned in the spec does not exist.

### §18 Save / load
- Status: ⚠️ Partial
- Evidence: `crates/engine/src/snapshot/format.rs` — `MAGIC`, `FORMAT_VERSION`, `HEADER_BYTES=64`, `SnapshotHeader` with magic+schema_hash+tick+seed+format_version. `crates/engine/src/snapshot/mod.rs` — save/load functions. Tests in `crates/engine/tests/snapshot_*.rs`.
- Notes: Header does not include `kernel_catalog_hash` field — the spec's header block lists it but the implementation has only magic+schema_hash+tick+seed+format_version+reserved. GPU backend save/load (`upload_from_mirror`) is not present; `crates/engine_gpu/src/snapshot.rs` implements `GpuSnapshot` (an observation snapshot for the observer API) but not the save/load contract described in §18. Several fields intentionally not snapshotted (noted in `format.rs` comments): `cold_channels`, `EventRing` entries, `views`, `ability_registry`, `terrain`, `config`.

### §19 Invariant runtime
- Status: ✅
- Evidence: `crates/engine/src/invariant/trait_.rs` — `Invariant` trait. `crates/engine/src/invariant/builtins.rs` — `MaskValidityInvariant`, `PoolNonOverlapInvariant`. `crates/engine/src/invariant/registry.rs` — `InvariantRegistry::check_all`. Tests in `crates/engine/tests/mask_validity.rs`.
- Notes: `requires_state` method is not on the `Invariant` trait (spec §19 says it should be). The trait has only `name`, `failure_mode`, `check`. The `event_hash_stable` dev-only invariant is not implemented as a builtin (though the hash infrastructure exists). GPU sync-before-invariant is not explicitly wired since `step_full` is unimplemented.

### §20 Probe harness
- Status: ❌ Not implemented (effectively)
- Evidence: `crates/engine/src/probe/mod.rs` — `Probe` struct and `run_probe` function exist, but `run_probe` is `unimplemented!()` ("Plan B1' Task 11"). Tests that use it (`probe_determinism.rs`, `probe_harness.rs`) are `#[ignore]`d.
- Notes: The probe infrastructure is present as a skeleton but not functional. The `#[probe(backend = "serial")]` attribute mentioned in the spec for pinning backend does not exist.

### §21 Telemetry sink
- Status: ✅
- Evidence: `crates/engine/src/telemetry/sink.rs` — `TelemetrySink` trait (`emit`, `emit_histogram`, `emit_counter`). `crates/engine/src/telemetry/metrics.rs` — `TICK_MS`, `EVENT_COUNT`, `AGENT_ALIVE`, `CASCADE_ITERATIONS`, `MASK_TRUE_FRAC`. `NullSink`, `VecSink`, `FileSink` in `crates/engine/src/telemetry/sinks.rs`.
- Notes: GPU-specific metrics (`engine.gpu.upload_ms`, `engine.gpu.download_ms`, `engine.gpu.kernel_ms`) are **not defined** in `metrics.rs` and not emitted by the GpuBackend. The backend exposes `PhaseTimings` (microsecond fields) for internal use but does not push them through the `TelemetrySink` API. The spec requires these under the `engine.gpu.*` prefix.

### §22 Schema hash
- Status: 🤔 Spec mismatch
- Evidence: `crates/engine/src/schema_hash.rs` — `schema_hash()` covers SoA layout, all enum variants, event kinds, lane ordinals, `MAX_CASCADE_ITERATIONS`, RNG constants (`0xA5A5…`, `0xDEAD_BEEF…` etc.), builtin metrics/invariants, snapshot format version, feature sources.
- Notes: The spec requires including `kernel_catalog_hash` (per-kernel SPIR-V content hash) in the schema hash. Because there are no pre-compiled SPIR-V files and no `KernelCatalog` struct, this input is absent. The schema hash is otherwise thorough and exceeds the spec's minimum. The RNG constant strings in `schema_hash.rs` match the `per_agent_u64` ahash seed values — consistent internally, but the spec frames them as PCG constants.

### §23 Observation packing
- Status: ✅
- Evidence: `crates/engine/src/obs/packer.rs` — `ObsPacker` + `FeatureSource` trait. `crates/engine/src/obs/sources.rs` — vitals, position, neighbor feature sources. Tests in `crates/engine/tests/obs_*.rs`.
- Notes: GPU-parallel kernel path for observation packing described in the spec is not present. The `ObsPacker` is CPU-only; no "GPU obs_field" or SPIR-V pack kernel. This matches the "Serial complete" status noted in `status.md`.

### §24 Debug & trace runtime
- Status: ✅
- Evidence: `crates/engine/src/debug/` — `trace_mask.rs`, `causal_tree.rs`, `tick_stepper.rs`, `tick_profile.rs`, `agent_history.rs`, `repro_bundle.rs`. `DebugConfig` struct. Tests in `crates/engine/tests/debug_*.rs`.
- Notes: All six debug components are present and tested. GPU-specific path (per-phase `debug_readback` toggle, kernel-scoped timing) is not implemented — `tick_profile` only records phase names, not per-kernel GPU timestamps. This is consistent with GPU still being a partially-wired backend.

### §25 Backend selection + fallback policy
- Status: ❌ Not implemented
- Evidence: No `new_backend()`, `new_backend_auto()`, or `BackendKind` enum found anywhere in `crates/`.
- Notes: Callers construct backends directly: `SerialBackend` (via `engine_rules::SerialBackend`) or `GpuBackend::new()`. The spec's unified entry-point with automatic fallback and `GpuInitError` → `SerialBackend` path does not exist. Feature flags also differ: no `serial` / `gpu` features on `crates/engine`; engine_gpu has `default = []` / `gpu = [...]` but no `serial` feature; the spec's `default = ["serial", "gpu"]` pattern is absent.

### §26 What's NOT in the engine
- Status: ✅ (non-implementation section)
- Evidence: DSL parser in `crates/dsl_compiler/`. Cascade rules emitted by compiler into `crates/engine_rules/`. Chronicle templates in game/world-sim layer. Curriculum in `scripts/`.
- Notes: Boundaries are respected as described.

---

## Cross-cutting observations

1. **GLSL vs WGSL.** The spec consistently refers to GLSL source files (`*.glsl`) and SPIR-V bytecode (`.spv`). The entire GPU stack uses WGSL (via `naga` + `wgpu`) compiled at runtime. Every spec reference to `rng.glsl`, `include_bytes!("...spv")`, `#define RNG_KEY_1 ...` should be updated to WGSL equivalents.

2. **`voxel_engine` vs `wgpu`.** The spec says the GPU backend depends on `voxel_engine::VulkanContext + GpuHarness`. The actual `crates/engine_gpu` does not depend on `voxel_engine`; it uses `wgpu` directly. The spec's §8 struct definition referencing `VulkanContext` and `GpuHarness` is stale.

3. **Plan B1' Task 11 churn.** A significant number of "implemented" items in `crates/engine/src/step.rs` are stub `unimplemented!()` functions left for test compilation. The real step logic is in `crates/engine_rules/src/step.rs` (generated). The spec does not acknowledge this crate split; `step_full` in particular is described as the canonical entry point but only exists as a panic stub.

4. **SerialBackend lives in `engine_rules`, not `engine`.** The spec says `crates/engine/src/backend/serial.rs`. Actual: `crates/engine_rules/src/backend.rs` (generated by dsl_compiler). This is the correct architecture for rules-as-data but diverges from the spec's file map.

5. **GPU backend is sync-path hybrid, not GPU-authoritative.** Throughout §3–§4 the spec describes GPU as the authoritative owner of hot fields with a dirty-mirror mechanism. The actual implementation packs full `SimState` into GPU buffers at the top of each `step` call and unpacks the result back — a synchronous copy, not a lazy dirty-tracked mirror. The distinction matters for performance and for the described sync-point semantics.
