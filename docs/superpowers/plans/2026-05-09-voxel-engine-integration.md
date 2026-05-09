# Voxel-Engine Integration — Bridge `~/Projects/voxel_engine` to the Sim Engine

> **Status: COMPLETE (2026-05-09)** — All 5 phases shipped. PR train: #69 (Phase A) → #70 (Phase B) → #71 (Phase C) → #72 (Phase D) → this PR (Phase E). After this slice `place_voxel` / `harvest` / terrain-aware queries do real work on both CPU and GPU; one production fixture (`wave_defense_runtime`) opts in via settler `BuildPalisade` casts; per-tick mirror upload cost baselined at 5.83 µs mean / 325 µs max in `docs/perf/2026-05-09-stress-ceilings.md`.

> Goal: connect the Vulkan-based `voxel_engine` repo to this sim engine via the existing `TerrainQuery` seam + chronicle event consumers + a **GPU-resident voxel mirror** that sim kernels can read directly. After this slice `place_voxel` / `harvest` / terrain-aware queries do real work, AND the GPU dispatcher can ask terrain questions without a CPU round-trip.

## Goal

Today the sim engine has a 3-method `TerrainQuery` trait (`crates/engine/src/terrain.rs`) with one impl: `FlatPlane` returning yes-to-everything. The DSL surface for `place_voxel <kind_hash>` and `harvest <kind_hash> <amount>` lowers cleanly through the compiler and emits chronicle events `EffectPlaceVoxelApplied=60` / `EffectHarvestApplied=59`, but no runtime consumes them. Meanwhile `~/Projects/voxel_engine/` has a full Vulkan voxel world (chunks, spatial hash, GPU compute) that nothing in this project links to.

**Today's terrain access pattern (verified):** ONE caller — `crates/engine/src/policy/utility.rs:358` invokes `state.terrain.line_of_sight(from, to)` from the CPU policy path. Zero WGSL emit sites lower terrain queries today; the DSL surface in `dsl_ast/src/resolve.rs` documents the methods but no kernel uses them. So a CPU-only `Arc<dyn TerrainQuery>` adapter would compile + work today.

**Why a CPU-only adapter is the wrong stopping point:** the moment a sim kernel needs terrain (settlers checking line-of-sight to monsters during scoring; agents avoiding solid voxels in pathing), CPU-side trait dispatch becomes a per-tick × per-agent CPU↔GPU round-trip. At 200k agents × 10 Hz that's 2M trait-object calls/sec across the bus. Catastrophic. The slice has to land GPU-side voxel access in the same plan, not as an unbounded "future work."

**After this slice:**
- `Forager.ability` actually places palisades and harvests berries.
- `TerrainQuery::height_at(x, y)` returns the highest occupied voxel in column (x, y) on CPU.
- `walkable(pos, Walk)` returns `false` inside solid rock on CPU.
- A wgpu storage-buffer mirror of the voxel grid is bound into kernels as a `KernelBindingsContext` source; WGSL emit lowers `terrain.line_of_sight(from, to)` to a function call that reads the buffer.
- 4 behavioral pins (semantic, not threshold) catch the "FlatPlane silently passes" probe-fooling pattern. A 5th catches the CPU/GPU mirror divergence pattern.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/terrain.rs` — `TerrainQuery` trait + `FlatPlane` default impl + `Arc<dyn TerrainQuery>` field on `SimState` (line 234).
  - `crates/engine/src/state/mod.rs:234` — `pub terrain: Arc<dyn TerrainQuery + Send + Sync>` already plumbed.
  - `~/Projects/voxel_engine/src/{lib,world/{chunks,spatial,mod},terrain_compute}.rs` — concrete voxel world. Vulkan-based (ash/gpu-allocator deps); separate workspace.
  - `crates/engine/src/ability/program.rs::EffectOp::PlaceVoxel` (variant 26) + `EffectOp::Harvest` (variant 25) — IR variants exist + dispatcher emits chronicle records.
  - `EventKindId::EffectPlaceVoxelApplied = 60` + `EffectHarvestApplied = 59` — event ordinals reserved.
  - `assets/ability_test/dsl_coverage/Forager.ability` — already-authored fixture exercising both verbs (ForageBerries, QuarryStrike, RaisePalisade, StockpileRun).
  - `docs/superpowers/notes/2026-04-22-terrain-integration-gap.md` — the design doc that recommended Option B (separate adapter crate).

  Search method: `rg` + `Read` + `find`.

- **Decision:** new `crates/engine_voxel/` adapter crate. Wraps `~/Projects/voxel_engine` (added as a path dependency in workspace Cargo.toml). Implements `TerrainQuery` for a `VoxelTerrain` struct that owns a CPU-side voxel world (uses voxel_engine's `world::*` modules; deliberately skips `vulkan::*` to avoid pulling Vulkan deps into the sim engine's wgpu world). Per-fixture runtimes that want voxel terrain construct `Arc::new(VoxelTerrain::new())` and pass it to `SimState::with_terrain(...)` instead of accepting the `FlatPlane` default.

  The adapter crate owns three responsibilities:
  1. **CPU-side voxel world** — for the legacy CPU policy callers (line_of_sight at policy/utility.rs:358).
  2. **Chronicle event consumer** — `apply_voxel_chronicle_record(&mut self, rec: &[u32; STRIDE])` drains `EffectPlaceVoxelApplied` / `EffectHarvestApplied` into voxel mutations.
  3. **GPU-resident mirror** — a `wgpu::Buffer` that holds a flat encoding of the voxel grid (chunked or hashmap-of-chunks). Updated on every chronicle drain. Exposed as a `&'a wgpu::Buffer` field on `KernelBindingsContext` so compiler-emitted `from_context()` constructors pull it in just like `agent_pos_buf` etc.

  **Why a new crate, not in-tree:** voxel_engine pulls Vulkan deps (`ash`, `gpu-allocator`, `shaderc` build dep) when its `vulkan::*` modules are used. We use only `world::*` here; even so, the adapter crate isolates the dep tree so engine consumers without voxel terrain don't pay the voxel_engine compile cost.

  **Why CPU-mirror-then-upload, not GPU↔GPU interop:** voxel_engine and sim engine are both GPU-resident, but in different APIs (Vulkan via ash vs. wgpu). True interop options exist (shared `VkDevice` via `wgpu_hal::vulkan::Adapter::from_raw_device`, external memory FDs via `VK_KHR_external_memory_fd`, or flipping to a single-engine architecture where voxel_engine owns the device). All of them are bigger surface than the slice can absorb, and the per-mutation upload cost of the CPU mirror is bounded by chronicle event rate (PlaceVoxel + Harvest fire only on cast — single-digit kHz at most for the fixtures we're planning, vs. millions of GPU reads per tick). Pick the simpler architecture; revisit when measurement says it's the bottleneck.

- **Rule-compiler touchpoints:**
  - DSL inputs added: a new fixture `assets/sim/voxel_probe.sim` (small) + `assets/ability_test/voxel_probe/{PlaceTestVoxel,HarvestTestVoxel}.ability`. Existing `Forager.ability` stays as a corpus fixture; the probe fixture is the behavioral-pin testbed.
  - Generated outputs emitted: per-runtime build.rs auto-regen for the new probe runtime; no other regen needed.
  - **No compiler changes.** The chronicle events + WGSL emit already exist (Slice γ tail, May 7).

- **Hand-written downstream code:**
  - `crates/engine_voxel/src/lib.rs` — `VoxelTerrain` struct + `TerrainQuery` impl + `apply_voxel_chronicle_record()` consumer. Justified: the engine deliberately doesn't depend on voxel_engine; this adapter is the seam. NOT compiler-emittable (it's a host-side adapter, not a kernel).
  - `crates/voxel_probe_runtime/src/lib.rs` — fixture runtime, follows the existing per-runtime pattern. Per-tick: dispatch the GPU kernels (compiler-emitted), then drain chronicle records into the voxel terrain via the adapter.

- **Constitution check:**
  - P1 (Compiler-First): PASS — chronicle event production is compiler-emitted; the consumer is a *host-side adapter*, not a hand-written `Rule`. The trait-object `Arc<dyn TerrainQuery>` pattern is already established (FlatPlane is a hand-written impl too — adapter is the same shape, just with a real backend).
  - P2 (Schema-Hash): PASS — voxel state lives in the adapter crate, NOT in `SimState` SoA. The `Arc<dyn TerrainQuery>` field already exists on `SimState`; we're just supplying a different impl. Schema unchanged.
  - P3 (Cross-Backend Parity): PASS — `TerrainQuery` is CPU-side; both `SerialBackend` and `GpuBackend` consult the same trait through the same Arc. Parity by construction.
  - P4 (`EffectOp` Size Budget): N/A — no new variants.
  - P5 (Determinism via Keyed PCG): PASS — voxel mutations are pure functions of chronicle records (which are themselves deterministic). Adapter must use deterministic data structures (BTreeMap, or sort keys before iterating Vec). voxel_engine's chunks may use HashMap — verify and either replace or sort-then-fold.
  - P6 (Events Are the Mutation Channel): PASS — voxel state mutates *only* in response to drained chronicle records. No direct `&mut SimState` writes.
  - P7 (Replayability Flagged): N/A — events already declared `@replayable`.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — phase tasks each close with a commit.
  - P10 (No Runtime Panic): PASS — `VoxelTerrain::new()` returns `Result`; queries return defaults on out-of-bounds (e.g. `height_at` on unloaded chunk returns 0).
  - P11 (Reduction Determinism): PASS — chronicle record drain order is determined by ring slot order (deterministic by tick); voxel writes are commutative within a tick (last write per cell wins, but writes to *different* cells commute trivially). If a tick produces multiple writes to the *same* cell, sort by source AgentId then apply — same convention as the AOE bitonic sort (PR #39).

- **Runtime gate:** the load-bearing test against probe-fooling. Three behavioral pins, each catching a class of FlatPlane-passes-trivially failure:
  - `placed_voxel_changes_height_at` — start with empty terrain, fire `place_voxel` for a cell at (5, 5, 5), assert subsequent `terrain.height_at(5, 5)` returns ≥5 (FlatPlane returns 0 → fails).
  - `solid_voxel_blocks_walkable` — place voxel at (10, 10, 10), assert `terrain.walkable(Vec3(10, 10, 10), Walk)` returns false (FlatPlane returns true → fails).
  - `voxel_blocks_line_of_sight` — place voxel at (5, 5, 5), assert `terrain.line_of_sight(Vec3(0,0,5), Vec3(10,10,5))` returns false because the segment crosses the placed cell (FlatPlane returns true → fails).

  Plus a determinism pin: `same_seed_same_voxel_world` — run the probe runtime twice with the same seed, assert the final voxel world hashes identically.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design.

---

## Phasing — 5 PRs

The slice is too large for one PR. Each phase ships a viable end state. Phases A+B prove the CPU side; phase C is the perf-fix that makes the slice viable for real fixtures; phase D wires GPU emit; phase E proves it on a production fixture.

### Phase A — Adapter crate skeleton + CPU-side `VoxelTerrain` (~400 LOC)

Goal: prove the wiring without changing any behavior. CPU-only; GPU mirror comes in phase C.

- New `crates/engine_voxel/{Cargo.toml, src/lib.rs}`. Add to workspace members. `voxel_engine = { path = "/home/ricky/Projects/voxel_engine" }` as path dep, restricted to `world::*` modules (no `vulkan::*` — verify Cargo features can isolate this; if not, document the dep cost).
- `pub struct VoxelTerrain { world: voxel_engine::world::VoxelWorld /* or equivalent */ }`. Implement `TerrainQuery` for it: `height_at` walks the column at (x, y); `walkable(pos, mode)` checks the cell at `floor(pos)`; `line_of_sight(from, to)` raycasts via voxel_engine's terrain_compute or a CPU port.
- Add `pub fn apply_voxel_chronicle_record(&mut self, rec: &[u32; STRIDE]) -> Option<()>` returning `None` (no-op) — to be filled in phase B.
- **Determinism audit**: voxel_engine's chunk storage may use HashMap. Verify in this phase. If yes, replace with `BTreeMap<ChunkCoord, Chunk>` at the adapter boundary or sort keys before any iteration. P5 / P11 violation risk if missed.
- Tests: `VoxelTerrain::new()` constructs cleanly; the 3 trait methods on an empty world produce sensible defaults (height 0, walkable true, LOS true — same as FlatPlane initially).

**Behavioral pin:** existing fixtures still pass (no runtime opts in yet). Schema unchanged.

### Phase B — Voxel probe runtime + chronicle event consumer (~700 LOC)

Goal: PlaceVoxel + Harvest events actually mutate the voxel world. CPU-side only; GPU access still pending.

- New `crates/voxel_probe_runtime/{Cargo.toml, build.rs, src/lib.rs}` mirroring `debug_probe_runtime`'s shape. Constructs `Arc::new(VoxelTerrain::new())` and passes to `SimState::with_terrain(...)`. After each tick, drains chronicle records via `apply_voxel_chronicle_record()` (drain ordering deterministic per the chronicle ring's tick-relative slot order).
- New fixtures `assets/sim/voxel_probe.sim` + `assets/ability_test/voxel_probe/{PlaceTestVoxel,HarvestTestVoxel}.ability`. One agent fires `place_voxel` then `harvest` over a 3-tick sequence.
- Adapter fills in `apply_voxel_chronicle_record` for both kind=60 (`EffectPlaceVoxelApplied`) and kind=59 (`EffectHarvestApplied`).
- **Two semantic behavioral pins** (the FlatPlane-killers):
  - `placed_voxel_changes_height_at` — fire `place_voxel` for cell (5, 5, 5); subsequent `terrain.height_at(5, 5)` ≥ 5. FlatPlane returns 0 → fails. Catches "events fire into a void."
  - `harvested_voxel_disappears` — place voxel, harvest it, assert `terrain.height_at` returns to ground. Catches "harvest emits an event but doesn't actually remove anything."

### Phase C — GPU-resident voxel mirror (the perf fix) (~700 LOC)

Goal: voxel state lives in a wgpu storage buffer, available to GPU kernels via the existing `KernelBindingsContext`. **This phase is what makes the slice viable for real fixtures.**

- Adapter holds a `VoxelMirror` struct: a `wgpu::Buffer` whose layout encodes the voxel grid (flat 3D array for small worlds; chunked for larger). Plus per-chunk dirty flags so per-tick uploads only touch changed chunks.
- On `apply_voxel_chronicle_record`: write to BOTH the CPU-side `VoxelWorld` AND the matching cell in the GPU mirror's host-side staging. At end of drain, single `Queue::write_buffer` flushes the dirty chunks to GPU.
- Extend `KernelBindingsContext<'a>` with an optional `pub voxel_grid: Option<&'a wgpu::Buffer>` field (analogous to the optional `debug` field). Compiler's classifier table (from the Bindings constructors plan) gets a new entry: `voxel_grid` field name → `ctx.voxel_grid.expect(...)`.
- WGSL helper functions for terrain queries that consume the buffer: `voxel_at(grid: ptr<storage, ...>, x: i32, y: i32, z: i32) -> u32`. Lives in the WGSL preamble emit.
- **Behavioral pin (catches mirror divergence)**: `cpu_gpu_voxel_state_matches` — after a sequence of place/harvest, read N random cells via the GPU mirror (compute kernel that writes the cell value to a readback buffer) and assert each equals the CPU-side `VoxelWorld.cell_at(...)`. Catches "host wrote to CPU but forgot to write to GPU."

### Phase D — DSL → WGSL emit for terrain queries (~600 LOC)

Goal: a `.sim` rule can write `if (terrain.line_of_sight(self.pos, target.pos)) { ... }` and the compiler lowers it to a WGSL function call against the voxel mirror.

- Today the DSL surface for `terrain.line_of_sight(from, to)` exists in `dsl_ast/src/resolve.rs` but no WGSL lowering. Implement the lowering in `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`: the resolved IR call becomes a WGSL `voxel_line_of_sight(&voxel_grid, from, to)` invocation. Same for `terrain.height_at(x, y)` and `terrain.walkable(pos, mode)`.
- Helper functions emitted into the WGSL preamble for any kernel whose body references them.
- Two semantic behavioral pins:
  - `solid_voxel_blocks_walkable` — place voxel at (10, 10, 10), assert `terrain.walkable(Vec3(10, 10, 10), Walk)` returns false (FlatPlane returns true → fails). Test the CPU side.
  - `voxel_blocks_line_of_sight` — place voxel at (5, 5, 5), assert `terrain.line_of_sight(Vec3(0,0,5), Vec3(10,10,5))` returns false. Test the CPU side.
  - `gpu_terrain_query_matches_cpu` — same scenario via a tiny GPU compute kernel that calls the new WGSL helper; assert it returns the same answer as the CPU oracle. Catches CPU/GPU semantic divergence.

### Phase E — Production fixture opts in + determinism (~400 LOC)

Goal: prove the slice doesn't regress determinism + one real fixture uses voxel terrain.

- Determinism pin: `same_seed_same_voxel_world` — probe runtime, two runs, hash the voxel world's flat byte array, assert equal.
- Opt `wave_defense_runtime` into VoxelTerrain. Use case: monsters can't pathfind through walls, settlers can place defensive walls (the deferred wave_defense polish item lands here). The wave defense `same_seed_same_death_tick` pin must still hold.
- Update `docs/perf/2026-05-09-stress-ceilings.md` with measured per-tick voxel mirror upload cost at the wave_defense fixture's typical place/harvest rate. Establishes a baseline so future PlaceVoxel-heavy fixtures know what they're paying.

---

## Cross-cutting risks

1. **Per-mutation GPU upload cost (Phase C's load-bearing question).** Each chronicle drain does a `Queue::write_buffer` for every dirty chunk. At `wave_defense` cast rates (a handful of place_voxel events per tick from settlers) this is trivial. At a hypothetical "fire-spreads-every-tick" rate (thousands of voxels mutating per tick) it could become a bottleneck. Phase E records the actual cost; if it exceeds 1 ms/tick at the production-fixture rate, that's the trigger to revisit GPU↔GPU interop.

2. **HashMap ordering = nondeterminism.** voxel_engine's chunks may iterate by HashMap. Verify in Phase A; replace with `BTreeMap<ChunkCoord, Chunk>` or sort keys at boundary. P5 violation risk if missed. The `same_seed_same_voxel_world` pin in Phase E catches it.

3. **Coordinate system mismatch.** Sim engine uses `Vec3` in arbitrary R³. voxel_engine uses chunk-relative `IVec3` per cell. Adapter handles the conversion (floor for placement, exact for queries). Check that voxel_engine's chunks have consistent cell-size + chunk-extent constants. Document the chosen world-units-per-cell at the adapter boundary so DSL authors know what `place_voxel` granularity actually means.

4. **CPU/GPU mirror divergence.** Phase C's mirror-as-buffer pattern has the same drift risk as any cache-coherence problem: a mutation might land on CPU but not GPU (or vice versa). The `cpu_gpu_voxel_state_matches` pin (Phase C) and `gpu_terrain_query_matches_cpu` pin (Phase D) directly attack this. Don't substitute "both readbacks return non-zero" for "both readbacks return the SAME value at the same cell."

5. **Behavioral pin coverage.** Per the prior probe-fooling discussion: the 5 behavioral pins listed (placed_voxel_changes_height_at, harvested_voxel_disappears, solid_voxel_blocks_walkable, voxel_blocks_line_of_sight, same_seed_same_voxel_world) are deliberately *semantic*, not threshold-based. Each catches a class of "FlatPlane silently passes" failure. Plus the two divergence pins (cpu_gpu_voxel_state_matches, gpu_terrain_query_matches_cpu) catch the new mirror-shape failures Phase C introduces. **Don't substitute counter-based pins.**

6. **Voxel_engine's API stability.** It's a separate repo at edition=2024 with its own evolution path. Pinning to a specific commit (or vendoring a snapshot into a workspace member) trades update friction for stability. Phase A picks one; document the choice.

## Out of scope (deferred)

- **True GPU↔GPU interop (zero-copy voxel access).** Three real architectures exist beyond the CPU-mirror approach this slice ships:
  1. **Shared `VkDevice`** — `wgpu_hal::vulkan::Adapter::from_raw_device(...)` wraps the same device voxel_engine creates; both engines reference each other's `VkBuffer` handles directly. Raw-hal API surface (unstable); manual Vulkan semaphores for cross-API sync.
  2. **External memory FDs** — `VK_KHR_external_memory_fd` lets voxel_engine export buffers as Linux FDs that wgpu (via raw-hal) imports. Linux-only; manual FD lifetime; cross-device sync.
  3. **Flip the relationship** — voxel_engine owns the Vulkan device; sim engine becomes ash-native and runs as compute passes scheduled into voxel_engine's frame loop. Massive port; biggest perf ceiling.

  Defer until the CPU-mirror's per-mutation upload cost is the measured bottleneck. The phase E findings doc will record the actual cost so we can decide.
- **Voxel rendering / visualization.** voxel_engine has full Vulkan rendering; we don't expose it here. A future viz slice (separate crate, like the deleted wolf-sim viz was) can connect.
- **Destructible terrain damage propagation.** PlaceVoxel + Harvest cover the basic mutate-by-cast surface; richer effects (explosion AOE breaks all voxels in radius, fire spreads through solid material) are per-fixture follow-ups.
- **Cover bonuses on attack scoring** (mentioned in the original gap doc as a "future feature hangs off this seam"). Not in scope here, but Phase D's WGSL emit work means it's a small follow-up — scoring can call `terrain.line_of_sight` directly.
- **Migrating Forager.ability from `dsl_coverage` to a real fixture.** It stays in dsl_coverage as a parser-corpus fixture; voxel_probe runs the new minimal probe fixture for behavioral pins.
- **Voxel-aware pathfinding for monsters** in wave_defense. Phase E opts the fixture in but only uses voxels for static walls + `walkable` queries. Real navmesh-style pathfinding is a separate slice.

## Why this slice now

Four reasons:

1. **The seam exists** — TerrainQuery + Arc<dyn> + FlatPlane default. The adapter crate is exactly the planned next step (per the design doc from April 22). No re-architecting.
2. **The voxel_engine repo is real and self-contained** — not a stub. Bridging is mechanical, not exploratory.
3. **The probe-fooling concern is concrete here.** Every existing TerrainQuery test passes against FlatPlane. Voxel ops emit chronicle events into a void. The 7 semantic behavioral pins designed above (5 across phases A/B/D/E + 2 divergence pins in C/D) directly attack the false-pass pattern. Shipping this slice without the pins would just deepen the same bug class; with the pins, the slice models the right way to extend the engine.
4. **GPU access lands in the same plan, not as future work.** The original draft of this plan stopped at CPU-only adapter; the user flagged it would perform terribly at scale (per-tick × per-agent CPU↔GPU round-trip is unworkable). Phase C is the perf-fix that makes the slice viable for real fixtures. Without Phase C the slice would be performative — looks integrated, dies under load.

This is also a natural place to demo the **invariant-based probe pattern** from the meta-discussion — every behavioral pin is an algebraic property (place→height changes, harvest→cell empty, solid→walkable false, voxel→LOS blocked, CPU mirror == GPU mirror, GPU query == CPU oracle, same seed → same world) rather than a counter threshold.
