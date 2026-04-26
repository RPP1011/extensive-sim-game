# Voxel Engine Architecture: Regions, Indices, and the Epistemic Split

**Status:** Design draft for review
**Date:** 2026-04-25
**Scope:** voxel_engine architecture + game/engine query API contract
**Companion:** `superpowers/notes/2026-04-22-terrain-integration-gap.md`

## 1. Problem statement

The game (`/home/ricky/Projects/game`) is a deterministic, tick-based agent simulation targeting 20k–200k agents on commodity desktop. Per `game/overview.md` and `terrain-integration-gap.md`, the simulation currently runs in featureless R³: agents have no terrain awareness, no line-of-sight, no walkability, no destructible structure. The voxel_engine repository (`/home/ricky/Projects/voxel_engine`) ships the substrate (SDF terrain, chunk pool, voxel-bodied entities, fluid, navmesh, raycasting) but the game's `engine` crate doesn't consume it. Closing this gap requires a coherent architecture for:

- **Storage** of voxel state at scale: 5k–20k loaded chunks at 0.2 m voxel scale, supporting frequent localized modification.
- **Physics queries** against that state: 200k agents producing point reads, ray queries, region scans, and writes per tick.
- **Rendering** of that state, async after the simulation tick, decoupled from sim cadence.
- **Epistemic discipline:** agents aren't omniscient. Decision-time queries must use agent beliefs; only execution-time queries hit ground truth.

Aokana (arXiv 2505.02017) was investigated as a reference architecture. It assumes a static world preprocessed offline; it has no story for spiky local modifications and treats the world as one large preprocessed asset rather than many smaller dynamic ones. Its specific contributions (multiple shallow SVDAGs, GPU-driven rendering pipeline, LOD streaming) remain useful as *components* of certain index types but cannot be the spine of the architecture.

### 1.1 What today doesn't work

Per `terrain-integration-gap.md`: the game's `engine` crate (combat sim, scoring, mask, cascade) contains zero references to `voxel_engine`. Agent positions are `Vec3`s in unbounded R³. The wolves+humans canonical fixture, the parity anchor, all four example scenes, and the entire DSL run "in the void." This blocks 13 mechanics enumerated in that note: high-ground bonuses, chokepoints, cover, flanking, real flight advantage, terrain-speed modifiers, destructible terrain, placement/structures, spatial memory, vision/scouting/fog of war, ambushes, environmental hazards, and pathfinding. The minimal first slice — a `TerrainQuery` trait, a `FlatPlane` default impl, one new mask primitive (`line_of_sight`), and one scoring predicate (height bonus + LOS) — unblocks the entire chain. This spec generalises that minimal slice into a coherent execution-time API surface (§4.2) and the index/registry machinery (§6–§10) that lets follow-on mechanics land as ≤500-LoC additions per item.

## 2. Constraints

| Dimension | Value |
|---|---|
| Voxel scale | 0.2 m per voxel (post-rescale for construction) |
| Chunk dimensions | 64³ voxels (12.8 m physical) |
| Loaded chunks at peak | 5k–20k unique (heavily clustered agents, 3×3 chunk neighborhood per agent) |
| Sim tick rate | 30 tick/s @ 20k agents → 2 tick/s @ 200k agents |
| Renderer cadence | Independent of sim; async snapshot consumer |
| Sim determinism | Bit-equivalent across CPU and GPU backends |
| Render determinism | Not required |
| Modification pattern | Spiky and local: 1–10³ voxels per modification; ~10⁵–5·10⁵ writes per tick global |
| Modification frequency | Sub-second: most ticks have writes in 1–2k of the loaded chunks |
| GPU memory target | 8 GB class |
| Physics query mix | Per-tick: ~10⁶ point reads, ~3·10⁶ raycast voxel-steps, ~10⁶ region-read voxels (pre-cache) |

### 2.1 Memory budget and per-voxel cost

The L3 hot tier stores **2 bytes per voxel**: 1 byte palette index + 1 byte for occupancy/material flags (allows uniform-block detection without auxiliary structures). Per chunk: 64³ × 2 = 512 KB. Plus the existing 3-level mip chain at ~36 KB total per chunk. Round to **~550 KB per L3 chunk** including mips.

| Loaded chunks | L3 footprint | Verdict on 8 GB target |
|---|---|---|
| 5,000 | 2.7 GB | Fits comfortably; leaves >5 GB for indices, render state, sim SoA, OS overhead |
| 10,000 | 5.4 GB | Tight but viable on 8 GB; indices and render state must compete |
| 15,000 | 8.2 GB | Exceeds 8 GB without L4 demotion |
| 20,000 | 10.9 GB | Requires L4 (compressed warm tier) or 16 GB+ card |

L4 is therefore mandatory above ~12k loaded chunks; v1 ships with L3 + L6 (regen-from-seed) only and accepts the 12k cap, with L4 staged for v2. Worldgen scratch (§5.4) is a transient 256 KB buffer per chunk-being-generated, freed at commit; it doesn't compound steady-state memory pressure since at most a handful of chunks are mid-generation simultaneously.

## 3. Architectural pillars

Four pillars carry the design. They are independent enough to evolve separately, and the contract between them is what this spec defines.

1. **Tiered storage hierarchy** with per-chunk write epoch as the universal cache key.
2. **Region registry** of game-owned, named volumes, with DSL-declared bounded indices over them.
3. **Engine primitive library** for index construction and consumer queries.
4. **Epistemic split** in the consumer API: decision-time queries use agent beliefs over events; execution-time queries use ground truth.

The single insight that ties them together: **most queries never reach voxel storage.** Region indices serve the bulk; per-chunk summaries serve a large fraction of what indices miss; cold storage serves the long tail. The voxel-storage layer's primary job is to be the regenerable source of truth from which indices are derived.

## 4. The two-phase epistemic split

This is the most important architectural rule because it determines which APIs the engine exposes at all.

### 4.1 Decision-time vs execution-time

| Phase | Used by | Allowed sources |
|---|---|---|
| **Decision-time** | `mask`, `scoring`, action selection inside DSL `verb`s | Agent's own materialized views (beliefs); shared faction views; immediate sense events from this tick |
| **Execution-time** | `physics` rules resolving chosen intents; movement resolution; renderer | Ground truth: storage tiers, indices, raw queries |

A wolf considering "should I hunt the deer 5 km east?" runs at decision-time. It may only consult its `known_targets` view. There is no engine API it can call to discover deer it hasn't observed. If `known_targets(self, _) sum == 0` for deer-class targets, the wolf cannot score `Hunt`; it must score `Scout` or `Wander` instead.

The same wolf, having decided `Travel(remembered_position)`, runs through a `physics` rule that resolves the actual movement. That rule calls `engine::resolve_movement` against the real navgrid. If the road is washed out, the wolf discovers this in execution and emits a `MovementBlocked` event back to its beliefs.

### 4.2 Engine API stratification

The engine exposes two disjoint API surfaces:

**Decision-time API (read-only, belief-mediated):**
- `view::*` lookups over the agent's own materialized views (existing mechanism)
- That's it. No engine spatial queries.

**Execution-time API (ground truth):**
- `engine::voxel_at(world_pos)` — index-fast-path, voxel-storage fallback
- `engine::walkable(world_pos, mode)` — navgrid fast-path, voxel fallback
- `engine::region_at(world_pos)` — point-in-region; returns covering regions
- `engine::raycast(origin, dir, max)` — used by physics rules and renderer
- `engine::resolve_movement(agent, intent)` — runs navgrid + collision; emits movement events
- `engine::can_observe(observer, target)` — sense gate (LOS + range + cover); used by emission code
- `engine::emit_to_observable_agents(template, source_pos, kind, filter)` — perception fan-out
- `engine::emit_via_propagation(template, source_pos, propagation_kind, intensity)` — sound/scent/rumor fan-out
- `engine::apply_voxel_modification(region, op)` — destruction/construction writes; bumps chunk epochs

There is no `engine::nearest_*`, no `engine::find_*`, no `engine::path(a, b)` exposed to scoring. These would let agents perform god-mode pulls.

The DSL compiler enforces this at compile time: any DSL expression appearing in a `scoring`, `mask`, or `verb`-selection body that resolves to an execution-time API call is rejected with a "decision-time code may not call ground-truth queries — use a `view` over emitted events" error. The validator that already rejects rule bodies that can't lift to GPU (`game/overview.md` §2) extends with this additional gate.

### 4.3 Where beliefs come from

Beliefs are materialized views over events the engine emits. The engine's job is event emission with epistemically appropriate gating; the DSL declares what folds into what view.

New events the engine emits to support spatial cognition:

| Event | Replayability | Emitted when | Carries | Lands in phase |
|---|---|---|---|---|
| `VoxelRegionEntered` | `@replayable` | Agent crosses into a region | observer, region | Phase 2 |
| `VoxelRegionLeft` | `@replayable` | Agent crosses out of a region | observer, region | Phase 2 |
| `MovementBlocked` | `@replayable` | `resolve_movement` fails to advance an agent | agent, attempted_pos, blocker_kind | Phase 1 |
| `Saw` | `@non_replayable` | Sense gate clears for a visible event source | observer, target, position, tick | Phase 3 |
| `HeardSound` | `@non_replayable` | Sound propagation reaches a listener above audible threshold | listener, direction, intensity, source_kind | Phase 3 |
| `SmelledScent` | `@non_replayable` | Scent propagation reaches an agent above detection threshold | observer, direction, scent_kind, freshness | Phase 3 |
| `FeltImpact` | `@non_replayable` | Physical contact (damage, push, vibration) | observer, source, magnitude, direction | Phase 3 |
| `VoxelRegionObservedAtDistance` | `@non_replayable` | Agent gains visual on a region's exterior without entering | observer, region, classification | Phase 3 |

These events feed game-side belief views that are already expressible in the existing DSL view syntax (`@materialized`, `@decay`, `topk`).

**Replayability rationale:**
- `@replayable` events are state-bearing — they affect what happens next deterministically and feed `replayable_sha256`. `VoxelRegionEntered/Left` and `MovementBlocked` shape future behavior (region membership drives index selection; blocking re-routes pathing) and must be in the replay log.
- `@non_replayable` events are perception side-channels (mirroring `ChronicleEntry`'s pre-deletion role): they exist for belief views to fold over, but two runs of the same seed can re-derive them from replayable state without storing them. Saving a snapshot drops them; loading regenerates by replaying.

**Schema-hash batching across migration phases:** rather than a `.schema_hash` bump per event, group bumps per phase to minimize parity-baseline churn:
- **Phase 1 bump:** `MovementBlocked` only (1 variant added).
- **Phase 2 bump:** `VoxelRegionEntered`, `VoxelRegionLeft` (2 variants added).
- **Phase 3 bump:** `Saw`, `HeardSound`, `SmelledScent`, `FeltImpact`, `VoxelRegionObservedAtDistance` (5 variants added in one commit).

Three baseline-update commits total, each landing atomically with its phase's new event variants. Each bump invalidates retained replay traces — coordinate with whoever maintains the trace fixtures (currently the wolves+humans parity anchor).

## 5. Tiered storage hierarchy

### 5.1 The hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ L0:  per-agent / per-ray result cache    (game-owned)   │
├─────────────────────────────────────────────────────────┤
│ L1:  per-chunk summaries                  (engine)      │
│      — uniform flag, max_z, material_histogram, etc.    │
├─────────────────────────────────────────────────────────┤
│ L2:  LOD pyramid (mips)                   (engine)      │
├─────────────────────────────────────────────────────────┤
│ L3:  HOT primary storage  — dense 64³     (engine)      │
├─────────────────────────────────────────────────────────┤
│ L4:  WARM primary storage — palette+RLE   (engine, v2)  │
│      or 8³ brick pool                                   │
├─────────────────────────────────────────────────────────┤
│ L5:  COLD primary storage — SVDAG         (deferred)    │
├─────────────────────────────────────────────────────────┤
│ L6:  DISK or regen-from-worldgen-seed     (engine)      │
└─────────────────────────────────────────────────────────┘
```

### 5.2 The epoch primitive

Every chunk slot carries a monotonic `u64` write epoch. Any voxel write to a chunk bumps its epoch. Every derived structure (summaries, mips, indices, render-state caches) records the epoch it was built against. On query, a stale epoch triggers (or schedules) rebuild.

```rust
pub struct ChunkSlot {
    chunk_id: ChunkId,
    epoch: u64,
    storage: ChunkStorage,  // L3/L4/L5 backend
    last_query_tick: u64,
}
```

Epoch bumping is a single atomic increment per write batch (one batch per chunk per tick at most). Cross-chunk derived structures key by the *set* of `(chunk_id, epoch)` they depend on; any constituent advance invalidates.

### 5.2.1 Where the per-tick batching lives

The batching layer lives in **voxel_engine**, owned by the same module that hosts `apply_voxel_modification` (§4.2). All writes from any source — `physics` rules calling `apply_voxel_modification`, internal worldgen, fluid simulation, fragment-body destruction — funnel through this single entry point. The module accumulates per-chunk write batches during the tick and commits them at tick end, bumping the epoch counter once per chunk that received any writes.

The contract surfaced to the game's `engine` crate:

```rust
pub trait VoxelStorage {
    fn current_epoch(&self, chunk_id: ChunkId) -> u64;
    fn epochs_advanced_since(&self, last_seen: &EpochSnapshot) -> Vec<(ChunkId, u64)>;
    fn apply_modification(&mut self, region: VoxelRegionId, op: VoxelOp);
}
```

`epochs_advanced_since` is the index-invalidation hook: every per-tick subscriber (`VoxelRegionRegistry`, render visibility cache, etc.) holds an `EpochSnapshot` from its last visit and asks the storage what changed. This makes the bump-on-batch contract honest at the API boundary — `engine` can rely on epoch monotonicity without coupling to voxel_engine internals.

Tick-end commit ordering matters for determinism: writes are applied in canonical agent-order (the engine's existing `step.rs` ordering), so two backends agree on final state per tick. Epoch bumps happen during the host-side commit phase, not on the GPU sim path.

### 5.3 Promotion / demotion

| Trigger | Action |
|---|---|
| Agent within 3×3 chunk neighborhood | Stay at ≥ L4; promote to L3 if recent writes |
| Recent write (last `WRITE_HOT_TTL` ticks, default 60) | L3 |
| Camera frustum or active focus | L3 |
| Sim-loaded but quiet for `QUIET_DEMOTE_TTL` ticks (default 600) | Demote L3 → L4 |
| Unloaded by all agents | Eligible for L4 → L5 → eviction |
| Cache miss in L3 from sim physics query | Faulted up from L4; stays L3 if access repeats |

The TTL constants are tunable; defaults assume 30 tick/s sim. At 2 tick/s the same wall-clock thresholds correspond to 4 and 40 ticks respectively — TTL should be specified in wall-clock seconds and converted internally.

Default policy in v1: only L3 + L6, with L4 staged for v2. Per §2.1, L3 alone covers up to ~12k loaded chunks on an 8 GB card; above that, either L4 ships or the player needs a larger card. The 5k–10k working set fits comfortably in v1.

### 5.4 Worldgen interaction

Complex multi-pass worldgen always writes into L3 dense (256 KB scratch, then committed in place). Multi-pass internal reads stay in L3. Demotion to L4 is asynchronous and decoupled. Worldgen complexity is orthogonal to the storage choice.

## 6. Regions

### 6.1 Definition

A **region** is a logical, named volume of voxel space with an identity and lifecycle that outlives any single index built over it. Regions are *the unit of indexed coverage*; indices attach to regions, not to chunks.

```rust
pub struct VoxelRegionId(u64);  // generational

pub struct VoxelRegion {
    id: VoxelRegionId,
    bounds: VoxelRegionBounds,
    chunks: SmallVec<[ChunkId; 8]>,  // resolved membership
    classification: VoxelRegionKind,       // game-defined tag
    indices: BTreeMap<IndexKind, IndexHandle>,
    epoch_floor: u64,                 // max chunk epoch at last full index build
    created_at_tick: u64,
}

pub enum VoxelRegionBounds {
    Aabb(Aabb),
    ChunkSet(SmallVec<[ChunkId; 16]>),
    Mask { aabb: Aabb, mask: Arc<BitGrid> },
    Sphere { center: Vec3, radius: f32 },
}

pub struct VoxelRegionKind(u32);  // opaque to engine; game-defined
```

Properties:

- **Bounds need not align with chunks.** A settlement can cover parts of 7 chunks; the wilderness around it isn't a region at all.
- **Regions can overlap.** A building inside a settlement is two nested regions, each with its own indices. `covering_regions(world_pos)` returns all coverers, ranked.
- **Generational ID.** Survives chunk epoch bumps; freed only when the region itself is destroyed.
- **Classification is opaque to engine.** The engine treats it as a tag that drives "which indices does this kind of region typically have"; the kind → index-set mapping is DSL-declared (see §6.1.1).

### 6.1.1 Type naming

The voxel-region types use a `Voxel`-prefixed namespace to avoid collision with future engine types in other layers (e.g., a "region" could later mean an event-emit blast radius in the rules layer). Spec B' D13 establishes the precedent: shared engine code avoids generic type names.

In code throughout this spec:
- `VoxelRegion`, `VoxelRegionId`, `VoxelRegionKind`, `VoxelRegionBounds`
- Internal types follow the same prefix: `VoxelRegionRegistry`, `VoxelRegionGraph`, `VoxelRegionEdge`, `VoxelRegionNode`
- Events: `VoxelRegionRegistered`, `VoxelRegionDestroyed`, `VoxelRegionEntered`, `VoxelRegionLeft`, `VoxelRegionObservedAtDistance`

Local variables and field names use unprefixed `region` / `region_id` / `region_kind` since their scope is unambiguous.

### 6.1.2 Region-kind → index-set mapping

The mapping from `VoxelRegionKind` to its default set of indices is declared in DSL via a new top-level `region_indices` declaration. The compiler emits a `static REGION_INDEX_MAP: phf::Map<VoxelRegionKind, &[IndexKind]>` into `engine_data`; the registry consults this map at `register_region` time to schedule the appropriate index builds.

```
region_indices Settlement     { Navgrid, Vismap, CoverMap, SurfaceMesh }
region_indices Building       { Navgrid, SurfaceMesh }
region_indices WildernessTile { SurfaceMesh }
region_indices BattleSite     { Vismap, CoverMap }
```

Properties:

- **All instances of a given `VoxelRegionKind` get the same indices.** This is by design — variation should flow through finer-grained `VoxelRegionKind` taxonomy, not per-instance overrides. Mirrors how the rest of the deterministic sim handles per-kind behavior.
- **Static memory bound.** The compiler computes `total_index_pool_bytes = SUM_over_index_kinds(max_active_regions_using[index_kind] × per_region_storage[index_kind])` at build time. Each `VoxelRegionKind` declares its `max_active` (sibling DSL declaration, see below). Overflow is a build-time error attributed to the specific (region_kind, index_kind) pair.
- **IndexKind catalogue is DSL-visible.** The names `Navgrid`, `Vismap`, etc. inside `region_indices` bodies must resolve to declared `index <name>(region: ...)` declarations from Phase 6a. The `region_indices` declaration depends on the index declarations being parsed first.
- **Schema-hash discipline.** A change to any `region_indices` declaration is a single `.schema_hash` bump, attributed cleanly to the policy change. Adding a new index to `Settlement` is one declaration edit, one bump, one parity-baseline update.

Sibling declaration for max-active counts:

```
region_kind Settlement     { max_active = 64 }
region_kind Building       { max_active = 512 }
region_kind WildernessTile { max_active = 4096 }
region_kind BattleSite     { max_active = 32 }
```

These two declarations together (`region_kind` + `region_indices`) provide everything the compiler needs to size pools, validate budgets, and emit the static lookup.

**Migration to per-instance variation (option C, deferred):** if a real use case emerges where two instances of the same kind genuinely need different indices, the migration is purely additive — `register_region` gains an optional `extra_indices: &[IndexKind]` parameter that augments the default mapping. No breaking change. Defer until a concrete scenario forces the question; the current decision is to start without per-instance overrides.

### 6.2 Lifecycle: game-owned

Region creation and destruction are game events emitted by `physics` rules in the cascade. The engine does not infer regions; it only registers what the game declares.

```
event VoxelRegionRegistered {
    region_id: VoxelRegionId,
    kind: VoxelRegionKind,
    bounds: VoxelRegionBounds,
}

event VoxelRegionDestroyed {
    region_id: VoxelRegionId,
}
```

These events are processed by an engine-side handler that mutates the registry.

### 6.3 DSL detection helpers

While region creation is game-driven, the DSL provides predicates the game's rules can call to *decide* when to register. These compile to compute-shader functions over voxel state.

```
// Illustrative — actual surface depends on stdlib finalisation
fn is_dense_construction(area: Aabb) -> bool {
    let solid_fraction = engine::region_reduce(area, |v| v.is_solid()).fraction();
    solid_fraction > 0.4 && engine::region_reduce(area, |v| v.material).distinct_count() >= 3
}

fn connected_interior_volume(seed: WorldPos, max_volume: u32) -> Option<Aabb> {
    let visited = engine::floodfill(seed, |v| v.is_air(), max_volume);
    if visited.count() > MIN_INTERIOR_VOXELS { Some(visited.bounds()) } else { None }
}
```

A `physics` rule can call these and emit `VoxelRegionRegistered` events accordingly. The detection logic is data-driven (DSL); the registration policy is data-driven (DSL); only the registry implementation is engine code.

These helpers run during execution-time `physics` rules (typically gated to fire at low frequency, e.g. once per 100 ticks per candidate area) — they are not decision-time queries. The result becomes a region whose presence then influences agent beliefs through the events its existence causes (e.g., `VoxelRegionObservedAtDistance` when an agent sees the new construction).

## 7. Indices

### 7.1 Definition

An **index** is a derived data structure attached to a region, declared in the DSL with a static memory bound. Indices are how the engine answers most queries without touching voxel storage.

```rust
pub struct Index {
    handle: IndexHandle,
    kind: IndexKind,
    region: VoxelRegionId,
    storage: IndexStorage,         // GPU buffer / texture / mesh / bitset
    built_at_epoch: u64,           // max(chunk_epoch over region) at build time
    cost_class: CostClass,         // Cheap | Medium | Heavy | Glacial
    state: IndexState,             // Pending | Building | Ready | Stale
    last_query_tick: u64,
}

pub enum CostClass {
    Cheap,    // <1 ms, build inline
    Medium,   // 1–10 ms, single-frame async
    Heavy,    // 10 ms – 1 s, time-sliced over many frames
    Glacial,  // >1 s, build offline, serialize to disk
}
```

### 7.2 DSL declaration

Each index kind is declared in the DSL with bounded storage, build expression, rebuild trigger, and cost class. The compiler validates the memory bound and emits the build kernel + invalidation hookup.

```
index navgrid(region: VoxelRegion) -> Walkable {
    storage: per_cell_2d(max_cells = 16_384, bytes_per_cell = 4),
    cost_class: Cheap,
    rebuild_on: chunk_epoch_advance(region.chunks),
    build {
        let height = engine::column_reduce_xz(region);
        let walk = engine::per_cell_classify(height, classify_walkable);
        engine::connect_neighbors(walk, AGENT_STEP_HEIGHT)
    }
}

index vismap(region: VoxelRegion) -> Vismap {
    storage: bitset_pairs(max_cells = 4_096),  // 4096² bits = 2 MB
    cost_class: Heavy,
    rebuild_on: chunk_epoch_advance(region.chunks),
    build {
        let cells = region.subdivide_view_cells(VIEW_CELL_SIZE);
        let pairs = cells.pairs_within_radius(MAX_VIS_RANGE);
        let rays = pairs.flat_map(|(a, b)| sample_rays_between(a, b, RAYS_PER_PAIR));
        let hits = engine::raycast_batch(rays);
        bitset_pairs::from_hit_results(cells, hits)
    }
}
```

Storage shapes the compiler validates:
- `per_cell_2d(max_cells, bytes_per_cell)` — 2D texture, peak `max_cells × bytes_per_cell`
- `per_cell_3d(max_cells, bytes_per_cell)` — 3D texture
- `bitset_pairs(max_cells)` — pair-membership bitset, peak `max_cells² / 8` bytes
- `mesh_buffer(max_vertices, max_indices)` — vertex/index buffer pair
- `sparse_grid(max_cells, bytes_per_cell)` — hash-table-backed sparse storage

Memory budget is the union over all index kinds × max-active regions per kind, enforced at compile time.

> **Note:** the `index` grammar form is substantial — comparable in scope to Spec B' Task 11 (~600 LoC emit). See §13 Phase 6a–6d for the broken-down plan, or hoist Phase 6a–b into a dedicated `dsl-index-grammar-design.md` if grammar finalisation slips past Phase 1–4.

### 7.3 Built-in index kinds (v1)

| Index | Purpose | Cost class | Storage |
|---|---|---|---|
| `Navgrid` | Walkability + path costs | Cheap | per_cell_2d |
| `SurfaceMesh` | Renderer rasterization | Cheap | mesh_buffer |
| `RegionAabb` | Cached membership for spatial hash | Cheap | trivial |

### 7.4 Built-in index kinds (v2)

| Index | Purpose | Cost class | Storage |
|---|---|---|---|
| `Vismap` (PVS) | Cell-pair visibility | Heavy | bitset_pairs |
| `CoverMap` | Per-cell cover direction | Medium | per_cell_2d |
| `SoundGraph` | Sound propagation cost graph | Medium | sparse_grid |
| `Sdf` | Distance to nearest solid | Medium | per_cell_3d |

### 7.5 Index kinds (deferred, evaluate later)

| Index | Purpose |
|---|---|
| `InfluenceField` | Faction influence per cell — but per-faction, may explode storage |
| `ScentGraph` | Per-scent propagation graph |
| `SVDAG` (Aokana-style) | Far-distance render LOD |

## 8. Engine primitive library

The primitives compose to express index builds. They are the engine's commitment that game/DSL code never writes raw voxel-iteration loops.

### 8.1 Tier 1 — Storage and lifecycle

```rust
pub trait IndexStorage {
    fn alloc(layout: StorageLayout, lifetime: Lifetime) -> Self;
    fn recycle(self);
}

pub struct ScratchBuffer { /* leased per-build, freed at end */ }

impl ScratchBuffer {
    pub fn lease(size: usize) -> Self;
}

pub struct EpochSubscription {
    pub fn on(chunk_set: &[ChunkId], callback: impl Fn(/* invalidated */));
}
```

### 8.2 Tier 2 — Voxel iteration over a region

- `voxels_in_region(region) → ChunkCoherentIterator` — visits chunks in cache-friendly order.
- `surface_voxels(region) → SurfaceIterator` — surface-only (≥1 face neighbor differs).
- `column_reduce_xz(region, reducer)` — per-(x,z) column reduce; e.g. find max solid Y.
- `predicate_iter(region, predicate)` — filter (`material == X`, `is_flammable`, etc.).

### 8.3 Tier 3 — Spatial kernels

- `raycast_voxels(origin, dir, max_steps)` — single-ray DDA (existing).
- `raycast_batch(rays: &[Ray]) → &[RayHit]` — N rays in one dispatch.
- `floodfill(seed, predicate, max_cells)` — connected-component primitive.
- `distance_transform(region, seeds, max_dist)` — JFA-based GPU SDF.
- `surface_extract(region, options)` — greedy meshing / MC / dual-contour.
- `neighbor_stencil(region, kernel)` — 3³ or 5³ stencil compute.

### 8.4 Tier 4 — Reductions and aggregates

- `region_reduce<T>(region, reducer)` — material histogram, max-z, total non-air count.
- `per_cell_classify(grid, classifier)` — applies classification fn per cell.
- `connect_neighbors(grid, predicate)` — adjacency-graph build.

### 8.5 Tier 5 — Composition / pipeline

- `build_pipeline(passes: [Pass])` — chains compute passes with epoch propagation.
- `bind_index_query(kind, build_fn, query_fn)` — registers an index kind.

### 8.6 Tier 6 — Query interface (consumer-side, execution-time)

- `voxel_at(world_pos) → Material` — index fast-path, voxel fallback.
- `walkable(world_pos, mode) → Walk` — navgrid fast-path, voxel fallback.
- `raycast(origin, dir, max) → Option<Hit>` — SDF-skipped if covered, dense DDA fallback.
- `region_at(world_pos) → impl Iterator<VoxelRegion>` — point-in-region.
- `apply_voxel_modification(region, op)` — write path; bumps chunk epochs.

These are the *only* execution-time queries exposed to game code (DSL `physics` rules and engine internals).

### 8.7 Validation: primitive coverage

Walking the planned index list against the primitive set:

| Index | Composes |
|---|---|
| Navgrid | column_reduce_xz (T2), per_cell_classify (T4), connect_neighbors (T4) |
| SurfaceMesh | surface_extract (T3) |
| Vismap | raycast_batch (T3), bitset construction (T1 storage) |
| CoverMap | raycast_batch × 8 dirs (T3), per_cell_classify (T4) |
| SoundGraph | floodfill (T3), region_reduce (T4) per-cell attenuation |
| Sdf | distance_transform (T3) |
| InfluenceField | distance_transform (T3) per-faction |
| SVDAG | surface_voxels (T2), `tree_reduce` (new, would extend T4) |

Only `tree_reduce` (bottom-up octree fold with hash-dedup) is missing for v1. Add only if SVDAG indices are ever shipped.

## 9. Cross-region machinery

### 9.1 Region graph

Nodes = regions, edges = adjacency (regions whose bounds touch), edge attributes = portal points and per-medium traversal cost.

```rust
pub struct VoxelRegionGraph {
    nodes: HashMap<VoxelRegionId, VoxelRegionNode>,
    edges: HashMap<(VoxelRegionId, VoxelRegionId), VoxelRegionEdge>,
}

pub struct VoxelRegionEdge {
    portals: SmallVec<[Portal; 4]>,         // boundary crossing points
    cost_per_medium: HashMap<MovementMode, f32>,  // computed from indices on both sides
    epoch_dep: (u64, u64),                  // last (region_a.epoch_floor, region_b.epoch_floor) used
}
```

The region graph is itself a singleton index — rebuilt incrementally on `VoxelRegionRegistered` / `VoxelRegionDestroyed` events; per-edge attributes invalidate when either side's `epoch_floor` advances.

### 9.2 Geometric decomposition

```rust
pub fn decompose_ray_by_region(a: WorldPos, b: WorldPos)
    -> SmallVec<[(Option<VoxelRegionId>, Range<f32>); 4]>;
```

Splits a ray into ordered (region, parametric range) segments. Cheap (uses spatial hash on region bounds). Used by sense gates and the renderer.

### 9.3 Internal cross-region algorithms

These exist inside the engine and are *not* directly exposed to DSL rules. They serve execution-time engine functions.

- `region_graph::astar(start, goal, edge_cost_fn, heuristic_fn)` — used by `resolve_movement` when an agent's chosen destination is in a different region.
- `region_graph::broadcast(source, value, attenuation_fn, terminate_fn)` — used by `propagate_sound`, `propagate_scent`, `propagate_rumor`.

There is **no** `region_graph::bfs_expand` in the DSL-facing API. "Find nearest X" is an agent-belief query (over the agent's own `known_places` view), not an engine query.

The sharper restatement of §4.2: `astar`, `broadcast`, and `bfs_expand` are internal to execution-time engine implementations; they are not in the DSL stdlib surface and not callable from `physics` rule bodies. The "no `engine::find_*`" rule applies to DSL-facing surface only — internal pathfinders that resolve already-decided movement are fine. The compile-time validator (§4.2) gates on stdlib visibility, so this distinction is enforced at the boundary, not by naming.

### 9.4 Sense gates and emission

The engine provides sense gates and emission primitives that respect the epistemic split.

```rust
pub fn can_observe(observer: AgentId, target_pos: WorldPos, kind: ObservationKind) -> bool;
// Returns true iff: in vision cone, range ≤ vision_range, line of sight clear,
// not occluded by cover, lighting sufficient. Uses indices when available, raycast fallback.

pub fn emit_to_observable_agents(
    template: EventTemplate,
    source_pos: WorldPos,
    kind: ObservationKind,
    filter: impl Fn(AgentId) -> bool,
);
// Iterates candidate observers in range; for each that satisfies can_observe + filter,
// emits the templated event with per-listener parameters (relative direction).

pub fn emit_via_propagation(
    template: EventTemplate,
    source_pos: WorldPos,
    propagation: PropagationKind,  // Sound | Scent | Rumor | Magical | ...
    initial_intensity: f32,
);
// Walks region_graph::broadcast from source's region; for each reachable region with
// intensity above threshold, iterates listener candidates within and emits per-listener
// events with attenuated intensity.

pub fn resolve_movement(
    agent: AgentId,
    intent: MovementIntent,
) -> MovementResolution;
// Resolves a chosen movement against ground truth: navgrid, collision, terrain-speed
// modifiers. Emits MovementBlocked, MovementCompleted, EnteredRegion, LeftRegion events
// as appropriate.
```

These are the surfaces game-side `physics` rules use. Decision-time code (`scoring`, `mask`) sees none of these.

### 9.4.1 DSL surface for emission

`physics` rule bodies invoke these via stdlib functions added to `spec/stdlib.md`. Sketch:

```
physics on Attacked { actor: a, target: t, location: pos } {
    engine::emit_to_observable_agents(
        template = Saw { observer: $observer, target: a, position: pos, tick: now() },
        source_pos = pos,
        kind = ObservationKind::Combat,
        filter = |observer| observer.faction != a.faction,
    );
}

physics on DragonRoar { source: dragon, position: pos } {
    engine::emit_via_propagation(
        template = HeardSound { listener: $listener, direction: $relative_dir, intensity: $atten, source_kind: SoundKind::DragonRoar },
        source_pos = pos,
        propagation = PropagationKind::Sound,
        intensity = 100.0,
    );
}
```

The `$observer`, `$listener`, `$relative_dir`, `$atten` are template-binding sigils — placeholders the engine fills per emitted instance. The DSL compiler lowers these to backend-specific code: SerialBackend produces a Rust closure that iterates candidates and pushes events; GpuBackend emits a compute kernel that does the same with parallel writes to the event ring. The lowering is mechanical once the stdlib syntax is fixed; `compiler.md` gains a new "emission lowering" subsection.

This integrates with the per-tick batching layer (§5.2.1): emitted events accumulate in a per-tick scratch buffer and commit at tick-end alongside voxel-write batches.

## 10. Registry and scheduling

### 10.1 Registry structure

```rust
pub struct VoxelRegionRegistry {
    regions: SlotMap<VoxelRegionId, VoxelRegion>,
    chunk_to_regions: HashMap<ChunkId, SmallVec<[VoxelRegionId; 4]>>,
    spatial_hash: SpatialHash<VoxelRegionId>,
    region_graph: VoxelRegionGraph,

    indices: SlotMap<IndexHandle, Index>,
    index_pool: IndexAllocator,

    scheduler: BuildScheduler,
    epoch_inbox: EpochInbox,
}
```

Three core ops:
- `covering_regions(world_pos) → impl Iterator<VoxelRegionRef>` — point query, ~3 candidates checked.
- `get_index(region_id, kind) → Option<IndexRef>` — high hit rate; misses fault into builder.
- `invalidate_chunk(chunk_id, new_epoch)` — called by the voxel write path; cascades to dependent indices via `chunk_to_regions`.

### 10.2 Scheduler

The scheduler holds a priority queue of `(IndexHandle, dirty_since_tick)` entries. Per-frame budget per cost class:

| Cost class | Per-frame budget | Strategy |
|---|---|---|
| Cheap | 4 ms | Build all dirty inline |
| Medium | 8 ms | Build oldest-first to fill budget |
| Heavy | 4 ms | Time-slice; resume across frames |
| Glacial | Background thread + disk cache | Build offline, serialize, load on demand |

Priority signal:
```
priority(index) =
    α · staleness_ticks
  + β · 1 / distance_to_nearest_camera_or_agent
  + γ · query_demand_recent_ticks
```

Defaults: α=1.0, β=10.0 m, γ=2.0. Tunable per cost class.

### 10.3 Eviction

Indices are evicted when:
- `last_query_tick > current_tick - INDEX_TTL_TICKS` AND no agent within their region's bounds, OR
- `index_pool` pressure forces LRU eviction within their cost class.

Eviction frees `IndexStorage`. The next query against the index re-faults a build.

## 11. The full query stack

To make the contract concrete, here is what happens when a `physics` rule calls `engine::raycast(origin, dir, max)` during execution:

1. **L0 (per-ray cache, optional, opt-in):** if a recent identical ray returned no-hit and no chunk on the path has advanced epoch, return cached result.
2. **decompose_ray_by_region(origin, origin + dir × max)** → segments.
3. **For each segment:**
   - If segment in region with `Sdf` index: SDF-skip raymarch. May terminate early on hit.
   - If segment in region with `Vismap` and ray endpoints align with cell centers: vismap lookup.
   - If segment in region with surface mesh: BVH ray-mesh test.
   - Otherwise fall through to **L1 (per-chunk summaries):** if all chunks on segment are uniform-air, skip. If any uniform-solid, hit at chunk surface.
   - Otherwise fall through to **L2 (mip-skip DDA)** then **L3 (dense DDA)** then **L4/L5** (faulting up).
4. Combine segment hits; return earliest.

For `engine::voxel_at(world_pos)`:

1. `region_at(world_pos)` → covering regions.
2. For each region with relevant index (e.g., surface mesh covers point): return.
3. Otherwise direct chunk fetch from L3 (faulting up from L4/L5 if needed).

For `engine::walkable(world_pos, mode)`:

1. `region_at(world_pos)` → covering regions.
2. If any covering region has `Navgrid` index: cell lookup.
3. Otherwise compute walkability ad-hoc from voxel storage (slow path; signals "this region should be indexed").

## 12. Renderer integration

The renderer is async, downstream of the sim tick. It reads a snapshot of voxel state + region indices.

### 12.1 Render path

```
For each visible region:
    If region has SurfaceMesh index → rasterize directly.
    Else if region's chunks are L3 hot → DDA on cube proxies (existing path).
    Else fault chunks up to L3 → DDA.
```

### 12.2 Visibility cache

Per-region render-state cache: `(epoch_floor_at_build, lod_level_chosen, draw_cmd_indices)`. If `epoch_floor` unchanged AND camera-relative metrics unchanged enough to not flip LOD, last frame's draw command is still valid. Most frames most regions hit this cache.

### 12.2.1 Snapshot mechanism

The snapshot reuses the existing engine infrastructure (`spec/gpu.md` §2): double-buffered, non-blocking. The renderer calls `voxel_storage::snapshot()`, receives a `VoxelSnapshot` analogous to `GpuSnapshot`, and reads against it for the duration of the frame. Like `GpuSnapshot`, contents reflect state as-of the previous snapshot call (one frame lag); front staging buffers populated by `map_async(Read)` while the back continues to receive sim writes. Region indices follow the same flip — the renderer's `VoxelRegionRegistryView` carries the chunk-epoch snapshot it was paired with, so visibility-cache invalidation aligns to the same tick boundary.

### 12.3 LOD

- LOD 0: dense DDA / direct mesh rasterization
- LOD 1: shallow surface mesh (greedy)
- LOD 2 (deferred): SVDAG raymarch (Aokana-style; only if v2+ targets long view distances)

LOD is per-region, selected by camera distance and screen-space size.

## 13. Migration path

### 13.0 Sequencing relative to other planned work

This spec defines one workstream of many tracked under Spec C (project-DAG / TaskWarrior). External dependencies and ordering constraints:

| Dependency | Direction | Notes |
|---|---|---|
| **B2 — chronicle / engagement DSL re-emission** | **Blocks Phase 3** | Sense-gate events (`Saw`, `HeardSound`, etc.) emit through the same emission infrastructure that chronicle deletion took out. Phase 3 cannot land before B2 restores the emission lowering machinery. |
| **Spec B' — engine crate split** | **Concurrent with Phase 0–1** | The `engine_data` boundary that Spec B' establishes is where `engine_data::tunables::spatial` (§16a) lives. Phase 0 trait surfaces should target the post-split crate layout. |
| **Spec C — project-DAG** | **Tracks this workstream** | Add the 7-phase plan as a workstream once this spec is approved. |
| **Critic skills** (P-rules) | **Gates every phase** | New events (`@replayable` / `@non_replayable`) trigger schema-hash bumps (P2). New scoring/mask primitives trigger P5 RNG-routing review. Plans for each phase need an AIS preamble (P8) per `docs/architecture/plan-template-ais.md`. |

Open question: confirm B2 status — is it landed, in-progress, or still planned? If still planned, Phase 3 sequencing depends on B2's own timeline.

### Phase 0 — Prerequisites in voxel_engine

Reviewer flagged Phase 0 as undersized; it splits into two sub-phases that Phase 1 depends on.

#### Phase 0a — Epoch counters in voxel_engine
- Add per-chunk epoch counter to existing `terrain_compute::ComputeSlot`.
- Implement the per-tick batching layer (§5.2.1) with the `VoxelStorage` trait surface.
- `epochs_advanced_since` returns deltas via `EpochSnapshot`.
- Tests: epoch monotonicity, batch-end commit timing, snapshot semantics.

#### Phase 0b — Registry skeleton + API trait surface
- Stub `VoxelRegionRegistry` types: `VoxelRegion`, `VoxelRegionId`, `VoxelRegionBounds`, `VoxelRegionKind`, `Index`, `IndexHandle`, `IndexStorage`, `EpochInbox`, `IndexAllocator`.
- No-op `BuildScheduler` (returns immediately, builds nothing).
- Execution-time API trait surface defined in `engine_data` (per Spec B' boundary).
- `FlatPlane` default impl in `crates/engine/src/terrain.rs` (per `terrain-integration-gap.md` §A) so headless tests pass with no voxel_engine dep.
- Tests: trait round-trip, FlatPlane default behavior matches current featureless-R³ semantics.

### Phase 1 — Single-region default + Navgrid
- Implement L3 hot pool with epoch wiring.
- Implement `Navgrid` index over a single full-world region (smoke test).
- Game's `engine` crate consumes `walkable(pos, mode)`.
- Wires up the first DSL terrain-mechanic: high-ground bonus from `terrain-integration-gap.md` minimal slice.

### Phase 2 — Multi-region + DSL detection helpers
- Game emits `VoxelRegionRegistered` events for hand-authored test settlements.
- Multiple regions coexist; `covering_regions` works.
- DSL detection helpers (`is_dense_construction`, `connected_interior_volume`).

### Phase 3 — Sense gates + propagation
- `can_observe` implementation backed by raycast + (later) `Vismap`.
- `emit_to_observable_agents` wired to combat events; first DSL `Saw` view fold.
- `emit_via_propagation` for sound; first DSL `HeardSound` view fold.
- Drives the cover, LOS, fog-of-war mechanics from `terrain-integration-gap.md`.

### Phase 4 — SurfaceMesh + renderer migration
- `SurfaceMesh` index per region.
- Renderer reads mesh for indexed regions, falls back to DDA for unindexed.
- Visibility cache keyed by epoch.

### Phase 5 — L4 warm tier
- Triggered when peak loaded chunks exceed 15k.
- Palette + RLE compression for quiet chunks.
- Promote on write, demote on tick-since-last-write threshold.

### Phase 6 — DSL index declarations

The reviewer flagged this phase as undersized; it splits into four sub-phases, each its own implementation plan. The compiler scope is comparable to Spec B' Task 11 (~600 LoC of emit code).

#### Phase 6a — Grammar + parser + IR
- New `index` top-level declaration form added to `spec/language.md`.
- Storage-shape sub-syntax (`per_cell_2d(...)`, `bitset_pairs(...)`, `mesh_buffer(...)`, etc.).
- IR: `Index`, `IndexStorage`, `CostClass`, `RebuildTrigger` types.
- Parser produces IR nodes.
- No emit yet; round-trip parse-and-print test only.

#### Phase 6b — Validation + memory bounds
- Static memory-bound arithmetic at compile time (per-instance × max-active-regions).
- Pool-budget validator: union over all index kinds × max-active.
- Reject overflows with clear "would exceed budget" diagnostic.
- Schema-hash extension covers index declarations.

#### Phase 6c — Rust emit (CPU SerialBackend)
- `emit_index` module: 4–6 files (kernel emit, cost-class wiring, invalidation hookup, registry registration, build-pipeline assembly).
- Each declared index kind generates a `IndexBuilder` impl that the registry can dispatch.
- Vismap, CoverMap, SoundGraph land here for the SerialBackend.

#### Phase 6d — GPU emit (GpuBackend)
- SPIR-V kernel generation per index kind.
- Same primitives library (§8) callable from emitted compute shaders.
- Determinism contract enforced (per §15.5): CPU and GPU emits use identical canonical algorithms.
- Vismap, CoverMap, SoundGraph parity verified against SerialBackend.

Phase 6a–b can land before Phase 5; 6c–d follow once L4 stability is proven.

**Alternative:** hoist Phase 6a–b into a dedicated `dsl-index-grammar-design.md` sub-spec, since the grammar work is independent of the runtime architecture this spec covers. Recommend doing this if Phase 6a slips past the planned Phase 1–4 timeline; the runtime-side work doesn't block on grammar finalisation if Phase 1–4 ship with hand-written `IndexBuilder` impls in Rust.

### Phase 7 — Full DSL integration
- All built-in index kinds DSL-declared.
- Engine primitives stable; new index kinds are pure DSL additions.
- Cold tier (L5 SVDAG) evaluated on real workload data; ship only if measurements support it.

## 14. What this design does NOT do

To keep scope honest, explicit non-goals:

- **No infinite world streaming.** Targeted at finite, bounded maps (DF-style). A streaming-from-disk story for true open world is deferred until requirements force it.
- **No Aokana port.** Aokana's pipeline is referenced as a possible LOD-index implementation in some far future; it is not the spine.
- **No god-mode agent queries.** Decision-time code cannot ask the engine spatial questions.
- **No render determinism.** Renderer is a free-running observer.
- **No runtime DSL changes.** Index declarations are compile-time; the index pool is sized statically.
- **No cross-tick ray result caching in v1.** L0 per-ray cache is opt-in, deferred until measurements show it pays.
- **No Glacial-class indices in v1.** PVS for entire settlements (minutes of build time) is deferred; offline build + serialization story comes when needed.
- **No first-frame-latency mitigation.** When an agent enters previously-unexposed terrain, the worldgen + index-build cost is paid synchronously on first access. Prefetch from agent intent, predictive preload, and loading-screen orchestration are deferred until player-camera mode requires them. See §15.7 for the deferred design.

## 15. Open questions

These warrant resolution during implementation but do not block this spec:

1. **Surface extraction algorithm.** Greedy meshing, marching cubes, dual contouring, or transvoxel? Each has rendering / collision / boundary-stitching tradeoffs. Recommend greedy initially for simplicity; revisit when dragon-breath-melts-stone shows seams.
2. **Index serialization format.** For save games and world sharing, we'll need a stable on-disk format for at least Navgrid and SurfaceMesh. Spec out when save/load ships.
3. **Region overlap semantics for queries.** When two indexed regions cover a point and both have Navgrid, which wins? Innermost? Most-recently-built? Specify when first overlapping case arises.
4. **Concurrent build limits.** How many indices may build simultaneously on async compute queues without VRAM pressure? Empirical, set during profiling.
5. **Determinism of index build (constraint, not question).** Index *outputs* feeding execution-time queries that emit replayable events MUST be deterministic across runs and across CPU/GPU backends, even when build *scheduling* is non-deterministic. Two indices built in different orders must produce bit-identical query results. This is implied by §2's sim-determinism requirement: `resolve_movement` emits replayable events (`MovementBlocked`, `VoxelRegionEntered`, etc.) into `replayable_sha256`, and those events depend on indices via `walkable()` / navgrid lookups. The constraint binds CPU+GPU implementations to identical algorithms (not merely identical primitives). Render-only indices (LOD meshes, visibility cache) are exempt — they don't feed replayable events. The open part: deciding the canonical algorithm for each non-trivial index (greedy mesh tie-breaking, JFA pass count, A* tie-breaking) and codifying it in `spec/runtime.md`'s determinism contract.
6. **Region graph rebuild cost.** Adding/removing regions modifies the graph; at high churn (constant region churn from destruction), is incremental update or full rebuild cheaper? Profile.
7. **First-frame latency on chunk exposure.** Worldgen + index builds for a freshly-exposed chunk neighborhood: how do we hide latency? Pre-emptive build-ahead from agent intent? Loading-screen for player-traversal? Specify when player-camera mode lands.

## 16. Acceptance criteria

This design is implementable when the following can be answered yes:

- [ ] Can a `physics` rule emit a `VoxelRegionRegistered` event and have `covering_regions(pos)` return that region on the next tick?
- [ ] Can a DSL `index navgrid(...)` declaration produce a working navgrid that `walkable(pos, mode)` consults?
- [ ] Does a voxel write to a chunk inside an indexed region invalidate that region's index epoch and trigger a rebuild?
- [ ] Does `can_observe(observer, target)` consult the region's `Vismap` when present and fall back to raycast otherwise?
- [ ] Does `emit_via_propagation` produce per-listener events with attenuated intensity values consistent across CPU and GPU sim backends?
- [ ] Does a `scoring` expression that calls `engine::nearest_*` fail at compile time?
- [ ] Does the renderer rasterize a region's SurfaceMesh when present, fall back to DDA otherwise?
- [ ] Does index pool memory occupancy stay within the compile-time bound under stress (50 active settlements, 200 active buildings)?

## 16a. Tunables

Centralised list of constants for the spatial layer. All time values are in **wall-clock seconds**; sim ticks at variable rates (30 tps at 20k agents → 2 tps at 200k), so tick-count thresholds are derived at runtime from `tps × seconds`. All tunables live in `engine_data::tunables::spatial` (loaded from a config table at engine init), with named-constant defaults in code and override paths via the standard config mechanism.

| Tunable | Default | Section | Derivation |
|---|---|---|---|
| `write_hot_ttl_secs` | 2.0 s | §5.3 | A chunk that wrote 2s ago is recent enough that a re-write within the window is plausible; cheaper to keep hot than churn |
| `quiet_demote_ttl_secs` | 20.0 s | §5.3 | An order of magnitude over `write_hot_ttl`; aligns with typical "scene shifted" gameplay timescales |
| `index_ttl_secs` | 30.0 s | §10.3 | Indices held one full quiet-demote period after last query |
| `min_interior_voxels` | 64 | §6.3 | 4³ block — smallest "room" worth indexing |
| `rays_per_pair` | 4 | §7.2 vismap | Sample diversity vs build cost; tune empirically |
| `max_vis_range_m` | 100.0 m | §7.2 vismap | Caps PVS edges; matches typical scoring vision range |
| `view_cell_size_m` | 4.0 m | §7.2 vismap | 20× voxel scale; coarse enough that one cell ~ one tactical position |
| `agent_step_height_m` | 0.6 m | §7.2 navgrid | 3 voxels at 0.2 m scale; matches average humanoid step |
| `priority_weight_staleness` (α) | 1.0 | §10.2 | Default; tune during profiling |
| `priority_weight_proximity` (β) | 10.0 m | §10.2 | Distance scale where proximity matters comparable to age |
| `priority_weight_demand` (γ) | 2.0 | §10.2 | Recent queries weigh ~2× staleness |
| `budget_cheap_ms` | 4.0 ms | §10.2 | ~25% of 16.7 ms frame budget at 60 fps render |
| `budget_medium_ms` | 8.0 ms | §10.2 | ~50% of frame budget; can spike on rebuild bursts |
| `budget_heavy_ms` | 4.0 ms | §10.2 | Time-sliced; heavy builds amortise over many frames |

Tuning runbook lives at `docs/runbooks/spatial-layer-tuning.md` (TBD), structured around the workload characteristics measured in early profiling sessions.

## 17. Cross-references

- `superpowers/notes/2026-04-22-terrain-integration-gap.md` — origin of the integration question; minimal first slice.
- `game/overview.md` — game scope, target scale, layer map.
- `spec/runtime.md` — engine contract, view system, cascade discipline.
- `spec/language.md` — DSL grammar; `index` declaration is a new top-level form to add.
- `spec/state.md` — field catalog; new spatial-belief views land here.
- `spec/gpu.md` — GPU backend contract; index build kernels are a new category.
- arXiv 2505.02017 (Aokana) — surveyed; informs LOD-index design only.
