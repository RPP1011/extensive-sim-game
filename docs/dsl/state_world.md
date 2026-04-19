# ECS DSL — World / Spatial / Terrain State Catalog

The environment layer: data defining terrain, structural tiles, voxels, spatial indices, and shared caches. Not per-entity, not per-settlement/faction.

Primary files:
- `src/world_sim/state.rs` — `WorldState`, `Tile`, `FidelityZone`, `BuildSeed`, `SimScratch`, `GroupIndex`, `StructuralEvent`
- `src/world_sim/voxel.rs` — `VoxelWorld`, `Chunk`, `Voxel`, `VoxelMaterial`
- `src/world_sim/terrain/region_plan.rs` — `RegionPlan`, `RegionCell`, `SettlementPlan`, `DungeonPlan`, `RiverPath`, `RoadSegment`
- `src/world_sim/systems/exploration.rs` — `FlatSurfaceGrid`, `SurfaceCache`, `CellCensus`
- `src/world_sim/nav_grid.rs` — `NavGrid`, `NavNode`
- `src/world_sim/constants.rs` — grid/voxel constants

---

## WorldState top-level fields (world/spatial only)

Per-entity and per-settlement/faction fields are documented in the other two docs. This section covers the rest.

### Scalar / identity

| Field | Type | Meaning | Persistence | Updated by | Read by |
|---|---|---|---|---|---|
| tick | u64 | Monotonic tick counter | Primary | `WorldSim::tick` (once per step) | all time-gated systems |
| rng_state | u64 | PCG-style RNG state; sole randomness source | Primary | `next_rand_u32` / `next_rand` | all stochastic systems |
| next_id | u32 | Monotonic entity ID counter | Primary | `next_entity_id()` | entity spawn |
| max_entity_id | u32 | Highest ID seen (sizes `entity_index`) | Derived (cache sizing) | `rebuild_entity_cache`, `rebuild_group_index` | indexed lookups |

### Derived indices (all `#[serde(skip)]` — rebuilt on load)

| Field | Type | Purpose | Rebuilt by | Read by |
|---|---|---|---|---|
| entity_index | `Vec<u32>` | `id → idx` into `entities/hot/cold`. Size = `max_entity_id+1`. Sentinel `u32::MAX` | `rebuild_entity_cache`, `rebuild_group_index` | `entity()`, `entity_mut()`, `hot_entity()`, `cold_entity()` |
| group_index | `GroupIndex` | Contiguous per-settlement + per-party entity ranges (see below) | `rebuild_group_index` | system dispatch (runtime, exploration, supply) |
| settlement_index | `Vec<u32>` | `settlement_id → idx` into `settlements` vec | `rebuild_settlement_index` | `settlement()`, `settlement_mut()` |

Note: all three are strictly **derived** from primary state and re-buildable from `entities`/`settlements`. They exist to avoid linear scans.

### Spatial / terrain state

| Field | Type | Persistence | Purpose |
|---|---|---|---|
| tiles | `HashMap<TilePos, Tile, ahash>` | Primary | Sparse 2D tile grid |
| fidelity_zones | `Vec<FidelityZone>` | Primary (entity_ids membership is derived, re-computed each tick) | Sim-fidelity bubbles |
| build_seeds | `Vec<BuildSeed>` | Primary | Pending room-growth seeds |
| voxel_world | `VoxelWorld` | Primary | 3D chunked voxel world (physical truth) |
| nav_grids | `Vec<NavGrid>` | Derived cache | Baked 2D walkable surfaces per settlement |
| region_plan | `Option<Arc<RegionPlan>>` (`#[serde(skip)]`) | Primary (regen from seed) | Biome/elevation plan driving terrain generation |
| structural_events | `Vec<StructuralEvent>` | Per-tick buffer | Collapse/fracture events this tick |
| chronicle | `Vec<ChronicleEntry>` | Primary (bounded ring) | Narrative log |
| world_events | `Vec<WorldEvent>` | Primary (bounded) | Recent events for system queries |

### Cache fields (all `#[serde(skip)]`)

| Field | Type | Rebuilt by | Purpose |
|---|---|---|---|
| surface_cache | `SurfaceCache` (HashMap<u64, i32>) | `scan_voxel_resources_cached` (lazy) | Fallback analytical surface-height cache |
| surface_grid | `FlatSurfaceGrid` | `warm_surface_cache` | Dense per-settlement surface-height tiles |
| cell_census | `CellCensus` (HashMap<(i32,i32),[u32;6]>) | `scan_voxel_resources_cached` (lazy) | Per-cell target-material counts for NPC discovery |
| sim_scratch | `SimScratch` | Caller clears before each use | Pooled scratch buffers, NOT persistent |

---

## WorldState.tiles

**Type:** `HashMap<TilePos, Tile, ahash::RandomState>` (state.rs:505)
**Purpose:** Sparse 2D grid of placed tiles — floors, walls, doors, furniture, farmland, workspaces, ditches. Only modified tiles stored; unmodified positions are empty. 2.0 world units per tile.

### Tile fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| tile_type | TileType | What kind of tile | `construction.rs:71,125,143` (wall/door placement), `action_eval.rs` (PlaceTile action) | flood_fill, interiors, buildings, pathing, movement_cost lookups |
| placed_by | Option\<u32\> | Entity that placed it (attribution) | construction, action_eval | rarely queried |
| placed_tick | u64 | Tick placed | construction | sometimes used for decay/age checks |
| floor_level | u8 | Floor index within a multi-story building (0 = ground level, 1 = second floor, etc.). Outdoor tiles use 0. Multi-story buildings occupy one `Tile` entry per (TilePos, floor_level). | construction (per-floor placement) | interior navigation (`floor_height(pos, building)`), rendering, mask predicates (overhear: same building → any floor; planar_distance + z_separation apply across floors) |

Tiles remain 2D per-floor: the `HashMap<TilePos, Tile>` keys are still `TilePos { x, y }`, but the stored `Tile` carries `floor_level`. Multi-story buildings store one `Tile` entry per occupied (x, y, floor) triple via distinct map insertions keyed on a `(TilePos, floor_level)` tuple (implementation detail; logical schema above treats `floor_level` as a `Tile` field).

### TileType variants (state.rs:205)
Exhaustive list:
- **Terrain:** `Dirt`, `Stone`, `Water`
- **Structural:** `Floor(TileMaterial)`, `Wall(TileMaterial)`, `Door`, `Window`
- **Infrastructure:** `Path`, `Bridge`, `Fence` (blocks monsters only)
- **Agricultural:** `Farmland`
- **Furniture (in-room):** `Workspace(WorkspaceType)`, `Bed`, `Altar`, `Bookshelf`, `StorageContainer`, `MarketStall`, `WeaponRack`, `TrainingDummy`, `Hearth`
- **Defensive:** `Moat`, `TowerBase`, `GateHouse`, `ArcherPosition`, `Trap`

Per-variant gameplay role is encoded in `movement_cost()`, `is_solid()`, `is_wall()`, `is_floor()`, `is_furniture()`, `blocks_monsters_only()` on `TileType`.

### TileMaterial variants (state.rs:246)
`Wood`, `Stone`, `Iron`. Used to parametrise wall/floor types.

### WorkspaceType variants (state.rs:254)
`Forge`, `Anvil`, `Loom`, `AlchemyBench`, `Kitchen`, `Sawbench`.

### BuildingFunction enum (state.rs:316)
`Shelter`, `Production`, `Worship`, `Knowledge`, `Defense`, `Trade`, `Storage`. Used by `BuildSeed.intended_function` and to compute `minimum_interior()` requirements. Not stored in `Tile` directly — a property of planned rooms.

### Characteristics
- **Update frequency:** incremental, sparse writes. Only when NPCs `PlaceTile`/construction runs room-growth. Rarely removed (door placement rewrites single tiles).
- **GPU-friendliness:** hostile — HashMap lookup per neighbor in flood_fill. `is_enclosed`, `has_door`, `find_door_position` all do 4/8-connected scans via HashMap.
- **Derived?** No — primary state.
- **Candidates for flat-grid conversion:** yes. Spatially dense within settlements; currently paying hash cost per neighbor lookup. A chunked flat grid keyed by TilePos would be faster.

### TilePos helpers (state.rs:156)
- `TilePos { x: i32, y: i32 }` with hand-packed `Hash` (pack to u64, single `write_u64` — ~2× faster than derive(Hash)).
- `TilePos::from_world(wx, wy) → (wx/2).floor()`
- `TilePos::to_world()` — returns tile center
- `neighbors8()`, `neighbors4()`

---

## WorldState.voxel_world (VoxelWorld)

**Type:** `VoxelWorld` (voxel.rs:443)
**Purpose:** 3D chunked voxel world — physical source of truth. Sparse; only loaded chunks stored.

### VoxelWorld fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| chunks | `HashMap<ChunkPos, Chunk, ahash>` (type alias `ChunkMap`) | Sparse chunk storage | `generate_chunk`, `set_voxel`, `damage_voxel`, `mine_voxel`, `remove_box/sphere`, `fill_box/sphere`, `replace_in_box` | `get_voxel`, surface_height chunk path, structural_tick, exploration scan, voxel_construction, voxel_harvest |
| sea_level | i32 | Global water z-level (default 350 at 0.10 VOXEL_SCALE = 35m) | Init only | terrain gen, surface_height fallback, flood detection |
| region_plan | `Option<RegionPlan>` | Biome plan driving materializer | Init only | `surface_height` fast path, `generate_chunk` |

### Constants (constants.rs)
| Constant | Value | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 64 | Voxels per chunk edge |
| `CHUNK_VOLUME` | 262,144 | `CHUNK_SIZE³` — voxels per chunk |
| `VOXEL_SCALE` | 0.10 | Meters per voxel |
| `MAX_SURFACE_Z` | 2000 | Upper bound on terrain Z (for surface scans) |
| `SEA_LEVEL` | 350 | Default water level in voxel-z |
| `MEGA` | (chunks-per-mega-axis) | Mega-chunk grouping (rendering) |

### ChunkPos (voxel.rs:31)
| Field | Type | Meaning |
|---|---|---|
| x, y, z | i32 | Chunk-space coord (each chunk covers CHUNK_SIZE³ voxels) |

Derive-Hash is used (3× write_i32) — not hand-optimised like TilePos, but `ahash` mitigates.

### Chunk (voxel.rs:385)
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| voxels | `Vec<Voxel>` (len = CHUNK_VOLUME = 262,144) | Dense voxel array, row-major `(z·CS + y)·CS + x` | `Chunk::set`, VoxelWorld mutations | renderer, structural_tick, scans |
| pos | ChunkPos | Identity | Constructor | mesh gen |
| dirty | bool | Any voxel changed — regen SDF/mesh | `set`, `set_voxel`, `damage_voxel`, `mine_voxel`, filled by structural_tick to `false` after processing | mesh regen, structural_tick (picks dirty chunks) |

### Voxel (voxel.rs:326)
`#[repr(C)]`, ~16 bytes/voxel.
| Field | Type | Meaning |
|---|---|---|
| material | VoxelMaterial (u8 repr) | Material enum |
| light | u8 | Light level 0–15 |
| damage | u8 | Mining damage (breaks when ≥ hardness) |
| flags | u8 | bits 0–3: water level; 4–5: flow direction; 6: is_source; 7: is_support |
| integrity | f32 | Structural health [0,1]. 1.0=intact, 0.0=collapsed (load-bearing voxels preserve material at 0.0) |
| building_id | Option\<u32\> | Building this voxel belongs to, if any |
| zone | VoxelZone | Residential/Commercial/Industrial/Military/Agricultural/Sacred/Underground/None |

Structural health + support are per-voxel: `effective_hp() = integrity * material.properties().hp_multiplier`. `damage_voxel` at ≤0 HP sets integrity=0 (load-bearing) or replaces with Air, then triggers `cascade_collapse` upward.

### VoxelMaterial variants (voxel.rs:90)
~45 variants: Air, natural terrain (Dirt/Stone/Granite/Sand/Clay/Gravel/Grass), fluids (Water/Lava/Ice/Snow), ores (IronOre/CopperOre/GoldOre/Coal/Crystal), placed materials (WoodLog/WoodPlanks/StoneBlock/StoneBrick/Thatch/Iron/Glass), agricultural (Farmland/Crop), additional (Basalt/Sandstone/Marble/Bone/Brick/CutStone/Concrete/Ceramic/Steel/Bronze/Obsidian), biome surfaces (JungleMoss/MudGrass/RedSand/Peat/TallGrass/Leaves), entity markers (NpcIdle/Walking/Working/Fighting/MonsterMarker — non-solid, rendering only).

Each material has: `is_solid()`, `is_fluid()`, `is_transparent()`, `hardness() → u32`, `mine_yield() → Option<(commodity_idx,f32)>`, `properties() → MaterialProperties` (hp_multiplier, fire_resistance, load_bearing, weight, rubble_move_cost, construction_cost, blast_resistance).

### VoxelZone (voxel.rs:307)
`None`, `Residential`, `Commercial`, `Industrial`, `Military`, `Agricultural`, `Sacred`, `Underground`. For building zone tracking.

### Surface-height paths

Signature: `surface_height: vec2 → f32` (outdoor only). Returns world-space z of the topmost solid voxel surface at planar coordinate `(vx, vy)`. Interior navigation uses `floor_height: (vec3, building_id) → f32` instead, derived from the tile `floor_level` layer the agent is on.

Three code paths, in priority order:
1. **Analytical fbm path** (`terrain::materialize::surface_height_at(vx, vy, plan, seed)`, voxel.rs:492) — used when `region_plan` is `Some`. Pure function of (vx, vy, plan, seed); **zero chunk lookups**. Vastly cheaper than chunk-walking (flamegraph: was 72% of program time pre-optimisation).
2. **Chunk-walking fallback** (`surface_height_from_chunks`, voxel.rs:503) — used when no region_plan. Walks chunk-z-slices top-down; one HashMap lookup per chunk-z. Returns `sea_level` if no solid found.
3. **Cached paths** (`surface_grid` dense tile → `surface_cache` sparse HashMap) — used by exploration scans to avoid recomputing even the analytical path ~16K times per cell.

The ground-snap cascade rule (see `proposal_dsl_surface.md` §7) reads `surface_height(pos.xy)` for outdoor agents and `floor_height(pos, building_id)` for indoor agents, then sets `pos.z = h + creature_height/2`.

### Characteristics
- **Update frequency:** chunks added lazily (`generate_chunk` / `ensure_loaded_around`). Voxels mutated by harvest, construction, structural_tick, damage.
- **GPU-friendliness:** chunks are inherently chunked (already the right shape). HashMap wrapper is hostile; indexed/pool-allocated chunks would be GPU-friendly. Dense `Vec<Voxel>` per chunk is GPU-upload-ready (16 bytes/voxel, 4 MB/chunk).
- **Derived?** No — primary state. `region_plan` + seed regenerates chunks, but once voxels are edited, those edits are primary.

**Role for 3D agent positions.** `Agent.pos: vec3` is authoritative. Agents with volumetric `creature_type` (Dragon, Fish, Bat) can be placed anywhere inside the voxel chunk grid — inside caverns, underwater, in a flying island, inside a dungeon chamber — without a separate indoor/outdoor state flag. Ground-locked agents resolve their `pos.z` via the snap cascade (outdoor `surface_height` or indoor `floor_height` per `inside_building_id`). Spatial queries (`query::nearby_agents`) chunk against the voxel grid in 3D by default. [OPEN] whether the spatial hash is 3D-chunked across full voxel space, or 2D-grid keyed on chunk-column with per-cell z-range buckets — pick one per `proposal_dsl_surface.md` §9.

---

## WorldState.region_plan (RegionPlan)

**Type:** `Option<RegionPlan>` in WorldState (region_plan.rs:67). Also stored by value in `VoxelWorld.region_plan`.
**Purpose:** Continental-scale biome plan — grid of `RegionCell`s classifying terrain, settlements, dungeons, plus polyline rivers and road segments. Stored so chunk generation can reference the same plan after world creation.

### RegionPlan fields
| Field | Type | Meaning |
|---|---|---|
| cols, rows | usize | Grid dimensions |
| cells | `Vec<RegionCell>` | Row-major biome grid |
| rivers | `Vec<RiverPath>` | Polyline rivers in voxel-space |
| roads | `Vec<RoadSegment>` | Road segments in voxel-space |
| seed | u64 | Worldgen seed (determinism anchor) |

### RegionCell (region_plan.rs:52)
| Field | Type | Meaning |
|---|---|---|
| height | f32 | Normalised elevation [0,1] |
| moisture | f32 | Normalised moisture [0,1] |
| temperature | f32 | Normalised temperature [0,1] |
| terrain | Terrain | Biome classification (Plains, Forest, Mountains, Coast, Swamp, Desert, Tundra, Volcano, DeepOcean, Jungle, Glacier, Caverns, Badlands, FlyingIslands, DeathZone, AncientRuins, CoralReef) |
| sub_biome | SubBiome | Variant within biome (Standard, LightForest, DenseForest, AncientForest, SandDunes, RockyDesert, HotSprings, GlowingMarsh, TempleJungle, …) |
| settlement | `Option<SettlementPlan>` | Planned settlement in this cell |
| dungeons | `Vec<DungeonPlan>` | Planned dungeons |
| has_road | bool | Whether a road passes through |

### SettlementPlan (region_plan.rs:12)
| Field | Type | Meaning |
|---|---|---|
| kind | SettlementKind | `Town` / `Castle` / `Camp` / `Ruin` |
| local_pos | (f32, f32) | Normalised position within cell [0,1) |

### DungeonPlan (region_plan.rs:27)
| Field | Type | Meaning |
|---|---|---|
| local_pos | (f32, f32) | Normalised position within cell |
| depth | u8 | Dungeon depth tier |

### RiverPath (region_plan.rs:35)
| Field | Type | Meaning |
|---|---|---|
| points | `Vec<(f32, f32)>` | Polyline in voxel-space |
| widths | `Vec<f32>` | Per-vertex widths (parallel array to points) |

### RoadSegment (region_plan.rs:42)
Straight-line segment: `from`, `to` as `(f32, f32)` in voxel-space.

### Characteristics
- **Update frequency:** static after init. `#[serde(skip)]` on WorldState — regenerated from seed on load.
- **GPU-friendliness:** friendly. Cells are a flat `Vec`, rivers/roads can flatten via `to_gpu_cells()`, `to_gpu_rivers()` (already implemented for `feature="app"`).
- **Derived?** Deterministic function of seed — effectively primary (source of all terrain gen), but regenerable.

---

## WorldState.nav_grids

**Type:** `Vec<NavGrid>` (state.rs:545, nav_grid.rs:15)
**Purpose:** Baked 2D walkable surfaces derived from `VoxelWorld`. One per settlement area. Pathfinding (A*, flow fields) operates on NavGrid, not on VoxelWorld.

### NavGrid fields
| Field | Type | Meaning |
|---|---|---|
| origin_vx, origin_vy | i32 | Min corner in voxel-space |
| width, height | u32 | Grid dimensions |
| nodes | `Vec<NavNode>` | Row-major `(dy·width + dx)` |

### NavNode
| Field | Type | Meaning |
|---|---|---|
| walkable | bool | Whether a walker can stand here |
| surface_z | i32 | Z of walkable surface |
| move_cost | f32 | Material-based cost; 0 for non-walkable |

### Characteristics
- **Update frequency:** rebuilt when `VoxelWorld` has structural changes in the area. `buildings.rs:rebake_nav_grids` walks the column from max_z down for each (x,y) in the footprint. Called after construction completes and when test setups add buildings.
- **Writers:** `buildings.rs:212,213` (push new nav), `buildings.rs:960,961` (test path), `buildings.rs:197` (iterate-and-rebake).
- **GPU-friendliness:** friendly — flat `Vec<NavNode>` row-major.
- **Derived?** **Yes — pure function of `voxel_world` contents.** Can be rebuilt at any time via `NavGrid::bake(world, origin, w, h, max_z)`. Serialized for convenience.

---

## FidelityZone

**Type:** `FidelityZone` (state.rs:4009), stored in `WorldState.fidelity_zones: Vec<FidelityZone>`
**Purpose:** Proximity bubble controlling simulation fidelity around a point. Entities inside the zone run at its fidelity level (High/Medium/Low/Background).

### Fields
| Field | Type | Meaning | Updated by | Read by |
|---|---|---|---|---|
| id | u32 | Unique zone ID | spawn (`tick.rs:366,384,405`, `compute_high.rs:168`, `world_sim_cmd.rs:2907`) | grid/zone lookups |
| fidelity | Fidelity | `High` / `Medium` / `Low` / `Background` | `exploration::compute_exploration_for_settlement` (EscalateFidelity delta), threat systems | tick fidelity dispatch (`runtime.rs:hot_entity_fidelity`) |
| center | vec3 | World-space center | spawn | `update_grid_membership` (proximity test; radius is 3D Euclidean) |
| radius | f32 | Zone radius | spawn | membership |
| entity_ids | `Vec<u32>` | Entity IDs currently inside | `runtime::update_grid_membership` (rewritten each tick) | `has_hostiles`, `has_friendlies` queries |

### Characteristics
- **Update frequency:** `entity_ids` re-populated every tick in `update_grid_membership`. Zones themselves added at init or on escalation.
- **GPU-friendliness:** mixed — scalar fields are friendly; `entity_ids` is a `Vec<u32>` per zone. Acceptable.
- **Derived?** `entity_ids` is **derived** (recomputed from entity positions + zone center/radius each tick). The zone definition (id/fidelity/center/radius) is primary.

---

## BuildSeed

**Type:** `BuildSeed` (state.rs:399), stored in `WorldState.build_seeds: Vec<BuildSeed>`
**Purpose:** A placed room-growth seed. NPC sets a seed, the room-growth automaton (construction.rs) enlarges outward until enclosed.

### Fields
| Field | Type | Meaning |
|---|---|---|
| pos | TilePos | Seed tile position |
| intended_function | BuildingFunction | What room function this targets (drives `minimum_interior`) |
| minimum_interior | u32 | Required interior tile count for completion |
| placed_by | u32 | Entity ID that set the seed |
| tick | u64 | Tick placed |
| complete | bool | Marked true when grown successfully, OR when stalled past `MAX_SEED_ATTEMPTS` |
| attempts | u16 | Room-growth attempts performed — stall detection |
| last_interior_size | u16 | Last observed interior size; if unchanged across attempts, seed is stalled |

### Characteristics
- **Update frequency:** added by `action_eval.rs:1178` (NPC PlaceBuildSeed action). Mutated by `construction.rs` room-growth (attempts, last_interior_size, complete). Pruned when `complete=true`.
- **GPU-friendliness:** tiny — flat `Vec<BuildSeed>`, ~40 bytes each, easily GPU-shaped.
- **Derived?** Primary state — must persist.

---

## StructuralEvent

**Type:** `StructuralEvent` (state.rs:436), stored in `WorldState.structural_events: Vec<StructuralEvent>`
**Purpose:** Per-tick events emitted by voxel collapse/fracture. **Cleared at tick start** (`runtime.rs:1540`) — ephemeral.

### Variants
- `FragmentCollapse { chunk_x, chunk_y, chunk_z, fragment_voxel_count: u32, cause: CollapseCase }`
- `StressFracture { chunk_x, chunk_y, chunk_z, cluster_mass: f32, material_strength: f32 }`

### CollapseCase enum (state.rs:428)
- `NpcHarvest` — caused by voxel_harvest (tree felling, mining)
- `NpcConstruction` — placed a voxel that destabilised neighbours
- `Natural` — organic collapse from structural_tick

### Characteristics
- **Update frequency:** per-tick buffer. Cleared at start of each tick (`runtime::tick`). Appended by `structural_tick.rs:71`.
- **GPU-friendliness:** small — tagged union. OK.
- **Derived?** Consumed same tick; technically a per-tick output buffer, not persistent state.

---

## SimScratch (NOT persistent state)

**Type:** `SimScratch` (state.rs:344)
**Purpose:** Pooled scratch buffers reused across tick systems. **Not persistent. Cleared + refilled within a single function call.** Avoids per-tick Vec/HashMap allocations (pre-pooling: ~55 page faults per tick, 220KB/tick allocator churn).

`Clone` returns `Default` intentionally — cloning a WorldState shouldn't duplicate scratch allocations.

### Sub-buffer ownership
| Buffer | Owner | Purpose |
|---|---|---|
| snaps | `action_eval::evaluate_and_act` | Read-only entity snapshots for scoring |
| snap_grid | `action_eval` | Snap indices by spatial cell |
| snap_grids_typed | `action_eval` | Kind-typed spatial grids (resources/buildings/combatants) |
| deferred | `action_eval` | Deferred action decisions |
| npc_indices | `exploration::scan_all_npc_resources` | NPC indices to scan |
| npc_pos_voxel | `exploration` | NPC voxel positions cached for step 3 |
| visible_cells | `exploration` | Visible cell set |
| flood_visited | `construction::flood_fill_with_boundary` | Generational visited tag (128×128 flat grid, u16 gen) |
| flood_current_gen | `construction` | Current generation tag |
| flood_queue | `construction` | BFS queue |
| flood_interior | `construction` | BFS result interior |
| flood_boundary | `construction` | BFS result boundary |

### Characteristics
- **Update frequency:** cleared + filled within one function call each.
- **GPU-friendliness:** N/A — CPU-side pools.
- **Derived?** Ephemeral. Never read across boundaries.

---

## GroupIndex

**Type:** `GroupIndex` (state.rs:1032)
**Purpose:** Contiguous per-settlement / per-party entity ranges. After `rebuild_group_index()`, entities are sorted by `(settlement_id, party_id)` so settlement members are adjacent. Systems iterate a slice instead of scanning all entities.

### Fields
| Field | Type | Meaning |
|---|---|---|
| settlement_ranges | `Vec<(u32,u32)>` | `[sid] = (start, end)` into entities — all kinds |
| settlement_npc_ranges | `Vec<(u32,u32)>` | Per-settlement NPC sub-range |
| settlement_building_ranges | `Vec<(u32,u32)>` | Per-settlement Building sub-range |
| settlement_monster_ranges | `Vec<(u32,u32)>` | Per-settlement Monster sub-range |
| party_ranges | `Vec<(u32,u32)>` | Per-party ranges |
| unaffiliated_range | (u32,u32) | Entities not assigned to any settlement |

### Accessors
`settlement_entities(sid)`, `settlement_npcs(sid)`, `settlement_buildings(sid)`, `settlement_monsters(sid)`, `party_entities(pid)`, `unaffiliated_entities()` — all return `Range<usize>`.

### Characteristics
- **Update frequency:** rebuilt by `rebuild_group_index()` after structural entity changes. Called in `rebuild_all_indices` after spawn/despawn. Runtime invokes when `entities.len() != hot.len()` (runtime.rs:1764).
- **GPU-friendliness:** friendly — flat `Vec<(u32,u32)>` arrays.
- **Derived?** **Yes — purely derived from `entities` + `settlement_id`/`party_id` membership.** `#[serde(skip)]`-able (currently serialised with Default).

---

## SurfaceCache (exploration.rs:222)

**Type:** `pub type SurfaceCache = HashMap<u64, i32, ahash::RandomState>`
**Key:** packed `(vx as u32) << 32 | vy as u32` via `pack_xy`.
**Value:** surface z-height.

- Populated lazily by `scan_voxel_resources_cached` on HashMap miss.
- Fallback for positions outside any FlatSurfaceTile in `surface_grid`.
- Persistent across ticks (world is mostly static; valid as long as region_plan + seed unchanged).
- **Derived — pure function of (vx, vy, region_plan, seed).** Regenerates on demand.
- `#[serde(skip)]`.

## FlatSurfaceGrid / FlatSurfaceTile (exploration.rs:229, 253)

### FlatSurfaceTile
| Field | Type | Meaning |
|---|---|---|
| origin_x, origin_y | i32 | Tile origin in voxel-space |
| width, height | i32 | Tile dimensions |
| heights | `Vec<i16>` | Row-major `(dy·width + dx)` surface z-heights |

### FlatSurfaceGrid
| Field | Type | Meaning |
|---|---|---|
| tiles | `Vec<FlatSurfaceTile>` | One tile per settlement region |

- Populated by `warm_surface_cache` per settlement (exploration.rs:497).
- `FlatSurfaceGrid::get(vx,vy)` linearly scans tiles — fine for small N (1–10 settlements).
- **30× faster than HashMap lookup** and 10× less memory; on-cache-hit path for per-cell census.
- **Derived** — rebuilt from analytical path.
- **GPU-friendly** — flat `Vec<i16>` row-major.
- `#[serde(skip)]`.

## CellCensus (exploration.rs:203)

**Type:** `pub type CellCensus = HashMap<(i32,i32), [u32; 6], ahash::RandomState>`
**Key:** `(cell_x, cell_y)` where each cell spans `RESOURCE_CELL_SIZE = 128` voxels.
**Value:** `[count_wood, count_iron, count_copper, count_gold, count_coal, count_crystal]` — 6 target materials.

- Populated lazily when any NPC can see the cell (`scan_voxel_resources_cached`, exploration.rs:705).
- Persistent for the run. Invalidation on voxel edits is a known tech-debt refinement; resources change slowly and NPCs reconfirm on harvest.
- **Derived** — pure function of chunk contents in cell's surface band. Rebuildable.
- **GPU-friendliness:** hostile (HashMap). Could migrate to a flat grid since cells align to 2×CHUNK_SIZE.
- `#[serde(skip)]`.

---

## WorldState.chronicle (narrative log)

**Type:** `Vec<ChronicleEntry>` (state.rs:571; entry at state.rs:4893)
**Purpose:** Bounded ring buffer of narrative events — what happened in the world. Human-readable.

### ChronicleEntry
| Field | Type | Meaning |
|---|---|---|
| tick | u64 | When it happened |
| category | ChronicleCategory | Battle / Quest / Diplomacy / Economy / Death / Discovery / Crisis / Achievement / Narrative |
| text | String | Human-readable text |
| entity_ids | `Vec<u32>` | Entities involved |

### Characteristics
- **Update frequency:** appended by ~20 systems (battles, quests, legends, family, death, warfare, settlement_founding, prophecy, oaths, outlaws, sea_travel, etc.).
- **GPU-friendliness:** hostile — contains `String`.
- **Derived?** Primary (lore state; append-only), bounded.

## WorldState.world_events

**Type:** `Vec<WorldEvent>` (state.rs:579; enum at state.rs:4919)
**Purpose:** Recent events for system queries (bounded).

### WorldEvent variants
`Generic{category,text}`, `EntityDied{entity_id,cause}`, `QuestChanged{quest_id,new_status}`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.

### Characteristics
- **Update frequency:** appended by relevant systems, bounded.
- **GPU-friendliness:** hostile — some variants contain `String` (`Generic`, `EntityDied`).
- **Derived?** Primary (event log).

---

## Summary

- **Primary state** (irreplaceable): `tick`, `rng_state`, `next_id`, `tiles`, `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan` (regenerable from seed), `build_seeds`, `chronicle`, `world_events`, `fidelity_zones` (zone definitions — id/fidelity/center/radius), `structural_events` (per-tick buffer).
- **Derived state** (rebuildable from primary): `entity_index`, `group_index`, `settlement_index`, `surface_cache`, `surface_grid`, `cell_census`, `nav_grids`, `max_entity_id`, `fidelity_zones[].entity_ids` (recomputed every tick in `update_grid_membership`).
- **Scratch state** (ephemeral, pooled, NOT persistent): `SimScratch` and all sub-buffers — cleared + filled + consumed within one function call.
- **GPU-hostile shapes:** `HashMap<TilePos, Tile>`, `HashMap<ChunkPos, Chunk>`, `HashMap<u64, i32>` (surface_cache), `HashMap<(i32,i32),[u32;6]>` (cell_census), `Vec<ChronicleEntry>`/`Vec<WorldEvent>` (contain `String`), `VecDeque<TilePos>` (flood_queue).
- **GPU-friendly shapes already in place:** `Chunk.voxels: Vec<Voxel>` (dense, 16B/voxel), `NavGrid.nodes: Vec<NavNode>` (row-major), `FlatSurfaceTile.heights: Vec<i16>` (row-major), `RegionPlan.cells: Vec<RegionCell>` (flat, with `to_gpu_cells()` already implemented), `RiverPath.points: Vec<(f32,f32)>` with `to_gpu_rivers()`, `GroupIndex` range arrays.
- **Candidates for flat-grid conversion:**
  - `tiles` (spatially dense within settlements; currently hashing per neighbor in flood_fill / is_enclosed / has_door).
  - `voxel_world.chunks` (inherently chunked — just needs indexed/slab-pooled chunk storage rather than HashMap).
  - `cell_census` (cells align to 2×CHUNK_SIZE; a flat grid keyed by cell coord would be GPU-friendly).
  - `surface_cache` is already superseded by `surface_grid` on the hot path.
