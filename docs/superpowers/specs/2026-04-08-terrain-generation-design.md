# 3D Terrain Generation — Voxel World Overhaul

**Date:** 2026-04-08
**Status:** Draft

## Goal

Replace the flat, boring voxel terrain generation with a rich 3D world featuring biomes, caves, dungeons, mountains, flying islands, and rivers. Delete the 2D `OverworldGrid` entirely — the voxel world becomes the single source of truth, driven by a lightweight region plan.

## Architecture

Two-phase generation with 3D spatial partitioning:

```
Phase 1: Continental Plan (runs once)
  FBM noise → heightmap + moisture → biome classification
  Mountain ridges, river networks, settlements, roads
  Stored as lightweight 2D grid (~50×30 sector columns)

Phase 2: Chunk Materialization (on sector load)
  Sample region plan → resolve 3D biome volume → build voxel column
  Carve caves, rivers, dungeons → place features (trees, boulders, etc.)
```

### Scale Reference

NPC = 10 voxels wide (~2m real-world equivalent).

| Thing | Size (voxels) | Real-world equiv |
|-------|--------------|------------------|
| NPC | 10 wide, 20 tall | ~2m |
| Tree trunk | ~5 wide | ~1m |
| Tree canopy | ~20 wide | ~4m |
| Road | ~20 wide | ~4m |
| River | 50–100 wide | 10–20m |
| House | 150×150 | ~30m |
| Village | 1000–2000 | 200–400m |
| Battlefield | 2000–4000 | 400–800m |
| Mountain peak | 500–1500 tall | 100–300m |
| Region of interest | ~5000–10000 | 1–2km |
| Full continent | ~200,000×120,000 | ~40×24km |

## 3D Sectors

Sectors are the spatial partition unit for chunk management and LOD. They are 3D volumes, not 2D columns.

**Sector dimensions:** 4096×4096×4096 voxels (256×256×256 chunks per sector).

**Sector grid:** Continent is ~50×30 sector columns. Vertically, sectors stack:

| Sector Z | Contents | Typically active? |
|----------|----------|-------------------|
| +1 | Sky — flying islands, cloud layer | Rarely |
| 0 | Surface — terrain, buildings, most entities | Often |
| -1 | Shallow underground — caves, dungeons, mines | When explored |
| -2 | Deep underground — lava tubes, abyss, crystal caves | Rarely |

**Sector is a detail-level boundary, not a simulation boundary.** All entities tick globally via the existing world sim delta system regardless of sector activation. Sectors control which areas have loaded voxel chunks.

### Simulation LOD Tiers

| Tier | Radius | Loaded state | Physics |
|------|--------|-------------|---------|
| Active zone | ~2000–4000 units around action | Full voxel chunks, teardown physics | Structural integrity, destruction, debris, fluids |
| Loaded zone | ~10,000 units | Voxel chunks loaded, entities tick normally | Collision only, no structural physics |
| Abstract zone | Rest of continent | Region plan only, no voxels | Entities simulated abstractly (position, inventory, goals, economy) |

## Region Plan

The region plan replaces `OverworldGrid` as the authoritative 2D map. Generated once at world creation, stored in `WorldState`.

**Grid:** ~50×30 cells (one per sector column). Each cell covers 4096×4096 voxels horizontally.

**Per-cell data:**

```rust
pub struct RegionCell {
    pub height: f32,               // base elevation (0.0–1.0), scaled to voxel Z during materialization
    pub moisture: f32,             // drives biome selection with height
    pub terrain: Terrain,          // biome — reuses existing 17-variant enum from state.rs
    pub sub_biome: SubBiome,       // variation within biome
    pub river: Option<RiverSegment>, // direction + width if river passes through
    pub road: bool,                // road present
    pub settlement: Option<SettlementPlan>, // type + size if settlement here
    pub dungeon_sites: Vec<DungeonSite>,   // underground structures
    pub elevation_detail: f32,     // additional high-frequency height variation
}
```

### Continental Plan Generation (Phase 1)

Ported from existing `terrain_gen.rs` 6-pass pipeline, expanded:

1. **Continent shape** — large-scale noise mask defines land vs ocean. Irregular coastline with peninsulas, bays, offshore islands. No rectangle.
2. **Heightmap** — multi-octave FBM noise (5+ octaves, up from current 3). Produces mountain ranges, valleys, plains. Height range maps to voxel Z 0–400.
3. **Moisture map** — separate FBM. Combined with height → biome selection via `classify_terrain(height, moisture)`.
4. **Tectonic features** — mountain ridge tracing (ported from existing pass 2). Volcano placement at ridge intersections.
5. **River network** — trace downhill from mountain peaks to coast, merging tributaries. Stored as world-space polylines with width increasing downstream.
6. **Settlement placement** — ported from existing pass 5. Size/type influenced by biome, river proximity, and elevation.
7. **Road network** — ported from existing pass 6. Bresenham lines connecting settlements.

Output: `RegionPlan` (the 2D grid) plus polyline data for rivers and roads stored separately.

## 3D Biome Volumes

Biome is a function of (x, y, z), not just (x, y). The region plan provides the surface biome. Vertical position + noise select the volume biome.

### Biome Resolution

```rust
fn resolve_biome(plan: &RegionPlan, x: i32, y: i32, z: i32, seed: u64) -> BiomeVolume
```

| Depth Zone | Source | Examples |
|------------|--------|----------|
| Sky (z > surface + 200) | Surface biome + noise | Flying islands, empty air |
| Surface (±40 of terrain height) | Region plan directly | Plains, forest, desert, mountain, etc. |
| Shallow underground (surface-40 to surface-80) | Surface biome influence + depth | Root caves under forest, sand caves under desert |
| Deep underground (below surface-80) | Depth + 3D noise, surface influence fades | Mushroom caverns, lava tubes, crystal caves, aquifers |
| Abyss (z < -100) | Pure depth-driven | Magma, bedrock |

### Underground Biome Variants

Extend `SubBiome` with underground types:

| Underground Biome | Walls | Features | Found under |
|-------------------|-------|----------|-------------|
| NaturalCave | Stone | Stalactites, stalagmites | Default everywhere |
| LavaTubes | Basalt | Lava pools, obsidian | Volcano, mountains |
| FrozenCavern | Ice | Frozen lakes, icicles | Tundra, glacier |
| MushroomGrove | Stone + dirt | Bioluminescent mushrooms, organic | Forest, jungle |
| CrystalVein | Stone | Crystal clusters, high ore density | Rare, noise-gated |
| Aquifer | Clay + stone | Flooded chambers, water | Swamp, coast |
| BoneOssuary | Bone + stone | Ancient remains | Death zone, ancient ruins |

### Surface Biome Materials

| Biome | Subsoil | Surface | Features |
|-------|---------|---------|----------|
| Plains/Grassland | Dirt | Grass | Occasional boulders |
| Forest | Dirt | Grass | Trees (trunk + canopy volumes) |
| Desert | Sand | Sand | Dunes (heightmap warping), sandstone below |
| Mountains | Stone | Stone/Snow | Steep slopes, exposed rock faces, snow above threshold |
| Swamp/Marsh | Clay | Grass/Water | Shallow pools, dead trees |
| Tundra/Glacier | Gravel | Snow/Ice | Frozen lakes |
| Volcano | Basalt | Basalt/Lava | Crater, lava pools |
| Coast | Sand | Sand | Beach slope toward water |
| Jungle | Dirt | Grass | Dense tall trees, undergrowth |
| Caverns | Stone | Stone | Surface holes leading to cave systems below |
| FlyingIslands | Normal ground below | Normal ground below | Sky sector gets floating chunks with vegetation on top, waterfalls off edges |
| DeepOcean | Sand/Clay | Water | Submerged terrain, coral in CoralReef variant |
| Badlands | Sandstone | Sandstone/Gravel | Eroded pillars, canyons |
| AncientRuins | Stone | Stone/Grass | Crumbling structures, overgrown |
| DeathZone | Bone/Stone | Barren | Dead vegetation, corruption |

## Chunk Materialization (Phase 2)

When a 3D sector activates, generate its chunks on demand.

### Per-chunk generation steps

1. **Sample region plan** at each (x,y) column → biome, interpolated height, moisture, river/road info.
2. **Build base column** — determine surface Z from heightmap. Fill layers bottom-up:
   - Bedrock (granite) at bottom
   - Deep stone with ore veins (ore type varies by depth + biome)
   - Subsoil (biome-dependent material)
   - Surface (biome-dependent top layer, 1–2 voxels)
   - Air above surface
   - Water fills to sea level where surface < sea level
3. **Resolve 3D biome volume** — for each voxel below surface, determine underground biome from surface biome + depth + 3D noise.
4. **Carve caves** — dual-threshold 3D worm noise. Where both noise fields exceed threshold, carve to air. Cave frequency and size vary by underground biome. Cavern biome regions get extra-large open chambers.
5. **Place surface features** — trees, boulders, dunes, ice formations, dead trees per surface biome. Deterministic from seed + position.
6. **Carve rivers** — follow polyline from region plan. Cut a valley into terrain, fill channel with water. Width from plan, depth proportional to width.
7. **Generate flying islands** (sky sectors only) — inverted-cone SDF shapes with noise perturbation at z 300+. Grass and trees on top, water columns (waterfalls) dripping off edges.
8. **Stamp dungeons** — at `DungeonSite` locations, carve room + corridor structures underground. Place entrance at surface.

### Cave Generation Detail

Caves use the "worm" technique: two independent 3D noise fields (different seeds). A cave exists where both fields are within a narrow band (e.g., `|noise_a| < 0.05 && |noise_b| < 0.05`). This produces connected worm-like tunnels.

Parameters vary by underground biome:
- **NaturalCave:** standard frequency, moderate tunnel width
- **LavaTubes:** lower frequency (longer straight sections), wider
- **MushroomGrove:** very large open chambers (wider threshold band)
- **CrystalVein:** tight veins, high frequency
- **Aquifer:** moderate caves, flooded with water

### Mountain Detail

Mountains are primarily driven by the heightmap (tall columns), but chunk materialization adds:
- **Cliff faces** — where horizontal height gradient is steep, expose raw stone instead of soil/grass
- **Ridgeline noise** — small-scale noise along ridge crests for jagged peaks
- **Snow line** — above a Z threshold (varies by latitude/biome), surface switches to snow
- **Scree slopes** — gravel on steep non-cliff faces

### River Carving Detail

Rivers are stored as world-space polylines with per-point width. During chunk materialization:
- For each (x,y) column, find distance to nearest river segment
- If within river width: carve terrain down to river bed depth, fill with water
- River banks slope gradually (not vertical cuts)
- Width increases downstream (tributaries merge)

## Noise Functions

Port and expand the existing noise library from `terrain_gen.rs`:

| Function | Use | Notes |
|----------|-----|-------|
| `hash_3d(x, y, z, seed)` | Base hash | Already exists in voxel.rs |
| `value_noise_2d(x, y, seed)` | Smooth 2D interpolation | Port from terrain_gen.rs |
| `value_noise_3d(x, y, z, seed)` | Smooth 3D interpolation | New |
| `fbm_2d(x, y, seed, octaves, lacunarity, gain)` | Heightmap, moisture | Port + parameterize |
| `fbm_3d(x, y, z, seed, octaves, lacunarity, gain)` | Cave shapes, underground biome boundaries | New |
| `worm_noise(x, y, z, seed_a, seed_b)` | Cave carving | New — dual-field technique |
| `domain_warp(x, y, seed)` | River meander, coastline distortion | New |

All noise is deterministic from seed. No external RNG.

## Deletions

### Remove entirely:
- `src/overworld_grid/mod.rs` — `OverworldGrid`, `TerrainType` enum, `MapCell`
- `src/overworld_grid/terrain_gen.rs` — logic ported to new `terrain/region_plan.rs`
- Any remaining references to `OverworldGrid` in world sim systems

### Replace:
- `VoxelWorld::generate_chunk()` in `voxel.rs` — current flat terrain gen replaced by call into `terrain/materialize.rs`

### Keep:
- `Terrain` enum in `world_sim/state.rs` (17 variants) — becomes the biome enum for region plan
- `SubBiome` enum — extended with underground variants
- `VoxelMaterial` enum — used as-is, potentially add: `Mushroom`, `Vine`, `Stalactite`, `Coral`
- `VoxelWorld`, `Chunk`, coordinate math — all kept
- `RegionState` in `state.rs` — adapted to read from region plan
- World sim systems (geography.rs, terrain_events.rs) — adapted to read from region plan instead of overworld grid

## New File Layout

```
src/world_sim/
  terrain/
    mod.rs            — public API: generate_continent(), materialize_chunk()
    region_plan.rs    — RegionPlan grid, per-cell data, Phase 1 generation passes
    biome.rs          — 3D biome volume resolution, biome→material tables
    materialize.rs    — chunk materialization (Phase 2), column building
    caves.rs          — 3D cave carving (worm noise, cavern types)
    rivers.rs         — river valley carving from polyline data
    features.rs       — surface features (trees, boulders, dunes, etc.)
    sky.rs            — flying island generation (SDF + noise)
    dungeons.rs       — underground structure generation (rooms + corridors)
    noise.rs          — noise functions (FBM, worm, value, domain warp, hash)
  sector.rs          — Sector struct, 3D sector grid, activation/deactivation, LOD tiers
  voxel.rs           — existing, generate_chunk() delegates to terrain/materialize.rs
```

## Integration with Voxel Engine Spec

This spec is complementary to the "Voxel Engine Integration" spec (2026-04-08). That spec defines how voxels get rendered (VoxelBridge → Scene). This spec improves what `generate_chunk()` produces. The interface is unchanged: `VoxelWorld` still has `Chunk`s with `Voxel`s, the bridge still syncs dirty chunks to the Scene.

## Success Criteria

1. Standing at any point in the world, the terrain looks distinct based on biome — not just flat grass everywhere.
2. Mountains have real elevation (hundreds of voxels tall), with snow caps, cliff faces, and ridgelines.
3. Rivers carve through terrain with sloped banks and water-filled channels.
4. Going underground reveals caves with biome-appropriate character (lava tubes under volcanoes, mushroom caves under forests, etc.).
5. Flying islands float in the sky with vegetation and waterfalls.
6. Dungeons exist underground with carved rooms and corridors.
7. Biome transitions are smooth, not hard grid lines.
8. All generation is deterministic from seed.
9. `OverworldGrid` is fully removed. No 2D grid remains.
10. Existing world sim systems continue to function, reading biome/terrain data from the region plan.
11. Chunk generation is fast enough that sector loading doesn't cause visible hitches (<50ms per chunk).
