# `--world small` Test Scene

**Date:** 2026-04-11
**Purpose:** A self-contained 9x9x9 chunk forest world with a seeded settlement, for iterating on NPC building and settlement behavior without continent-scale generation.

## Overview

The voxel app gains a `--world small` CLI flag. When set, the app creates a fixed 576³-voxel (57.6m per side) world, pre-generates all 729 chunks, seeds a settlement with NPCs at the center clearing, and runs the normal simulation loop. The infinite-world path is unchanged when `--world` is absent.

## World Generation

### Region plan

A dedicated `create_small_world_plan(seed)` function builds a `RegionPlan` sized to cover exactly the 9x9x9 chunk footprint (576×576 voxels horizontally). The plan uses a single `CELL_SIZE`-aligned cell (or a small grid if the footprint exceeds one cell) with:

- `terrain: Terrain::Forest`
- `sub_biome: SubBiome::LightForest`
- `settlement: Some(SettlementPlan)` on the center cell

The seed is fixed (deterministic scene) or user-provided via `--seed`.

### Clearing

The settlement center is at world position `(288, 288)` — the horizontal midpoint of the 576×576 footprint. During feature placement (both CPU `place_surface_features` and GPU `main_features`), tree/boulder/pillar density is suppressed within a radius around the settlement center. The suppression radius is ~2 chunks (~128 voxels, ~12.8m) to give NPCs open ground while leaving 3+ chunks of dense forest buffer on each side. The suppression is a smooth falloff (linear ramp over ~32 voxels at the edge), not a hard circle, so the forest-to-clearing transition looks natural.

Implementation: `FeatureParams` density values are scaled by a distance factor from the settlement center. At distance 0 the factor is 0 (no trees); at the suppression radius the factor reaches 1 (full density). This check runs per-origin in the feature pass, before the density hash.

### Chunk pre-generation

On startup, all 729 chunks are generated via the CPU path (`materialize_chunk`) and inserted into `VoxelWorld.chunks`. The GPU terrain compute pipeline then processes them for rendering. The disk loader's spiral is replaced by a single bulk submission of all 729 chunk positions.

## Chunk Boundaries

### No chunks outside the world

The world extent is `[0, 9)` in chunk coordinates on all three axes, or `[0, 576)` in voxel coordinates. `generate_chunk` rejects positions outside this range. `generate_camera_chunks` clips its disk to the world extent — any chunk position outside `[0, 9)` is skipped. No pool slots are wasted on out-of-bounds chunks.

### Camera clamping

Each frame, the camera's eye position is clamped to the world's voxel AABB in sim coordinates: `[margin, 576 - margin]` on each axis, where `margin` is a small buffer (e.g., 2 voxels) to avoid clipping into the boundary face. This is applied in `update_camera` after movement integration.

## Settlement Seeding

### Settlement entity

A `Settlement` entity is created at world center `(288, 288, surface_z)` with:

- A unique `settlement_id`
- Name (e.g., "Clearbrook")
- Initial zone layout: a single `Residential` zone covering the clearing

### NPCs

8–12 NPCs are spawned at the settlement center at startup:

- All assigned `home_settlement_id` pointing to the settlement
- Mix of skills appropriate for early settlement (builder, woodcutter, farmer, etc.)
- `GoalKind::Build` set as initial goal for builders
- Starting positions scattered within the clearing

### Build progression

The existing `process_npc_builds` system runs each sim tick. NPCs with build goals select build sites within their settlement's zones, stamp buildings into voxels, and assign themselves to completed structures. No new NPC or building logic is needed — the test scene exercises the existing systems.

## CLI Integration

### Argument parsing

The voxel app's startup (in `voxel_app.rs` or its caller) parses:

```
--world <name>    World preset (default: infinite)
--seed <u64>      World seed (default: fixed value for reproducibility)
```

Recognized `--world` values:

- `small` — 9x9x9 forest + settlement scene (this spec)
- (absent or `infinite`) — current behavior, unchanged

### Startup path

When `--world small`:

1. `create_small_world_plan(seed)` → `RegionPlan`
2. Pre-generate all 729 chunks via CPU path
3. Create settlement entity + NPCs
4. Set `VoxelWorld.region_plan`
5. Set world extent on the app state (used by camera clamp + chunk loader clip)
6. Submit all chunks to GPU pool
7. Enter normal run loop

## What's NOT in Scope

- New building types, NPC behaviors, or AI changes
- Terrain LOD or distant rendering (everything is loaded)
- Persistence / save-load (in-memory, regenerated each launch)
- Other `--world` presets beyond `small`
- Multiplayer or networking considerations
