# VoxelGrid Redesign — Design Spec

**Date:** 2026-04-04
**Goal:** Replace the 2D settlement-tied CityGrid with a 3D octree-based VoxelGrid, rename LocalGrid to FidelityZone, and separate pathfinding into a baked NavGrid.

---

## 1. VoxelGrid Core

`VoxelGrid` is a sparse 3D voxel volume positioned in world space. Storage is an octree with a `Natural` variant for procedurally-generated terrain that hasn't been modified.

```rust
pub struct VoxelGrid {
    pub id: u32,
    pub origin: (f32, f32, f32),  // world-space anchor (min corner)
    pub root_size: u32,           // power of 2 (e.g. 128)
    pub max_depth: u8,            // leaf size = root_size / 2^max_depth
    pub cell_size: f32,           // world units per leaf cell
    pub root: OctreeNode,
}

pub enum OctreeNode {
    /// Unmodified region — fall through to terrain function
    Natural,
    /// Uniform modified region (all cells same voxel)
    Leaf(Voxel),
    /// Subdivided — 8 children [−x−y−z, +x−y−z, −x+y−z, +x+y−z, −x−y+z, +x−y+z, −x+y+z, +x+y+z]
    Branch(Box<[OctreeNode; 8]>),
}
```

A fresh grid is a single `Natural` root — zero voxel data. Modifications subdivide only the affected octants. Grids are not tied to settlements; any spatial area can have a VoxelGrid.

### Voxel

```rust
pub struct Voxel {
    pub material: Material,
    pub integrity: f32,             // 0.0 (destroyed) to 1.0 (full)
    pub building_id: Option<u32>,   // if part of a building
    pub zone: VoxelZone,            // functional designation
}

pub enum VoxelZone {
    None,
    Residential,
    Commercial,
    Industrial,
    Military,
    Agricultural,
    Sacred,
    Underground,
}
```

---

## 2. Material System

26 materials across 5 categories. Properties are a flat struct — adding a new property means adding a field.

```rust
pub enum Material {
    // Natural
    Air, Stone, Granite, Basalt, Sandstone, Marble,
    Dirt, Clay, Sand, Gravel, Ice,
    // Organic
    Wood, Thatch, Bone,
    // Worked
    Brick, CutStone, Concrete, Glass, Ceramic,
    // Metal
    Iron, Steel, Bronze, Copper, Gold,
    // Exotic
    Obsidian, Crystal,
}

pub struct MaterialProperties {
    pub hp_multiplier: f32,
    pub fire_resistance: f32,
    pub load_bearing: bool,
    pub weight: f32,
    pub rubble_move_cost: f32,
    pub construction_cost: f32,
    pub blast_resistance: f32,
}
```

Each `Material` variant maps to a `MaterialProperties` via a `fn properties(material: Material) -> MaterialProperties` lookup. All properties are on the struct — no per-variant trait implementations.

---

## 3. Coordinate System

Grids are world-positioned via `origin`. No settlement parameter needed.

```rust
impl VoxelGrid {
    /// Grid coords (u32, u32, u32) → world position (f32, f32, f32)
    pub fn grid_to_world(&self, gx: u32, gy: u32, gz: u32) -> (f32, f32, f32) {
        (
            self.origin.0 + gx as f32 * self.cell_size,
            self.origin.1 + gy as f32 * self.cell_size,
            self.origin.2 + gz as f32 * self.cell_size,
        )
    }

    /// World position → grid coords (clamped to grid bounds)
    pub fn world_to_grid(&self, wx: f32, wy: f32, wz: f32) -> (u32, u32, u32) {
        let gx = ((wx - self.origin.0) / self.cell_size).floor().max(0.0) as u32;
        let gy = ((wy - self.origin.1) / self.cell_size).floor().max(0.0) as u32;
        let gz = ((wz - self.origin.2) / self.cell_size).floor().max(0.0) as u32;
        (
            gx.min(self.root_size - 1),
            gy.min(self.root_size - 1),
            gz.min(self.root_size - 1),
        )
    }
}
```

Convention: X = east, Y = north, Z = up. Z=0 is the deepest underground layer of the grid.

---

## 4. Destructible Terrain

Per-cell HP via `integrity` field (0.0–1.0). Effective HP = `integrity * material.hp_multiplier`.

### Damage

```
effective_hp = integrity * properties(material).hp_multiplier
new_integrity = (effective_hp - damage) / properties(material).hp_multiplier
if new_integrity <= 0 → voxel becomes rubble or air
```

### Cascading Collapse

When a voxel is destroyed, check structural support for voxels above it.

**Support rule:** A voxel is supported if:
- The voxel directly below is solid (load-bearing material with integrity > 0), OR
- At least 2 of the 4 horizontal neighbors are solid

**Underground exception:** Voxels at Z=0 are always supported (bedrock).

**Cascade:** When a voxel loses support:
1. Set integrity to 0, convert to rubble
2. Check the voxel above — if it was supported only by this voxel, it also collapses
3. Repeat upward until a supported voxel is found or the top of the grid is reached

### Rubble

Destroyed load-bearing voxels become rubble (same material, integrity=0, zone=None). Rubble has a movement cost penalty via `material.rubble_move_cost`. Non-load-bearing voxels (Air, Glass) become Air when destroyed.

---

## 5. Natural Terrain Function

Procedural terrain is computed on demand, not stored in the octree.

```rust
/// Pure function: seed + world position → natural voxel.
/// Deterministic — same inputs always produce the same voxel.
fn natural_voxel_at(seed: u64, world_pos: (f32, f32, f32)) -> Voxel
```

### Octree Query

```rust
impl VoxelGrid {
    pub fn get(&self, pos: (u32, u32, u32), seed: u64) -> Voxel {
        match self.lookup(pos) {
            OctreeNode::Natural => natural_voxel_at(seed, self.grid_to_world(pos.0, pos.1, pos.2)),
            OctreeNode::Leaf(v) => v,
            // Branch: recurse into child octant
        }
    }
}
```

### Materialize on Write

Both building placement and terrain damage follow the same pattern:

1. If the target octant is `Natural`, query `natural_voxel_at()` to get the baseline
2. Subdivide the `Natural` node into a `Branch` with 7 `Natural` siblings + 1 target child
3. Write the modified voxel into the target child
4. Repeat subdivision down to leaf depth as needed

**Merge-back:** After modification, if all 8 children of a `Branch` are identical `Leaf` values, collapse to a single `Leaf`. Never merge back to `Natural` — once modified, always explicit.

### Key Methods

```rust
impl VoxelGrid {
    fn get(&self, pos: (u32, u32, u32), seed: u64) -> Voxel;
    fn set(&mut self, pos: (u32, u32, u32), voxel: Voxel);    // auto-splits Natural
    fn damage(&mut self, pos: (u32, u32, u32), amount: f32, seed: u64);  // materialize + damage
    fn is_modified(&self, pos: (u32, u32, u32)) -> bool;       // false for Natural regions
    fn modified_count(&self) -> usize;                          // number of non-Natural leaves
}
```

---

## 6. NavGrid

Baked walkable surface graph derived from VoxelGrid. Effectively 2D — each node is a surface position (x, y) with an implied z (the highest walkable surface at that column).

```rust
pub struct NavGrid {
    pub grid_id: u32,              // which VoxelGrid this was baked from
    pub width: u32,
    pub height: u32,
    pub nodes: Vec<NavNode>,       // width * height, indexed by (x, y)
}

pub struct NavNode {
    pub walkable: bool,
    pub surface_z: u32,            // z of the walkable surface
    pub move_cost: f32,            // base cost (material + rubble penalty)
}
```

### Baking

Walk each (x, y) column top-down. The first solid voxel found is the surface; the cell above it is the walkable position. `move_cost` comes from the surface material's `rubble_move_cost` (0 for normal surfaces).

### Rebuild

NavGrid is rebaked when the VoxelGrid undergoes structural changes (building placed, terrain destroyed, collapse). Only the affected columns need rebaking — tracked via a dirty flag or change list from the octree write.

### Pathfinding

A* and flow fields operate on NavGrid, not VoxelGrid directly. Same algorithms as the current CityGrid pathfinding, but sourced from NavGrid nodes.

Convenience methods for 2D queries:

```rust
impl NavGrid {
    fn find_path(&self, start: (u32, u32), end: (u32, u32)) -> Option<Vec<(u32, u32)>>;
    fn compute_flow_field(&self, target: (u32, u32)) -> Vec<(f32, f32)>;
    fn is_walkable(&self, x: u32, y: u32) -> bool;
    fn surface_z_at(&self, x: u32, y: u32) -> u32;
}
```

Flying entities would use NavClouds (3D navigation volumes) — future work, not in scope.

---

## 7. FidelityZone

Renamed from `LocalGrid`. The struct is unchanged — just a proximity bubble that controls simulation detail level. Not a grid.

```rust
pub struct FidelityZone {
    pub id: u32,
    pub fidelity: FidelityLevel,
    pub center: (f32, f32),
    pub radius: f32,
    pub entity_ids: Vec<u32>,
}
```

All references to `LocalGrid` become `FidelityZone`. The `local_grids` field on `WorldState` becomes `fidelity_zones`.

---

## 8. WorldState Storage

```rust
pub struct WorldState {
    // ... existing fields ...
    pub voxel_grids: Vec<VoxelGrid>,       // one per spatial volume of interest
    pub nav_grids: Vec<NavGrid>,           // one per voxel grid
    pub fidelity_zones: Vec<FidelityZone>, // renamed from local_grids
}
```

Each settlement gets its own VoxelGrid. Dungeons, siege camps, or other spatial areas can get their own grids. Different grids can have different Z ranges (deep mines vs flat villages).

---

## 9. Migration from CityGrid

### Renames

| Old | New |
|-----|-----|
| `CityGrid` | `VoxelGrid` |
| `Cell` | `Voxel` |
| `LocalGrid` | `FidelityZone` |
| `city_grids` (field) | `voxel_grids` |
| `local_grids` (field) | `fidelity_zones` |

### File Changes

| File | Change |
|------|--------|
| `city_grid.rs` | Full rewrite → `voxel_grid.rs` (octree replaces flat array) |
| `state.rs` | Field renames, `LocalGrid` → `FidelityZone` |
| `systems/buildings.rs` | Building placement via `VoxelGrid::set()` |
| `systems/action_eval.rs` | Grid lookups via `VoxelGrid::get()` |
| `building_ai/` | Action space indexes into octree, scoring uses `get()` |
| `apply.rs` | Delta application for grid changes |
| `delta.rs` | Grid-related delta variants updated |
| `mod.rs` | Re-exports |

### API Mapping

| CityGrid method | VoxelGrid equivalent |
|---|---|
| `get_cell(x, y)` | `get((x, y, z), seed)` |
| `set_cell(x, y, cell)` | `set((x, y, z), voxel)` |
| `grid_to_world(gx, gy, settlement_pos)` | `grid_to_world(gx, gy, gz)` (origin-based) |
| `find_path(start, end)` | `NavGrid::find_path(start, end)` |
| `compute_flow_field(target)` | `NavGrid::compute_flow_field(target)` |
| `place_walls(settlement)` | Building AI places wall voxels via `set()` |

### Dropped Fields

`Cell.density`, `Cell.age` — no equivalent in the new system. If needed later, they become voxel metadata fields.

---

## 10. Future Work (Not In Scope)

- **NavClouds** — 3D navigation volumes for flying entities
- **Chunk streaming** — load/unload VoxelGrids based on FidelityZone proximity
- **LOD rendering** — octree depth as level-of-detail for visualization
- **Multi-grid pathfinding** — navigation across VoxelGrid boundaries
- **Tech tree gating** — material availability tied to civilization progression
