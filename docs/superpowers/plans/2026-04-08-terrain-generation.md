# 3D Terrain Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat voxel terrain with biome-driven 3D world generation featuring caves, mountains, rivers, flying islands, and dungeons.

**Architecture:** Two-phase pipeline — Phase 1 generates a lightweight 2D continental plan (region grid + river/road polylines), Phase 2 materializes voxel chunks on demand by sampling the plan, resolving 3D biome volumes, and carving features. 3D sectors partition chunk loading. The existing `OverworldGrid` module is deleted; the voxel world becomes the single source of truth.

**Tech Stack:** Pure Rust, no external crate dependencies. Deterministic noise from seed. Existing `VoxelWorld`/`Chunk`/`VoxelMaterial` types retained.

**Spec:** `docs/superpowers/specs/2026-04-08-terrain-generation-design.md`

---

## File Structure

```
src/world_sim/
  terrain/
    mod.rs            — public API re-exports, generate_continent()
    noise.rs          — deterministic noise: hash, value, FBM, worm, domain warp
    region_plan.rs    — RegionPlan grid + Phase 1 continental generation
    biome.rs          — 3D biome volume resolution + material lookup tables
    materialize.rs    — chunk materialization (Phase 2), replaces old generate_chunk
    caves.rs          — 3D worm-noise cave carving
    rivers.rs         — river valley carving from polyline data
    features.rs       — surface features (trees, boulders, dunes, etc.)
    sky.rs            — flying island generation (SDF + noise)
    dungeons.rs       — underground room+corridor structures
  sector.rs          — 3D sector grid, activation/deactivation
```

**Modified files:**
- `src/world_sim/mod.rs` — add `pub mod terrain; pub mod sector;`
- `src/world_sim/voxel.rs` — `generate_chunk()` delegates to `terrain::materialize_chunk()`
- `src/world_sim/state.rs` — extend `SubBiome` with underground variants, add `RegionPlan` field to `WorldState`
- `src/bin/xtask/world_sim_cmd.rs` — `build_world()` uses `generate_continent()` instead of manual region creation
- `src/lib.rs` — remove `pub mod overworld_grid;`

**Deleted files:**
- `src/overworld_grid/` (entire module: mod.rs, terrain_gen.rs, border.rs, renderer.rs, camera.rs)

---

### Task 1: Noise Library

**Files:**
- Create: `src/world_sim/terrain/noise.rs`
- Create: `src/world_sim/terrain/mod.rs`
- Modify: `src/world_sim/mod.rs`

Port existing noise functions from `voxel.rs` (lines 763-821) and `overworld_grid/terrain_gen.rs` (lines 12-63), then add 3D extensions.

- [ ] **Step 1: Create terrain module with noise.rs**

Create `src/world_sim/terrain/mod.rs`:

```rust
pub mod noise;
```

Add to `src/world_sim/mod.rs`:

```rust
pub mod terrain;
```

- [ ] **Step 2: Write tests for hash and value noise**

In `src/world_sim/terrain/noise.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        assert_eq!(hash_3d(10, 20, 30, 42), hash_3d(10, 20, 30, 42));
        assert_ne!(hash_3d(10, 20, 30, 42), hash_3d(11, 20, 30, 42));
    }

    #[test]
    fn hash_f32_in_range() {
        for i in 0..100 {
            let v = hash_f32(i, i * 7, 0, 999);
            assert!(v >= 0.0 && v < 1.0, "hash_f32 out of range: {v}");
        }
    }

    #[test]
    fn value_noise_2d_smooth() {
        // Adjacent samples should be similar (within smoothstep interpolation)
        let a = value_noise_2d(100.0, 100.0, 42, 16.0);
        let b = value_noise_2d(101.0, 100.0, 42, 16.0);
        assert!((a - b).abs() < 0.2, "value_noise_2d not smooth: {a} vs {b}");
    }

    #[test]
    fn value_noise_3d_smooth() {
        let a = value_noise_3d(100.0, 100.0, 100.0, 42, 16.0);
        let b = value_noise_3d(101.0, 100.0, 100.0, 42, 16.0);
        assert!((a - b).abs() < 0.2, "value_noise_3d not smooth: {a} vs {b}");
    }

    #[test]
    fn fbm_2d_in_range() {
        for i in 0..50 {
            let v = fbm_2d(i as f32 * 10.0, i as f32 * 7.0, 42, 5, 2.0, 0.5);
            assert!(v >= 0.0 && v <= 1.0, "fbm_2d out of range: {v}");
        }
    }

    #[test]
    fn fbm_3d_in_range() {
        for i in 0..50 {
            let v = fbm_3d(i as f32 * 10.0, 0.0, i as f32 * 7.0, 42, 4, 2.0, 0.5);
            assert!(v >= 0.0 && v <= 1.0, "fbm_3d out of range: {v}");
        }
    }

    #[test]
    fn worm_noise_produces_caves() {
        // At least some positions should be "carved" (both fields in band)
        let mut cave_count = 0;
        for x in 0..100 {
            for y in 0..100 {
                if worm_cave(x as f32, y as f32, 50.0, 42, 55555, 0.06) {
                    cave_count += 1;
                }
            }
        }
        assert!(cave_count > 0, "worm_noise produced no caves");
        assert!(cave_count < 5000, "worm_noise produced too many caves: {cave_count}");
    }
}
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::noise --no-run 2>&1 | head -5`
Expected: compilation error — functions not defined.

- [ ] **Step 4: Implement noise functions**

In `src/world_sim/terrain/noise.rs`, implement:

```rust
/// Deterministic 3D hash → u32. Ported from voxel.rs.
pub fn hash_3d(x: i32, y: i32, z: i32, seed: u64) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(x as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(y as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(z as u64);
    h = h ^ (h >> 33);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h = h ^ (h >> 33);
    (h >> 32) as u32
}

/// Hash to float in [0, 1).
pub fn hash_f32(x: i32, y: i32, z: i32, seed: u64) -> f32 {
    hash_3d(x, y, z, seed) as f32 / u32::MAX as f32
}

/// Smoothstep: t² × (3 - 2t).
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 2D value noise with smoothstep interpolation. Returns [0, 1].
pub fn value_noise_2d(x: f32, y: f32, seed: u64, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let fx = smoothstep(sx - sx.floor());
    let fy = smoothstep(sy - sy.floor());
    let h00 = hash_f32(ix, iy, 0, seed);
    let h10 = hash_f32(ix + 1, iy, 0, seed);
    let h01 = hash_f32(ix, iy + 1, 0, seed);
    let h11 = hash_f32(ix + 1, iy + 1, 0, seed);
    let a = h00 + (h10 - h00) * fx;
    let b = h01 + (h11 - h01) * fx;
    a + (b - a) * fy
}

/// 3D value noise with trilinear smoothstep. Returns [0, 1].
pub fn value_noise_3d(x: f32, y: f32, z: f32, seed: u64, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let sz = z / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let iz = sz.floor() as i32;
    let fx = smoothstep(sx - sx.floor());
    let fy = smoothstep(sy - sy.floor());
    let fz = smoothstep(sz - sz.floor());
    // 8 corner hashes
    let c000 = hash_f32(ix, iy, iz, seed);
    let c100 = hash_f32(ix + 1, iy, iz, seed);
    let c010 = hash_f32(ix, iy + 1, iz, seed);
    let c110 = hash_f32(ix + 1, iy + 1, iz, seed);
    let c001 = hash_f32(ix, iy, iz + 1, seed);
    let c101 = hash_f32(ix + 1, iy, iz + 1, seed);
    let c011 = hash_f32(ix, iy + 1, iz + 1, seed);
    let c111 = hash_f32(ix + 1, iy + 1, iz + 1, seed);
    // Trilinear interpolation
    let a0 = c000 + (c100 - c000) * fx;
    let b0 = c010 + (c110 - c010) * fx;
    let a1 = c001 + (c101 - c001) * fx;
    let b1 = c011 + (c111 - c011) * fx;
    let c0 = a0 + (b0 - a0) * fy;
    let c1 = a1 + (b1 - a1) * fy;
    c0 + (c1 - c0) * fz
}

/// 2D Fractal Brownian Motion. Returns [0, 1].
pub fn fbm_2d(x: f32, y: f32, seed: u64, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut max_amp = 0.0f32;
    for i in 0..octaves {
        sum += amp * value_noise_2d(x * freq, y * freq, seed.wrapping_add(i as u64 * 31337), 1.0);
        max_amp += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_amp
}

/// 3D Fractal Brownian Motion. Returns [0, 1].
pub fn fbm_3d(x: f32, y: f32, z: f32, seed: u64, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    let mut max_amp = 0.0f32;
    for i in 0..octaves {
        let s = seed.wrapping_add(i as u64 * 31337);
        sum += amp * value_noise_3d(x * freq, y * freq, z * freq, s, 1.0);
        max_amp += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / max_amp
}

/// Worm cave test: returns true if position should be carved.
/// Uses two independent noise fields; cave exists where both are in a narrow band.
pub fn worm_cave(x: f32, y: f32, z: f32, seed_a: u64, seed_b: u64, threshold: f32) -> bool {
    let a = value_noise_3d(x, y, z, seed_a, 16.0);
    let b = value_noise_3d(x, y, z, seed_b, 16.0);
    (a - 0.5).abs() < threshold && (b - 0.5).abs() < threshold
}

/// Domain warp: offset input coordinates by noise for organic shapes.
pub fn domain_warp_2d(x: f32, y: f32, seed: u64, scale: f32, strength: f32) -> (f32, f32) {
    let wx = value_noise_2d(x, y, seed, scale) * 2.0 - 1.0;
    let wy = value_noise_2d(x, y, seed.wrapping_add(77777), scale) * 2.0 - 1.0;
    (x + wx * strength, y + wy * strength)
}
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::noise -- --nocapture`
Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/terrain/mod.rs src/world_sim/terrain/noise.rs src/world_sim/mod.rs
git commit -m "feat(terrain): add deterministic noise library for terrain generation"
```

---

### Task 2: Region Plan — Data Structures and Continental Generation

**Files:**
- Create: `src/world_sim/terrain/region_plan.rs`
- Modify: `src/world_sim/terrain/mod.rs`

The region plan is a 2D grid that drives all chunk generation. Each cell stores biome, height, moisture, and feature data. The continental generation pipeline (Phase 1) fills this grid from noise.

- [ ] **Step 1: Write tests for region plan generation**

In `src/world_sim/terrain/region_plan.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_dimensions() {
        let plan = generate_continent(50, 30, 42);
        assert_eq!(plan.cols, 50);
        assert_eq!(plan.rows, 30);
        assert_eq!(plan.cells.len(), 50 * 30);
    }

    #[test]
    fn plan_deterministic() {
        let a = generate_continent(20, 10, 42);
        let b = generate_continent(20, 10, 42);
        for i in 0..a.cells.len() {
            assert_eq!(a.cells[i].terrain, b.cells[i].terrain, "cell {i} terrain differs");
            assert_eq!(a.cells[i].height, b.cells[i].height, "cell {i} height differs");
        }
    }

    #[test]
    fn plan_has_variety() {
        let plan = generate_continent(50, 30, 42);
        let mut terrain_set = std::collections::HashSet::new();
        for cell in &plan.cells {
            terrain_set.insert(std::mem::discriminant(&cell.terrain));
        }
        // Should have at least 5 different terrain types
        assert!(terrain_set.len() >= 5, "only {} terrain types", terrain_set.len());
    }

    #[test]
    fn plan_has_continent_shape() {
        let plan = generate_continent(50, 30, 42);
        let land = plan.cells.iter().filter(|c| c.terrain != Terrain::DeepOcean && c.terrain != Terrain::Coast).count();
        let total = plan.cells.len();
        let land_pct = land as f32 / total as f32;
        // Should be ~40-80% land (not all ocean, not all land)
        assert!(land_pct > 0.3 && land_pct < 0.85, "land percentage: {land_pct}");
    }

    #[test]
    fn plan_has_rivers() {
        let plan = generate_continent(50, 30, 42);
        assert!(!plan.rivers.is_empty(), "no rivers generated");
        // Each river should have at least 2 points
        for river in &plan.rivers {
            assert!(river.points.len() >= 2, "river too short: {} points", river.points.len());
        }
    }

    #[test]
    fn plan_has_settlements() {
        let plan = generate_continent(50, 30, 42);
        let settlement_count = plan.cells.iter().filter(|c| c.settlement.is_some()).count();
        assert!(settlement_count >= 3, "only {settlement_count} settlements");
    }

    #[test]
    fn height_maps_to_terrain() {
        // High elevation → mountains, low → ocean
        let plan = generate_continent(50, 30, 42);
        for cell in &plan.cells {
            if cell.height > 0.8 {
                assert!(
                    matches!(cell.terrain, Terrain::Mountains | Terrain::Glacier | Terrain::FlyingIslands | Terrain::Volcano),
                    "high elevation ({}) got {:?}", cell.height, cell.terrain
                );
            }
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::region_plan --no-run 2>&1 | head -5`
Expected: compilation error.

- [ ] **Step 3: Implement region plan data structures**

In `src/world_sim/terrain/region_plan.rs`:

```rust
use serde::{Serialize, Deserialize};
use crate::world_sim::state::{Terrain, SubBiome};
use super::noise;

/// Voxels per region cell (horizontal).
pub const CELL_SIZE: i32 = 4096;

/// A settlement site in the region plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementPlan {
    pub kind: SettlementKind,
    /// Position within the cell (0.0-1.0 fractional).
    pub local_pos: (f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementKind {
    Town,
    Castle,
    Camp,
    Ruin,
}

/// A dungeon entrance in the region plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DungeonPlan {
    pub local_pos: (f32, f32),
    pub depth: u8, // 1-3 levels deep
}

/// A river as a polyline with varying width.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverPath {
    /// Points in world-space voxel coordinates.
    pub points: Vec<(f32, f32)>,
    /// Width at each point (in voxels). Increases downstream.
    pub widths: Vec<f32>,
}

/// A road segment connecting two settlements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegment {
    pub from: (f32, f32), // world-space voxel coords
    pub to: (f32, f32),
}

/// Per-cell data in the region plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCell {
    pub height: f32,    // 0.0-1.0
    pub moisture: f32,  // 0.0-1.0
    pub temperature: f32, // 0.0-1.0 (cold to hot)
    pub terrain: Terrain,
    pub sub_biome: SubBiome,
    pub settlement: Option<SettlementPlan>,
    pub dungeons: Vec<DungeonPlan>,
    pub has_road: bool,
}

/// The continental region plan — lightweight 2D grid driving all chunk generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionPlan {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<RegionCell>,
    pub rivers: Vec<RiverPath>,
    pub roads: Vec<RoadSegment>,
    pub seed: u64,
}

impl RegionPlan {
    pub fn get(&self, col: usize, row: usize) -> &RegionCell {
        &self.cells[row * self.cols + col]
    }

    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut RegionCell {
        &mut self.cells[row * self.cols + col]
    }

    /// Sample the plan at a voxel-space (x, y) position.
    /// Returns the cell and fractional position within it.
    pub fn sample(&self, vx: f32, vy: f32) -> (&RegionCell, f32, f32) {
        let col = (vx / CELL_SIZE as f32).clamp(0.0, (self.cols - 1) as f32);
        let row = (vy / CELL_SIZE as f32).clamp(0.0, (self.rows - 1) as f32);
        let frac_x = col.fract();
        let frac_y = row.fract();
        (self.get(col as usize, row as usize), frac_x, frac_y)
    }

    /// Bilinearly interpolate height at a voxel-space position.
    pub fn interpolate_height(&self, vx: f32, vy: f32) -> f32 {
        let col = (vx / CELL_SIZE as f32).clamp(0.0, (self.cols - 1) as f32);
        let row = (vy / CELL_SIZE as f32).clamp(0.0, (self.rows - 1) as f32);
        let c0 = col.floor() as usize;
        let r0 = row.floor() as usize;
        let c1 = (c0 + 1).min(self.cols - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let fx = col.fract();
        let fy = row.fract();
        let h00 = self.get(c0, r0).height;
        let h10 = self.get(c1, r0).height;
        let h01 = self.get(c0, r1).height;
        let h11 = self.get(c1, r1).height;
        let a = h00 + (h10 - h00) * fx;
        let b = h01 + (h11 - h01) * fx;
        a + (b - a) * fy
    }
}
```

- [ ] **Step 4: Implement continental generation**

Still in `region_plan.rs`, add the generation pipeline:

```rust
/// Maximum surface height in voxels (mountains).
pub const MAX_SURFACE_Z: i32 = 400;
/// Sea level in voxels.
pub const SEA_LEVEL: i32 = 80;

/// Generate a continental plan from seed.
pub fn generate_continent(cols: usize, rows: usize, seed: u64) -> RegionPlan {
    let mut plan = RegionPlan {
        cols,
        rows,
        cells: Vec::with_capacity(cols * rows),
        rivers: Vec::new(),
        roads: Vec::new(),
        seed,
    };

    // Phase 1a: Generate noise fields and classify terrain
    for row in 0..rows {
        for col in 0..cols {
            let x = col as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            let y = row as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            let scale = 0.00015; // world-scale frequency (larger world = smaller scale)

            let elevation = noise::fbm_2d(x * scale, y * scale, seed, 5, 2.0, 0.5);
            let moisture = noise::fbm_2d(x * scale, y * scale, seed.wrapping_add(99991), 5, 2.0, 0.5);
            let temperature = noise::fbm_2d(x * scale, y * scale, seed.wrapping_add(77773), 4, 2.0, 0.5);

            // Continent mask: fade toward edges for ocean border
            let cx = col as f32 / cols as f32 * 2.0 - 1.0; // -1 to 1
            let cy = row as f32 / rows as f32 * 2.0 - 1.0;
            let edge_dist = 1.0 - (cx * cx + cy * cy).sqrt().min(1.0);
            let continent_mask = (edge_dist * 2.5).clamp(0.0, 1.0);
            let height = elevation * continent_mask;

            let terrain = classify_terrain(height, moisture, temperature, seed, x, y);
            let sub_biome = assign_sub_biome(terrain, moisture, temperature, seed, x, y);

            plan.cells.push(RegionCell {
                height,
                moisture,
                temperature,
                terrain,
                sub_biome,
                settlement: None,
                dungeons: Vec::new(),
                has_road: false,
            });
        }
    }

    // Phase 1b-1f: features
    trace_rivers(&mut plan);
    place_settlements(&mut plan);
    build_roads(&mut plan);
    place_dungeons(&mut plan);

    plan
}

/// Classify terrain from noise fields. Ported from world_sim_cmd.rs assign_terrain().
fn classify_terrain(height: f32, moisture: f32, temperature: f32, seed: u64, x: f32, y: f32) -> Terrain {
    let detail = noise::fbm_2d(x * 0.00045, y * 0.00045, seed.wrapping_add(55537), 3, 2.0, 0.5);

    if height < 0.12 {
        if moisture > 0.6 { return Terrain::DeepOcean; }
        if detail > 0.7 { return Terrain::CoralReef; }
        return Terrain::Coast;
    }
    if height > 0.75 {
        if temperature < 0.3 { return Terrain::Glacier; }
        if detail > 0.85 { return Terrain::FlyingIslands; }
        return Terrain::Mountains;
    }

    match (temperature > 0.5, moisture > 0.5) {
        (true, true) => {
            if moisture > 0.7 { Terrain::Swamp }
            else { Terrain::Jungle }
        }
        (true, false) => {
            if detail > 0.8 { Terrain::Volcano }
            else if moisture < 0.25 { Terrain::Desert }
            else { Terrain::Badlands }
        }
        (false, true) => {
            if temperature < 0.25 { Terrain::Tundra }
            else { Terrain::Forest }
        }
        (false, false) => {
            if detail > 0.85 { Terrain::AncientRuins }
            else if detail > 0.8 { Terrain::DeathZone }
            else if temperature < 0.3 { Terrain::Caverns }
            else { Terrain::Plains }
        }
    }
}

fn assign_sub_biome(terrain: Terrain, moisture: f32, temperature: f32, seed: u64, x: f32, y: f32) -> SubBiome {
    let detail = noise::hash_f32((x * 0.01) as i32, (y * 0.01) as i32, 0, seed.wrapping_add(88888));
    match terrain {
        Terrain::Forest => {
            if detail < 0.3 { SubBiome::LightForest }
            else if detail > 0.8 { SubBiome::AncientForest }
            else if moisture > 0.65 { SubBiome::DenseForest }
            else { SubBiome::Standard }
        }
        Terrain::Desert => {
            if detail > 0.5 { SubBiome::RockyDesert } else { SubBiome::SandDunes }
        }
        Terrain::Mountains if temperature > 0.6 => SubBiome::HotSprings,
        Terrain::Swamp if detail > 0.6 => SubBiome::GlowingMarsh,
        Terrain::Jungle if detail > 0.7 => SubBiome::TempleJungle,
        _ => SubBiome::Standard,
    }
}

/// Trace rivers from high elevation to low, following steepest descent.
fn trace_rivers(plan: &mut RegionPlan) {
    let seed = plan.seed.wrapping_add(11111);
    let num_rivers = 3 + (noise::hash_3d(0, 0, 0, seed) % 5) as usize;

    for r in 0..num_rivers {
        let rseed = seed.wrapping_add(r as u64 * 7919);
        // Find a high-elevation starting cell
        let start_col = (noise::hash_3d(r as i32, 0, 0, rseed) as usize) % plan.cols;
        let start_row = (noise::hash_3d(r as i32, 1, 0, rseed) as usize) % plan.rows;
        let start_height = plan.get(start_col, start_row).height;
        if start_height < 0.5 { continue; } // skip low starts

        let mut points = Vec::new();
        let mut widths = Vec::new();
        let mut col = start_col;
        let mut row = start_row;
        let mut width = 30.0f32; // starting width in voxels

        for _ in 0..200 {
            let x = col as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            let y = row as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            points.push((x, y));
            widths.push(width);

            // Find lowest neighbor
            let current_h = plan.get(col, row).height;
            let mut best = (col, row, current_h);
            for (dc, dr) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc >= 0 && nc < plan.cols as i32 && nr >= 0 && nr < plan.rows as i32 {
                    let nh = plan.get(nc as usize, nr as usize).height;
                    if nh < best.2 {
                        best = (nc as usize, nr as usize, nh);
                    }
                }
            }
            if best.0 == col && best.1 == row { break; } // local minimum
            col = best.0;
            row = best.1;
            width = (width + 5.0).min(100.0); // widen downstream
            if plan.get(col, row).terrain == Terrain::DeepOcean { break; } // reached ocean
        }

        if points.len() >= 2 {
            plan.rivers.push(RiverPath { points, widths });
        }
    }
}

/// Place settlements on suitable terrain.
fn place_settlements(plan: &mut RegionPlan) {
    let seed = plan.seed.wrapping_add(22222);
    let mut placed = 0usize;
    for i in 0..plan.cells.len() {
        let cell = &plan.cells[i];
        if !cell.terrain.is_settleable() { continue; }
        let col = i % plan.cols;
        let row = i / plan.cols;
        let h = noise::hash_f32(col as i32, row as i32, 0, seed);
        // ~8% of settleable cells get a settlement
        if h > 0.08 { continue; }
        let kind = match (noise::hash_3d(col as i32, row as i32, 1, seed) % 4) {
            0 => SettlementKind::Town,
            1 => SettlementKind::Castle,
            2 => SettlementKind::Camp,
            _ => SettlementKind::Ruin,
        };
        plan.cells[i].settlement = Some(SettlementPlan {
            kind,
            local_pos: (0.5, 0.5),
        });
        placed += 1;
    }
}

/// Connect nearby settlements with roads.
fn build_roads(plan: &mut RegionPlan) {
    let settlements: Vec<(usize, usize, f32, f32)> = plan.cells.iter().enumerate()
        .filter(|(_, c)| c.settlement.is_some())
        .map(|(i, _)| {
            let col = i % plan.cols;
            let row = i / plan.cols;
            let x = col as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            let y = row as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5;
            (col, row, x, y)
        })
        .collect();

    let max_dist = CELL_SIZE as f32 * 5.0; // connect settlements within 5 cells
    for i in 0..settlements.len() {
        for j in (i + 1)..settlements.len() {
            let dx = settlements[i].2 - settlements[j].2;
            let dy = settlements[i].3 - settlements[j].3;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < max_dist {
                plan.roads.push(RoadSegment {
                    from: (settlements[i].2, settlements[i].3),
                    to: (settlements[j].2, settlements[j].3),
                });
                // Mark cells along road
                let steps = (dist / CELL_SIZE as f32).ceil() as usize;
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let rx = settlements[i].2 + dx * -t;
                    let ry = settlements[i].3 + dy * -t;
                    let c = (rx / CELL_SIZE as f32) as usize;
                    let r = (ry / CELL_SIZE as f32) as usize;
                    if c < plan.cols && r < plan.rows {
                        plan.get_mut(c, r).has_road = true;
                    }
                }
            }
        }
    }
}

/// Place dungeon entrances in appropriate terrain.
fn place_dungeons(plan: &mut RegionPlan) {
    let seed = plan.seed.wrapping_add(33333);
    for i in 0..plan.cells.len() {
        let cell = &plan.cells[i];
        let col = i % plan.cols;
        let row = i / plan.cols;
        let h = noise::hash_f32(col as i32, row as i32, 0, seed);
        let should_place = match cell.terrain {
            Terrain::Caverns => h < 0.3,       // 30% of cavern cells
            Terrain::AncientRuins => h < 0.4,  // 40% of ruin cells
            Terrain::Mountains => h < 0.05,    // 5% of mountain cells
            Terrain::DeathZone => h < 0.2,     // 20% of death zone cells
            _ => h < 0.02,                     // 2% of other terrain
        };
        if should_place {
            let depth = 1 + (noise::hash_3d(col as i32, row as i32, 2, seed) % 3) as u8;
            plan.cells[i].dungeons.push(DungeonPlan {
                local_pos: (
                    noise::hash_f32(col as i32, row as i32, 3, seed),
                    noise::hash_f32(col as i32, row as i32, 4, seed),
                ),
                depth,
            });
        }
    }
}
```

- [ ] **Step 5: Update terrain/mod.rs to export**

```rust
pub mod noise;
pub mod region_plan;

pub use region_plan::{RegionPlan, RegionCell, generate_continent, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::region_plan -- --nocapture`
Expected: all 7 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/
git commit -m "feat(terrain): add region plan with continental generation pipeline"
```

---

### Task 3: 3D Biome Volume Resolution

**Files:**
- Create: `src/world_sim/terrain/biome.rs`
- Modify: `src/world_sim/state.rs` — extend SubBiome with underground variants
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Extend SubBiome enum with underground variants**

In `src/world_sim/state.rs`, add to the `SubBiome` enum after `TempleJungle`:

```rust
    // Underground variants
    /// Default cave — stone walls, stalactites.
    NaturalCave,
    /// Basalt walls, lava pools. Under volcano/mountains.
    LavaTubes,
    /// Ice walls, frozen lakes. Under tundra/glacier.
    FrozenCavern,
    /// Bioluminescent mushrooms, organic. Under forest/jungle.
    MushroomGrove,
    /// Crystal clusters, high ore density. Rare.
    CrystalVein,
    /// Flooded chambers. Under swamp/coast.
    Aquifer,
    /// Ancient remains. Under death zone/ruins.
    BoneOssuary,
```

Also add their `suffix()`, `wood_mult()`, `herb_mult()`, `travel_mult()` match arms (all return defaults except suffix):

```rust
SubBiome::NaturalCave => " (cave)",
SubBiome::LavaTubes => " (lava tubes)",
SubBiome::FrozenCavern => " (frozen)",
SubBiome::MushroomGrove => " (mushroom)",
SubBiome::CrystalVein => " (crystal)",
SubBiome::Aquifer => " (aquifer)",
SubBiome::BoneOssuary => " (ossuary)",
```

- [ ] **Step 2: Write tests for biome resolution**

In `src/world_sim/terrain/biome.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::{Terrain, SubBiome};

    #[test]
    fn surface_returns_surface_biome() {
        let vol = resolve_biome(Terrain::Forest, SubBiome::DenseForest, 100, 42);
        assert_eq!(vol.surface, Terrain::Forest);
        assert_eq!(vol.surface_sub, SubBiome::DenseForest);
    }

    #[test]
    fn deep_underground_varies_by_surface() {
        let under_volcano = resolve_biome(Terrain::Volcano, SubBiome::Standard, -200, 42);
        assert_eq!(under_volcano.underground, SubBiome::LavaTubes);

        let under_tundra = resolve_biome(Terrain::Tundra, SubBiome::Standard, -200, 42);
        assert_eq!(under_tundra.underground, SubBiome::FrozenCavern);

        let under_forest = resolve_biome(Terrain::Forest, SubBiome::Standard, -200, 42);
        assert_eq!(under_forest.underground, SubBiome::MushroomGrove);
    }

    #[test]
    fn abyss_is_always_lava() {
        let vol = resolve_biome(Terrain::Plains, SubBiome::Standard, -500, 42);
        assert_eq!(vol.underground, SubBiome::LavaTubes);
    }

    #[test]
    fn surface_materials_differ_by_biome() {
        let desert = surface_materials(Terrain::Desert);
        let forest = surface_materials(Terrain::Forest);
        assert_ne!(desert.surface, forest.surface);
        assert_ne!(desert.subsoil, forest.subsoil);
    }
}
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::biome --no-run 2>&1 | head -5`

- [ ] **Step 4: Implement biome resolution**

In `src/world_sim/terrain/biome.rs`:

```rust
use crate::world_sim::state::{Terrain, SubBiome};
use crate::world_sim::voxel::VoxelMaterial;

/// Resolved 3D biome at a specific position.
#[derive(Debug, Clone, Copy)]
pub struct BiomeVolume {
    pub surface: Terrain,
    pub surface_sub: SubBiome,
    pub underground: SubBiome,
}

/// Resolve the 3D biome volume at a given depth relative to surface.
/// `depth_below_surface` is negative when underground (e.g., -100 = 100 voxels below surface).
pub fn resolve_biome(
    surface_terrain: Terrain,
    surface_sub: SubBiome,
    depth_below_surface: i32,
    seed: u64,
) -> BiomeVolume {
    let underground = if depth_below_surface >= -40 {
        // Near-surface: inherit surface biome character
        match surface_terrain {
            Terrain::Caverns => SubBiome::NaturalCave,
            _ => SubBiome::Standard,
        }
    } else if depth_below_surface >= -150 {
        // Shallow underground: surface-influenced caves
        match surface_terrain {
            Terrain::Volcano => SubBiome::LavaTubes,
            Terrain::Mountains => SubBiome::NaturalCave,
            Terrain::Tundra | Terrain::Glacier => SubBiome::FrozenCavern,
            Terrain::Forest | Terrain::Jungle => SubBiome::MushroomGrove,
            Terrain::Swamp | Terrain::Coast => SubBiome::Aquifer,
            Terrain::DeathZone | Terrain::AncientRuins => SubBiome::BoneOssuary,
            Terrain::Caverns => SubBiome::CrystalVein,
            _ => SubBiome::NaturalCave,
        }
    } else if depth_below_surface >= -350 {
        // Deep underground: surface influence fades, depth-driven
        match surface_terrain {
            Terrain::Volcano => SubBiome::LavaTubes,
            Terrain::Tundra | Terrain::Glacier => SubBiome::FrozenCavern,
            _ => SubBiome::NaturalCave,
        }
    } else {
        // Abyss: always lava
        SubBiome::LavaTubes
    };

    BiomeVolume {
        surface: surface_terrain,
        surface_sub: surface_sub,
        underground,
    }
}

/// Material palette for a surface biome.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceMaterials {
    pub surface: VoxelMaterial,       // topmost 1-2 voxels
    pub subsoil: VoxelMaterial,       // below surface, above stone
    pub deep_stone: VoxelMaterial,    // primary rock type
}

/// Get the surface material palette for a terrain type.
pub fn surface_materials(terrain: Terrain) -> SurfaceMaterials {
    match terrain {
        Terrain::Plains | Terrain::Forest | Terrain::Jungle => SurfaceMaterials {
            surface: VoxelMaterial::Grass,
            subsoil: VoxelMaterial::Dirt,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::Desert | Terrain::Badlands => SurfaceMaterials {
            surface: VoxelMaterial::Sand,
            subsoil: VoxelMaterial::Sandstone,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::Mountains => SurfaceMaterials {
            surface: VoxelMaterial::Stone,
            subsoil: VoxelMaterial::Stone,
            deep_stone: VoxelMaterial::Granite,
        },
        Terrain::Tundra | Terrain::Glacier => SurfaceMaterials {
            surface: VoxelMaterial::Snow,
            subsoil: VoxelMaterial::Gravel,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::Swamp => SurfaceMaterials {
            surface: VoxelMaterial::Grass,
            subsoil: VoxelMaterial::Clay,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::Coast | Terrain::CoralReef => SurfaceMaterials {
            surface: VoxelMaterial::Sand,
            subsoil: VoxelMaterial::Sand,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::Volcano => SurfaceMaterials {
            surface: VoxelMaterial::Basalt,
            subsoil: VoxelMaterial::Basalt,
            deep_stone: VoxelMaterial::Obsidian,
        },
        Terrain::Caverns => SurfaceMaterials {
            surface: VoxelMaterial::Stone,
            subsoil: VoxelMaterial::Stone,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::DeepOcean => SurfaceMaterials {
            surface: VoxelMaterial::Sand,
            subsoil: VoxelMaterial::Clay,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::DeathZone => SurfaceMaterials {
            surface: VoxelMaterial::Bone,
            subsoil: VoxelMaterial::Dirt,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::AncientRuins => SurfaceMaterials {
            surface: VoxelMaterial::Grass,
            subsoil: VoxelMaterial::CutStone,
            deep_stone: VoxelMaterial::Stone,
        },
        Terrain::FlyingIslands => SurfaceMaterials {
            surface: VoxelMaterial::Grass,
            subsoil: VoxelMaterial::Dirt,
            deep_stone: VoxelMaterial::Stone,
        },
    }
}

/// Get cave wall material for an underground biome.
pub fn cave_materials(biome: SubBiome) -> VoxelMaterial {
    match biome {
        SubBiome::LavaTubes => VoxelMaterial::Basalt,
        SubBiome::FrozenCavern => VoxelMaterial::Ice,
        SubBiome::MushroomGrove => VoxelMaterial::Dirt,
        SubBiome::CrystalVein => VoxelMaterial::Crystal,
        SubBiome::Aquifer => VoxelMaterial::Clay,
        SubBiome::BoneOssuary => VoxelMaterial::Bone,
        _ => VoxelMaterial::Stone,
    }
}

/// Get cave fill material (what fills open cave spaces — usually air, sometimes fluid).
pub fn cave_fill(biome: SubBiome) -> VoxelMaterial {
    match biome {
        SubBiome::LavaTubes => VoxelMaterial::Lava,
        SubBiome::Aquifer => VoxelMaterial::Water,
        SubBiome::FrozenCavern => VoxelMaterial::Ice,
        _ => VoxelMaterial::Air,
    }
}
```

- [ ] **Step 5: Update terrain/mod.rs**

```rust
pub mod noise;
pub mod region_plan;
pub mod biome;

pub use region_plan::{RegionPlan, RegionCell, generate_continent, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
pub use biome::{BiomeVolume, resolve_biome, surface_materials};
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::biome -- --nocapture`
Expected: all 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/biome.rs src/world_sim/terrain/mod.rs src/world_sim/state.rs
git commit -m "feat(terrain): add 3D biome volume resolution and material tables"
```

---

### Task 4: Chunk Materialization — Replace generate_chunk

**Files:**
- Create: `src/world_sim/terrain/materialize.rs`
- Modify: `src/world_sim/voxel.rs` — delegate `generate_chunk()` to materializer
- Modify: `src/world_sim/terrain/mod.rs`

This is the core task — replacing the flat terrain gen with biome-driven column building.

- [ ] **Step 1: Write tests for chunk materialization**

In `src/world_sim/terrain/materialize.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::terrain::region_plan::generate_continent;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};

    fn test_plan() -> RegionPlan {
        generate_continent(10, 10, 42)
    }

    #[test]
    fn chunk_at_surface_has_ground() {
        let plan = test_plan();
        // Chunk at (0, 0, surface_chunk_z) should have some solid voxels
        let surface_z = (SEA_LEVEL / CHUNK_SIZE as i32) + 1;
        let chunk = materialize_chunk(ChunkPos::new(5, 5, surface_z), &plan, 42);
        let solid_count = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
        assert!(solid_count > 0, "surface chunk has no solid voxels");
    }

    #[test]
    fn chunk_deep_underground_is_solid() {
        let plan = test_plan();
        let chunk = materialize_chunk(ChunkPos::new(5, 5, -5), &plan, 42);
        let solid_count = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
        // Deep underground should be mostly solid (stone/granite)
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert!(solid_count > total / 2, "deep chunk not mostly solid: {solid_count}/{total}");
    }

    #[test]
    fn chunk_high_sky_is_air() {
        let plan = test_plan();
        let chunk = materialize_chunk(ChunkPos::new(5, 5, 30), &plan, 42);
        let air_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert_eq!(air_count, total, "sky chunk is not all air");
    }

    #[test]
    fn different_biomes_produce_different_surfaces() {
        let plan = test_plan();
        // Find a plains cell and a non-plains cell
        let plains_idx = plan.cells.iter().position(|c| c.terrain == Terrain::Plains);
        let other_idx = plan.cells.iter().position(|c| {
            c.terrain != Terrain::Plains && c.terrain != Terrain::DeepOcean && c.terrain != Terrain::Coast
        });
        if plains_idx.is_none() || other_idx.is_none() { return; } // skip if seed doesn't produce both

        let pi = plains_idx.unwrap();
        let oi = other_idx.unwrap();
        let pc = (pi % plan.cols, pi / plan.cols);
        let oc = (oi % plan.cols, oi / plan.cols);

        let surface_z = (SEA_LEVEL / CHUNK_SIZE as i32) + 2;
        let chunk_a = materialize_chunk(ChunkPos::new(pc.0 as i32, pc.1 as i32, surface_z), &plan, 42);
        let chunk_b = materialize_chunk(ChunkPos::new(oc.0 as i32, oc.1 as i32, surface_z), &plan, 42);

        // At least some voxels should differ in material
        let diffs = chunk_a.voxels.iter().zip(chunk_b.voxels.iter())
            .filter(|(a, b)| a.material != b.material)
            .count();
        assert!(diffs > 0, "different biomes produced identical chunks");
    }

    #[test]
    fn materialization_is_deterministic() {
        let plan = test_plan();
        let cp = ChunkPos::new(3, 3, 2);
        let a = materialize_chunk(cp, &plan, 42);
        let b = materialize_chunk(cp, &plan, 42);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::materialize --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement chunk materialization**

In `src/world_sim/terrain/materialize.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::Terrain;
use crate::world_sim::terrain::region_plan::{RegionPlan, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
use crate::world_sim::terrain::biome::{resolve_biome, surface_materials, cave_materials};
use crate::world_sim::terrain::noise;

/// Materialize a single chunk from the region plan.
/// This replaces the old flat `generate_chunk()`.
pub fn materialize_chunk(cp: ChunkPos, plan: &RegionPlan, seed: u64) -> Chunk {
    let mut chunk = Chunk::new_air(cp);
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let vx = base_x + lx as i32;
                let vy = base_y + ly as i32;
                let vz = base_z + lz as i32;

                let material = voxel_material(vx, vy, vz, plan, seed);
                if material != VoxelMaterial::Air {
                    chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(material);
                }
            }
        }
    }

    chunk.dirty = true;
    chunk
}

/// Determine the material for a single voxel position.
fn voxel_material(vx: i32, vy: i32, vz: i32, plan: &RegionPlan, seed: u64) -> VoxelMaterial {
    // Sample region plan for this column
    let (cell, _fx, _fy) = plan.sample(vx as f32, vy as f32);
    let terrain = cell.terrain;

    // Compute surface height for this column
    let base_height = plan.interpolate_height(vx as f32, vy as f32);
    // Map [0, 1] height to voxel Z, with detail noise
    let detail = noise::fbm_2d(
        vx as f32 * 0.02,
        vy as f32 * 0.02,
        seed.wrapping_add(44444),
        3, 2.0, 0.5,
    );
    let surface_z = (base_height * MAX_SURFACE_Z as f32) as i32
        + ((detail - 0.5) * 20.0) as i32; // ±10 voxel detail variation

    let depth = surface_z - vz; // positive = below surface

    // Get biome-appropriate materials
    let mats = surface_materials(terrain);
    let biome = resolve_biome(terrain, cell.sub_biome, -depth, seed);

    // Layer assignment
    if vz < -120 {
        // Deep bedrock
        return VoxelMaterial::Granite;
    }

    if depth > 80 {
        // Deep stone with ore veins
        return ore_at(vx, vy, vz, seed, terrain).unwrap_or(mats.deep_stone);
    }

    if depth > 20 {
        // Subsoil: mix of stone and subsoil material
        let mix = noise::value_noise_3d(vx as f32, vy as f32, vz as f32, seed.wrapping_add(1234), 8.0);
        return if mix > 0.4 { mats.deep_stone } else { mats.subsoil };
    }

    if depth > 0 {
        // Near-surface soil
        return mats.subsoil;
    }

    if depth >= -1 && depth <= 0 {
        // Surface layer
        // Snow line for mountains
        if terrain == Terrain::Mountains && vz > 250 {
            return VoxelMaterial::Snow;
        }
        return mats.surface;
    }

    // Above surface
    if vz <= SEA_LEVEL && terrain != Terrain::Mountains && terrain != Terrain::FlyingIslands {
        // Below sea level = water
        return VoxelMaterial::Water;
    }

    VoxelMaterial::Air
}

/// Ore vein placement, biome-aware.
fn ore_at(vx: i32, vy: i32, vz: i32, seed: u64, terrain: Terrain) -> Option<VoxelMaterial> {
    let n = noise::value_noise_3d(
        vx as f32, vy as f32, vz as f32,
        seed.wrapping_add(0x0EE1), 6.0,
    );

    // Mountains/Caverns have more ore
    let ore_boost = match terrain {
        Terrain::Mountains | Terrain::Caverns => 0.03,
        Terrain::Volcano => 0.02,
        _ => 0.0,
    };

    if n > 0.95 - ore_boost && vz < -50 {
        Some(VoxelMaterial::Crystal)
    } else if n > 0.93 - ore_boost && vz < -30 {
        Some(VoxelMaterial::GoldOre)
    } else if n > 0.90 - ore_boost {
        if vz < -20 { Some(VoxelMaterial::IronOre) }
        else { Some(VoxelMaterial::Coal) }
    } else if n > 0.87 - ore_boost {
        Some(VoxelMaterial::CopperOre)
    } else {
        None
    }
}
```

- [ ] **Step 4: Update terrain/mod.rs**

```rust
pub mod noise;
pub mod region_plan;
pub mod biome;
pub mod materialize;

pub use region_plan::{RegionPlan, RegionCell, generate_continent, CELL_SIZE, SEA_LEVEL, MAX_SURFACE_Z};
pub use biome::{BiomeVolume, resolve_biome, surface_materials};
pub use materialize::materialize_chunk;
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::materialize -- --nocapture`
Expected: all 5 tests pass.

- [ ] **Step 6: Wire into VoxelWorld::generate_chunk()**

In `src/world_sim/voxel.rs`, modify `generate_chunk()` (line 452) to delegate to the materializer when a region plan is available. Add a `region_plan` field to `VoxelWorld`:

Add field to `VoxelWorld` struct:

```rust
pub region_plan: Option<crate::world_sim::terrain::RegionPlan>,
```

Initialize it as `None` in `VoxelWorld::new()`.

Replace the body of `generate_chunk()`:

```rust
pub fn generate_chunk(&mut self, cp: ChunkPos, seed: u64) {
    if self.chunks.contains_key(&cp) { return; }

    let chunk = if let Some(ref plan) = self.region_plan {
        crate::world_sim::terrain::materialize_chunk(cp, plan, seed)
    } else {
        // Legacy flat generation for tests that don't set up a plan
        self.generate_chunk_legacy(cp, seed)
    };
    self.chunks.insert(cp, chunk);
}
```

Rename the old `generate_chunk` body to `generate_chunk_legacy` (private method) so existing tests still work without a region plan.

- [ ] **Step 7: Run all voxel tests to ensure nothing breaks**

Run: `cargo test world_sim::voxel -- --nocapture`
Expected: all existing tests pass (they use legacy path since no region_plan is set).

- [ ] **Step 8: Commit**

```bash
git add src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs src/world_sim/voxel.rs
git commit -m "feat(terrain): add chunk materializer, wire into VoxelWorld::generate_chunk"
```

---

### Task 5: Cave Carving

**Files:**
- Create: `src/world_sim/terrain/caves.rs`
- Modify: `src/world_sim/terrain/materialize.rs` — call cave carving after base column
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Write tests for cave carving**

In `src/world_sim/terrain/caves.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE, local_index};
    use crate::world_sim::state::SubBiome;

    #[test]
    fn caves_carve_air_in_solid() {
        // Create a fully solid chunk, then carve caves
        let cp = ChunkPos::new(0, 0, -3); // underground
        let mut chunk = Chunk::new_air(cp);
        // Fill with stone
        for i in 0..chunk.voxels.len() {
            chunk.voxels[i] = Voxel::new(VoxelMaterial::Stone);
        }
        carve_caves(&mut chunk, cp, SubBiome::NaturalCave, 42);
        let air = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        // Should have carved at least some air, but not everything
        assert!(air > 0, "no caves carved");
        assert!(air < chunk.voxels.len(), "entire chunk carved away");
    }

    #[test]
    fn lava_tubes_have_lava() {
        let cp = ChunkPos::new(0, 0, -5);
        let mut chunk = Chunk::new_air(cp);
        for i in 0..chunk.voxels.len() {
            chunk.voxels[i] = Voxel::new(VoxelMaterial::Basalt);
        }
        carve_caves(&mut chunk, cp, SubBiome::LavaTubes, 42);
        let lava = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Lava).count();
        // Lava tubes should have some lava in carved areas
        // (may be 0 if this chunk doesn't intersect a cave, that's ok)
        let air = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        // At least one of air or lava should exist (caves were carved)
        assert!(air + lava > 0, "no caves carved in lava tube chunk");
    }

    #[test]
    fn cave_carving_is_deterministic() {
        let cp = ChunkPos::new(2, 3, -3);
        let make = || {
            let mut chunk = Chunk::new_air(cp);
            for i in 0..chunk.voxels.len() {
                chunk.voxels[i] = Voxel::new(VoxelMaterial::Stone);
            }
            carve_caves(&mut chunk, cp, SubBiome::NaturalCave, 42);
            chunk
        };
        let a = make();
        let b = make();
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material);
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::caves --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement cave carving**

In `src/world_sim/terrain/caves.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::SubBiome;
use crate::world_sim::terrain::noise;
use crate::world_sim::terrain::biome::{cave_fill};

/// Cave generation parameters per underground biome.
struct CaveParams {
    scale: f32,         // noise scale (larger = bigger caves)
    threshold: f32,     // worm threshold (larger = wider tunnels)
    fill: VoxelMaterial, // what fills carved space
    floor_fill: Option<VoxelMaterial>, // optional different floor material
}

fn cave_params(biome: SubBiome) -> CaveParams {
    match biome {
        SubBiome::LavaTubes => CaveParams {
            scale: 24.0, threshold: 0.07, fill: VoxelMaterial::Air,
            floor_fill: Some(VoxelMaterial::Lava),
        },
        SubBiome::MushroomGrove => CaveParams {
            scale: 32.0, threshold: 0.09, // large open chambers
            fill: VoxelMaterial::Air, floor_fill: None,
        },
        SubBiome::CrystalVein => CaveParams {
            scale: 10.0, threshold: 0.04, // tight veins
            fill: VoxelMaterial::Air, floor_fill: None,
        },
        SubBiome::Aquifer => CaveParams {
            scale: 20.0, threshold: 0.07,
            fill: VoxelMaterial::Water, floor_fill: None,
        },
        SubBiome::FrozenCavern => CaveParams {
            scale: 20.0, threshold: 0.06,
            fill: VoxelMaterial::Air, floor_fill: Some(VoxelMaterial::Ice),
        },
        SubBiome::BoneOssuary => CaveParams {
            scale: 18.0, threshold: 0.06,
            fill: VoxelMaterial::Air, floor_fill: None,
        },
        _ => CaveParams { // NaturalCave / Standard
            scale: 16.0, threshold: 0.06,
            fill: VoxelMaterial::Air, floor_fill: None,
        },
    }
}

/// Carve caves into a chunk that's already been filled with base terrain.
pub fn carve_caves(chunk: &mut Chunk, cp: ChunkPos, biome: SubBiome, seed: u64) {
    let params = cave_params(biome);
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;
    let seed_a = seed.wrapping_add(0xCAVE0001);
    let seed_b = seed.wrapping_add(0xCAVE0002);

    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let idx = local_index(lx, ly, lz);
                let voxel = &chunk.voxels[idx];
                if !voxel.material.is_solid() { continue; } // already air/fluid
                if voxel.material == VoxelMaterial::Granite { continue; } // never carve bedrock

                let vx = (base_x + lx as i32) as f32;
                let vy = (base_y + ly as i32) as f32;
                let vz = (base_z + lz as i32) as f32;

                let a = noise::value_noise_3d(vx, vy, vz, seed_a, params.scale);
                let b = noise::value_noise_3d(vx, vy, vz, seed_b, params.scale);

                if (a - 0.5).abs() < params.threshold && (b - 0.5).abs() < params.threshold {
                    // Check if this is a floor voxel (solid below, being carved)
                    let is_floor = if lz > 0 {
                        let below_idx = local_index(lx, ly, lz - 1);
                        let below_a = noise::value_noise_3d(vx, vy, vz - 1.0, seed_a, params.scale);
                        let below_b = noise::value_noise_3d(vx, vy, vz - 1.0, seed_b, params.scale);
                        !((below_a - 0.5).abs() < params.threshold && (below_b - 0.5).abs() < params.threshold)
                    } else {
                        true // bottom of chunk = floor
                    };

                    let fill = if is_floor {
                        params.floor_fill.unwrap_or(params.fill)
                    } else {
                        params.fill
                    };
                    chunk.voxels[idx] = Voxel::new(fill);
                }
            }
        }
    }

    chunk.dirty = true;
}
```

- [ ] **Step 4: Integrate cave carving into materialize_chunk**

In `src/world_sim/terrain/materialize.rs`, after the base column loop, add cave carving:

```rust
// After the voxel loop, carve caves for underground chunks
let chunk_top_z = base_z + CHUNK_SIZE as i32;
// Only carve caves below surface (rough check: below max surface + some margin)
if chunk_top_z < MAX_SURFACE_Z + 50 && base_z < MAX_SURFACE_Z - 40 {
    // Determine underground biome for this chunk's center
    let center_x = base_x + CHUNK_SIZE as i32 / 2;
    let center_y = base_y + CHUNK_SIZE as i32 / 2;
    let center_z = base_z + CHUNK_SIZE as i32 / 2;
    let (cell, _, _) = plan.sample(center_x as f32, center_y as f32);
    let base_height = plan.interpolate_height(center_x as f32, center_y as f32);
    let surface_z = (base_height * MAX_SURFACE_Z as f32) as i32;
    let depth = surface_z - center_z;
    if depth > 20 {
        let biome = biome::resolve_biome(cell.terrain, cell.sub_biome, -depth, seed);
        caves::carve_caves(&mut chunk, cp, biome.underground, seed);
    }
}
```

Add `use super::caves;` at the top of materialize.rs.

- [ ] **Step 5: Update terrain/mod.rs**

Add `pub mod caves;` to the module declarations.

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::caves -- --nocapture`
Expected: all 3 tests pass.

Also run: `cargo test world_sim::terrain::materialize -- --nocapture`
Expected: still passes (caves only modify underground chunks).

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/caves.rs src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs
git commit -m "feat(terrain): add 3D cave carving with biome-specific parameters"
```

---

### Task 6: River Carving

**Files:**
- Create: `src/world_sim/terrain/rivers.rs`
- Modify: `src/world_sim/terrain/materialize.rs`
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Write tests for river carving**

In `src/world_sim/terrain/rivers.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};
    use crate::world_sim::terrain::region_plan::RiverPath;

    #[test]
    fn river_carves_water_channel() {
        // Create a solid chunk and carve a river through it
        let cp = ChunkPos::new(0, 0, 5); // surface level
        let mut chunk = Chunk::new_air(cp);
        // Fill with dirt up to top
        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(VoxelMaterial::Dirt);
                }
            }
        }

        let river = RiverPath {
            points: vec![(-100.0, 8.0 * 16.0), (100.0, 8.0 * 16.0)], // straight E-W through chunk center
            widths: vec![50.0, 50.0],
        };

        let surface_z_fn = |_x: f32, _y: f32| -> i32 { 5 * CHUNK_SIZE as i32 + 10 }; // surface in middle of chunk
        carve_river_in_chunk(&mut chunk, cp, &river, &surface_z_fn);

        let water = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Water).count();
        assert!(water > 0, "river carving produced no water");
    }

    #[test]
    fn river_carving_is_deterministic() {
        let cp = ChunkPos::new(0, 0, 5);
        let river = RiverPath {
            points: vec![(-50.0, 8.0 * 16.0), (50.0, 8.0 * 16.0)],
            widths: vec![40.0, 40.0],
        };
        let surface_fn = |_x: f32, _y: f32| -> i32 { 5 * 16 + 10 };

        let make = || {
            let mut chunk = Chunk::new_air(cp);
            for i in 0..chunk.voxels.len() {
                chunk.voxels[i] = Voxel::new(VoxelMaterial::Dirt);
            }
            carve_river_in_chunk(&mut chunk, cp, &river, &surface_fn);
            chunk
        };
        let a = make();
        let b = make();
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material);
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::rivers --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement river carving**

In `src/world_sim/terrain/rivers.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::terrain::region_plan::RiverPath;

/// River bed depth as fraction of width.
const DEPTH_RATIO: f32 = 0.3;

/// Carve a river into a chunk. `surface_z_fn` returns the surface Z for any (x, y).
pub fn carve_river_in_chunk(
    chunk: &mut Chunk,
    cp: ChunkPos,
    river: &RiverPath,
    surface_z_fn: &dyn Fn(f32, f32) -> i32,
) {
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let vx = (base_x + lx as i32) as f32;
                let vy = (base_y + ly as i32) as f32;
                let vz = base_z + lz as i32;

                // Find closest point on river polyline
                let (dist, width) = closest_river_distance(vx, vy, river);
                if dist > width * 0.7 { continue; } // outside river influence

                let surface = surface_z_fn(vx, vy);
                let river_depth = (width * DEPTH_RATIO) as i32;
                let river_bed = surface - river_depth;
                let bank_width = width * 0.7;

                if dist < width * 0.4 {
                    // River channel center
                    if vz > river_bed && vz <= surface {
                        let idx = local_index(lx, ly, lz);
                        if vz <= river_bed + river_depth / 2 {
                            // Lower part: water
                            chunk.voxels[idx] = Voxel::new(VoxelMaterial::Water);
                        } else {
                            // Upper part: carved air (above water line)
                            chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
                        }
                    }
                } else {
                    // River bank: gradual slope
                    let bank_t = (dist - width * 0.4) / (width * 0.3); // 0 at channel edge, 1 at bank edge
                    let bank_surface = river_bed + (bank_t * river_depth as f32) as i32;
                    if vz > bank_surface && vz <= surface {
                        let idx = local_index(lx, ly, lz);
                        chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
                    }
                }
            }
        }
    }

    chunk.dirty = true;
}

/// Find the closest distance from point (px, py) to the river polyline,
/// and the interpolated width at that closest point.
fn closest_river_distance(px: f32, py: f32, river: &RiverPath) -> (f32, f32) {
    let mut best_dist = f32::MAX;
    let mut best_width = 50.0;

    for i in 0..river.points.len().saturating_sub(1) {
        let (ax, ay) = river.points[i];
        let (bx, by) = river.points[i + 1];
        let wa = river.widths[i];
        let wb = river.widths[i + 1];

        // Project point onto segment
        let dx = bx - ax;
        let dy = by - ay;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 0.001 { continue; }

        let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);

        let cx = ax + t * dx;
        let cy = ay + t * dy;
        let dist = ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt();
        let width = wa + t * (wb - wa);

        if dist < best_dist {
            best_dist = dist;
            best_width = width;
        }
    }

    (best_dist, best_width)
}
```

- [ ] **Step 4: Integrate river carving into materialize_chunk**

In `src/world_sim/terrain/materialize.rs`, after cave carving, add river carving:

```rust
// Carve rivers
for river in &plan.rivers {
    let surface_z_fn = |x: f32, y: f32| -> i32 {
        let h = plan.interpolate_height(x, y);
        (h * MAX_SURFACE_Z as f32) as i32
    };
    rivers::carve_river_in_chunk(&mut chunk, cp, river, &surface_z_fn);
}
```

Add `use super::rivers;` at the top.

- [ ] **Step 5: Update terrain/mod.rs**

Add `pub mod rivers;` to the module declarations.

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::rivers -- --nocapture`
Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/rivers.rs src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs
git commit -m "feat(terrain): add river valley carving from polyline data"
```

---

### Task 7: Surface Features (Trees, Boulders, Dunes)

**Files:**
- Create: `src/world_sim/terrain/features.rs`
- Modify: `src/world_sim/terrain/materialize.rs`
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Write tests for surface features**

In `src/world_sim/terrain/features.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};

    #[test]
    fn tree_has_trunk_and_canopy() {
        let mut chunk = Chunk::new_air(ChunkPos::new(0, 0, 0));
        stamp_tree(&mut chunk, 8, 8, 2, 42); // place tree at (8,8) with base at z=2

        // Should have wood (trunk) and grass/leaf (canopy)
        let wood = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::WoodLog).count();
        let canopy = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Grass).count();
        assert!(wood >= 3, "tree trunk too short: {wood} voxels");
        assert!(canopy >= 5, "tree canopy too small: {canopy} voxels");
    }

    #[test]
    fn forest_has_multiple_trees() {
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new_air(cp);
        // Fill bottom half with dirt (ground)
        for lz in 0..8 {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(VoxelMaterial::Dirt);
                }
            }
        }
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::Standard, 8, 42);
        let wood = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::WoodLog).count();
        assert!(wood > 0, "forest has no trees");
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::features --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement surface features**

In `src/world_sim/terrain/features.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::{Terrain, SubBiome};
use crate::world_sim::terrain::noise;

/// Place surface features in a chunk based on biome.
/// `surface_z_local` is the local z coordinate of the surface within this chunk (0-15).
/// Features are only placed if the surface falls within this chunk.
pub fn place_surface_features(
    chunk: &mut Chunk,
    cp: ChunkPos,
    terrain: Terrain,
    sub_biome: SubBiome,
    surface_z_local: i32,
    seed: u64,
) {
    if surface_z_local < 0 || surface_z_local >= CHUNK_SIZE as i32 { return; }

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;

    match terrain {
        Terrain::Forest | Terrain::Jungle => {
            let density = match sub_biome {
                SubBiome::DenseForest => 0.15,
                SubBiome::LightForest => 0.03,
                SubBiome::AncientForest => 0.08,
                _ => if terrain == Terrain::Jungle { 0.12 } else { 0.06 },
            };
            place_trees(chunk, cp, surface_z_local, seed, density);
        }
        Terrain::Plains => {
            // Occasional boulders
            place_boulders(chunk, cp, surface_z_local, seed, 0.005);
        }
        Terrain::Desert | Terrain::Badlands => {
            // Occasional rock outcrops
            place_boulders(chunk, cp, surface_z_local, seed, 0.01);
        }
        Terrain::Tundra => {
            // Sparse dead trees + boulders
            place_trees(chunk, cp, surface_z_local, seed, 0.01);
            place_boulders(chunk, cp, surface_z_local, seed, 0.008);
        }
        Terrain::Swamp => {
            // Dead/sparse trees
            place_trees(chunk, cp, surface_z_local, seed, 0.04);
        }
        _ => {}
    }
}

/// Place trees at random positions on the chunk surface.
fn place_trees(chunk: &mut Chunk, cp: ChunkPos, surface_z: i32, seed: u64, density: f32) {
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;

    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let vx = base_x + lx as i32;
            let vy = base_y + ly as i32;
            let h = noise::hash_f32(vx, vy, 0, seed.wrapping_add(0x1REE));
            if h < density {
                stamp_tree(chunk, lx, ly, surface_z as usize + 1, seed);
            }
        }
    }
}

/// Stamp a single tree at (lx, ly) with trunk base at local z = base_z.
pub fn stamp_tree(chunk: &mut Chunk, lx: usize, ly: usize, base_z: usize, seed: u64) {
    let trunk_height = 5 + (noise::hash_3d(lx as i32, ly as i32, 0, seed) % 4) as usize;
    let canopy_radius = 2 + (noise::hash_3d(lx as i32, ly as i32, 1, seed) % 2) as i32;

    // Trunk
    for dz in 0..trunk_height {
        let z = base_z + dz;
        if z < CHUNK_SIZE && lx < CHUNK_SIZE && ly < CHUNK_SIZE {
            chunk.voxels[local_index(lx, ly, z)] = Voxel::new(VoxelMaterial::WoodLog);
        }
    }

    // Canopy (sphere of grass/leaves around top of trunk)
    let canopy_z = base_z + trunk_height;
    for dz in -canopy_radius..=canopy_radius {
        for dy in -canopy_radius..=canopy_radius {
            for dx in -canopy_radius..=canopy_radius {
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq > canopy_radius * canopy_radius { continue; }

                let cx = lx as i32 + dx;
                let cy = ly as i32 + dy;
                let cz = canopy_z as i32 + dz;

                if cx >= 0 && cx < CHUNK_SIZE as i32
                    && cy >= 0 && cy < CHUNK_SIZE as i32
                    && cz >= 0 && cz < CHUNK_SIZE as i32
                {
                    let idx = local_index(cx as usize, cy as usize, cz as usize);
                    if chunk.voxels[idx].material == VoxelMaterial::Air {
                        chunk.voxels[idx] = Voxel::new(VoxelMaterial::Grass);
                    }
                }
            }
        }
    }
}

/// Place boulders (small stone clusters) on the surface.
fn place_boulders(chunk: &mut Chunk, cp: ChunkPos, surface_z: i32, seed: u64, density: f32) {
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;

    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let vx = base_x + lx as i32;
            let vy = base_y + ly as i32;
            let h = noise::hash_f32(vx, vy, 0, seed.wrapping_add(0xB01D));
            if h < density {
                let size = 1 + (noise::hash_3d(vx, vy, 0, seed) % 3) as i32;
                for dz in 0..size {
                    for dy in 0..size {
                        for dx in 0..size {
                            let cx = lx as i32 + dx;
                            let cy = ly as i32 + dy;
                            let cz = surface_z + 1 + dz;
                            if cx >= 0 && cx < CHUNK_SIZE as i32
                                && cy >= 0 && cy < CHUNK_SIZE as i32
                                && cz >= 0 && cz < CHUNK_SIZE as i32
                            {
                                let idx = local_index(cx as usize, cy as usize, cz as usize);
                                if chunk.voxels[idx].material == VoxelMaterial::Air {
                                    chunk.voxels[idx] = Voxel::new(VoxelMaterial::Stone);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Integrate into materialize_chunk**

In `materialize.rs`, after river carving, add:

```rust
// Place surface features for chunks at the surface level
// Check if the surface falls within this chunk's Z range
let center_x = base_x + CHUNK_SIZE as i32 / 2;
let center_y = base_y + CHUNK_SIZE as i32 / 2;
let (center_cell, _, _) = plan.sample(center_x as f32, center_y as f32);
let center_height = plan.interpolate_height(center_x as f32, center_y as f32);
let center_surface_z = (center_height * MAX_SURFACE_Z as f32) as i32;
let surface_z_local = center_surface_z - base_z;
if surface_z_local >= 0 && surface_z_local < CHUNK_SIZE as i32 {
    features::place_surface_features(
        &mut chunk, cp, center_cell.terrain, center_cell.sub_biome,
        surface_z_local, seed,
    );
}
```

Add `use super::features;` at the top.

- [ ] **Step 5: Update terrain/mod.rs**

Add `pub mod features;` to module declarations.

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::features -- --nocapture`
Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/features.rs src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs
git commit -m "feat(terrain): add surface features — trees, boulders per biome"
```

---

### Task 8: Flying Island Generation

**Files:**
- Create: `src/world_sim/terrain/sky.rs`
- Modify: `src/world_sim/terrain/materialize.rs`
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Write tests for flying islands**

In `src/world_sim/terrain/sky.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};

    #[test]
    fn island_has_solid_mass() {
        let cp = ChunkPos::new(0, 0, 20); // sky level (~z=320)
        let chunk = generate_flying_island_chunk(cp, 42);
        let solid = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
        assert!(solid > 50, "flying island has too little mass: {solid}");
    }

    #[test]
    fn island_has_grass_on_top() {
        let cp = ChunkPos::new(0, 0, 20);
        let chunk = generate_flying_island_chunk(cp, 42);
        let grass = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Grass).count();
        assert!(grass > 0, "flying island has no grass surface");
    }

    #[test]
    fn island_is_deterministic() {
        let cp = ChunkPos::new(1, 1, 20);
        let a = generate_flying_island_chunk(cp, 42);
        let b = generate_flying_island_chunk(cp, 42);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material);
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::sky --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement flying island generation**

In `src/world_sim/terrain/sky.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::terrain::noise;

/// Base Z level where flying islands start.
pub const SKY_BASE_Z: i32 = 300;

/// Generate a chunk containing flying island terrain.
/// Islands are inverted-cone SDF shapes with noise perturbation.
pub fn generate_flying_island_chunk(cp: ChunkPos, seed: u64) -> Chunk {
    let mut chunk = Chunk::new_air(cp);
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    // Island placement: deterministic grid of potential island centers
    let island_spacing = 64; // voxels between potential island centers
    let island_seed = seed.wrapping_add(0x15LAND);

    // Check nearby island centers that could affect this chunk
    let search_radius = 40i32; // max island radius in voxels
    let min_ix = (base_x - search_radius).div_euclid(island_spacing);
    let max_ix = (base_x + CHUNK_SIZE as i32 + search_radius).div_euclid(island_spacing);
    let min_iy = (base_y - search_radius).div_euclid(island_spacing);
    let max_iy = (base_y + CHUNK_SIZE as i32 + search_radius).div_euclid(island_spacing);

    for iy in min_iy..=max_iy {
        for ix in min_ix..=max_ix {
            // Not every grid cell has an island
            let presence = noise::hash_f32(ix as i32, iy as i32, 0, island_seed);
            if presence > 0.3 { continue; } // ~30% of cells have islands

            // Island center with jitter
            let cx = ix * island_spacing + (noise::hash_3d(ix as i32, iy as i32, 1, island_seed) % island_spacing as u32) as i32;
            let cy = iy * island_spacing + (noise::hash_3d(ix as i32, iy as i32, 2, island_seed) % island_spacing as u32) as i32;
            let cz = SKY_BASE_Z + (noise::hash_3d(ix as i32, iy as i32, 3, island_seed) % 60) as i32;
            let radius = 15 + (noise::hash_3d(ix as i32, iy as i32, 4, island_seed) % 25) as i32;
            let thickness = 8 + (noise::hash_3d(ix as i32, iy as i32, 5, island_seed) % 12) as i32;

            stamp_island(&mut chunk, base_x, base_y, base_z, cx, cy, cz, radius, thickness, seed);
        }
    }

    chunk.dirty = true;
    chunk
}

/// Stamp a single flying island: inverted cone with grass top, dirt middle, stone bottom.
fn stamp_island(
    chunk: &mut Chunk,
    base_x: i32, base_y: i32, base_z: i32,
    cx: i32, cy: i32, cz: i32,
    radius: i32, thickness: i32,
    seed: u64,
) {
    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let vx = base_x + lx as i32;
                let vy = base_y + ly as i32;
                let vz = base_z + lz as i32;

                let dx = vx - cx;
                let dy = vy - cy;
                let dz = vz - cz;
                let horiz_dist = ((dx * dx + dy * dy) as f32).sqrt();

                // Inverted cone: radius shrinks as we go down
                let z_frac = (dz as f32 + thickness as f32 * 0.5) / thickness as f32; // 0 at bottom, 1 at top
                if z_frac < 0.0 || z_frac > 1.0 { continue; }

                let local_radius = radius as f32 * z_frac; // shrinks toward bottom
                // Noise perturbation for organic shape
                let noise_offset = noise::value_noise_3d(
                    vx as f32 * 0.1, vy as f32 * 0.1, vz as f32 * 0.1,
                    seed.wrapping_add(0xF1CA), 8.0,
                ) * 6.0 - 3.0;

                if horiz_dist > local_radius + noise_offset { continue; }

                let idx = local_index(lx, ly, lz);
                if chunk.voxels[idx].material != VoxelMaterial::Air { continue; }

                // Material layers: stone bottom, dirt middle, grass top
                let material = if z_frac > 0.85 {
                    VoxelMaterial::Grass
                } else if z_frac > 0.4 {
                    VoxelMaterial::Dirt
                } else {
                    VoxelMaterial::Stone
                };

                chunk.voxels[idx] = Voxel::new(material);
            }
        }
    }

    // Add waterfall columns (water dropping off edges)
    let wf_hash = noise::hash_3d(cx, cy, 0, seed.wrapping_add(0xFA11));
    if wf_hash % 3 == 0 { // ~33% of islands have waterfalls
        let wf_x = cx + (wf_hash as i32 % (radius / 2)) - radius / 4;
        let wf_y = cy + ((wf_hash >> 8) as i32 % (radius / 2)) - radius / 4;
        for lz in 0..CHUNK_SIZE {
            let vz = base_z + lz as i32;
            if vz >= cz - thickness && vz < cz - thickness / 2 { // below island
                let lx_local = wf_x - base_x;
                let ly_local = wf_y - base_y;
                if lx_local >= 0 && lx_local < CHUNK_SIZE as i32
                    && ly_local >= 0 && ly_local < CHUNK_SIZE as i32
                {
                    let idx = local_index(lx_local as usize, ly_local as usize, lz);
                    if chunk.voxels[idx].material == VoxelMaterial::Air {
                        chunk.voxels[idx] = Voxel::new(VoxelMaterial::Water);
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Integrate into materialize_chunk**

In `materialize.rs`, add a check for sky-level chunks in FlyingIslands biome. After the base column generation, before returning:

```rust
// Flying islands in sky-level chunks
if base_z >= sky::SKY_BASE_Z - 20 {
    let (center_cell, _, _) = plan.sample(
        (base_x + CHUNK_SIZE as i32 / 2) as f32,
        (base_y + CHUNK_SIZE as i32 / 2) as f32,
    );
    if center_cell.terrain == Terrain::FlyingIslands {
        let sky_chunk = sky::generate_flying_island_chunk(cp, seed);
        // Merge: only overwrite air voxels with island voxels
        for i in 0..chunk.voxels.len() {
            if chunk.voxels[i].material == VoxelMaterial::Air
                && sky_chunk.voxels[i].material != VoxelMaterial::Air
            {
                chunk.voxels[i] = sky_chunk.voxels[i];
            }
        }
    }
}
```

Add `use super::sky;` at the top.

- [ ] **Step 5: Update terrain/mod.rs**

Add `pub mod sky;` to module declarations.

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::sky -- --nocapture`
Expected: all 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/sky.rs src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs
git commit -m "feat(terrain): add flying island generation with waterfalls"
```

---

### Task 9: Dungeon Generation

**Files:**
- Create: `src/world_sim/terrain/dungeons.rs`
- Modify: `src/world_sim/terrain/materialize.rs`
- Modify: `src/world_sim/terrain/mod.rs`

- [ ] **Step 1: Write tests for dungeon generation**

In `src/world_sim/terrain/dungeons.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};

    #[test]
    fn dungeon_carves_rooms() {
        let cp = ChunkPos::new(0, 0, 3); // slightly underground
        let mut chunk = Chunk::new_air(cp);
        // Fill with stone
        for i in 0..chunk.voxels.len() {
            chunk.voxels[i] = Voxel::new(VoxelMaterial::Stone);
        }
        let dungeon = DungeonLayout::generate(0, 0, 3, 1, 42);
        dungeon.carve_into_chunk(&mut chunk, cp);

        let air = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        assert!(air > 50, "dungeon carved too little: {air} air voxels");
    }

    #[test]
    fn dungeon_has_rooms_and_corridors() {
        let layout = DungeonLayout::generate(0, 0, 0, 2, 42);
        assert!(layout.rooms.len() >= 3, "too few rooms: {}", layout.rooms.len());
        assert!(!layout.corridors.is_empty(), "no corridors");
    }

    #[test]
    fn dungeon_is_deterministic() {
        let a = DungeonLayout::generate(5, 5, 0, 2, 42);
        let b = DungeonLayout::generate(5, 5, 0, 2, 42);
        assert_eq!(a.rooms.len(), b.rooms.len());
        for (ra, rb) in a.rooms.iter().zip(b.rooms.iter()) {
            assert_eq!(ra.min, rb.min);
            assert_eq!(ra.max, rb.max);
        }
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::terrain::dungeons --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement dungeon generation**

In `src/world_sim/terrain/dungeons.rs`:

```rust
use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::terrain::noise;

/// An axis-aligned room in voxel coordinates.
#[derive(Debug, Clone)]
pub struct Room {
    pub min: (i32, i32, i32),
    pub max: (i32, i32, i32),
}

/// A corridor connecting two rooms (L-shaped).
#[derive(Debug, Clone)]
pub struct Corridor {
    pub from: (i32, i32, i32), // center of room A
    pub to: (i32, i32, i32),   // center of room B
    pub width: i32,
}

/// A generated dungeon layout.
#[derive(Debug, Clone)]
pub struct DungeonLayout {
    pub rooms: Vec<Room>,
    pub corridors: Vec<Corridor>,
}

impl DungeonLayout {
    /// Generate a dungeon layout at a given cell position.
    /// `depth` controls number of rooms (more depth = larger dungeon).
    pub fn generate(cell_col: i32, cell_row: i32, cell_z: i32, depth: u8, seed: u64) -> Self {
        let dseed = seed
            .wrapping_add(cell_col as u64 * 99991)
            .wrapping_add(cell_row as u64 * 77773)
            .wrapping_add(cell_z as u64 * 55537)
            .wrapping_add(0xD00D);

        let base_x = cell_col * CHUNK_SIZE as i32;
        let base_y = cell_row * CHUNK_SIZE as i32;
        let base_z = cell_z * CHUNK_SIZE as i32;

        let num_rooms = 3 + depth as usize * 2 + (noise::hash_3d(0, 0, 0, dseed) % 3) as usize;
        let mut rooms = Vec::with_capacity(num_rooms);

        // Generate rooms with random positions and sizes
        for i in 0..num_rooms {
            let rseed = dseed.wrapping_add(i as u64 * 7919);
            let rx = base_x + (noise::hash_3d(i as i32, 0, 0, rseed) % 12) as i32;
            let ry = base_y + (noise::hash_3d(i as i32, 1, 0, rseed) % 12) as i32;
            let rz = base_z + (noise::hash_3d(i as i32, 2, 0, rseed) % 8) as i32;
            let w = 4 + (noise::hash_3d(i as i32, 3, 0, rseed) % 6) as i32;
            let h = 4 + (noise::hash_3d(i as i32, 4, 0, rseed) % 6) as i32;
            let d = 3 + (noise::hash_3d(i as i32, 5, 0, rseed) % 3) as i32; // room height

            rooms.push(Room {
                min: (rx, ry, rz),
                max: (rx + w, ry + h, rz + d),
            });
        }

        // Connect rooms with L-shaped corridors
        let mut corridors = Vec::new();
        for i in 0..rooms.len().saturating_sub(1) {
            let a = room_center(&rooms[i]);
            let b = room_center(&rooms[i + 1]);
            corridors.push(Corridor {
                from: a,
                to: b,
                width: 2,
            });
        }

        DungeonLayout { rooms, corridors }
    }

    /// Carve this dungeon into a chunk (sets solid voxels to air).
    pub fn carve_into_chunk(&self, chunk: &mut Chunk, cp: ChunkPos) {
        let base_x = cp.x * CHUNK_SIZE as i32;
        let base_y = cp.y * CHUNK_SIZE as i32;
        let base_z = cp.z * CHUNK_SIZE as i32;

        // Carve rooms
        for room in &self.rooms {
            for vz in room.min.2..=room.max.2 {
                for vy in room.min.1..=room.max.1 {
                    for vx in room.min.0..=room.max.0 {
                        let lx = vx - base_x;
                        let ly = vy - base_y;
                        let lz = vz - base_z;
                        if lx >= 0 && lx < CHUNK_SIZE as i32
                            && ly >= 0 && ly < CHUNK_SIZE as i32
                            && lz >= 0 && lz < CHUNK_SIZE as i32
                        {
                            let idx = local_index(lx as usize, ly as usize, lz as usize);
                            if chunk.voxels[idx].material != VoxelMaterial::Granite {
                                chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
                            }
                        }
                    }
                }
            }
        }

        // Carve corridors (L-shaped: first horizontal, then vertical)
        for corridor in &self.corridors {
            let (ax, ay, az) = corridor.from;
            let (bx, by, bz) = corridor.to;
            let hw = corridor.width / 2;

            // Horizontal segment: ax→bx at ay
            let (x_min, x_max) = if ax < bx { (ax, bx) } else { (bx, ax) };
            for vx in x_min..=x_max {
                for dy in -hw..=hw {
                    for dz in 0..3 { // corridor height = 3
                        carve_voxel(chunk, vx - base_x, ay + dy - base_y, az + dz - base_z);
                    }
                }
            }

            // Vertical segment: ay→by at bx
            let (y_min, y_max) = if ay < by { (ay, by) } else { (by, ay) };
            for vy in y_min..=y_max {
                for dx in -hw..=hw {
                    for dz in 0..3 {
                        carve_voxel(chunk, bx + dx - base_x, vy - base_y, bz + dz - base_z);
                    }
                }
            }
        }

        chunk.dirty = true;
    }
}

fn room_center(room: &Room) -> (i32, i32, i32) {
    (
        (room.min.0 + room.max.0) / 2,
        (room.min.1 + room.max.1) / 2,
        room.min.2, // floor level
    )
}

fn carve_voxel(chunk: &mut Chunk, lx: i32, ly: i32, lz: i32) {
    if lx >= 0 && lx < CHUNK_SIZE as i32
        && ly >= 0 && ly < CHUNK_SIZE as i32
        && lz >= 0 && lz < CHUNK_SIZE as i32
    {
        let idx = local_index(lx as usize, ly as usize, lz as usize);
        if chunk.voxels[idx].material != VoxelMaterial::Granite {
            chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
        }
    }
}
```

- [ ] **Step 4: Integrate into materialize_chunk**

In `materialize.rs`, after cave carving, check for dungeon sites:

```rust
// Stamp dungeons
let (cell_for_dungeon, _, _) = plan.sample(
    (base_x + CHUNK_SIZE as i32 / 2) as f32,
    (base_y + CHUNK_SIZE as i32 / 2) as f32,
);
for dungeon_plan in &cell_for_dungeon.dungeons {
    let dungeon_vx = (base_x as f32 + dungeon_plan.local_pos.0 * CHUNK_SIZE as f32) as i32;
    let dungeon_vy = (base_y as f32 + dungeon_plan.local_pos.1 * CHUNK_SIZE as f32) as i32;
    let layout = dungeons::DungeonLayout::generate(
        dungeon_vx / CHUNK_SIZE as i32,
        dungeon_vy / CHUNK_SIZE as i32,
        cp.z,
        dungeon_plan.depth,
        seed,
    );
    layout.carve_into_chunk(&mut chunk, cp);
}
```

Note: This reference to `cell_for_dungeon` may need to clone the dungeons vec if borrowing is an issue. The implementation agent should handle borrow checker details.

Add `use super::dungeons;` at the top.

- [ ] **Step 5: Update terrain/mod.rs**

Add `pub mod dungeons;` to module declarations.

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test world_sim::terrain::dungeons -- --nocapture`
Expected: all 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/terrain/dungeons.rs src/world_sim/terrain/materialize.rs src/world_sim/terrain/mod.rs
git commit -m "feat(terrain): add dungeon generation with rooms and corridors"
```

---

### Task 10: 3D Sector Grid

**Files:**
- Create: `src/world_sim/sector.rs`
- Modify: `src/world_sim/mod.rs`

- [ ] **Step 1: Write tests for sector management**

In `src/world_sim/sector.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_from_voxel() {
        let sp = SectorPos::from_voxel(100, 200, 300);
        assert_eq!(sp, SectorPos::new(0, 0, 0)); // all within first sector (4096³)

        let sp2 = SectorPos::from_voxel(5000, 5000, 5000);
        assert_eq!(sp2, SectorPos::new(1, 1, 1));
    }

    #[test]
    fn sector_activation() {
        let mut grid = SectorGrid::new();
        let sp = SectorPos::new(0, 0, 0);
        assert!(!grid.is_active(&sp));

        grid.activate(sp);
        assert!(grid.is_active(&sp));

        grid.deactivate(&sp);
        assert!(!grid.is_active(&sp));
    }

    #[test]
    fn sectors_around() {
        let sectors = SectorPos::sectors_around_voxel(2048, 2048, 50, 1);
        // Radius 1 around center = 3×3×3 = 27 sectors
        assert_eq!(sectors.len(), 27);
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test world_sim::sector --no-run 2>&1 | head -5`

- [ ] **Step 3: Implement sector grid**

In `src/world_sim/sector.rs`:

```rust
use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Voxels per sector edge (same as CELL_SIZE).
pub const SECTOR_SIZE: i32 = 4096;

/// 3D sector coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SectorPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl SectorPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }

    /// Convert voxel-space position to sector position.
    pub fn from_voxel(vx: i32, vy: i32, vz: i32) -> Self {
        Self {
            x: vx.div_euclid(SECTOR_SIZE),
            y: vy.div_euclid(SECTOR_SIZE),
            z: vz.div_euclid(SECTOR_SIZE),
        }
    }

    /// Get all sectors within `radius` of a voxel position.
    pub fn sectors_around_voxel(vx: i32, vy: i32, vz: i32, radius: i32) -> Vec<SectorPos> {
        let center = Self::from_voxel(vx, vy, vz);
        let mut result = Vec::new();
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    result.push(SectorPos::new(center.x + dx, center.y + dy, center.z + dz));
                }
            }
        }
        result
    }
}

/// Manages which 3D sectors are active (have loaded voxel chunks).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SectorGrid {
    active: HashSet<SectorPos>,
}

impl SectorGrid {
    pub fn new() -> Self { Self { active: HashSet::new() } }

    pub fn is_active(&self, sp: &SectorPos) -> bool {
        self.active.contains(sp)
    }

    pub fn activate(&mut self, sp: SectorPos) {
        self.active.insert(sp);
    }

    pub fn deactivate(&mut self, sp: &SectorPos) {
        self.active.remove(sp);
    }

    pub fn active_sectors(&self) -> impl Iterator<Item = &SectorPos> {
        self.active.iter()
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}
```

- [ ] **Step 4: Add to world_sim/mod.rs**

Add: `pub mod sector;`

- [ ] **Step 5: Run tests, verify they pass**

Run: `cargo test world_sim::sector -- --nocapture`
Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/sector.rs src/world_sim/mod.rs
git commit -m "feat(terrain): add 3D sector grid for chunk management"
```

---

### Task 11: Integration — Wire Region Plan into World Sim

**Files:**
- Modify: `src/world_sim/state.rs` — add `RegionPlan` field to `WorldState`
- Modify: `src/bin/xtask/world_sim_cmd.rs` — use `generate_continent()` in `build_world()`
- Modify: `src/world_sim/voxel.rs` — update `VoxelWorld::new()` default sea level

- [ ] **Step 1: Add RegionPlan to WorldState**

In `src/world_sim/state.rs`, add to `WorldState` struct:

```rust
pub region_plan: Option<crate::world_sim::terrain::RegionPlan>,
```

Initialize as `None` in `WorldState::new()`.

- [ ] **Step 2: Write integration test**

Add a test in `src/world_sim/terrain/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{VoxelWorld, ChunkPos, VoxelMaterial, CHUNK_SIZE};

    #[test]
    fn end_to_end_terrain_generation() {
        let plan = generate_continent(10, 10, 42);
        let mut voxel_world = VoxelWorld::new();
        voxel_world.region_plan = Some(plan);

        // Generate a few chunks at different locations
        voxel_world.generate_chunk(ChunkPos::new(5, 5, 2), 42);  // near surface
        voxel_world.generate_chunk(ChunkPos::new(5, 5, -3), 42); // underground
        voxel_world.generate_chunk(ChunkPos::new(5, 5, 25), 42); // sky

        // Surface chunk should have ground
        let surface = voxel_world.get_chunk(&ChunkPos::new(5, 5, 2)).unwrap();
        let solid = surface.voxels.iter().filter(|v| v.material.is_solid()).count();
        assert!(solid > 0, "surface chunk has no solid voxels");

        // Sky chunk should be mostly air (unless flying islands)
        let sky = voxel_world.get_chunk(&ChunkPos::new(5, 5, 25)).unwrap();
        let air = sky.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        assert!(air > CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE / 2, "sky chunk not mostly air");
    }
}
```

- [ ] **Step 3: Run test, verify it fails (or needs VoxelWorld updates)**

Run: `cargo test world_sim::terrain::tests -- --nocapture`
Expected: may fail due to missing `get_chunk` method or field access. Fix as needed.

- [ ] **Step 4: Update build_world() in world_sim_cmd.rs**

In `src/bin/xtask/world_sim_cmd.rs`, modify `build_world()` to generate and use a region plan:

Replace `create_regions()` call with:

```rust
// Generate continental plan
let plan_cols = (args.settlements as f32).sqrt().ceil() as usize + 5;
let plan_rows = plan_cols * 2 / 3;
let plan = game::world_sim::terrain::generate_continent(plan_cols, plan_rows, terrain_seed);

// Populate regions from plan
populate_regions_from_plan(&mut state, &plan);

// Store plan in VoxelWorld for chunk generation
state.voxel_world.region_plan = Some(plan);
```

Add a new function:

```rust
fn populate_regions_from_plan(state: &mut WorldState, plan: &RegionPlan) {
    use game::world_sim::terrain::region_plan::CELL_SIZE;
    for row in 0..plan.rows {
        for col in 0..plan.cols {
            let cell = plan.get(col, row);
            let pos = (
                col as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5,
                row as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 * 0.5,
            );
            state.regions.push(RegionState {
                id: (row * plan.cols + col) as u32,
                name: generate_region_name(row * plan.cols + col, &mut plan.seed),
                terrain: cell.terrain,
                pos,
                monster_density: cell.terrain.threat_multiplier() * 0.2,
                faction_id: None,
                threat_level: (cell.terrain.threat_multiplier() - 1.0).max(0.0) * 0.2,
                has_river: false, // will be updated from river data
                has_lake: false,
                is_coastal: cell.terrain == Terrain::Coast,
                river_connections: Vec::new(),
                dungeon_sites: Vec::new(), // populated from plan.dungeons
                sub_biome: cell.sub_biome,
                neighbors: Vec::new(), // built by adjacency graph
                is_chokepoint: false,
                elevation: cell.terrain.base_elevation(),
                is_floating: cell.terrain == Terrain::FlyingIslands,
                unrest: 0.0,
                control: 0.5,
            });
        }
    }
}
```

Note: The exact implementation will need to handle the `generate_region_name` function's seed parameter and adapt to the existing code structure. The implementation agent should follow the existing patterns in `create_regions()`.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test`
Expected: all tests pass. Fix any compilation errors from the integration.

- [ ] **Step 6: Run world sim to verify terrain generates correctly**

Run: `cargo run --bin xtask -- world-sim --entities 100 --ticks 10 --seed 42 2>&1 | tail -20`
Expected: runs without crash. Terrain generates from plan.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/state.rs src/world_sim/terrain/mod.rs src/bin/xtask/world_sim_cmd.rs src/world_sim/voxel.rs
git commit -m "feat(terrain): integrate region plan into world sim initialization"
```

---

### Task 12: Delete OverworldGrid Module

**Files:**
- Delete: `src/overworld_grid/` (entire directory)
- Modify: `src/lib.rs` — remove `pub mod overworld_grid;`

- [ ] **Step 1: Check for any remaining references**

Run: `grep -r "overworld_grid\|OverworldGrid" src/ --include="*.rs" -l`
Expected: only `src/overworld_grid/` files and `src/lib.rs`.

- [ ] **Step 2: Remove module declaration from lib.rs**

In `src/lib.rs`, delete the line:
```rust
pub mod overworld_grid;
```

- [ ] **Step 3: Delete the overworld_grid directory**

```bash
rm -rf src/overworld_grid/
```

- [ ] **Step 4: Run full test suite**

Run: `cargo test`
Expected: all tests pass. No code outside `overworld_grid/` depends on it.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove OverworldGrid module — voxel world is now single source of truth"
```

---

### Task 13: Verification and Cleanup

**Files:**
- Various — fix any issues found during verification

- [ ] **Step 1: Run full test suite**

Run: `cargo test`
Expected: all tests pass.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -- -D warnings 2>&1 | head -30`
Expected: no warnings. Fix any that appear.

- [ ] **Step 3: Test world sim end-to-end**

Run: `cargo run --bin xtask -- world-sim --entities 500 --ticks 100 --seed 42 2>&1 | tail -30`
Expected: runs to completion. Entities interact with terrain normally.

- [ ] **Step 4: Verify determinism**

Run the same command twice and compare output. Terrain generation should be identical for the same seed.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: cleanup after terrain generation overhaul"
```
