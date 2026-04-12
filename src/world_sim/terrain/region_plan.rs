use serde::{Serialize, Deserialize};
use crate::world_sim::state::{Terrain, SubBiome};
use super::noise;

pub use crate::world_sim::constants::{CELL_SIZE, MAX_SURFACE_Z, SEA_LEVEL};

// ---------------------------------------------------------------------------
// Settlement / dungeon / road / river data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementPlan {
    pub kind: SettlementKind,
    /// Position within the cell, normalised [0, 1).
    pub local_pos: (f32, f32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementKind {
    Town,
    Castle,
    Camp,
    Ruin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DungeonPlan {
    /// Position within the cell, normalised [0, 1).
    pub local_pos: (f32, f32),
    pub depth: u8,
}

/// A river stored as a polyline in world-space voxel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverPath {
    pub points: Vec<(f32, f32)>,
    pub widths: Vec<f32>,
}

/// A road segment in world-space voxel coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegment {
    pub from: (f32, f32),
    pub to: (f32, f32),
}

// ---------------------------------------------------------------------------
// RegionCell / RegionPlan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCell {
    /// Normalised elevation [0, 1].
    pub height: f32,
    /// Normalised moisture [0, 1].
    pub moisture: f32,
    /// Normalised temperature [0, 1].
    pub temperature: f32,
    pub terrain: Terrain,
    pub sub_biome: SubBiome,
    pub settlement: Option<SettlementPlan>,
    pub dungeons: Vec<DungeonPlan>,
    pub has_road: bool,
}

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

    /// Sample at voxel-space position. Returns (cell_ref, local_x, local_y)
    /// where local coords are in [0, CELL_SIZE).
    pub fn sample(&self, vx: f32, vy: f32) -> (&RegionCell, f32, f32) {
        let col = (vx / CELL_SIZE as f32).floor().clamp(0.0, (self.cols - 1) as f32) as usize;
        let row = (vy / CELL_SIZE as f32).floor().clamp(0.0, (self.rows - 1) as f32) as usize;
        let local_x = vx - col as f32 * CELL_SIZE as f32;
        let local_y = vy - row as f32 * CELL_SIZE as f32;
        (self.get(col, row), local_x, local_y)
    }

    /// Convert the region cell grid to GPU-uploadable format.
    /// Callers pass the result to `TerrainComputePipeline::upload_region_plan`.
    #[cfg(feature = "app")]
    pub fn to_gpu_cells(&self) -> Vec<voxel_engine::terrain_compute::RegionCellGpu> {
        self.cells
            .iter()
            .map(|c| voxel_engine::terrain_compute::RegionCellGpu {
                height: c.height,
                moisture: c.moisture,
                temperature: c.temperature,
                terrain: c.terrain.to_u32(),
                sub_biome: c.sub_biome.to_u32(),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            })
            .collect()
    }

    /// Flatten the river polylines into GPU buffer format:
    /// a single `points` buffer plus a `headers` buffer describing each
    /// river's `(start_idx, length)` range in the points buffer.
    /// Pass the result to `TerrainComputePipeline::upload_rivers`.
    #[cfg(feature = "app")]
    pub fn to_gpu_rivers(
        &self,
    ) -> (
        Vec<voxel_engine::terrain_compute::RiverPointGpu>,
        Vec<voxel_engine::terrain_compute::RiverHeaderGpu>,
    ) {
        let mut points = Vec::new();
        let mut headers = Vec::new();
        for river in &self.rivers {
            let start_idx = points.len() as u32;
            for (i, &(x, y)) in river.points.iter().enumerate() {
                let width = river.widths.get(i).copied().unwrap_or(1.0);
                points.push(voxel_engine::terrain_compute::RiverPointGpu {
                    x,
                    y,
                    width,
                    _pad: 0.0,
                });
            }
            headers.push(voxel_engine::terrain_compute::RiverHeaderGpu {
                start_idx,
                length: river.points.len() as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }
        (points, headers)
    }

    /// Bilinear interpolation of height at a voxel-space position.
    pub fn interpolate_height(&self, vx: f32, vy: f32) -> f32 {
        let cx = vx / CELL_SIZE as f32;
        let cy = vy / CELL_SIZE as f32;
        let col0 = cx.floor().clamp(0.0, (self.cols - 1) as f32) as usize;
        let row0 = cy.floor().clamp(0.0, (self.rows - 1) as f32) as usize;
        let col1 = (col0 + 1).min(self.cols - 1);
        let row1 = (row0 + 1).min(self.rows - 1);
        let tx = cx - cx.floor();
        let ty = cy - cy.floor();
        let h00 = self.get(col0, row0).height;
        let h10 = self.get(col1, row0).height;
        let h01 = self.get(col0, row1).height;
        let h11 = self.get(col1, row1).height;
        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        h0 + (h1 - h0) * ty
    }
}

// ---------------------------------------------------------------------------
// Noise helpers (wrap fbm_2d with the world_sim_cmd.rs signature convention)
// ---------------------------------------------------------------------------

/// fbm with lacunarity=2.03, gain=0.5, returns raw sum (not normalised).
/// Matches world_sim_cmd.rs fbm() behaviour for terrain classification.
fn fbm_raw(x: f32, y: f32, seed: u64, octaves: u32) -> f32 {
    let mut v = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f32;
    for i in 0..octaves {
        v += amp * noise::value_noise_2d(x * freq, y * freq, seed.wrapping_add(i as u64 * 7919), 1.0);
        amp *= 0.5;
        freq *= 2.03;
    }
    v
}

// ---------------------------------------------------------------------------
// Terrain / sub-biome classification
// ---------------------------------------------------------------------------

fn classify_terrain(
    elevation: f32,
    moisture: f32,
    temperature: f32,
    detail: f32,
) -> Terrain {
    // fbm_raw peaks at ~0.5, continent mask preserves that at center,
    // so practical elevation range is [0, ~0.5]. Thresholds scaled accordingly.
    if elevation > 0.42 {
        if temperature < 0.3 {
            return Terrain::Glacier;
        }
        if detail > 0.85 {
            return Terrain::FlyingIslands;
        }
        return Terrain::Mountains;
    }
    if elevation < 0.12 {
        if moisture > 0.6 {
            return Terrain::DeepOcean;
        }
        if detail > 0.7 {
            return Terrain::CoralReef;
        }
        return Terrain::Coast;
    }

    match (temperature > 0.5, moisture > 0.5) {
        (true, true) => {
            if moisture > 0.7 { Terrain::Swamp } else { Terrain::Jungle }
        }
        (true, false) => {
            if detail > 0.8 { Terrain::Volcano }
            else if moisture < 0.25 { Terrain::Desert }
            else { Terrain::Badlands }
        }
        (false, true) => {
            if temperature < 0.35 { Terrain::Tundra } else { Terrain::Forest }
        }
        (false, false) => {
            if detail > 0.85 { Terrain::AncientRuins }
            else if detail > 0.8 { Terrain::DeathZone }
            else if temperature < 0.3 { Terrain::Caverns }
            else { Terrain::Plains }
        }
    }
}

fn classify_sub_biome(terrain: Terrain, hash: u32) -> SubBiome {
    match terrain {
        Terrain::Forest => match hash % 10 {
            0..=3 => SubBiome::LightForest,
            4..=6 => SubBiome::DenseForest,
            _ => SubBiome::AncientForest,
        },
        Terrain::Desert => {
            if hash % 3 == 0 { SubBiome::SandDunes } else { SubBiome::RockyDesert }
        }
        Terrain::Mountains => {
            if hash % 5 == 0 { SubBiome::HotSprings } else { SubBiome::Standard }
        }
        Terrain::Swamp => {
            if hash % 4 == 0 { SubBiome::GlowingMarsh } else { SubBiome::Standard }
        }
        Terrain::Jungle => {
            if hash % 3 == 0 { SubBiome::TempleJungle } else { SubBiome::Standard }
        }
        _ => SubBiome::Standard,
    }
}

// ---------------------------------------------------------------------------
// Simple deterministic LCG for non-simulation RNG (placement decisions).
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) ^ (*state >> 17)) as u32
}

fn lcg_f32(state: &mut u64) -> f32 {
    lcg_next(state) as f32 / u32::MAX as f32
}

// ---------------------------------------------------------------------------
// River tracing
// ---------------------------------------------------------------------------

fn trace_rivers(cells: &[RegionCell], cols: usize, rows: usize, seed: u64) -> Vec<RiverPath> {
    let mut rivers = Vec::new();
    let mut rng = seed.wrapping_add(0xdead_beef_cafe_babe);

    // Start one river per 300 cells (approx), from high-elevation cells.
    let target_rivers = ((cols * rows) / 150).max(1);

    // Collect high-elevation cell indices sorted by height descending.
    // Lower threshold to 0.45 so rivers still generate even on smaller grids.
    let mut candidates: Vec<(usize, f32)> = cells
        .iter()
        .enumerate()
        .filter(|(_, c)| c.height > 0.45 && !matches!(c.terrain, Terrain::DeepOcean | Terrain::Coast | Terrain::CoralReef))
        .map(|(i, c)| (i, c.height))
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Use every Nth candidate to spread river sources.
    let stride = (candidates.len() / target_rivers.max(1)).max(1);

    for k in 0..target_rivers {
        let start_idx = k * stride;
        if start_idx >= candidates.len() {
            break;
        }
        let (ci, _) = candidates[start_idx];
        let mut col = ci % cols;
        let mut row = ci / cols;

        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut widths: Vec<f32> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let width = 1.0f32;

        for step in 0..200 {
            if visited.contains(&(col, row)) {
                break;
            }
            visited.insert((col, row));

            let cx = (col as f32 + 0.5) * CELL_SIZE as f32;
            let cy = (row as f32 + 0.5) * CELL_SIZE as f32;
            points.push((cx, cy));
            widths.push(width + step as f32 * 0.5);

            // Check if we've reached ocean/deep water — done.
            let terrain = cells[row * cols + col].terrain;
            if matches!(terrain, Terrain::DeepOcean | Terrain::Coast | Terrain::CoralReef) {
                break;
            }

            // Find neighbour with lowest height (steepest descent).
            let mut best_col = col;
            let mut best_row = row;
            let mut best_h = cells[row * cols + col].height;

            let jitter_seed = rng;
            let _ = lcg_next(&mut rng);

            for &(dc, dr) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc < 0 || nr < 0 || nc >= cols as i32 || nr >= rows as i32 {
                    continue;
                }
                let nc = nc as usize;
                let nr = nr as usize;
                if visited.contains(&(nc, nr)) {
                    continue;
                }
                let nh = cells[nr * cols + nc].height
                    + noise::hash_f32(nc as i32, nr as i32, step as i32, jitter_seed) * 0.02;
                if nh < best_h {
                    best_h = nh;
                    best_col = nc;
                    best_row = nr;
                }
            }

            if best_col == col && best_row == row {
                // Flat area — random walk.
                let dir = lcg_next(&mut rng) % 4;
                let (dc, dr): (i32, i32) = match dir {
                    0 => (-1, 0),
                    1 => (1, 0),
                    2 => (0, -1),
                    _ => (0, 1),
                };
                let nc = (col as i32 + dc).clamp(0, cols as i32 - 1) as usize;
                let nr = (row as i32 + dr).clamp(0, rows as i32 - 1) as usize;
                col = nc;
                row = nr;
            } else {
                col = best_col;
                row = best_row;
            }
        }

        if points.len() >= 2 {
            rivers.push(RiverPath { points, widths });
        }
    }

    rivers
}

// ---------------------------------------------------------------------------
// Settlement placement
// ---------------------------------------------------------------------------

fn place_settlements(cells: &mut Vec<RegionCell>, cols: usize, rows: usize, seed: u64) {
    let mut rng = seed.wrapping_add(0x5e77_1e_c0de);

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let terrain = cells[idx].terrain;

            if !terrain.is_settleable() {
                continue;
            }
            // Skip cells below sea level (height < 0.25 → surface below SEA_LEVEL).
            if cells[idx].height < 0.25 {
                continue;
            }

            // Deterministic hash for this cell.
            let cell_hash = noise::hash_3d(col as i32, row as i32, 0, seed);
            // ~8% chance.
            if cell_hash % 100 >= 8 {
                continue;
            }

            let kind = match terrain {
                Terrain::AncientRuins => SettlementKind::Ruin,
                Terrain::Mountains | Terrain::Glacier => SettlementKind::Castle,
                Terrain::Badlands | Terrain::DeathZone | Terrain::Caverns => SettlementKind::Camp,
                _ => {
                    let v = lcg_next(&mut rng) % 3;
                    match v {
                        0 => SettlementKind::Town,
                        1 => SettlementKind::Castle,
                        _ => SettlementKind::Camp,
                    }
                }
            };

            let local_x = lcg_f32(&mut rng);
            let local_y = lcg_f32(&mut rng);

            cells[idx].settlement = Some(SettlementPlan {
                kind,
                local_pos: (local_x, local_y),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Road building
// ---------------------------------------------------------------------------

fn build_roads(cells: &mut Vec<RegionCell>, cols: usize, rows: usize, roads: &mut Vec<RoadSegment>, seed: u64) {
    // Collect settlement positions.
    let settlements: Vec<(usize, usize)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (c, r)))
        .filter(|(c, r)| cells[r * cols + c].settlement.is_some())
        .collect();

    // Connect pairs within 5 cells.
    for i in 0..settlements.len() {
        for j in (i + 1)..settlements.len() {
            let (c0, r0) = settlements[i];
            let (c1, r1) = settlements[j];
            let dc = (c0 as i32 - c1 as i32).unsigned_abs() as usize;
            let dr = (r0 as i32 - r1 as i32).unsigned_abs() as usize;
            if dc + dr > 5 {
                continue;
            }

            let from = ((c0 as f32 + 0.5) * CELL_SIZE as f32, (r0 as f32 + 0.5) * CELL_SIZE as f32);
            let to   = ((c1 as f32 + 0.5) * CELL_SIZE as f32, (r1 as f32 + 0.5) * CELL_SIZE as f32);
            roads.push(RoadSegment { from, to });

            // Mark cells along the road (Bresenham-style).
            let steps = dc.max(dr).max(1);
            let _ = seed; // not needed here, determinism is structural
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let rc = (c0 as f32 + (c1 as f32 - c0 as f32) * t).round() as usize;
                let rr = (r0 as f32 + (r1 as f32 - r0 as f32) * t).round() as usize;
                if rc < cols && rr < rows {
                    cells[rr * cols + rc].has_road = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dungeon placement
// ---------------------------------------------------------------------------

fn place_dungeons(cells: &mut Vec<RegionCell>, cols: usize, rows: usize, seed: u64) {
    let mut rng = seed.wrapping_add(0xdead_c0de_f00d);

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let terrain = cells[idx].terrain;

            let chance_pct: u32 = match terrain {
                Terrain::Caverns => 30,
                Terrain::AncientRuins => 40,
                Terrain::Mountains => 5,
                _ => 2,
            };

            let cell_hash = noise::hash_3d(col as i32, row as i32, 1, seed);
            if cell_hash % 100 >= chance_pct {
                continue;
            }

            let local_x = lcg_f32(&mut rng);
            let local_y = lcg_f32(&mut rng);
            let depth_raw = lcg_next(&mut rng) % 8 + 1;
            let depth = depth_raw as u8;

            cells[idx].dungeons.push(DungeonPlan { local_pos: (local_x, local_y), depth });
        }
    }
}

// ---------------------------------------------------------------------------
// Continental generation (public entry point)
// ---------------------------------------------------------------------------

/// Generate a full region plan from noise.
pub fn generate_continent(cols: usize, rows: usize, seed: u64) -> RegionPlan {
    assert!(cols >= 2 && rows >= 2, "need at least 2×2 grid");

    let half_cols = cols as f32 * 0.5;
    let half_rows = rows as f32 * 0.5;
    let max_dist = (half_cols * half_cols + half_rows * half_rows).sqrt();

    // World-scale frequency: we want ~4-6 noise wavelengths across the grid
    // for good biome variety. Scale adapts to grid size.
    let max_dim = cols.max(rows) as f32;
    let scale = 4.0 / max_dim; // ~4 wavelengths across the largest dimension

    // Phase 1: noise fields + continent mask.
    let mut cells: Vec<RegionCell> = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                let wx = col as f32 * scale;
                let wy = row as f32 * scale;

                let elevation   = fbm_raw(wx, wy, seed.wrapping_add(77773), 4);
                let moisture    = fbm_raw(wx, wy, seed.wrapping_add(99991), 5);
                let temperature = fbm_raw(wx, wy, seed, 5);
                let detail      = fbm_raw(wx * 3.0, wy * 3.0, seed.wrapping_add(55537), 3);

                // Continent mask: fade to ocean near edges.
                // We bias elevation toward 0 at edges so ocean forms a border.
                // Formula: elevation = noise * mask_boost - (1 - mask) * edge_pull
                // where mask_boost amplifies land in the center and edge_pull
                // subtracts at the periphery, pushing sub-threshold cells to ocean.
                let dx = col as f32 - half_cols;
                let dy = row as f32 - half_rows;
                let dist = (dx * dx + dy * dy).sqrt();
                // mask in [0, 1]: 1 at center, 0 at edge.
                let mask = (1.0 - (dist / max_dist).powf(1.4)).max(0.0);
                // Multiply raw noise by mask, then subtract a large edge penalty.
                // At center (mask=1): elevation = noise * 1.0 - 0 → full noise.
                // At 80% dist (mask≈0.3): elevation = noise * 0.3 - 0.7 * 0.3 ≈ very low.
                let edge_pull = 1.0 - mask;
                let elevation = (elevation * mask - edge_pull * 0.4).clamp(0.0, 1.0);

                // Phase 2: classify terrain.
                let terrain = classify_terrain(elevation, moisture, temperature, detail);

                // Phase 3: sub-biome.
                let hash = noise::hash_3d(col as i32, row as i32, 42, seed);
                let sub_biome = classify_sub_biome(terrain, hash);

                RegionCell {
                    height: elevation,
                    moisture,
                    temperature,
                    terrain,
                    sub_biome,
                    settlement: None,
                    dungeons: Vec::new(),
                    has_road: false,
                }
            })
        })
        .collect();

    // Phase 4: trace rivers.
    let rivers = trace_rivers(&cells, cols, rows, seed);

    // Phase 5: place settlements.
    place_settlements(&mut cells, cols, rows, seed);

    // Phase 6: build roads.
    let mut roads = Vec::new();
    build_roads(&mut cells, cols, rows, &mut roads, seed);

    // Phase 7: place dungeons.
    place_dungeons(&mut cells, cols, rows, seed);

    RegionPlan { cols, rows, cells, rivers, roads, seed }
}

// ---------------------------------------------------------------------------
// Small-world preset
// ---------------------------------------------------------------------------

/// Create a minimal RegionPlan for the `--world small` test scene.
pub fn create_small_world_plan(seed: u64, _center: (f32, f32)) -> RegionPlan {
    use crate::world_sim::state::{Terrain, SubBiome};

    let cell = RegionCell {
        height: 0.3,
        moisture: 0.6,
        temperature: 0.5,
        terrain: Terrain::Forest,
        sub_biome: SubBiome::LightForest,
        settlement: Some(SettlementPlan {
            kind: SettlementKind::Town,
            local_pos: (0.5, 0.5),
        }),
        dungeons: vec![],
        has_road: false,
    };

    RegionPlan {
        cols: 1,
        rows: 1,
        cells: vec![cell],
        rivers: vec![],
        roads: vec![],
        seed,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            assert_eq!(a.cells[i].terrain, b.cells[i].terrain);
            assert_eq!(a.cells[i].height, b.cells[i].height);
        }
    }

    #[test]
    fn plan_has_variety() {
        let plan = generate_continent(50, 30, 42);
        let mut terrain_set = std::collections::HashSet::new();
        for cell in &plan.cells {
            terrain_set.insert(std::mem::discriminant(&cell.terrain));
        }
        assert!(terrain_set.len() >= 5, "only {} terrain types", terrain_set.len());
    }

    #[test]
    fn plan_has_continent_shape() {
        let plan = generate_continent(50, 30, 42);
        let land = plan.cells.iter().filter(|c| {
            c.terrain != Terrain::DeepOcean && c.terrain != Terrain::Coast
        }).count();
        let total = plan.cells.len();
        let land_pct = land as f32 / total as f32;
        assert!(land_pct > 0.3 && land_pct < 0.85, "land percentage: {land_pct}");
    }

    #[test]
    fn plan_has_rivers() {
        let plan = generate_continent(50, 30, 42);
        assert!(!plan.rivers.is_empty(), "no rivers generated");
        for river in &plan.rivers {
            assert!(river.points.len() >= 2);
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
        let plan = generate_continent(50, 30, 42);
        for cell in &plan.cells {
            if cell.height > 0.45 {
                assert!(
                    matches!(cell.terrain, Terrain::Mountains | Terrain::Glacier | Terrain::FlyingIslands | Terrain::Volcano),
                    "high elevation ({}) got {:?}", cell.height, cell.terrain
                );
            }
        }
    }
}
